"""Фазы B и C ETL: сборка витрины признаков (W4, SPEC §5.3).

Порядок шагов зафиксирован и важен именно в этом виде:

1. агрегаты уровня клиента (RFM + поведенческие блоки) по транзакциям из
   ``[feature_start, feature_end)``;
2. присоединение ``uplift_train`` (``treatment``, ``target``) и исхода;
3. режим дисбаланса (``data.treatment_share``) — **до** сплита, иначе доли
   воздействия в train/valid/test разъедутся;
4. стратифицированный сплит ``(T, Y)`` → колонка ``split``;
5. кодирование категорий, зажатие выбросов и Z-score — **все параметры
   считаются только на train** (ETL-4, ETL-5);
6. фильтр мультиколлинеарности по train-подвыборке;
7. проверка контракта ETL-6 и запись ``data/processed/feature_store.parquet``.

Шаги 5–6 после сплита — не стилистическое решение: любой другой порядок
переносит информацию о тесте в обучение, и leakage-тест §11 это ловит.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.config import Config
from causal_uplift.etl.aggregate import (
    load_assortment,
    load_outcome_revenue,
    load_transactions,
    resolve_window,
)
from causal_uplift.etl.cleaning import ImbalanceReport, ObservationWindow, apply_treatment_share
from causal_uplift.etl.loaders import load_table
from causal_uplift.etl.schema import FEATURE_STORE_RESERVED, FeatureStoreContract, covariate_columns
from causal_uplift.etl.synthetic import load_truth
from causal_uplift.features.behavioral import build_behavioral_block
from causal_uplift.features.correlation import (
    CorrelationReport,
    save_correlation_artifacts,
    select_decorrelated,
)
from causal_uplift.features.encoding import CategoricalEncoder
from causal_uplift.features.rfm import build_rfm_features
from causal_uplift.features.scaling import ZScoreScaler, make_scaler
from causal_uplift.io_utils import save_json, save_parquet
from causal_uplift.logging_setup import get_logger, log_stage
from causal_uplift.splitting import SPLIT_COLUMN, SplitSummary, assign_splits

logger = get_logger(__name__)

#: Имена артефактов фазы C
SCALER_ARTIFACT = "scaler.json"
ENCODER_ARTIFACT = "encoders.json"
CLIP_ARTIFACT = "clip_bounds.json"
FEATURE_META_ARTIFACT = "feature_meta.json"


@dataclass
class FeatureBuildResult:
    """Что получилось на выходе фазы C — попадает в журнал прогона (ML-5)."""

    path: Path
    n_clients: int
    covariates: list[str]
    window: ObservationWindow
    imbalance: ImbalanceReport
    split: SplitSummary
    correlation: CorrelationReport
    block_sizes: dict[str, int] = field(default_factory=dict)
    clients_without_transactions: int = 0
    constant_dropped: list[str] = field(default_factory=list)
    outcome: str = "target"
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def n_covariates(self) -> int:
        return len(self.covariates)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_store": str(self.path),
            "n_clients": self.n_clients,
            "n_covariates": self.n_covariates,
            "outcome": self.outcome,
            "clients_without_transactions": self.clients_without_transactions,
            "constant_dropped": self.constant_dropped,
            "block_sizes": self.block_sizes,
            "window": self.window.as_dict(),
            "imbalance": self.imbalance.as_dict(),
            "split": self.split.as_dict(),
            "multicollinearity": {
                "threshold": self.correlation.threshold,
                "method": self.correlation.method,
                "n_dropped": len(self.correlation.dropped),
                "dropped": self.correlation.dropped_names,
            },
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
        }


# ---------------------------------------------------------------------------
# Фаза B: клиентские агрегаты
# ---------------------------------------------------------------------------


def build_client_features(
    cfg: Config, window: ObservationWindow
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Собрать все блоки признаков уровня клиента.

    Returns:
        Кадр, индексированный ``client_id``, и размеры блоков для отчёта.
    """
    transactions = load_transactions(cfg)
    assortment = load_assortment(cfg)
    clients, _ = load_table(cfg, "clients")

    rfm = build_rfm_features(transactions, window)
    behavior = build_behavioral_block(transactions, assortment, clients, rfm, window)

    features = rfm.join(behavior, how="outer")
    block_sizes = {
        "rfm": rfm.shape[1],
        "behavioral": behavior.shape[1],
        "assortment_raw": assortment.shape[1],
    }
    logger.info("фаза B: %d признаков на %d клиентов", features.shape[1], len(features))
    return features, block_sizes


def attach_labels(
    cfg: Config, features: pd.DataFrame, *, outcome: str | None = None
) -> tuple[pd.DataFrame, int]:
    """Присоединить ``treatment``/``target``/``revenue`` к клиентским признакам.

    Ведущая таблица — ``uplift_train``: воздействие известно только для её
    клиентов. Клиент без транзакций в окне признаков не отбрасывается молча —
    его признаки заполняются нейтральными нулями, а сам факт отсутствия
    истории кодируется признаком ``has_transactions`` (профиль пропусков
    фиксируется в EDA, W3).
    """
    labels, _ = load_table(cfg, "uplift_train")
    labels = labels.set_index("client_id")

    aligned = features.reindex(labels.index)
    # Признак наличия истории определяется по транзакционному счётчику, а не по
    # «любая колонка заполнена»: соц-дем приходит из clients и есть у всех.
    anchor = "n_transactions_total" if "n_transactions_total" in aligned.columns else None
    has_tx = (
        aligned[anchor].notna() & aligned[anchor].gt(0)
        if anchor
        else aligned.notna().any(axis=1)
    )
    n_missing = int((~has_tx).sum())

    numeric_cols = aligned.select_dtypes(include=[np.number]).columns
    aligned[numeric_cols] = aligned[numeric_cols].fillna(0.0)
    aligned["has_transactions"] = has_tx.astype("float32")

    out = aligned.copy()
    # В RetailHero флаг воздействия называется treatment_flg; внутреннее имя —
    # treatment (оно же в DAG и в контракте витрины ETL-6).
    out["treatment"] = labels["treatment_flg"].astype("int8")
    out["target"] = labels["target"].astype("int8")

    # revenue пишется всегда: при outcome=target это дополнительная метрика
    # для экономического слоя, при outcome=revenue — целевая переменная.
    revenue = load_outcome_revenue(cfg)
    out["revenue"] = revenue.reindex(out.index).fillna(0.0).astype("float32")

    resolved_outcome = outcome or cfg.data.outcome
    if resolved_outcome == "revenue" and not revenue.notna().any():
        logger.warning(
            "outcome=revenue, но в окне исхода не найдено ни одного чека: "
            "проверьте data.outcome_window_days"
        )

    truth = load_truth(cfg)
    if truth is not None:
        # Синтетика: ground truth переносится в витрину для sanity-теста §11
        truth = truth.set_index("client_id")
        for col in ("true_tau", "true_mu0"):
            if col in truth.columns:
                out[col] = truth[col].reindex(out.index).astype("float32")
        logger.info("присоединён ground truth синтетики (true_tau, true_mu0)")

    out = out.reset_index()
    if n_missing:
        logger.warning(
            "%d из %d клиентов uplift_train не имеют транзакций в окне признаков "
            "(признаки = 0, флаг has_transactions = 0)",
            n_missing,
            len(out),
        )
    return out, n_missing


# ---------------------------------------------------------------------------
# Фаза C: кодирование, нормализация, фильтры
# ---------------------------------------------------------------------------


def fit_clip_bounds(
    train: pd.DataFrame, columns: Sequence[str], quantiles: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    """Границы зажатия выбросов по train-подвыборке (ETL-4: без утечки)."""
    lower, upper = quantiles
    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        if col not in train.columns or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        lo, hi = train[col].quantile([lower, upper])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            bounds[col] = (float(lo), float(hi))
    return bounds


def apply_clip_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Применить train-границы ко всей витрине."""
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lo, hi)
    return out


def _guard_min_covariates(
    report: CorrelationReport, *, min_covariates: int
) -> CorrelationReport:
    """Не дать фильтру корреляций нарушить критерий приёмки 3.

    Если после отбрасывания скоррелированных признаков их осталось меньше
    ``min_covariates``, часть возвращается — начиная с наименее
    скоррелированных. Витрина с 28 ковариатами формально корректна, но не
    удовлетворяет контракту ETL-6, а молча падать на последнем шаге хуже,
    чем вернуть несколько почти дублирующих признаков и сказать об этом.
    """
    if len(report.kept) >= min_covariates or not report.dropped:
        return report

    deficit = min_covariates - len(report.kept)
    restore = sorted(report.dropped, key=lambda d: float(d["mean_abs_corr"]))[:deficit]
    restore_names = {str(d["feature"]) for d in restore}
    logger.warning(
        "после фильтра |r| > %.2f осталось %d ковариат (< %d): возвращаю %s",
        report.threshold,
        len(report.kept),
        min_covariates,
        ", ".join(sorted(restore_names)),
    )
    kept = [c for c in report.matrix.columns if c in set(report.kept) | restore_names]
    dropped = [d for d in report.dropped if str(d["feature"]) not in restore_names]
    return CorrelationReport(
        matrix=report.matrix,
        dropped=dropped,
        kept=kept,
        threshold=report.threshold,
        method=report.method,
    )


def build_feature_store(
    cfg: Config,
    *,
    treatment_share: float | None | str = "config",
    write: bool = True,
) -> FeatureBuildResult:
    """Собрать и записать ``data/processed/feature_store.parquet``.

    Args:
        cfg: конфигурация прогона.
        treatment_share: режим дисбаланса; ``"config"`` — взять
            ``data.treatment_share``, иначе переопределить (используется
            экспериментом §5.1 для режима ``scarce_15``).
        write: записывать артефакты на диск (``False`` — для тестов).
    """
    cfg.ensure_dirs()
    window = resolve_window(cfg)

    with log_stage(logger, "фаза B: клиентские агрегаты"):
        features, block_sizes = build_client_features(cfg, window)

    with log_stage(logger, "фаза B: присоединение разметки"):
        df, n_missing = attach_labels(cfg, features)

    share = cfg.data.treatment_share if treatment_share == "config" else treatment_share
    with log_stage(logger, "режим дисбаланса"):
        df, imbalance = apply_treatment_share(
            df, share, seed=cfg.seed, stratify_on=("target",)
        )

    with log_stage(logger, "стратифицированный сплит"):
        df, split_summary = assign_splits(
            df,
            test_size=cfg.split.test_size,
            valid_size=cfg.split.valid_size,
            stratify_on=cfg.split.stratify_on,
            seed=cfg.seed,
        )

    train_mask = df[SPLIT_COLUMN] == "train"
    if not train_mask.any():
        msg = "train-подвыборка пуста: проверьте split.test_size / split.valid_size"
        raise ValueError(msg)

    # --- кодирование категорий (ETL-5): категории берутся с train -------
    with log_stage(logger, "кодирование категорий"):
        encoder = CategoricalEncoder().fit(
            df.loc[train_mask],
            one_hot=cfg.features.one_hot,
            frequency=cfg.features.high_cardinality,
        )
        df = encoder.transform(df)

    # Любые оставшиеся нечисловые ковариаты — ошибка контракта, а не повод
    # молча их выбросить: имя такой колонки попадает в сообщение.
    leftover = [
        c
        for c in covariate_columns(df, extra_exclude=(SPLIT_COLUMN,))
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if leftover:
        msg = (
            "нечисловые ковариаты остались незакодированными: "
            f"{', '.join(leftover)} — добавьте их в features.one_hot "
            "или features.high_cardinality"
        )
        raise ValueError(msg)

    covariates = covariate_columns(df, extra_exclude=(SPLIT_COLUMN,))

    # --- зажатие выбросов по train-квантилям ---------------------------
    clip_bounds: dict[str, tuple[float, float]] = {}
    if cfg.features.clip_quantiles is not None:
        with log_stage(logger, "зажатие выбросов"):
            clip_bounds = fit_clip_bounds(
                df.loc[train_mask], covariates, cfg.features.clip_quantiles
            )
            df = apply_clip_bounds(df, clip_bounds)

    # --- мультиколлинеарность (W4) -------------------------------------
    # Константные на train колонки убираем до матрицы корреляций: их
    # корреляция не определена (std = 0), и они всё равно не несут сигнала.
    constant = [
        c
        for c in covariates
        if float(df.loc[train_mask, c].std(ddof=0)) < 1e-12
        and c not in set(cfg.features.protected)
    ]
    if constant:
        logger.warning(
            "константные на train ковариаты исключены (%d): %s",
            len(constant),
            ", ".join(constant[:10]),
        )
        df = df.drop(columns=constant)
        covariates = [c for c in covariates if c not in set(constant)]

    with log_stage(logger, "фильтр мультиколлинеарности"):
        corr_report = select_decorrelated(
            df.loc[train_mask],
            covariates,
            threshold=cfg.features.corr_threshold,
            method=cfg.features.corr_method,
            protected=cfg.features.protected,
        )
        corr_report = _guard_min_covariates(
            corr_report, min_covariates=cfg.features.min_covariates
        )
        dropped = corr_report.dropped_names
        if dropped:
            df = df.drop(columns=dropped)
        covariates = corr_report.kept

    # --- Z-score (ETL-4): параметры только с train ---------------------
    scaler: ZScoreScaler = make_scaler(cfg.features.scaler)
    with log_stage(logger, "Z-score нормализация"):
        scaler.fit(df.loc[train_mask], covariates)
        df = scaler.transform(df)

    # --- контракт витрины (ETL-6) --------------------------------------
    contract = FeatureStoreContract(
        min_covariates=cfg.features.min_covariates, outcome=cfg.data.outcome
    )
    check_frame = df.drop(columns=[SPLIT_COLUMN])
    contract.assert_valid(check_frame)
    logger.info(
        "контракт ETL-6 выполнен: %d ковариат, %d клиентов", len(covariates), len(df)
    )

    artifacts: dict[str, Path] = {}
    path = cfg.feature_store_path
    if write:
        save_parquet(df, path)
        artifacts["feature_store"] = path
        artifacts["scaler"] = scaler.save(cfg.artifact(SCALER_ARTIFACT))
        artifacts["encoders"] = encoder.save(cfg.artifact(ENCODER_ARTIFACT))
        artifacts["clip_bounds"] = save_json(
            {
                "quantiles": list(cfg.features.clip_quantiles)
                if cfg.features.clip_quantiles
                else None,
                "bounds": {k: list(v) for k, v in clip_bounds.items()},
                "fitted_on_rows": int(train_mask.sum()),
            },
            cfg.artifact(CLIP_ARTIFACT),
        )
        corr_paths = save_correlation_artifacts(
            corr_report,
            png_path=cfg.report("corr_matrix.png"),
            json_path=cfg.report("dropped_features.json"),
            csv_path=cfg.report("corr_matrix.csv"),
        )
        artifacts["corr_matrix_png"] = corr_paths["png"]
        artifacts["dropped_features"] = corr_paths["json"]
        artifacts["corr_matrix_csv"] = corr_paths["csv"]

    result = FeatureBuildResult(
        path=path,
        n_clients=len(df),
        covariates=list(covariates),
        window=window,
        imbalance=imbalance,
        split=split_summary,
        correlation=corr_report,
        block_sizes={**block_sizes, "final_covariates": len(covariates)},
        clients_without_transactions=n_missing,
        constant_dropped=constant,
        outcome=cfg.data.outcome,
        artifacts=artifacts,
    )

    if write:
        artifacts["feature_meta"] = save_json(
            {
                **result.as_dict(),
                "covariates": list(covariates),
                "reserved_columns": list(FEATURE_STORE_RESERVED),
            },
            cfg.artifact(FEATURE_META_ARTIFACT),
        )
    return result


# ---------------------------------------------------------------------------
# Чтение витрины
# ---------------------------------------------------------------------------


def load_feature_store(cfg: Config) -> pd.DataFrame:
    """Прочитать витрину; понятная ошибка вместо ``FileNotFoundError`` pyarrow."""
    path = cfg.feature_store_path
    if not path.is_file():
        msg = f"не найдена витрина {path}: выполните `make features`"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path, engine="pyarrow")


def feature_store_covariates(df: pd.DataFrame) -> list[str]:
    """Ковариаты ``X`` витрины (служебные колонки и сплит исключены)."""
    return covariate_columns(df, extra_exclude=(SPLIT_COLUMN,))


__all__ = [
    "CLIP_ARTIFACT",
    "ENCODER_ARTIFACT",
    "FEATURE_META_ARTIFACT",
    "SCALER_ARTIFACT",
    "FeatureBuildResult",
    "apply_clip_bounds",
    "attach_labels",
    "build_client_features",
    "build_feature_store",
    "feature_store_covariates",
    "fit_clip_bounds",
    "load_feature_store",
]
