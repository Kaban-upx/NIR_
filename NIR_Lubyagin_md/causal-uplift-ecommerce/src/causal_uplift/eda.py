"""Разведочный анализ данных (W3, SPEC §5.2).

Назначение модуля — не «посмотреть на данные», а зафиксировать в отчёте те
факты, от которых зависят решения следующих слоёв:

* **фактическое соотношение ``T=1`` / ``T=0``** — критерий приёмки 2; именно
  оно определяет, нужен ли режим дефицита §5.1 и можно ли доверять
  propensity-весам;
* наблюдаемая разность отклика ``Δ_corr`` — величина, которую причинный слой
  обязан скорректировать (§6.3);
* профиль пропусков и выбросов — обоснование импутации и зажатия квантилей;
* статистика карантина ETL-1 — сколько строк отброшено и почему.

Все числа берутся из уже собранных артефактов: EDA не пересчитывает ETL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.config import Config
from causal_uplift.etl.aggregate import load_transactions, resolve_window
from causal_uplift.etl.cleaning import ObservationWindow
from causal_uplift.features.build import feature_store_covariates, load_feature_store
from causal_uplift.io_utils import df_to_markdown, fmt_float, fmt_pct, save_json, write_markdown
from causal_uplift.logging_setup import get_logger
from causal_uplift.splitting import SPLIT_COLUMN

logger = get_logger(__name__)

EDA_REPORT = "eda_summary.md"
EDA_ARTIFACT = "eda_summary.json"

#: Сколько признаков показывать в таблицах «топ-N»
TOP_N = 15


@dataclass
class TreatmentBalance:
    """Фактическое соотношение групп и наблюдаемая разность отклика."""

    n_total: int
    n_treated: int
    n_control: int
    share_treated: float
    response_treated: float
    response_control: float
    delta_corr: float
    ratio_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "n_total": self.n_total,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "share_treated": round(self.share_treated, 6),
            "ratio": self.ratio_text,
            "response_rate_treated": round(self.response_treated, 6),
            "response_rate_control": round(self.response_control, 6),
            "delta_corr": round(self.delta_corr, 6),
        }


@dataclass
class EdaResult:
    """Сводка EDA: и в Markdown для ВКР, и в JSON для журнала прогона."""

    balance: TreatmentBalance
    window: ObservationWindow
    n_covariates: int
    missing: pd.DataFrame
    outliers: pd.DataFrame
    describe: pd.DataFrame
    quarantine: pd.DataFrame
    transactions_summary: dict[str, object]
    report_path: Path | None = None
    figures: dict[str, Path] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "balance": self.balance.as_dict(),
            "window": self.window.as_dict(),
            "n_covariates": self.n_covariates,
            "transactions": self.transactions_summary,
            "features_with_missing": int((self.missing["n_missing"] > 0).sum())
            if not self.missing.empty
            else 0,
            "quarantine_rows": int(self.quarantine["rows"].sum())
            if not self.quarantine.empty
            else 0,
            "figures": {k: str(v) for k, v in self.figures.items()},
        }


# ---------------------------------------------------------------------------
# Расчёты
# ---------------------------------------------------------------------------


def treatment_balance(df: pd.DataFrame, *, outcome: str = "target") -> TreatmentBalance:
    """Соотношение ``T=1``/``T=0`` и наблюдаемая разность исхода (критерий 2)."""
    t = df["treatment"].to_numpy()
    y = df[outcome].to_numpy(dtype="float64")
    n_treated = int((t == 1).sum())
    n_control = int((t == 0).sum())
    n_total = n_treated + n_control
    if n_treated == 0 or n_control == 0:
        msg = (
            f"в выборке отсутствует одна из групп (T=1: {n_treated}, T=0: {n_control}) — "
            "uplift-постановка невозможна"
        )
        raise ValueError(msg)

    resp_t = float(y[t == 1].mean())
    resp_c = float(y[t == 0].mean())
    # Отношение приводим к виду «1 : k» от меньшей группы — так читается сразу,
    # дефицитная ли treatment-группа.
    if n_treated <= n_control:
        ratio = f"T=1 : T=0 = 1 : {n_control / n_treated:.2f}"
    else:
        ratio = f"T=1 : T=0 = {n_treated / n_control:.2f} : 1"

    return TreatmentBalance(
        n_total=n_total,
        n_treated=n_treated,
        n_control=n_control,
        share_treated=n_treated / n_total,
        response_treated=resp_t,
        response_control=resp_c,
        delta_corr=resp_t - resp_c,
        ratio_text=ratio,
    )


def missing_profile(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Профиль пропусков: сколько и в каких колонках."""
    rows = []
    for col in columns:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "n_missing": n_missing,
                "share_missing": n_missing / len(df) if len(df) else 0.0,
                "dtype": str(df[col].dtype),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_missing", ascending=False).reset_index(drop=True)
    return out


def outlier_profile(
    df: pd.DataFrame, columns: list[str], *, iqr_factor: float = 3.0
) -> pd.DataFrame:
    """Профиль выбросов по правилу Тьюки с расширенным множителем.

    Множитель 3·IQR, а не стандартные 1.5: в ретейле правый хвост чеков
    толстый по природе, и при 1.5 «выбросом» объявляется каждый десятый
    клиент, что делает отчёт бесполезным.
    """
    rows = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        n_out = int(((series < lo) | (series > hi)).sum())
        if n_out == 0:
            continue
        rows.append(
            {
                "feature": col,
                "n_outliers": n_out,
                "share_outliers": n_out / len(df) if len(df) else 0.0,
                "lower_bound": float(lo),
                "upper_bound": float(hi),
                "max": float(series.max()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["feature", "n_outliers", "share_outliers", "lower_bound", "upper_bound", "max"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("share_outliers", ascending=False)
        .reset_index(drop=True)
    )


def quarantine_profile(cfg: Config) -> pd.DataFrame:
    """Сводка карантина ETL-1: сколько строк и по какой причине отброшено."""
    path = cfg.resolve(cfg.data.quarantine_path)
    if not path.is_file():
        return pd.DataFrame(columns=["table", "reason", "rows"])
    try:
        quarantine = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.warning("карантинный файл %s не читается как CSV", path)
        return pd.DataFrame(columns=["table", "reason", "rows"])
    if quarantine.empty or "_quarantine_reason" not in quarantine.columns:
        return pd.DataFrame(columns=["table", "reason", "rows"])
    group_cols = [c for c in ("_table", "_quarantine_reason") if c in quarantine.columns]
    out = (
        quarantine.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    renames = {"_table": "table", "_quarantine_reason": "reason"}
    out = out.rename(columns=renames)
    return out.reset_index(drop=True)


def transactions_summary(cfg: Config) -> dict[str, object]:
    """Короткая сводка по транзакционному кадру фазы A."""
    tx = load_transactions(cfg, columns=["client_id", "transaction_datetime", "purchase_sum"])
    ts = pd.to_datetime(tx["transaction_datetime"])
    return {
        "n_transactions": int(len(tx)),
        "n_clients_with_transactions": int(tx["client_id"].nunique()),
        "transactions_per_client": float(len(tx) / max(tx["client_id"].nunique(), 1)),
        "datetime_min": ts.min().isoformat(),
        "datetime_max": ts.max().isoformat(),
        "purchase_sum_mean": float(tx["purchase_sum"].mean()),
        "purchase_sum_median": float(tx["purchase_sum"].median()),
        "purchase_sum_p99": float(tx["purchase_sum"].quantile(0.99)),
    }


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def plot_eda_figures(
    df: pd.DataFrame, covariates: list[str], cfg: Config, *, outcome: str = "target"
) -> dict[str, Path]:
    """Базовые графики EDA: баланс групп, отклик, распределения признаков."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, Path] = {}
    reports = cfg.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    # 1. Баланс групп и отклик по группам
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["treatment"].value_counts().sort_index()
    axes[0].bar(["T=0", "T=1"], [counts.get(0, 0), counts.get(1, 0)], color=["#8fa8c8", "#c87b7b"])
    axes[0].set_title("Размеры групп")
    axes[0].set_ylabel("клиентов")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(i, v, f"{v:,}".replace(",", " "), ha="center", va="bottom", fontsize=9)

    rates = df.groupby("treatment", observed=True)[outcome].mean()
    axes[1].bar(
        ["T=0", "T=1"], [rates.get(0, np.nan), rates.get(1, np.nan)], color=["#8fa8c8", "#c87b7b"]
    )
    axes[1].set_title(f"Средний отклик ({outcome}) по группам")
    delta = float(rates.get(1, np.nan) - rates.get(0, np.nan))
    axes[1].set_xlabel(f"наблюдаемая разность Δ_corr = {delta:+.4f}")
    fig.tight_layout()
    path = reports / "eda_treatment_balance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["treatment_balance"] = path

    # 2. Распределения признаков с наибольшей дисперсией между группами:
    #    именно они — кандидаты в конфаундеры, и смотреть на них полезнее,
    #    чем на первые по алфавиту.
    treated = df.loc[df["treatment"] == 1, covariates]
    control = df.loc[df["treatment"] == 0, covariates]
    pooled_std = df[covariates].std(ddof=0).replace(0.0, np.nan)
    smd = ((treated.mean() - control.mean()) / pooled_std).abs().sort_values(ascending=False)
    top = [c for c in smd.head(9).index if c in df.columns]

    if top:
        n = len(top)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
        flat = np.atleast_1d(axes).ravel()
        for ax, col in zip(flat, top, strict=False):
            ax.hist(control[col].to_numpy(), bins=40, alpha=0.6, label="T=0", density=True)
            ax.hist(treated[col].to_numpy(), bins=40, alpha=0.6, label="T=1", density=True)
            ax.set_title(f"{col}\nSMD = {smd[col]:.3f}", fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in flat[len(top) :]:
            ax.axis("off")
        flat[0].legend(fontsize=8)
        fig.suptitle("Признаки с наибольшим стандартизованным расхождением групп", fontsize=11)
        fig.tight_layout()
        path = reports / "eda_covariate_shift.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures["covariate_shift"] = path

    return figures


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


def render_report(result: EdaResult, cfg: Config) -> list[str]:
    """Markdown-отчёт ``reports/eda_summary.md``."""
    b = result.balance
    # При outcome=revenue «отклик» измеряется в деньгах, а не в долях —
    # процентное форматирование здесь было бы просто неверным.
    rate = fmt_pct if cfg.data.outcome == "target" else fmt_float
    lines = [
        "# Разведочный анализ данных (EDA)",
        "",
        f"Источник данных: `{cfg.data.source}`; исход: `{cfg.data.outcome}`.",
        "",
        "## 1. Окно наблюдения",
        "",
        f"* окно признаков: `{result.window.feature_start.date()}` … "
        f"`{result.window.feature_end.date()}` ({result.window.feature_days} дн.)",
        f"* окно измерения исхода: `{result.window.feature_end.date()}` … "
        f"`{result.window.outcome_end.date()}` ({result.window.outcome_days} дн.)",
        "",
        "Разделение окон жёсткое: признак, посчитанный по транзакции из окна исхода,",
        "был бы прямой утечкой целевой переменной.",
        "",
        "## 2. Фактическое соотношение групп (критерий приёмки 2)",
        "",
        f"**{b.ratio_text}**",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Всего клиентов | {b.n_total:,} |".replace(",", " "),
        f"| Группа воздействия `T=1` | {b.n_treated:,} |".replace(",", " "),
        f"| Контрольная группа `T=0` | {b.n_control:,} |".replace(",", " "),
        f"| Доля `T=1` | {fmt_pct(b.share_treated)} |",
        f"| Отклик при `T=1` | {rate(b.response_treated)} |",
        f"| Отклик при `T=0` | {rate(b.response_control)} |",
        f"| Наблюдаемая разность `Δ_corr` | {fmt_float(b.delta_corr)} |",
        "",
        "`Δ_corr` — **смещённая** оценка: группы различаются по ковариатам, и часть",
        "разности объясняется отбором, а не воздействием. Её скорректированный",
        "аналог считает причинный слой (`reports/ate_summary.md`), а разность",
        "двух величин и есть измеренное смещение конфаундинга.",
        "",
        "## 3. Транзакционный слой",
        "",
        "| Показатель | Значение |",
        "|---|---|",
    ]
    tx = result.transactions_summary
    labels = {
        "n_transactions": "Транзакций (чеков) в окне признаков",
        "n_clients_with_transactions": "Клиентов с транзакциями",
        "transactions_per_client": "Транзакций на клиента",
        "datetime_min": "Первая транзакция",
        "datetime_max": "Последняя транзакция",
        "purchase_sum_mean": "Средний чек",
        "purchase_sum_median": "Медианный чек",
        "purchase_sum_p99": "99-й процентиль чека",
    }
    for key, label in labels.items():
        value = tx.get(key)
        rendered = fmt_float(value) if isinstance(value, float) else f"{value}"
        lines.append(f"| {label} | {rendered} |")

    lines += [
        "",
        "## 4. Профиль пропусков",
        "",
    ]
    missing_nonzero = result.missing.loc[result.missing["n_missing"] > 0]
    if missing_nonzero.empty:
        lines += [
            "Пропусков в витрине нет: импутация выполнена на этапе сборки признаков",
            "(для `age` и `first_redeem_date` добавлены флаги-индикаторы, так что",
            "сам факт отсутствия значения доступен модели как признак).",
        ]
    else:
        lines.append(df_to_markdown(missing_nonzero.head(TOP_N), floatfmt="{:.4f}"))

    lines += [
        "",
        "## 5. Профиль выбросов (правило 3·IQR)",
        "",
    ]
    if result.outliers.empty:
        lines.append("Признаков с выбросами по правилу 3·IQR не обнаружено.")
    else:
        lines += [
            f"Показаны {min(TOP_N, len(result.outliers))} признаков с наибольшей долей выбросов"
            f" из {len(result.outliers)}.",
            "",
            df_to_markdown(result.outliers.head(TOP_N), floatfmt="{:.4f}"),
            "",
            "Выбросы не удаляются: границы зажатия считаются по квантилям",
            f"`{cfg.features.clip_quantiles}` **на train-подвыборке** и применяются",
            "ко всей витрине (`artifacts/clip_bounds.json`).",
        ]

    lines += [
        "",
        "## 6. Карантин невалидных строк (ETL-1)",
        "",
    ]
    if result.quarantine.empty:
        lines.append("Невалидных строк не обнаружено — карантин пуст.")
    else:
        lines += [
            df_to_markdown(result.quarantine.head(TOP_N)),
            "",
            f"Полный список: `{cfg.data.quarantine_path}`. Строки не отбрасываются молча:"
            " каждая сохраняется вместе с причиной отклонения.",
        ]

    lines += [
        "",
        "## 7. Описательные статистики ковариат",
        "",
        f"Всего ковариат: **{result.n_covariates}** (контракт ETL-6 требует"
        f" не менее {cfg.features.min_covariates}).",
        "",
        f"Первые {min(TOP_N, len(result.describe))} признаков:",
        "",
        df_to_markdown(result.describe.head(TOP_N), floatfmt="{:.4f}", index=True),
    ]

    if result.figures:
        lines += ["", "## 8. Графики", ""]
        for name, path in result.figures.items():
            lines.append(f"* `{name}` — `{path.name}`")

    return lines


def run_eda(cfg: Config, *, write: bool = True, plots: bool = True) -> EdaResult:
    """Выполнить EDA по собранной витрине и записать отчёт (W3)."""
    df = load_feature_store(cfg)
    covariates = feature_store_covariates(df)
    window = resolve_window(cfg)

    balance = treatment_balance(df, outcome=cfg.data.outcome)
    missing = missing_profile(df, covariates)
    outliers = outlier_profile(df, covariates)
    quarantine = quarantine_profile(cfg)
    tx_summary = transactions_summary(cfg)

    describe = (
        df[covariates]
        .describe()
        .T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .sort_index()
    )

    figures = plot_eda_figures(df, covariates, cfg, outcome=cfg.data.outcome) if plots else {}

    result = EdaResult(
        balance=balance,
        window=window,
        n_covariates=len(covariates),
        missing=missing,
        outliers=outliers,
        describe=describe,
        quarantine=quarantine,
        transactions_summary=tx_summary,
        figures=figures,
    )

    if write:
        result.report_path = write_markdown(render_report(result, cfg), cfg.report(EDA_REPORT))
        save_json(result.as_dict(), cfg.artifact(EDA_ARTIFACT))
        logger.info("EDA-отчёт: %s", result.report_path)

    logger.info(
        "EDA: %s, отклик %.4f vs %.4f, Δ_corr = %+.4f",
        balance.ratio_text,
        balance.response_treated,
        balance.response_control,
        balance.delta_corr,
    )
    return result


def split_balance_table(df: pd.DataFrame, *, outcome: str = "target") -> pd.DataFrame:
    """Доли воздействия и отклика по частям сплита — проверка стратификации."""
    if SPLIT_COLUMN not in df.columns:
        msg = f"в витрине нет колонки {SPLIT_COLUMN!r}"
        raise KeyError(msg)
    grouped = df.groupby(SPLIT_COLUMN, observed=True)
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "share_treated": grouped["treatment"].mean(),
            f"{outcome}_rate": grouped[outcome].mean(),
        }
    ).reset_index()


__all__ = [
    "EDA_ARTIFACT",
    "EDA_REPORT",
    "EdaResult",
    "TreatmentBalance",
    "missing_profile",
    "outlier_profile",
    "plot_eda_figures",
    "quarantine_profile",
    "render_report",
    "run_eda",
    "split_balance_table",
    "transactions_summary",
    "treatment_balance",
]
