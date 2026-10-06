"""Очистка и приведение выборки: дедупликация, окна наблюдения, режим дисбаланса.

Реализует ETL-2 (дедупликация по ``(client_id, transaction_id, product_id)``),
ETL-3 (отсечение транзакций вне окна наблюдения) и стратифицированный
даунсэмплинг treatment-группы из SPEC §5.1 — тот самый явный параметр
``data.treatment_share``, которым воспроизводится заявленный в НИР дисбаланс
15% : 85%.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Дедупликация (ETL-2)
# ---------------------------------------------------------------------------


class CrossChunkDeduplicator:
    """Дедупликация по составному ключу, работающая поверх потока чанков.

    ``drop_duplicates`` внутри чанка не ловит дубль, разъехавшийся по границе
    чанков, поэтому хранится множество 64-битных хэшей уже увиденных ключей:
    это ~8 байт на строку вместо хранения самих ключей.
    """

    def __init__(self, key: Sequence[str]) -> None:
        self.key = list(key)
        self._seen: set[int] = set()
        self.dropped = 0
        self.total = 0

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        missing = [k for k in self.key if k not in df.columns]
        if missing:
            msg = f"для дедупликации нужны колонки {missing}, их нет в кадре"
            raise KeyError(msg)

        self.total += len(df)
        hashes = pd.util.hash_pandas_object(df[self.key], index=False).to_numpy(dtype="uint64")
        keep = np.fromiter(
            (h not in self._seen and (self._seen.add(h) or True) for h in hashes.tolist()),
            dtype=bool,
            count=hashes.size,
        )
        self.dropped += int((~keep).sum())
        return df.loc[keep]

    @property
    def stats(self) -> dict[str, int]:
        return {"rows_seen": self.total, "duplicates_dropped": self.dropped}


def deduplicate(df: pd.DataFrame, key: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Снять дубли по ключу внутри одного кадра (ETL-2)."""
    missing = [k for k in key if k not in df.columns]
    if missing:
        msg = f"для дедупликации нужны колонки {missing}"
        raise KeyError(msg)
    before = len(df)
    out = df.drop_duplicates(subset=list(key), keep="first")
    return out.reset_index(drop=True), before - len(out)


# ---------------------------------------------------------------------------
# Окна наблюдения (ETL-3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationWindow:
    """Окно наблюдения признаков и окно измерения исхода.

    * ``[feature_start, feature_end)`` — транзакции, по которым считаются
      ковариаты;
    * ``[feature_end, outcome_end)`` — окно, в котором измеряется ``Y``.

    Разделение жёсткое: признак, посчитанный по транзакции из окна исхода,
    был бы утечкой целевой переменной (риск §14).
    """

    feature_start: pd.Timestamp
    feature_end: pd.Timestamp
    outcome_end: pd.Timestamp

    @property
    def feature_days(self) -> int:
        return int((self.feature_end - self.feature_start).days)

    @property
    def outcome_days(self) -> int:
        return int((self.outcome_end - self.feature_end).days)

    def as_dict(self) -> dict[str, str | int]:
        return {
            "feature_start": self.feature_start.isoformat(),
            "feature_end": self.feature_end.isoformat(),
            "outcome_end": self.outcome_end.isoformat(),
            "observation_window_days": self.feature_days,
            "outcome_window_days": self.outcome_days,
        }


def build_window(
    data_end: pd.Timestamp,
    *,
    observation_window_days: int,
    outcome_window_days: int,
) -> ObservationWindow:
    """Построить окна от даты последнего события в данных.

    Конец окна признаков отодвигается от конца данных на ``outcome_window_days``:
    последние дни датасета — это и есть окно, в котором мог реализоваться исход.
    """
    feature_end = (data_end - pd.Timedelta(days=outcome_window_days)).normalize() + pd.Timedelta(
        days=1
    )
    feature_start = feature_end - pd.Timedelta(days=observation_window_days)
    outcome_end = feature_end + pd.Timedelta(days=outcome_window_days)
    return ObservationWindow(
        feature_start=feature_start, feature_end=feature_end, outcome_end=outcome_end
    )


def filter_observation_window(
    df: pd.DataFrame,
    window: ObservationWindow,
    *,
    datetime_column: str = "transaction_datetime",
) -> tuple[pd.DataFrame, int]:
    """Оставить транзакции внутри окна признаков (ETL-3)."""
    if datetime_column not in df.columns:
        msg = f"колонка {datetime_column!r} отсутствует"
        raise KeyError(msg)
    ts = pd.to_datetime(df[datetime_column], errors="coerce")
    mask = (ts >= window.feature_start) & (ts < window.feature_end)
    dropped = int((~mask).sum())
    return df.loc[mask], dropped


# ---------------------------------------------------------------------------
# Режим дисбаланса (SPEC §5.1)
# ---------------------------------------------------------------------------


@dataclass
class ImbalanceReport:
    """Что именно сделал даунсэмплинг — обязателен к логированию (SPEC §5.1)."""

    requested_share: float | None
    share_before: float
    share_after: float
    rows_before: int
    rows_after: int
    treatment_dropped: int
    applied: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_treatment_share": self.requested_share,
            "treatment_share_before": round(self.share_before, 6),
            "treatment_share_after": round(self.share_after, 6),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "treatment_rows_dropped": self.treatment_dropped,
            "applied": self.applied,
            "note": self.note,
        }


def apply_treatment_share(
    df: pd.DataFrame,
    share: float | None,
    *,
    seed: int,
    treatment_column: str = "treatment",
    stratify_on: Sequence[str] = ("target",),
) -> tuple[pd.DataFrame, ImbalanceReport]:
    """Привести долю treatment-группы к заданной стратифицированным даунсэмплингом.

    Контрольная группа не трогается — уменьшается только treatment-группа,
    пропорционально внутри страт (по умолчанию по ``target``), чтобы режим
    дефицита не искажал распределение исхода внутри группы воздействия.

    Args:
        df: выборка клиентского уровня с колонками ``treatment`` и стратами.
        share: требуемая доля ``T=1``; ``None`` — native-режим (без изменений).
        seed: сид (ML-1).
        treatment_column: имя колонки воздействия.
        stratify_on: колонки страт; отсутствующие игнорируются.

    Returns:
        Усечённая выборка и отчёт о фактических долях до/после.
    """
    t = df[treatment_column].to_numpy()
    n_before = len(df)
    share_before = float(np.mean(t)) if n_before else 0.0

    if share is None:
        return df, ImbalanceReport(
            requested_share=None,
            share_before=share_before,
            share_after=share_before,
            rows_before=n_before,
            rows_after=n_before,
            treatment_dropped=0,
            applied=False,
            note="native-режим: фактическая доля взята как есть",
        )

    n_control = int((t == 0).sum())
    n_treat = int((t == 1).sum())
    # Из N0 контролей доля share достигается при N1' = share/(1-share) · N0
    target_treat = int(round(share / (1.0 - share) * n_control))

    if target_treat >= n_treat:
        note = (
            f"запрошенная доля {share:.3f} недостижима даунсэмплингом treatment-группы: "
            f"фактическая доля {share_before:.3f} уже не выше требуемой "
            f"(T=1: {n_treat}, T=0: {n_control}); выборка оставлена без изменений"
        )
        logger.warning(note)
        return df, ImbalanceReport(
            requested_share=share,
            share_before=share_before,
            share_after=share_before,
            rows_before=n_before,
            rows_after=n_before,
            treatment_dropped=0,
            applied=False,
            note=note,
        )

    strata = [c for c in stratify_on if c in df.columns and c != treatment_column]
    treat_idx = df.index[df[treatment_column] == 1]
    rng = np.random.default_rng(seed)

    if strata:
        treat_df = df.loc[treat_idx, strata]
        keep: list[np.ndarray] = []
        groups = list(treat_df.groupby(strata, dropna=False, observed=True).groups.items())
        # Пропорциональное распределение квоты по стратам с детерминированным
        # добором остатка — иначе сумма по стратам не совпадёт с квотой.
        sizes = np.array([len(idx) for _, idx in groups], dtype=float)
        quotas = np.floor(sizes / sizes.sum() * target_treat).astype(int)
        remainder = target_treat - int(quotas.sum())
        if remainder > 0:
            order = np.argsort(-(sizes / sizes.sum() * target_treat - quotas))
            for pos in order[:remainder]:
                quotas[pos] += 1
        for (_, idx), quota in zip(groups, quotas, strict=True):
            idx_array = np.asarray(idx)
            quota = min(int(quota), idx_array.size)
            if quota > 0:
                keep.append(rng.choice(idx_array, size=quota, replace=False))
        kept_treat = np.concatenate(keep) if keep else np.array([], dtype=treat_idx.dtype)
    else:
        kept_treat = rng.choice(np.asarray(treat_idx), size=target_treat, replace=False)

    keep_mask = df[treatment_column].eq(0).to_numpy()
    keep_mask[df.index.get_indexer(pd.Index(kept_treat))] = True
    out = df.loc[keep_mask].reset_index(drop=True)

    share_after = float(out[treatment_column].mean())
    report = ImbalanceReport(
        requested_share=share,
        share_before=share_before,
        share_after=share_after,
        rows_before=n_before,
        rows_after=len(out),
        treatment_dropped=n_treat - int(kept_treat.size),
        applied=True,
        note=f"стратификация по {strata or ['—']}, сид {seed}",
    )
    logger.info(
        "режим дисбаланса: доля T=1 %.4f → %.4f (отброшено %d treatment-строк из %d)",
        share_before,
        share_after,
        report.treatment_dropped,
        n_treat,
    )
    return out, report


# ---------------------------------------------------------------------------
# Прочее
# ---------------------------------------------------------------------------


def clip_quantiles(
    df: pd.DataFrame, columns: Sequence[str], *, lower: float = 0.001, upper: float = 0.999
) -> pd.DataFrame:
    """Зажать выбросы по квантилям (профиль выбросов фиксируется в EDA, W3)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        lo, hi = out[col].quantile([lower, upper])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[col] = out[col].clip(lo, hi)
    return out


__all__ = [
    "CrossChunkDeduplicator",
    "ImbalanceReport",
    "ObservationWindow",
    "apply_treatment_share",
    "build_window",
    "clip_quantiles",
    "deduplicate",
    "filter_observation_window",
]
