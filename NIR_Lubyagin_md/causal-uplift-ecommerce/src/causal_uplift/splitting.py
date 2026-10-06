"""Стратифицированный сплит train/valid/test (ML-2).

Стратификация по паре ``(T, Y)``, а не по одному ``Y``: при дефицитной
treatment-группе случайный сплит легко оставляет в валидации слишком мало
объектов с ``T=1, Y=1``, и early stopping базовых моделей начинает шуметь.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

SPLIT_COLUMN = "split"
SPLIT_LABELS: tuple[str, str, str] = ("train", "valid", "test")

#: Минимальный размер страты. Три, а не два: сплит двухшаговый (test, затем
#: valid), и страта из двух объектов разваливается на втором шаге.
MIN_STRATUM_SIZE = 3


@dataclass
class SplitSummary:
    """Фактические размеры и доли воздействия по частям сплита."""

    sizes: dict[str, int]
    treatment_shares: dict[str, float]
    target_rates: dict[str, float]
    stratify_on: list[str]
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "sizes": self.sizes,
            "treatment_shares": {k: round(v, 6) for k, v in self.treatment_shares.items()},
            "target_rates": {k: round(v, 6) for k, v in self.target_rates.items()},
            "stratify_on": self.stratify_on,
            "seed": self.seed,
        }


def _stratify_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series | None:
    usable = [c for c in columns if c in df.columns]
    if not usable:
        return None
    key = df[usable[0]].astype(str)
    for col in usable[1:]:
        key = key + "_" + df[col].astype(str)
    # Мелкие страты ломают train_test_split — сливаем их в «прочее»
    counts = key.value_counts()
    rare = counts[counts < MIN_STRATUM_SIZE].index
    if len(rare) > 0:
        key = key.where(~key.isin(rare), other="__rare__")
        counts = key.value_counts()
    if int(counts.min()) < MIN_STRATUM_SIZE:
        # Слияние не помогло: даже объединённая страта слишком мала. Падать здесь
        # нельзя (прогон на малой выборке — законный режим отладки), молча
        # стратифицировать — тоже: сводка должна честно сказать, что сплит случайный.
        logger.warning(
            "страта %r содержит %d объектов (<%d): стратификация отключена, сплит случайный",
            counts.idxmin(),
            int(counts.min()),
            MIN_STRATUM_SIZE,
        )
        return None
    return key


def assign_splits(
    df: pd.DataFrame,
    *,
    test_size: float,
    valid_size: float,
    stratify_on: Sequence[str] = ("treatment", "target"),
    seed: int = 42,
    split_column: str = SPLIT_COLUMN,
) -> tuple[pd.DataFrame, SplitSummary]:
    """Разметить строки метками ``train``/``valid``/``test``.

    Сплит фиксируется колонкой в витрине, а не хранится отдельно: любой слой
    (обучение, оценка, дашборд) обязан видеть одно и то же разбиение, иначе
    регрессионные метрики §11 несопоставимы между прогонами.

    Returns:
        Кадр с добавленной колонкой сплита и сводка фактических долей.
    """
    if df.empty:
        msg = "нечего разбивать: пустая выборка"
        raise ValueError(msg)

    out = df.reset_index(drop=True).copy()
    index = np.arange(len(out))
    strat = _stratify_key(out, stratify_on)

    train_valid_idx, test_idx = train_test_split(
        index,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strat.to_numpy() if strat is not None else None,
    )

    if valid_size > 0:
        # valid_size задан как доля от всей выборки → пересчитываем в долю от train+valid
        relative_valid = valid_size / (1.0 - test_size)
        strat_tv = strat.to_numpy()[train_valid_idx] if strat is not None else None
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=relative_valid,
            random_state=seed,
            shuffle=True,
            stratify=strat_tv,
        )
    else:
        train_idx, valid_idx = train_valid_idx, np.array([], dtype=int)

    labels = np.empty(len(out), dtype=object)
    labels[train_idx] = "train"
    labels[valid_idx] = "valid"
    labels[test_idx] = "test"
    out[split_column] = pd.Categorical(labels, categories=list(SPLIT_LABELS))

    sizes = {label: int((out[split_column] == label).sum()) for label in SPLIT_LABELS}
    treatment_shares = {}
    target_rates = {}
    for label in SPLIT_LABELS:
        part = out.loc[out[split_column] == label]
        treatment_shares[label] = float(part["treatment"].mean()) if len(part) else float("nan")
        target_rates[label] = float(part["target"].mean()) if len(part) else float("nan")

    summary = SplitSummary(
        sizes=sizes,
        treatment_shares=treatment_shares,
        target_rates=target_rates,
        # Пусто, если стратификация не состоялась: журнал прогона не должен
        # обещать стратифицированный сплит там, где его не было
        stratify_on=[c for c in stratify_on if c in out.columns] if strat is not None else [],
        seed=seed,
    )
    logger.info(
        "сплит: train %d / valid %d / test %d; доля T=1 %.4f / %.4f / %.4f",
        sizes["train"],
        sizes["valid"],
        sizes["test"],
        treatment_shares["train"],
        treatment_shares["valid"],
        treatment_shares["test"],
    )
    return out, summary


def split_frames(
    df: pd.DataFrame, *, split_column: str = SPLIT_COLUMN
) -> dict[str, pd.DataFrame]:
    """Разложить витрину по частям сплита."""
    if split_column not in df.columns:
        msg = f"в витрине нет колонки {split_column!r}: сначала выполните `make features`"
        raise KeyError(msg)
    return {
        label: df.loc[df[split_column] == label].reset_index(drop=True) for label in SPLIT_LABELS
    }


def stratified_kfold_indices(
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Индексы фолдов для cross-fitting, стратифицированные по паре ``(T, Y)``."""
    from sklearn.model_selection import StratifiedKFold

    if n_splits < 2:
        idx = np.arange(len(treatment))
        return [(idx, idx)]
    strata = (treatment.astype(int) * 2 + (outcome > np.median(outcome)).astype(int)).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(strata)), strata)]


__all__ = [
    "MIN_STRATUM_SIZE",
    "SPLIT_COLUMN",
    "SPLIT_LABELS",
    "SplitSummary",
    "assign_splits",
    "split_frames",
    "stratified_kfold_indices",
]
