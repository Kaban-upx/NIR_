"""Тесты сплита train/valid/test (ML-2).

Главное свойство: разбиение детерминировано и сохраняет долю ``T=1`` и ``Y=1``
в каждой части. При дисбалансе 15:85 случайный сплит легко оставляет в valid
почти пустую группу ``T=1, Y=1``, и метрики §11 перестают быть сопоставимыми.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.splitting import (
    SPLIT_COLUMN,
    SPLIT_LABELS,
    assign_splits,
    split_frames,
    stratified_kfold_indices,
)


def _frame(n: int = 2000, treated_share: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < treated_share).astype(int)
    # Отклик зависит от воздействия — иначе страта (T,Y) ничего не стратифицирует
    target = (rng.random(n) < 0.2 + 0.1 * treatment).astype(int)
    return pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n)],
            "treatment": treatment,
            "target": target,
            "x": rng.normal(size=n),
        }
    )


def test_splits_cover_sample_without_overlap() -> None:
    df = _frame()
    out, summary = assign_splits(df, test_size=0.2, valid_size=0.1)
    assert out[SPLIT_COLUMN].notna().all()
    assert sum(summary.sizes.values()) == len(df)
    assert summary.sizes["test"] == pytest.approx(0.2 * len(df), abs=2)
    assert summary.sizes["valid"] == pytest.approx(0.1 * len(df), abs=2)


def test_split_column_is_categorical_with_fixed_labels() -> None:
    out, _ = assign_splits(_frame(), test_size=0.2, valid_size=0.1)
    assert list(out[SPLIT_COLUMN].cat.categories) == list(SPLIT_LABELS)


def test_treatment_share_preserved_in_every_part() -> None:
    df = _frame(n=4000, treated_share=0.15)
    _, summary = assign_splits(df, test_size=0.2, valid_size=0.1)
    for label in SPLIT_LABELS:
        assert summary.treatment_shares[label] == pytest.approx(0.15, abs=0.02)
        assert summary.target_rates[label] == pytest.approx(
            float(df["target"].mean()), abs=0.03
        )


def test_joint_stratum_is_preserved_not_just_marginals() -> None:
    """Стратификация именно по паре ``(T, Y)``: доля T=1,Y=1 держится везде."""
    df = _frame(n=4000, treated_share=0.15)
    out, _ = assign_splits(df, test_size=0.2, valid_size=0.1)
    overall = float(((df["treatment"] == 1) & (df["target"] == 1)).mean())
    for label in SPLIT_LABELS:
        part = out.loc[out[SPLIT_COLUMN] == label]
        share = float(((part["treatment"] == 1) & (part["target"] == 1)).mean())
        assert share == pytest.approx(overall, abs=0.02)


def test_split_is_deterministic_for_same_seed() -> None:
    df = _frame()
    first, _ = assign_splits(df, test_size=0.2, valid_size=0.1, seed=42)
    second, _ = assign_splits(df, test_size=0.2, valid_size=0.1, seed=42)
    other, _ = assign_splits(df, test_size=0.2, valid_size=0.1, seed=7)
    assert first[SPLIT_COLUMN].tolist() == second[SPLIT_COLUMN].tolist()
    assert first[SPLIT_COLUMN].tolist() != other[SPLIT_COLUMN].tolist()


def test_zero_valid_size_leaves_only_train_and_test() -> None:
    out, summary = assign_splits(_frame(), test_size=0.2, valid_size=0.0)
    assert summary.sizes["valid"] == 0
    assert set(out[SPLIT_COLUMN].dropna().unique()) == {"train", "test"}
    assert np.isnan(summary.treatment_shares["valid"])


def test_rare_strata_do_not_break_split() -> None:
    """Страта из одного объекта не должна ронять train_test_split."""
    df = _frame(n=300)
    df["treatment"] = 0
    df["target"] = 0
    df.loc[0, ["treatment", "target"]] = [1, 1]  # единственный объект своей страты
    out, summary = assign_splits(df, test_size=0.2, valid_size=0.1)
    assert sum(summary.sizes.values()) == len(df)
    assert out[SPLIT_COLUMN].notna().all()
    # Стратификация не состоялась — сводка обязана это показать
    assert summary.stratify_on == []


def test_missing_stratify_column_falls_back_to_random() -> None:
    out, summary = assign_splits(
        _frame(n=500), test_size=0.2, valid_size=0.1, stratify_on=("нет_такой",)
    )
    assert summary.stratify_on == []
    assert sum(summary.sizes.values()) == len(out)


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="пустая выборка"):
        assign_splits(_frame().head(0), test_size=0.2, valid_size=0.1)


def test_summary_is_serializable() -> None:
    _, summary = assign_splits(_frame(), test_size=0.2, valid_size=0.1, seed=11)
    payload = summary.as_dict()
    assert payload["seed"] == 11
    assert payload["stratify_on"] == ["treatment", "target"]


def test_split_frames_returns_all_parts() -> None:
    out, summary = assign_splits(_frame(), test_size=0.2, valid_size=0.1)
    parts = split_frames(out)
    assert set(parts) == set(SPLIT_LABELS)
    for label, part in parts.items():
        assert len(part) == summary.sizes[label]
        assert part.index.tolist() == list(range(len(part)))


def test_split_frames_requires_split_column() -> None:
    with pytest.raises(KeyError, match="make features"):
        split_frames(_frame())


# ------------------------------------------------------ cross-fitting (X-Learner)


def test_kfold_indices_partition_sample() -> None:
    rng = np.random.default_rng(3)
    treatment = (rng.random(600) < 0.3).astype(int)
    outcome = rng.normal(size=600)
    folds = stratified_kfold_indices(treatment, outcome, n_splits=5, seed=42)

    assert len(folds) == 5
    validation = np.concatenate([va for _, va in folds])
    assert np.array_equal(np.sort(validation), np.arange(600))
    for tr, va in folds:
        assert not set(tr.tolist()) & set(va.tolist())


def test_kfold_indices_keep_treatment_share_per_fold() -> None:
    rng = np.random.default_rng(3)
    treatment = (rng.random(1200) < 0.15).astype(int)
    outcome = rng.normal(size=1200)
    for _, va in stratified_kfold_indices(treatment, outcome, n_splits=4, seed=42):
        assert treatment[va].mean() == pytest.approx(0.15, abs=0.03)


def test_single_fold_degenerates_to_full_sample() -> None:
    """n_splits<2 — обучение без cross-fitting; индексы совпадают."""
    treatment = np.zeros(10, dtype=int)
    folds = stratified_kfold_indices(treatment, np.arange(10.0), n_splits=1, seed=42)
    assert len(folds) == 1
    assert np.array_equal(folds[0][0], folds[0][1])
