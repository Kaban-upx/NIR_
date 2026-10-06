"""Тесты очистки выборки: дедупликация, окна наблюдения, режим дисбаланса.

Это ETL-2, ETL-3 и §5.1. Проверяются именно те свойства, нарушение которых
не видно по итоговым числам: дубль на границе чанков, транзакция из окна
исхода, попавшая в признаки, и доля treatment, «достигнутая» выбрасыванием
контроля.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.etl.cleaning import (
    CrossChunkDeduplicator,
    ObservationWindow,
    apply_treatment_share,
    build_window,
    clip_quantiles,
    deduplicate,
    filter_observation_window,
)

KEY = ("client_id", "transaction_id", "product_id")


def _rows(*triples: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(triples, columns=list(KEY)).assign(
        purchase_sum=np.arange(len(triples), dtype="float64")
    )


# ------------------------------------------------------------------ ETL-2


def test_deduplicate_keeps_first_occurrence() -> None:
    df = _rows(("c1", "t1", "p1"), ("c1", "t1", "p1"), ("c1", "t1", "p2"))
    out, dropped = deduplicate(df, KEY)
    assert dropped == 1
    assert len(out) == 2
    # keep="first": остаться должна строка с purchase_sum=0, а не её дубль
    assert out.loc[0, "purchase_sum"] == pytest.approx(0.0)


def test_deduplicate_requires_key_columns() -> None:
    with pytest.raises(KeyError, match="product_id"):
        deduplicate(pd.DataFrame({"client_id": ["c1"], "transaction_id": ["t1"]}), KEY)


def test_same_transaction_different_products_is_not_duplicate() -> None:
    """Ключ составной: одна транзакция — много товарных позиций."""
    df = _rows(("c1", "t1", "p1"), ("c1", "t1", "p2"), ("c1", "t1", "p3"))
    _, dropped = deduplicate(df, KEY)
    assert dropped == 0


def test_cross_chunk_deduplicator_catches_duplicate_across_boundary() -> None:
    dedup = CrossChunkDeduplicator(KEY)
    first = dedup.apply(_rows(("c1", "t1", "p1"), ("c2", "t2", "p1")))
    second = dedup.apply(_rows(("c1", "t1", "p1"), ("c3", "t3", "p1")))

    assert len(first) == 2
    # Дубль первой строки приехал во втором чанке — внутри чанка его не видно
    assert len(second) == 1
    assert second["client_id"].tolist() == ["c3"]
    assert dedup.stats == {"rows_seen": 4, "duplicates_dropped": 1}


def test_cross_chunk_deduplicator_removes_duplicates_inside_chunk_too() -> None:
    dedup = CrossChunkDeduplicator(KEY)
    out = dedup.apply(_rows(("c1", "t1", "p1"), ("c1", "t1", "p1")))
    assert len(out) == 1
    assert dedup.dropped == 1


def test_cross_chunk_deduplicator_handles_empty_chunk() -> None:
    dedup = CrossChunkDeduplicator(KEY)
    assert dedup.apply(pd.DataFrame(columns=list(KEY))).empty
    assert dedup.stats["rows_seen"] == 0


def test_cross_chunk_deduplicator_is_callable() -> None:
    dedup = CrossChunkDeduplicator(KEY)
    assert len(dedup(_rows(("c1", "t1", "p1")))) == 1


# ------------------------------------------------------------------ ETL-3


def test_build_window_reserves_tail_for_outcome() -> None:
    window = build_window(
        pd.Timestamp("2019-03-31 23:40:00"), observation_window_days=90, outcome_window_days=30
    )
    # Последние 30 дней данных — окно исхода, признаки туда не заглядывают
    assert window.feature_end == pd.Timestamp("2019-03-02")
    assert window.feature_start == pd.Timestamp("2018-12-02")
    assert window.outcome_end == pd.Timestamp("2019-04-01")
    assert window.feature_days == 90
    assert window.outcome_days == 30


def test_windows_do_not_overlap() -> None:
    window = build_window(
        pd.Timestamp("2019-03-31"), observation_window_days=90, outcome_window_days=30
    )
    assert window.feature_start < window.feature_end <= window.outcome_end
    assert window.as_dict()["observation_window_days"] == 90


def test_filter_observation_window_is_half_open() -> None:
    window = ObservationWindow(
        feature_start=pd.Timestamp("2019-01-01"),
        feature_end=pd.Timestamp("2019-02-01"),
        outcome_end=pd.Timestamp("2019-03-01"),
    )
    df = pd.DataFrame(
        {
            "transaction_datetime": pd.to_datetime(
                [
                    "2018-12-31 23:59:59",  # до окна
                    "2019-01-01 00:00:00",  # граница включена
                    "2019-01-31 23:59:59",  # внутри
                    "2019-02-01 00:00:00",  # граница исключена — это уже окно исхода
                    "2019-02-15",  # окно исхода
                ]
            )
        }
    )
    out, dropped = filter_observation_window(df, window)
    assert dropped == 3
    assert out.index.tolist() == [1, 2]


def test_filter_observation_window_drops_unparsable_dates() -> None:
    window = ObservationWindow(
        feature_start=pd.Timestamp("2019-01-01"),
        feature_end=pd.Timestamp("2019-02-01"),
        outcome_end=pd.Timestamp("2019-03-01"),
    )
    df = pd.DataFrame({"transaction_datetime": ["2019-01-15", "не дата"]})
    out, dropped = filter_observation_window(df, window)
    assert len(out) == 1
    assert dropped == 1


def test_filter_observation_window_requires_column() -> None:
    window = build_window(
        pd.Timestamp("2019-03-31"), observation_window_days=90, outcome_window_days=30
    )
    with pytest.raises(KeyError, match="transaction_datetime"):
        filter_observation_window(pd.DataFrame({"client_id": ["c1"]}), window)


# ------------------------------------------------------- режим дисбаланса §5.1


def _population(n: int = 1000, treated_share: float = 0.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < treated_share).astype(int)
    return pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n)],
            "treatment": treatment,
            "target": (rng.random(n) < 0.3).astype(int),
        }
    )


def test_native_mode_leaves_sample_untouched() -> None:
    df = _population()
    out, report = apply_treatment_share(df, None, seed=42)
    assert out.equals(df)
    assert not report.applied
    assert report.requested_share is None
    assert report.share_after == pytest.approx(report.share_before)


def test_downsampling_reaches_requested_share() -> None:
    df = _population(n=2000, treated_share=0.5)
    out, report = apply_treatment_share(df, 0.15, seed=42)
    assert report.applied
    assert out["treatment"].mean() == pytest.approx(0.15, abs=0.005)
    assert report.treatment_dropped > 0


def test_downsampling_never_drops_control_rows() -> None:
    """Доля достигается только усечением treatment — контроль неприкосновенен."""
    df = _population(n=2000, treated_share=0.5)
    control_before = set(df.loc[df["treatment"] == 0, "client_id"])
    out, _ = apply_treatment_share(df, 0.15, seed=42)
    assert set(out.loc[out["treatment"] == 0, "client_id"]) == control_before


def test_downsampling_preserves_outcome_mix_inside_treatment() -> None:
    """Стратификация по target: режим дефицита не должен менять долю Y=1."""
    rng = np.random.default_rng(7)
    n = 4000
    treatment = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n)],
            "treatment": treatment,
            "target": (rng.random(n) < 0.4).astype(int),
        }
    )
    before = df.loc[df["treatment"] == 1, "target"].mean()
    out, _ = apply_treatment_share(df, 0.15, seed=42)
    after = out.loc[out["treatment"] == 1, "target"].mean()
    assert after == pytest.approx(before, abs=0.02)


def test_downsampling_is_deterministic_for_same_seed() -> None:
    df = _population(n=1500)
    first, _ = apply_treatment_share(df, 0.15, seed=42)
    second, _ = apply_treatment_share(df, 0.15, seed=42)
    other, _ = apply_treatment_share(df, 0.15, seed=43)
    assert first["client_id"].tolist() == second["client_id"].tolist()
    assert first["client_id"].tolist() != other["client_id"].tolist()


def test_unreachable_share_is_reported_not_faked() -> None:
    """Доля 0.5 из выборки с 15% treatment недостижима — честный отказ."""
    df = _population(n=1000, treated_share=0.15)
    out, report = apply_treatment_share(df, 0.5, seed=42)
    assert not report.applied
    assert len(out) == len(df)
    assert "недостижима" in report.note


def test_report_is_serializable_and_rounded() -> None:
    df = _population(n=1000)
    _, report = apply_treatment_share(df, 0.2, seed=42)
    payload = report.as_dict()
    assert payload["applied"] is True
    assert payload["requested_treatment_share"] == pytest.approx(0.2)
    assert payload["rows_after"] < payload["rows_before"]


def test_missing_stratum_column_is_ignored() -> None:
    df = _population(n=800).drop(columns=["target"])
    out, report = apply_treatment_share(df, 0.15, seed=42, stratify_on=("target",))
    assert report.applied
    assert out["treatment"].mean() == pytest.approx(0.15, abs=0.01)


# ------------------------------------------------------------------ прочее


def test_clip_quantiles_bounds_outliers_only() -> None:
    df = pd.DataFrame({"x": [*range(100), 10_000], "label": ["a"] * 101})
    out = clip_quantiles(df, ["x", "label", "нет такой"], lower=0.01, upper=0.99)
    assert out["x"].max() < 10_000
    assert out["x"].min() >= df["x"].min()
    # Нечисловая колонка и отсутствующая игнорируются без ошибки
    assert out["label"].tolist() == df["label"].tolist()


def test_clip_quantiles_noop_on_constant_column() -> None:
    df = pd.DataFrame({"x": [5.0] * 10})
    assert clip_quantiles(df, ["x"])["x"].tolist() == df["x"].tolist()
