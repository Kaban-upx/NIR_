"""Тесты препроцессинга витрины: нормализация, кодирование, декоррелирование.

Это ETL-4 и ETL-5. Проверяется не арифметика, а направление зависимости:
параметры преобразований обучаются **только на train** и применяются к valid и
test без изменения. Обратный порядок — та самая утечка распределения, из-за
которой uplift-метрики в §11 выглядят лучше, чем есть.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_uplift.features.correlation import (
    correlation_matrix,
    save_correlation_artifacts,
    select_decorrelated,
)
from causal_uplift.features.encoding import UNSEEN_CATEGORY, CategoricalEncoder
from causal_uplift.features.scaling import IdentityScaler, ZScoreScaler, make_scaler

# ----------------------------------------------------------------- ZScoreScaler


def _shifted_frame() -> pd.DataFrame:
    """train со средним ≈0 и test со средним ≈10 — сдвиг видно по результату."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0.0, 1.0, 400), "split": "train"})
    test = pd.DataFrame({"x": rng.normal(10.0, 1.0, 200), "split": "test"})
    return pd.concat([train, test], ignore_index=True)


def test_scaler_parameters_come_from_train_only() -> None:
    df = _shifted_frame()
    train = df.loc[df["split"] == "train"]
    scaler = ZScoreScaler().fit(train, ["x"])

    assert scaler.mean_["x"] == pytest.approx(float(train["x"].mean()), abs=1e-6)
    assert scaler.std_["x"] == pytest.approx(float(train["x"].std(ddof=0)), abs=1e-6)
    assert scaler.fitted_on_rows == len(train)


def test_scaler_does_not_centre_test_part() -> None:
    """Если test после нормализации тоже центрирован — параметры подсмотрели его."""
    df = _shifted_frame()
    scaler = ZScoreScaler().fit(df.loc[df["split"] == "train"], ["x"])
    out = scaler.transform(df)

    assert float(out.loc[out["split"] == "train", "x"].mean()) == pytest.approx(0.0, abs=0.05)
    assert float(out.loc[out["split"] == "test", "x"].mean()) > 5.0


def test_scaler_keeps_non_listed_columns_intact() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    out = ZScoreScaler().fit_transform(df, ["x"])
    assert out["y"].tolist() == df["y"].tolist()
    assert out["x"].dtype == np.float32


def test_constant_column_becomes_zero_not_nan() -> None:
    """std≈0 — деления на ноль быть не должно: контракт ETL-6 запрещает NaN в X."""
    df = pd.DataFrame({"c": [7.0] * 20})
    scaler = ZScoreScaler().fit(df, ["c"])
    out = scaler.transform(df)
    assert scaler.std_["c"] == pytest.approx(1.0)
    assert out["c"].abs().max() == pytest.approx(0.0)


def test_scaler_removes_nan_and_inf() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    scaler = ZScoreScaler().fit(train, ["x"])
    out = scaler.transform(pd.DataFrame({"x": [np.nan, np.inf, -np.inf, 2.0]}))
    assert np.isfinite(out["x"].to_numpy()).all()
    assert out["x"].tolist()[:3] == [0.0, 0.0, 0.0]


def test_scaler_defaults_to_numeric_columns() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
    scaler = ZScoreScaler().fit(df)
    assert scaler.columns == ["x"]


def test_transform_before_fit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="не обучен"):
        ZScoreScaler().transform(pd.DataFrame({"x": [1.0]}))


def test_transform_requires_all_fitted_columns() -> None:
    scaler = ZScoreScaler().fit(pd.DataFrame({"x": [1.0, 2.0]}), ["x"])
    with pytest.raises(KeyError, match="x"):
        scaler.transform(pd.DataFrame({"y": [1.0]}))


def test_scaler_roundtrip_through_json(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 5.0, 9.0]})
    scaler = ZScoreScaler().fit(df, ["x"])
    path = scaler.save(tmp_path / "scaler.json")

    restored = ZScoreScaler.load(path)
    assert restored.to_dict() == scaler.to_dict()
    pd.testing.assert_frame_equal(restored.transform(df), scaler.transform(df))


def test_identity_scaler_changes_nothing() -> None:
    df = pd.DataFrame({"x": [1.0, 100.0]})
    scaler = IdentityScaler().fit(df, ["x"])
    pd.testing.assert_frame_equal(scaler.transform(df), df)
    assert scaler.to_dict()["type"] == "identity"


def test_make_scaler_factory() -> None:
    assert isinstance(make_scaler("zscore"), ZScoreScaler)
    assert isinstance(make_scaler("none"), IdentityScaler)
    with pytest.raises(ValueError, match="неизвестный скейлер"):
        make_scaler("minmax")


# ------------------------------------------------------------ CategoricalEncoder


def test_one_hot_uses_train_categories_only() -> None:
    train = pd.DataFrame({"gender": ["F", "M", "F", "U"]})
    encoder = CategoricalEncoder().fit(train, one_hot=["gender"])
    assert encoder.categories_["gender"] == ["F", "M", "U"]

    out = encoder.transform(pd.DataFrame({"gender": ["F", "НОВОЕ"]}))
    assert "gender" not in out.columns  # drop_source по умолчанию
    assert out.loc[0, "gender_F"] == pytest.approx(1.0)
    # Невстреченная категория не создаёт новую колонку — иначе ломается ширина X
    assert set(out.columns) == {"gender_F", "gender_M", "gender_U"}
    assert out.loc[1].sum() == pytest.approx(0.0)


def test_frequency_encoding_normalized_on_train() -> None:
    train = pd.DataFrame({"favorite_store_id": ["s1"] * 6 + ["s2"] * 2 + ["s3"] * 2})
    encoder = CategoricalEncoder().fit(train, frequency=["favorite_store_id"])
    assert sum(encoder.frequencies_["favorite_store_id"].values()) == pytest.approx(1.0)
    assert encoder.frequencies_["favorite_store_id"]["s1"] == pytest.approx(0.6)


def test_unseen_high_cardinality_value_gets_zero_frequency() -> None:
    """«Такого раньше не было» — 0.0, а не средняя частота: иначе это утечка."""
    train = pd.DataFrame({"favorite_brand_id": ["b1", "b1", "b2", "b2"]})
    encoder = CategoricalEncoder().fit(train, frequency=["favorite_brand_id"])
    out = encoder.transform(pd.DataFrame({"favorite_brand_id": ["b1", "b_новый", None]}))
    assert out["favorite_brand_id_freq"].tolist() == [
        pytest.approx(0.5),
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]


def test_missing_value_is_its_own_category() -> None:
    train = pd.DataFrame({"gender": ["F", None, None, "M"]})
    encoder = CategoricalEncoder().fit(train, one_hot=["gender"])
    assert UNSEEN_CATEGORY in encoder.categories_["gender"]
    out = encoder.transform(train)
    assert out[f"gender_{UNSEEN_CATEGORY}"].sum() == pytest.approx(2.0)


def test_encoder_can_keep_source_columns() -> None:
    train = pd.DataFrame({"gender": ["F", "M"]})
    out = CategoricalEncoder().fit_transform(train, one_hot=["gender"], drop_source=False)
    assert "gender" in out.columns


def test_encoder_ignores_absent_columns() -> None:
    train = pd.DataFrame({"gender": ["F", "M"]})
    encoder = CategoricalEncoder().fit(train, one_hot=["gender"], frequency=["нет_такой"])
    assert encoder.frequency_columns == []
    assert encoder.output_columns() == ["gender_F", "gender_M"]


def test_encoder_output_columns_match_transform() -> None:
    train = pd.DataFrame({"gender": ["F", "M"], "favorite_store_id": ["s1", "s2"]})
    encoder = CategoricalEncoder().fit(
        train, one_hot=["gender"], frequency=["favorite_store_id"]
    )
    out = encoder.transform(train)
    assert set(encoder.output_columns()) <= set(out.columns)


def test_encoder_roundtrip_through_json(tmp_path: Path) -> None:
    train = pd.DataFrame({"gender": ["F", "M", "F"], "favorite_store_id": ["s1", "s1", "s2"]})
    encoder = CategoricalEncoder().fit(train, one_hot=["gender"], frequency=["favorite_store_id"])
    path = encoder.save(tmp_path / "encoders.json")

    restored = CategoricalEncoder.load(path)
    pd.testing.assert_frame_equal(restored.transform(train), encoder.transform(train))


# --------------------------------------------------------- мультиколлинеарность


def _correlated_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    base = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": base,
            "a_copy": base * 2.0 + 1.0,  # |r| = 1 с a
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )


def test_perfectly_correlated_pair_loses_exactly_one_feature() -> None:
    report = select_decorrelated(_correlated_frame(), ["a", "a_copy", "b", "c"], threshold=0.95)
    assert len(report.dropped) == 1
    assert report.dropped_names[0] in {"a", "a_copy"}
    assert len({"a", "a_copy"} & set(report.kept)) == 1
    assert {"b", "c"} <= set(report.kept)


def test_protected_feature_survives_collinearity() -> None:
    report = select_decorrelated(
        _correlated_frame(), ["a", "a_copy", "b", "c"], threshold=0.95, protected=["a_copy"]
    )
    assert "a_copy" in report.kept
    assert report.dropped_names == ["a"]


def test_high_threshold_drops_nothing() -> None:
    report = select_decorrelated(_correlated_frame(), ["a", "b", "c"], threshold=0.99)
    assert report.dropped == []
    assert report.kept == ["a", "b", "c"]  # исходный порядок колонок сохранён


def test_report_counts_features_before_and_after() -> None:
    report = select_decorrelated(_correlated_frame(), ["a", "a_copy", "b", "c"])
    payload = report.as_dict()
    assert payload["n_features_before"] == 4
    assert payload["n_features_after"] == 3
    assert payload["method"] == "pearson"


def test_dropped_entry_names_its_partner() -> None:
    report = select_decorrelated(_correlated_frame(), ["a", "a_copy", "b", "c"])
    entry = report.dropped[0]
    assert entry["correlated_with"] in {"a", "a_copy"}
    assert abs(float(entry["r"])) == pytest.approx(1.0, abs=1e-6)


def test_top_pairs_lists_strongest_first() -> None:
    pairs = select_decorrelated(_correlated_frame(), ["a", "a_copy", "b", "c"]).top_pairs(n=3)
    assert list(pairs.columns) == ["feature_a", "feature_b", "abs_r"]
    assert pairs.loc[0, "abs_r"] == pytest.approx(1.0, abs=1e-6)
    assert pairs["abs_r"].is_monotonic_decreasing


def test_correlation_matrix_requires_two_numeric_columns() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "label": ["x", "y"]})
    with pytest.raises(ValueError, match="минимум два"):
        correlation_matrix(df, ["a", "label"])


def test_spearman_method_is_supported() -> None:
    corr = correlation_matrix(_correlated_frame(), ["a", "a_copy", "b"], method="spearman")
    assert corr.loc["a", "a_copy"] == pytest.approx(1.0, abs=1e-9)


def test_correlation_artifacts_are_written(tmp_path: Path) -> None:
    report = select_decorrelated(_correlated_frame(), ["a", "a_copy", "b", "c"])
    paths = save_correlation_artifacts(
        report,
        png_path=tmp_path / "reports" / "corr_matrix.png",
        json_path=tmp_path / "reports" / "dropped_features.json",
        csv_path=tmp_path / "reports" / "corr_matrix.csv",
    )
    assert set(paths) == {"png", "json", "csv"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
