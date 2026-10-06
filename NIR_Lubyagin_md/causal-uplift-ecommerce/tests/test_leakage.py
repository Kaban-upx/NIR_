"""Leakage-тесты (критерий приёмки §11, риск §14).

Утечка в этом конвейере может произойти в двух местах, и оба проверяются здесь:

1. **По времени** — признак, посчитанный по транзакции из окна измерения
   исхода. Проверяется инвариантность: добавление транзакций в окно исхода не
   меняет ни одного признака.
2. **По выборке** — параметр преобразования (среднее, частота, граница
   зажатия), посчитанный с участием valid/test. Проверяется, что все
   артефакты фазы C обучены ровно на train-строках.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.config import Config
from causal_uplift.etl.aggregate import load_transactions, resolve_window
from causal_uplift.etl.cleaning import ObservationWindow, filter_observation_window
from causal_uplift.features.build import (
    CLIP_ARTIFACT,
    ENCODER_ARTIFACT,
    SCALER_ARTIFACT,
    apply_clip_bounds,
    feature_store_covariates,
    fit_clip_bounds,
    load_feature_store,
)
from causal_uplift.features.rfm import build_rfm_features
from causal_uplift.io_utils import load_json
from causal_uplift.splitting import SPLIT_COLUMN

WINDOW = ObservationWindow(
    feature_start=pd.Timestamp("2019-01-01"),
    feature_end=pd.Timestamp("2019-02-01"),
    outcome_end=pd.Timestamp("2019-03-01"),
)


def _transactions(dates: list[str], *, clients: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    return pd.DataFrame(
        {
            "client_id": clients,
            "transaction_id": [f"t{i}" for i in range(n)],
            "transaction_datetime": pd.to_datetime(dates),
            "purchase_sum": rng.uniform(100.0, 900.0, n),
            "n_items": rng.integers(1, 9, n).astype("float32"),
            "quantity_total": rng.integers(1, 15, n).astype("float32"),
            "n_unique_products": rng.integers(1, 7, n).astype("float32"),
        }
    )


def _feature_window_transactions() -> pd.DataFrame:
    return _transactions(
        ["2019-01-05", "2019-01-09", "2019-01-20", "2019-01-25", "2019-01-31"],
        clients=["c1", "c1", "c2", "c2", "c3"],
    )


def _outcome_window_transactions() -> pd.DataFrame:
    """Крупные чеки уже в окне исхода — именно они и «подсказали» бы ответ."""
    frame = _transactions(
        ["2019-02-02", "2019-02-10", "2019-02-20"],
        clients=["c1", "c2", "c3"],
        seed=7,
    )
    frame["purchase_sum"] *= 50.0
    frame["transaction_id"] = ["after1", "after2", "after3"]
    return frame


# ----------------------------------------------------- утечка по времени (ETL-3)


def test_outcome_window_transactions_do_not_change_any_feature() -> None:
    clean = _feature_window_transactions()
    polluted = pd.concat([clean, _outcome_window_transactions()], ignore_index=True)

    filtered, dropped = filter_observation_window(polluted, WINDOW)
    assert dropped == 3

    expected = build_rfm_features(clean, WINDOW)
    actual = build_rfm_features(filtered, WINDOW)
    pd.testing.assert_frame_equal(actual, expected)


def test_unfiltered_outcome_rows_would_change_features() -> None:
    """Контрольная проверка: без фильтра значения действительно «едут».

    Без неё предыдущий тест проходил бы и на коде, который просто игнорирует
    поздние транзакции по другой причине.
    """
    clean = _feature_window_transactions()
    polluted = pd.concat([clean, _outcome_window_transactions()], ignore_index=True)

    expected = build_rfm_features(clean, WINDOW)
    leaked = build_rfm_features(polluted, WINDOW)
    assert (leaked["total_spend"] > expected["total_spend"]).all()
    # Recency стала бы отрицательной: «последняя покупка» после конца окна
    assert (leaked["days_since_last_purchase"] < 0).any()


def test_recency_is_anchored_to_feature_end_not_to_data_max() -> None:
    """Отсчёт от ``feature_end``, иначе признак зависел бы от даты прогона."""
    tx = _feature_window_transactions()
    shifted = ObservationWindow(
        feature_start=WINDOW.feature_start + pd.Timedelta(days=10),
        feature_end=WINDOW.feature_end + pd.Timedelta(days=10),
        outcome_end=WINDOW.outcome_end + pd.Timedelta(days=10),
    )
    base = build_rfm_features(tx, WINDOW)
    later = build_rfm_features(tx, shifted)
    delta = later["days_since_last_purchase"] - base["days_since_last_purchase"]
    assert np.allclose(delta.to_numpy(), 10.0)


def test_short_window_counts_ignore_transactions_outside_window() -> None:
    tx = _feature_window_transactions()
    polluted, _ = filter_observation_window(
        pd.concat([tx, _outcome_window_transactions()], ignore_index=True), WINDOW
    )
    assert (
        build_rfm_features(polluted, WINDOW)["n_transactions_7d"].tolist()
        == build_rfm_features(tx, WINDOW)["n_transactions_7d"].tolist()
    )


# ------------------------------------------- утечка по выборке (ETL-4, ETL-5)


def test_clip_bounds_come_from_train_subsample() -> None:
    train = pd.DataFrame({"x": np.linspace(0.0, 100.0, 200)})
    test = pd.DataFrame({"x": np.linspace(10_000.0, 20_000.0, 50)})
    full = pd.concat([train, test], ignore_index=True)

    bounds = fit_clip_bounds(train, ["x"], (0.01, 0.99))
    clipped = apply_clip_bounds(full, bounds)

    # Граница определена train-выборкой, поэтому выбросы теста зажаты до ~100
    assert bounds["x"][1] < 101.0
    assert clipped["x"].max() == pytest.approx(bounds["x"][1])
    # Если бы границы считались по всей выборке, максимум остался бы десятками тысяч
    assert fit_clip_bounds(full, ["x"], (0.01, 0.99))["x"][1] > 1_000.0


def test_clip_bounds_skip_constant_and_non_numeric_columns() -> None:
    train = pd.DataFrame({"c": [1.0] * 50, "label": ["a"] * 50})
    assert fit_clip_bounds(train, ["c", "label", "нет_такой"], (0.01, 0.99)) == {}


# ------------------------------------------------ сквозные проверки на витрине


@pytest.mark.integration
@pytest.mark.slow
def test_interim_transactions_never_leave_feature_window(layer1: Config) -> None:
    window = resolve_window(layer1)
    transactions = load_transactions(layer1, columns=["transaction_datetime"])
    ts = pd.to_datetime(transactions["transaction_datetime"])
    assert ts.min() >= window.feature_start
    assert ts.max() < window.feature_end


@pytest.mark.integration
@pytest.mark.slow
def test_transform_parameters_fitted_on_train_rows_only(layer1: Config) -> None:
    store = load_feature_store(layer1)
    n_train = int((store[SPLIT_COLUMN] == "train").sum())
    assert n_train > 0

    for artifact in (SCALER_ARTIFACT, ENCODER_ARTIFACT, CLIP_ARTIFACT):
        payload = load_json(layer1.artifact(artifact))
        assert payload["fitted_on_rows"] == n_train, artifact


@pytest.mark.integration
@pytest.mark.slow
def test_outcome_columns_are_not_covariates(layer1: Config) -> None:
    """``revenue`` и ground truth измерены в окне исхода — в ``X`` им нельзя."""
    covariates = feature_store_covariates(load_feature_store(layer1))
    assert not {"revenue", "target", "true_tau", "true_mu0", SPLIT_COLUMN} & set(covariates)


@pytest.mark.integration
@pytest.mark.slow
def test_no_covariate_is_a_proxy_of_the_outcome(layer1: Config) -> None:
    """Ковариата с |r| ≈ 1 к исходу означала бы утечку Y, а не сильный признак."""
    store = load_feature_store(layer1)
    covariates = feature_store_covariates(store)
    corr = store[covariates].corrwith(store["target"].astype("float64")).abs()
    worst = corr.idxmax()
    assert corr[worst] < 0.9, f"подозрительная связь с исходом: {worst} r={corr[worst]:.3f}"
