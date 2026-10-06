"""Тесты воспроизводимости (ML-1, SPEC §11).

Требование ML-1 звучит как «один сид — один результат», и проверяется оно
буквально: конвейер прогоняется дважды в двух независимых корнях проекта, и
сравниваются сырые CSV, витрина, обученные параметры преобразований и
итоговый ATE. Сравнение именно двух полных прогонов, а не повторного чтения
одних и тех же артефактов — иначе тест проходил бы и на коде, который просто
кэширует результат на диске.

Отдельно проверяется обратное: при другом сиде данные обязаны отличаться.
Без этой проверки «воспроизводимость» могла бы означать, что генератор
игнорирует сид вовсе.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import build_layer1, make_project

from causal_uplift.causal.identify import CausalAnalysis, estimation_frame, run_causal_analysis
from causal_uplift.config import Config
from causal_uplift.etl.synthetic import TRUTH_FILENAME
from causal_uplift.features.build import (
    CLIP_ARTIFACT,
    ENCODER_ARTIFACT,
    SCALER_ARTIFACT,
    feature_store_covariates,
    load_feature_store,
)
from causal_uplift.io_utils import load_json
from causal_uplift.splitting import SPLIT_COLUMN

#: Клиентов меньше, чем в фикстуре ``layer1``: здесь важна не устойчивость
#: оценок, а побитовое совпадение двух прогонов, и его видно на любом размере.
N_CLIENTS = 1200
SEED = 42
OTHER_SEED = 2024

RAW_FILES = ("clients.csv", "products.csv", "uplift_train.csv", "purchases.csv")

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.regression]


@pytest.fixture(scope="session")
def twin_a(tmp_path_factory: pytest.TempPathFactory) -> Config:
    root = make_project(tmp_path_factory.mktemp("twin_a"))
    return build_layer1(root, n_clients=N_CLIENTS, seed=SEED)


@pytest.fixture(scope="session")
def twin_b(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Второй прогон с тем же сидом — в другом каталоге и другом процессе чтения."""
    root = make_project(tmp_path_factory.mktemp("twin_b"))
    return build_layer1(root, n_clients=N_CLIENTS, seed=SEED)


@pytest.fixture(scope="session")
def other(tmp_path_factory: pytest.TempPathFactory) -> Config:
    root = make_project(tmp_path_factory.mktemp("other_seed"))
    return build_layer1(root, n_clients=N_CLIENTS, seed=OTHER_SEED)


def _analysis(cfg: Config) -> CausalAnalysis:
    return run_causal_analysis(cfg)


# ------------------------------------------------------------- сырые данные


@pytest.mark.parametrize("name", RAW_FILES)
def test_same_seed_produces_byte_identical_raw_files(
    twin_a: Config, twin_b: Config, name: str
) -> None:
    """Генератор — чистая функция сида: даже порядок строк обязан совпасть."""
    a = (twin_a.raw_dir / name).read_bytes()
    b = (twin_b.raw_dir / name).read_bytes()
    assert a == b, f"{name}: прогоны с одним сидом разошлись"
    assert a, f"{name}: файл пуст"


def test_same_seed_produces_identical_ground_truth(twin_a: Config, twin_b: Config) -> None:
    """Ground truth (``true_tau``) — основа sanity-проверок, он тоже от сида."""
    a = pd.read_parquet(twin_a.raw_dir / TRUTH_FILENAME)
    b = pd.read_parquet(twin_b.raw_dir / TRUTH_FILENAME)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_produces_different_raw_data(twin_a: Config, other: Config) -> None:
    """Контрольная проверка: сид действительно используется."""
    assert (twin_a.raw_dir / "purchases.csv").read_bytes() != (
        other.raw_dir / "purchases.csv"
    ).read_bytes()
    assert (twin_a.raw_dir / "clients.csv").read_bytes() != (
        other.raw_dir / "clients.csv"
    ).read_bytes()


# ------------------------------------------------------------------ витрина


def test_same_seed_produces_identical_feature_store(twin_a: Config, twin_b: Config) -> None:
    """Главная проверка ML-1: витрина совпадает целиком — значения и типы."""
    a = load_feature_store(twin_a)
    b = load_feature_store(twin_b)
    pd.testing.assert_frame_equal(a, b)


def test_same_seed_produces_identical_interim_tables(twin_a: Config, twin_b: Config) -> None:
    for name in ("transactions.parquet", "assortment.parquet", "outcome_revenue.parquet"):
        a = pd.read_parquet(twin_a.interim_dir / name)
        b = pd.read_parquet(twin_b.interim_dir / name)
        pd.testing.assert_frame_equal(a, b, obj=name)


def test_same_seed_produces_identical_split_assignment(twin_a: Config, twin_b: Config) -> None:
    """Разбиение — отдельный источник случайности (ML-2), проверяем его явно."""
    a = load_feature_store(twin_a).set_index("client_id")[SPLIT_COLUMN].sort_index()
    b = load_feature_store(twin_b).set_index("client_id")[SPLIT_COLUMN].sort_index()
    pd.testing.assert_series_equal(a, b)


def test_same_seed_produces_identical_transform_parameters(
    twin_a: Config, twin_b: Config
) -> None:
    """Средние/границы обучены на train — совпадение подтверждает и состав train."""
    for artifact in (SCALER_ARTIFACT, ENCODER_ARTIFACT, CLIP_ARTIFACT):
        assert load_json(twin_a.artifact(artifact)) == load_json(
            twin_b.artifact(artifact)
        ), artifact


def test_different_seed_keeps_the_covariate_contract(twin_a: Config, other: Config) -> None:
    """Состав ковариат задан кодом, а не выборкой: от сида он зависеть не должен."""
    assert feature_store_covariates(load_feature_store(twin_a)) == feature_store_covariates(
        load_feature_store(other)
    )


def test_different_seed_changes_the_values(twin_a: Config, other: Config) -> None:
    a = load_feature_store(twin_a)
    b = load_feature_store(other)
    assert not a["total_spend"].equals(b["total_spend"])


# ---------------------------------------------------- подвыборка для DoWhy


def test_estimation_subsample_is_seed_stable(cfg: Config) -> None:
    """Прореживание для DoWhy — тоже случайность, и она зафиксирована сидом.

    Тест не требует витрины: достаточно кадра нужной формы, зато проверка
    остаётся быстрой и ловит потерянный ``default_rng(cfg.seed)``.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "treatment": rng.integers(0, 2, 2000),
            cfg.data.outcome: rng.normal(size=2000),
            "age": rng.normal(size=2000),
        }
    )
    first, n_full = estimation_frame(cfg, frame, ["age"], limit=500)
    second, _ = estimation_frame(cfg, frame, ["age"], limit=500)

    assert n_full == 2000
    assert len(first) <= 510  # квоты округляются по стратам
    pd.testing.assert_frame_equal(first, second)

    other_seed = cfg.model_copy(deep=True)
    other_seed.seed = cfg.seed + 1
    assert not estimation_frame(other_seed, frame, ["age"], limit=500)[0].index.equals(first.index)


# -------------------------------------------------------------- причинный слой


def test_same_seed_produces_identical_causal_estimates(
    twin_a: Config, twin_b: Config
) -> None:
    """ML-1 для слоя 2: одинаковые данные обязаны давать одинаковый ATE."""
    a = _analysis(twin_a)
    b = _analysis(twin_b)

    assert a.delta_corr == b.delta_corr
    assert a.n_rows == b.n_rows
    assert a.backdoor_variables == b.backdoor_variables
    assert a.ate == pytest.approx(b.ate, abs=1e-9)
    assert a.confounding_bias == pytest.approx(b.confounding_bias, abs=1e-9)
    for est_a, est_b in zip(a.estimates, b.estimates, strict=True):
        assert est_a.method == est_b.method
        assert est_a.failed == est_b.failed
        if est_a.failed is None:
            assert est_a.ate == pytest.approx(est_b.ate, abs=1e-9)


def test_saved_ate_artifact_matches_between_runs(twin_a: Config, twin_b: Config) -> None:
    """Сравниваются именно записанные артефакты: отчёт тоже должен быть повторим."""
    _analysis(twin_a)
    _analysis(twin_b)
    a = load_json(twin_a.artifact("ate.json"))
    b = load_json(twin_b.artifact("ate.json"))
    for key in ("delta_corr", "ate", "confounding_bias", "n_rows", "n_treated", "n_control"):
        assert a[key] == b[key], key


def test_conclusion_survives_a_change_of_seed(twin_a: Config, other: Config) -> None:
    """Структурная часть вывода от выборки не зависит.

    Численные оценки на 1200 клиентах шумны, поэтому сравниваются не числа, а
    то, что обязано быть устойчивым: множество корректировки, исключённая из
    него сезонность и знак истинного эффекта генератора.
    """
    a = _analysis(twin_a)
    b = _analysis(other)

    assert a.backdoor_variables == b.backdoor_variables
    assert a.graph.outcome_only == b.graph.outcome_only
    assert a.true_ate is not None
    assert b.true_ate is not None
    assert a.true_ate > 0
    assert b.true_ate > 0


def test_pipeline_is_idempotent_on_rerun(tmp_path: Path) -> None:
    """Повторный прогон в том же корне перезаписывает артефакты тем же содержимым.

    Проверка на скрытое состояние: конвейер не должен ни дописывать к старым
    файлам, ни зависеть от того, что в ``data/`` уже лежало.
    """
    root = make_project(tmp_path / "rerun")
    cfg = build_layer1(root, n_clients=400, seed=SEED)
    first = load_feature_store(cfg)
    scaler_first = load_json(cfg.artifact(SCALER_ARTIFACT))

    build_layer1(root, n_clients=400, seed=SEED)
    pd.testing.assert_frame_equal(load_feature_store(cfg), first)
    assert load_json(cfg.artifact(SCALER_ARTIFACT)) == scaler_first
