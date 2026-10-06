"""Общие фикстуры (W12, SPEC §11).

Каждый тест работает в собственном корне проекта: копия ``configs/`` плюс
пустой ``pyproject.toml``, по которому ``Config.load`` находит корень. Так
тесты не трогают рабочие ``data/``, ``artifacts/`` и ``reports/`` репозитория
и не зависят от того, что в них уже лежит от предыдущих ручных прогонов.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import pytest

from causal_uplift.config import Config

# EDA строит графики; без Agg matplotlib на macOS пытается открыть окно
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = ("config.yaml", "dag.gml", "feature_groups.yaml", "model_lgbm.yaml")

#: Параметры, делающие прогон быстрым: конвейер тот же, данные меньше.
#: Ничего, кроме размеров, не меняется — иначе тест проверял бы другой код.
FAST_OVERRIDES: tuple[str, ...] = (
    "data.synthetic.avg_transactions_per_client=8",
    "data.synthetic.products_catalog_size=200",
    "data.synthetic.stores_count=20",
    "data.chunk_size=5000",
    "causal.estimation_sample=1500",
    "causal.n_simulations=5",
    "refutation.sample_size=1200",
    "refutation.num_simulations=3",
    "evaluation.bootstrap.n_iterations=20",
    # лог-файл в artifacts не нужен: pytest и так захватывает вывод
    "logging.file=null",
)


def make_project(root: Path) -> Path:
    """Собрать изолированный корень проекта с копией конфигураций."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    # Config._find_project_root ищет ближайший pyproject.toml вверх от конфига
    (root / "pyproject.toml").write_text(
        '[project]\nname = "causal-uplift-test"\nversion = "0"\n', encoding="utf-8"
    )
    for name in CONFIG_FILES:
        shutil.copy(REPO_ROOT / "configs" / name, root / "configs" / name)
    return root


def build_config(
    root: Path, *, n_clients: int = 400, seed: int = 42, extra: Sequence[str] = ()
) -> Config:
    """Конфигурация тестового прогона: быстрые размеры + точечные правки."""
    overrides = [
        *FAST_OVERRIDES,
        f"data.synthetic.n_clients={n_clients}",
        f"seed={seed}",
        *extra,
    ]
    cfg = Config.load(root / "configs" / "config.yaml", overrides=overrides)
    cfg.ensure_dirs()
    return cfg


def build_layer1(root: Path, *, n_clients: int, seed: int = 42) -> Config:
    """Прогнать слой 1 (data → etl → features → eda) в указанном корне."""
    from causal_uplift.pipeline.run import run_stage

    cfg = build_config(root, n_clients=n_clients, seed=seed)
    for stage in ("data", "etl", "features", "eda"):
        run_stage(cfg, stage, journal=False)
    return cfg


ConfigFactory = Callable[..., Config]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Пустой изолированный корень проекта."""
    return make_project(tmp_path / "proj")


@pytest.fixture
def make_cfg(project: Path) -> ConfigFactory:
    """Фабрика конфигураций внутри ``project`` (для тестов валидации)."""

    def factory(*, n_clients: int = 400, seed: int = 42, extra: Sequence[str] = ()) -> Config:
        return build_config(project, n_clients=n_clients, seed=seed, extra=extra)

    return factory


@pytest.fixture
def cfg(make_cfg: ConfigFactory) -> Config:
    """Конфигурация без данных: для unit-тестов конфига и графа."""
    return make_cfg()


@pytest.fixture(scope="session")
def layer1(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Один раз собранная витрина реального размера (≥30 ковариат).

    Клиентов намеренно 4000, а не 400: контракт ETL-6 и причинный слой
    проверяются на выборке, где ковариаты не вырождаются в константы, а у
    DoWhy хватает данных на propensity-модель. Фикстура сессионная — сборка
    витрины занимает большую часть времени этих тестов.
    """
    root = make_project(tmp_path_factory.mktemp("layer1"))
    return build_layer1(root, n_clients=4000)
