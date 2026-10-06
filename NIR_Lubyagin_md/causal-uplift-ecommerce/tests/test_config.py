"""Тесты конфигурации (W1, SPEC §10).

Конфигурация — единственный источник параметров, поэтому её ошибки должны
падать на загрузке, а не всплывать расчётом на половине конвейера.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from causal_uplift.config import Config, Threshold

ConfigFactory = Callable[..., Config]


def test_default_config_loads(cfg: Config) -> None:
    assert cfg.seed == 42
    assert cfg.data.source == "synthetic"
    assert cfg.features.min_covariates >= 30  # критерий приёмки 3
    assert len(cfg.causal.estimators) >= 2  # CAU-3


def test_project_root_found_by_pyproject(project: Path, cfg: Config) -> None:
    assert cfg.project_root.resolve() == project.resolve()
    # Относительные пути конфигурации считаются от корня проекта, не от cwd
    assert cfg.feature_store_path.resolve() == (
        project / "data/processed/feature_store.parquet"
    ).resolve()
    assert cfg.artifact("estimand.json").name == "estimand.json"


def test_unknown_key_is_rejected(project: Path) -> None:
    """Опечатка в ключе — ошибка загрузки (``extra="forbid"``)."""
    path = project / "configs" / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["data"]["outcome_window_dayz"] = 7
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValidationError, match="outcome_window_dayz"):
        Config.load(path)


def test_override_is_parsed_as_yaml_scalar(make_cfg: ConfigFactory) -> None:
    cfg = make_cfg(extra=["data.treatment_share=0.15", "features.clip_quantiles=null"])
    assert cfg.data.treatment_share == pytest.approx(0.15)
    assert cfg.features.clip_quantiles is None


def test_override_creates_missing_nesting(make_cfg: ConfigFactory) -> None:
    cfg = make_cfg(extra=["model.xlearner.cross_fitting_folds=2"])
    assert cfg.model.xlearner.cross_fitting_folds == 2


def test_malformed_override_rejected(make_cfg: ConfigFactory) -> None:
    with pytest.raises(ValueError, match="key.path=value"):
        make_cfg(extra=["seed:7"])


def test_single_estimator_rejected(make_cfg: ConfigFactory) -> None:
    """CAU-3: одного оценщика ATE недостаточно."""
    with pytest.raises(ValidationError, match="CAU-3"):
        make_cfg(extra=["causal.estimators=[linear_regression]"])


def test_split_sizes_must_leave_train(make_cfg: ConfigFactory) -> None:
    with pytest.raises(ValidationError, match="train"):
        make_cfg(extra=["split.test_size=0.6", "split.valid_size=0.35"])


def test_propensity_clip_bounds_validated(make_cfg: ConfigFactory) -> None:
    with pytest.raises(ValidationError, match="propensity_clip"):
        make_cfg(extra=["model.xlearner.propensity_clip=[0.9, 0.1]"])


def test_target_model_must_be_in_models(make_cfg: ConfigFactory) -> None:
    with pytest.raises(ValidationError, match="target_model"):
        make_cfg(extra=["experiment.target_model=magic"])


def test_clip_quantiles_order_validated(make_cfg: ConfigFactory) -> None:
    with pytest.raises(ValidationError, match="clip_quantiles"):
        make_cfg(extra=["features.clip_quantiles=[0.9, 0.1]"])


def test_with_treatment_share_does_not_mutate_original(cfg: Config) -> None:
    clone = cfg.with_treatment_share(0.15)
    assert clone.data.treatment_share == pytest.approx(0.15)
    assert cfg.data.treatment_share is None


def test_dump_is_json_serializable_without_project_root(cfg: Config) -> None:
    payload = cfg.dump()
    assert "project_root" not in payload
    assert payload["causal"]["identification"] == "backdoor"


def test_ensure_dirs_is_idempotent(cfg: Config) -> None:
    cfg.ensure_dirs()
    cfg.ensure_dirs()
    for directory in (cfg.raw_dir, cfg.interim_dir, cfg.artifacts_dir, cfg.reports_dir):
        assert directory.is_dir()


# --------------------------------------------------------------------- Threshold


def test_threshold_absolute_value() -> None:
    t = Threshold(raw=0.3)
    assert not t.is_quantile
    assert t.resolve([0.0, 1.0]) == pytest.approx(0.3)


def test_threshold_quantile_resolved_on_sample() -> None:
    t = Threshold(raw="q75")
    assert t.is_quantile
    assert t.resolve([0.0, 1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.0)


def test_threshold_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="q75"):
        Threshold(raw="высокий")


def test_threshold_rejects_quantile_out_of_range() -> None:
    with pytest.raises(ValueError, match="квантиль"):
        Threshold(raw="q140")


def test_segmentation_thresholds_coerced_from_scalars(make_cfg: ConfigFactory) -> None:
    cfg = make_cfg(extra=["segmentation.theta_high=0.05"])
    assert cfg.segmentation.theta_high.value == pytest.approx(0.05)
    assert cfg.segmentation.theta_low.is_quantile
