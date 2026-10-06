"""Pydantic-схема конфигурации (SPEC §10, W1).

Единственный источник параметров конвейера: код не содержит магических чисел,
всё приходит из ``configs/config.yaml``. Схема строгая — опечатка в ключе
останавливает прогон, а не тихо игнорируется (``extra="forbid"``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Вспомогательное: порог, заданный числом или квантилем вида "q75"
# ---------------------------------------------------------------------------

_QUANTILE_RE = re.compile(r"^q(\d{1,3}(?:\.\d+)?)$", re.IGNORECASE)


class Threshold(BaseModel):
    """Порог: либо абсолютное значение, либо квантиль распределения (``q75``)."""

    model_config = ConfigDict(frozen=True)

    raw: str | float
    quantile: float | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _parse(self) -> Threshold:
        raw = self.raw
        if isinstance(raw, int | float):
            object.__setattr__(self, "value", float(raw))
            return self
        match = _QUANTILE_RE.match(str(raw).strip())
        if match is None:
            msg = f"порог должен быть числом или квантилем вида 'q75', получено: {raw!r}"
            raise ValueError(msg)
        q = float(match.group(1))
        if not 0.0 <= q <= 100.0:
            msg = f"квантиль вне диапазона [0, 100]: {raw!r}"
            raise ValueError(msg)
        object.__setattr__(self, "quantile", q / 100.0)
        return self

    @property
    def is_quantile(self) -> bool:
        return self.quantile is not None

    def resolve(self, sample: Sequence[float] | Any) -> float:
        """Вернуть числовой порог: для квантильного — посчитать по выборке."""
        if self.value is not None:
            return self.value
        import numpy as np

        arr = np.asarray(sample, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            msg = "невозможно вычислить квантильный порог по пустой выборке"
            raise ValueError(msg)
        assert self.quantile is not None  # noqa: S101 — гарантировано валидатором
        return float(np.quantile(arr, self.quantile))

    def __str__(self) -> str:  # pragma: no cover — только для отчётов
        return f"q{self.quantile * 100:g}" if self.is_quantile else f"{self.value:g}"


def _as_threshold(v: Any) -> Any:
    return v if isinstance(v, Threshold | dict) else Threshold(raw=v)


# ---------------------------------------------------------------------------
# Секции конфигурации
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(StrictModel):
    data_dir: Path = Path("data")
    interim_dir: Path = Path("data/interim")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")


class SyntheticConfig(StrictModel):
    n_clients: int = Field(40_000, gt=0)
    avg_transactions_per_client: float = Field(18.0, gt=0)
    products_catalog_size: int = Field(2_000, gt=0)
    stores_count: int = Field(150, gt=0)
    confounding_strength: float = Field(1.0, ge=0.0)
    uplift_scale: float = Field(0.12, gt=0.0, lt=1.0)
    sleeping_dogs_share: float = Field(0.12, ge=0.0, lt=1.0)


class DataConfig(StrictModel):
    source: Literal["synthetic", "retailhero"] = "synthetic"
    raw_dir: Path = Path("data/raw")
    processed_path: Path = Path("data/processed/feature_store.parquet")
    quarantine_path: Path = Path("data/interim/quarantine.csv")
    outcome: Literal["target", "revenue"] = "target"
    outcome_window_days: int = Field(7, gt=0)
    observation_window_days: int = Field(180, gt=0)
    treatment_share: float | None = Field(None, gt=0.0, lt=1.0)
    chunk_size: int = Field(200_000, gt=0)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)


class SplitConfig(StrictModel):
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    valid_size: float = Field(0.1, ge=0.0, lt=1.0)
    stratify_on: list[str] = Field(default_factory=lambda: ["treatment", "target"])

    @model_validator(mode="after")
    def _sizes_sane(self) -> SplitConfig:
        if self.test_size + self.valid_size >= 0.9:
            msg = "test_size + valid_size должны оставлять ≥10% данных под train"
            raise ValueError(msg)
        return self


class FeaturesConfig(StrictModel):
    corr_threshold: float = Field(0.95, gt=0.0, le=1.0)
    corr_method: Literal["pearson", "spearman"] = "pearson"
    scaler: Literal["zscore", "none"] = "zscore"
    high_cardinality: list[str] = Field(default_factory=list)
    one_hot: list[str] = Field(default_factory=list)
    min_covariates: int = Field(30, gt=0)
    #: Квантили зажатия выбросов; границы считаются на train (``null`` — не зажимать)
    clip_quantiles: tuple[float, float] | None = (0.001, 0.999)
    #: Признаки, которые не исключаются фильтром мультиколлинеарности
    protected: list[str] = Field(default_factory=list)

    @field_validator("clip_quantiles")
    @classmethod
    def _clip_valid(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is None:
            return None
        lo, hi = v
        if not 0.0 <= lo < hi <= 1.0:
            msg = f"features.clip_quantiles требует 0 <= lower < upper <= 1, получено {v}"
            raise ValueError(msg)
        return lo, hi


class CausalConfig(StrictModel):
    graph: Path = Path("configs/dag.gml")
    #: Карта «колонка витрины → узел-группа DAG» (SPEC §6.1)
    feature_groups: Path = Path("configs/feature_groups.yaml")
    identification: Literal["backdoor"] = "backdoor"
    estimators: list[str] = Field(
        default_factory=lambda: ["propensity_score_weighting", "linear_regression"]
    )
    estimation_sample: int | None = Field(50_000, gt=0)
    #: Бутстрэп-итерации DoWhy для ДИ и теста значимости ATE (у propensity-методов
    #: аналитической формулы нет, поэтому каждая итерация — переобучение модели)
    n_simulations: int = Field(50, gt=0)
    expanded_graph_path: Path = Path("artifacts/dag_expanded.gml")

    @field_validator("estimators")
    @classmethod
    def _at_least_two(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            msg = "CAU-3: ATE оценивается минимум двумя методами"
            raise ValueError(msg)
        return v


class LGBMParams(StrictModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    n_estimators: int = Field(500, gt=0)
    learning_rate: float = Field(0.05, gt=0.0)
    num_leaves: int = Field(31, gt=1)
    min_child_samples: int = Field(50, gt=0)
    early_stopping_rounds: int | None = Field(50, gt=0)
    verbose: int = -1

    def sklearn_kwargs(self) -> dict[str, Any]:
        """Параметры для конструктора LightGBM (без служебных ключей)."""
        data = self.model_dump()
        data.pop("early_stopping_rounds", None)
        return data


class XLearnerConfig(StrictModel):
    cross_fitting_folds: int = Field(5, ge=1)
    propensity_clip: tuple[float, float] = (0.01, 0.99)
    backend: Literal["native", "econml"] = "native"

    @field_validator("propensity_clip")
    @classmethod
    def _clip_valid(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not 0.0 < lo < hi < 1.0:
            msg = f"propensity_clip должен быть 0 < lo < hi < 1, получено {v}"
            raise ValueError(msg)
        return v


class ModelConfig(StrictModel):
    lgbm: LGBMParams = Field(default_factory=LGBMParams)
    propensity: LGBMParams = Field(default_factory=LGBMParams)
    xlearner: XLearnerConfig = Field(default_factory=XLearnerConfig)


class BootstrapConfig(StrictModel):
    n_iterations: int = Field(200, gt=0)
    confidence: float = Field(0.95, gt=0.0, lt=1.0)


class EvaluationConfig(StrictModel):
    uplift_at_k: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3])
    n_deciles: int = Field(10, gt=1)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    @field_validator("uplift_at_k")
    @classmethod
    def _shares(cls, v: list[float]) -> list[float]:
        bad = [k for k in v if not 0.0 < k <= 1.0]
        if bad:
            msg = f"uplift_at_k задаётся долями из (0, 1], некорректно: {bad}"
            raise ValueError(msg)
        return v


class RefutationThresholds(StrictModel):
    random_common_cause_rel_tol: float = Field(0.15, gt=0.0)
    placebo_abs_rel_tol: float = Field(0.15, gt=0.0)
    data_subset_rel_tol: float = Field(0.15, gt=0.0)


class RefutationConfig(StrictModel):
    tests: list[str] = Field(
        default_factory=lambda: [
            "random_common_cause",
            "placebo_treatment_refuter",
            "data_subset_refuter",
        ]
    )
    p_value_threshold: float = Field(0.05, gt=0.0, lt=1.0)
    sample_size: int | None = Field(20_000, gt=0)
    num_simulations: int = Field(20, gt=0)
    thresholds: RefutationThresholds = Field(default_factory=RefutationThresholds)
    subset_fraction: float = Field(0.85, gt=0.0, lt=1.0)
    add_unobserved_common_cause: bool = False


class SegmentationConfig(StrictModel):
    theta_high: Threshold = Field(default_factory=lambda: Threshold(raw="q75"))
    theta_low: Threshold = Field(default_factory=lambda: Threshold(raw="q25"))
    control_response_threshold: Threshold = Field(default_factory=lambda: Threshold(raw="q50"))

    _coerce = field_validator(
        "theta_high", "theta_low", "control_response_threshold", mode="before"
    )(_as_threshold)


class EconomicsConfig(StrictModel):
    cost_per_coupon: float = Field(100.0, gt=0.0)
    avg_margin: float = Field(500.0, gt=0.0)
    coverage_share: float | None = Field(None, gt=0.0, le=1.0)


class ExperimentMode(StrictModel):
    name: str
    treatment_share: float | None = Field(None, gt=0.0, lt=1.0)


class ExperimentConfig(StrictModel):
    modes: list[ExperimentMode] = Field(
        default_factory=lambda: [
            ExperimentMode(name="native", treatment_share=None),
            ExperimentMode(name="scarce_15", treatment_share=0.15),
        ]
    )
    models: list[str] = Field(
        default_factory=lambda: [
            "random",
            "naive_classifier",
            "s_learner",
            "t_learner",
            "x_learner",
        ]
    )
    target_model: str = "x_learner"

    @model_validator(mode="after")
    def _target_in_models(self) -> ExperimentConfig:
        if self.target_model not in self.models:
            msg = f"target_model={self.target_model!r} отсутствует в models={self.models}"
            raise ValueError(msg)
        if len(self.models) < 5:
            msg = "критерий приёмки 6: целевая модель + минимум 4 baseline"
            raise ValueError(msg)
        return self


class TrackingConfig(StrictModel):
    runs_dir: Path = Path("artifacts/runs")


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = Path("artifacts/pipeline.log")


# ---------------------------------------------------------------------------
# Корневая конфигурация
# ---------------------------------------------------------------------------


class Config(StrictModel):
    """Корневая схема ``configs/config.yaml``."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    refutation: RefutationConfig = Field(default_factory=RefutationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    economics: EconomicsConfig = Field(default_factory=EconomicsConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # корень проекта: заполняется при загрузке, не часть YAML
    project_root: Path = Field(default_factory=Path.cwd, exclude=True)

    # ------------------------------------------------------------------ загрузка
    @classmethod
    def load(
        cls,
        path: str | Path = "configs/config.yaml",
        *,
        overlays: Iterable[str | Path] = (),
        overrides: Iterable[str] = (),
    ) -> Config:
        """Прочитать YAML, наложить оверлеи и точечные ``--set`` переопределения.

        Args:
            path: путь к основному файлу конфигурации.
            overlays: дополнительные YAML, накладываемые поверх (например
                ``configs/model_lgbm.yaml`` — оверлей секции ``model``).
            overrides: строки вида ``model.lgbm.num_leaves=63``.
        """
        cfg_path = Path(path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.is_file():
            msg = f"файл конфигурации не найден: {cfg_path}"
            raise FileNotFoundError(msg)

        raw = _read_yaml(cfg_path)
        for overlay in overlays:
            overlay_path = Path(overlay)
            data = _read_yaml(overlay_path)
            # Оверлей model_lgbm.yaml описывает содержимое секции `model`
            if overlay_path.name == "model_lgbm.yaml" and "model" not in data:
                data = {"model": data}
            raw = _deep_merge(raw, data)
        for item in overrides:
            raw = _apply_override(raw, item)

        raw["project_root"] = _find_project_root(cfg_path)
        return cls.model_validate(raw)

    # ------------------------------------------------------------------ пути
    def resolve(self, path: str | Path) -> Path:
        """Абсолютный путь: относительные пути считаются от корня проекта."""
        p = Path(path)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.data.raw_dir)

    @property
    def interim_dir(self) -> Path:
        return self.resolve(self.paths.interim_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.paths.artifacts_dir)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.paths.reports_dir)

    @property
    def feature_store_path(self) -> Path:
        return self.resolve(self.data.processed_path)

    def artifact(self, name: str) -> Path:
        return self.artifacts_dir / name

    def report(self, name: str) -> Path:
        return self.reports_dir / name

    def ensure_dirs(self) -> None:
        """Создать все рабочие каталоги (идемпотентно)."""
        for d in (
            self.raw_dir,
            self.interim_dir,
            self.feature_store_path.parent,
            self.artifacts_dir,
            self.resolve(self.tracking.runs_dir),
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ сервис
    def dump(self) -> dict[str, Any]:
        """Сериализуемый снимок конфигурации (для журнала прогонов, ML-5)."""
        return json.loads(self.model_dump_json(exclude={"project_root"}))

    def with_treatment_share(self, share: float | None) -> Config:
        """Копия конфигурации с другим режимом дисбаланса (SPEC §5.1)."""
        clone = self.model_copy(deep=True)
        clone.data.treatment_share = share
        return clone


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"YAML-файл не найден: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        msg = f"ожидался словарь на верхнем уровне {path}, получено {type(data).__name__}"
        raise TypeError(msg)
    return data


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_override(raw: dict[str, Any], item: str) -> dict[str, Any]:
    """Применить ``a.b.c=value``; значение парсится как YAML-скаляр."""
    if "=" not in item:
        msg = f"переопределение должно иметь вид key.path=value, получено {item!r}"
        raise ValueError(msg)
    key_path, _, value_str = item.partition("=")
    keys = [k for k in key_path.strip().split(".") if k]
    if not keys:
        msg = f"пустой путь ключа в переопределении {item!r}"
        raise ValueError(msg)
    value = yaml.safe_load(value_str)
    out = dict(raw)
    node: dict[str, Any] = out
    for key in keys[:-1]:
        child = node.get(key)
        node[key] = dict(child) if isinstance(child, dict) else {}
        node = node[key]
    node[keys[-1]] = value
    return out


def _find_project_root(config_path: Path) -> Path:
    """Корень проекта — ближайший каталог вверх с ``pyproject.toml``."""
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent.parent


__all__ = [
    "BootstrapConfig",
    "CausalConfig",
    "Config",
    "DataConfig",
    "EconomicsConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExperimentMode",
    "FeaturesConfig",
    "LGBMParams",
    "ModelConfig",
    "RefutationConfig",
    "SegmentationConfig",
    "SplitConfig",
    "SyntheticConfig",
    "Threshold",
    "XLearnerConfig",
]
