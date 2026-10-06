"""Z-score нормализация с защитой от утечки (ETL-4).

Параметры (``mean``, ``std``) считаются **только на train-подвыборке** и
сохраняются в ``artifacts/scaler.json``. Любой другой порядок — утечка
распределения теста в обучение (риск §14, leakage-тест §11).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from causal_uplift.io_utils import load_json, save_json
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Ниже этого значения std считается нулевым (константная колонка)
_STD_EPS = 1e-9


@dataclass
class ZScoreScaler:
    """Нормализация ``(x − mean) / std`` по параметрам train-выборки."""

    columns: list[str] = field(default_factory=list)
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    fitted_on_rows: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, columns: Sequence[str] | None = None) -> Self:
        """Посчитать параметры по кадру (должен быть **train**-подвыборкой)."""
        cols = list(columns) if columns is not None else list(
            df.select_dtypes(include=[np.number]).columns
        )
        self.columns = cols
        self.fitted_on_rows = len(df)
        self.mean_ = {}
        self.std_ = {}
        constant: list[str] = []
        for col in cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
            mean = float(np.nanmean(values)) if values.size else 0.0
            std = float(np.nanstd(values)) if values.size else 0.0
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std < _STD_EPS:
                constant.append(col)
                std = 1.0
            self.mean_[col] = mean
            self.std_[col] = std
        if constant:
            logger.warning(
                "константные на train колонки (std≈0, нормируются как есть): %s",
                ", ".join(constant[:10]),
            )
        logger.info("ZScoreScaler обучен на %d строках, %d колонок", len(df), len(cols))
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            msg = "ZScoreScaler не обучен: вызовите fit() на train-подвыборке"
            raise RuntimeError(msg)
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            msg = f"в кадре отсутствуют нормируемые колонки: {', '.join(missing[:10])}"
            raise KeyError(msg)

        out = df.copy()
        for col in self.columns:
            mean = self.mean_[col]
            std = self.std_[col] or 1.0
            out[col] = ((pd.to_numeric(out[col], errors="coerce") - mean) / std).astype("float32")
        # Нормализация не должна порождать NaN/inf: контракт ETL-6
        out[self.columns] = out[self.columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    # ------------------------------------------------------------------ сериализация
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "zscore",
            "columns": self.columns,
            "mean": self.mean_,
            "std": self.std_,
            "fitted_on_rows": self.fitted_on_rows,
        }

    def save(self, path: str | Path) -> Path:
        return save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ZScoreScaler:
        return cls(
            columns=list(payload.get("columns", [])),
            mean_={k: float(v) for k, v in payload.get("mean", {}).items()},
            std_={k: float(v) for k, v in payload.get("std", {}).items()},
            fitted_on_rows=int(payload.get("fitted_on_rows", 0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> ZScoreScaler:
        return cls.from_dict(load_json(path))


class IdentityScaler(ZScoreScaler):
    """Заглушка для ``features.scaler: none`` — удобна в тестах метрик."""

    def fit(self, df: pd.DataFrame, columns: Sequence[str] | None = None) -> Self:
        self.columns = list(columns) if columns is not None else []
        self.mean_ = dict.fromkeys(self.columns, 0.0)
        self.std_ = dict.fromkeys(self.columns, 1.0)
        self.fitted_on_rows = len(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["type"] = "identity"
        return payload


def make_scaler(kind: str) -> ZScoreScaler:
    """Фабрика скейлера по значению ``features.scaler``."""
    if kind == "zscore":
        return ZScoreScaler()
    if kind == "none":
        return IdentityScaler()
    msg = f"неизвестный скейлер: {kind!r}"
    raise ValueError(msg)


__all__ = ["IdentityScaler", "ZScoreScaler", "make_scaler"]
