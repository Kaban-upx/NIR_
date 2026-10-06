"""Кодирование категориальных признаков (ETL-5).

* ``gender`` — one-hot по категориям, встреченным на train;
* высококардинальные (``favorite_store_id``, ``favorite_brand_id``) — частотное
  кодирование, посчитанное **на train-фолде**.

Целевое кодирование сознательно не используется: при uplift-постановке оно
протекает не только через ``Y``, но и через ``T`` (частоты различаются между
группами), и отделить утечку от сигнала сложнее, чем она того стоит. Частотное
кодирование даёт тот же выигрыш по размерности без контакта с целевой
переменной — см. docs/architecture.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd

from causal_uplift.io_utils import load_json, save_json
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Значение для категорий, не встреченных на train
UNSEEN_CATEGORY = "__unseen__"


@dataclass
class CategoricalEncoder:
    """One-hot для низкокардинальных и частотное кодирование для остальных."""

    one_hot_columns: list[str] = field(default_factory=list)
    frequency_columns: list[str] = field(default_factory=list)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    frequencies_: dict[str, dict[str, float]] = field(default_factory=dict)
    fitted_on_rows: int = 0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df: pd.DataFrame,
        *,
        one_hot: Sequence[str] = (),
        frequency: Sequence[str] = (),
    ) -> Self:
        """Запомнить категории и частоты по **train**-подвыборке."""
        self.one_hot_columns = [c for c in one_hot if c in df.columns]
        self.frequency_columns = [c for c in frequency if c in df.columns]
        self.fitted_on_rows = len(df)

        self.categories_ = {}
        for col in self.one_hot_columns:
            values = df[col].astype("string").fillna(UNSEEN_CATEGORY)
            self.categories_[col] = sorted(values.unique().tolist())

        self.frequencies_ = {}
        for col in self.frequency_columns:
            values = df[col].astype("string").fillna(UNSEEN_CATEGORY)
            freq = values.value_counts(normalize=True)
            self.frequencies_[col] = {str(k): float(v) for k, v in freq.items()}

        logger.info(
            "CategoricalEncoder обучен на %d строках: one-hot %s, частотное %s",
            len(df),
            self.one_hot_columns or "—",
            self.frequency_columns or "—",
        )
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, *, drop_source: bool = True) -> pd.DataFrame:
        out = df.copy()

        for col in self.one_hot_columns:
            if col not in out.columns:
                continue
            values = out[col].astype("string").fillna(UNSEEN_CATEGORY)
            for category in self.categories_.get(col, []):
                safe = str(category).replace(" ", "_")
                out[f"{col}_{safe}"] = (values == category).astype("float32")
            if drop_source:
                out = out.drop(columns=[col])

        for col in self.frequency_columns:
            if col not in out.columns:
                continue
            values = out[col].astype("string").fillna(UNSEEN_CATEGORY)
            mapping = self.frequencies_.get(col, {})
            # Невстреченная на train категория получает 0.0 — «такого раньше не было»
            out[f"{col}_freq"] = values.map(mapping).astype("float32").fillna(0.0)
            if drop_source:
                out = out.drop(columns=[col])

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        one_hot: Sequence[str] = (),
        frequency: Sequence[str] = (),
        drop_source: bool = True,
    ) -> pd.DataFrame:
        return self.fit(df, one_hot=one_hot, frequency=frequency).transform(
            df, drop_source=drop_source
        )

    # ------------------------------------------------------------------ сериализация
    def to_dict(self) -> dict[str, Any]:
        return {
            "one_hot_columns": self.one_hot_columns,
            "frequency_columns": self.frequency_columns,
            "categories": self.categories_,
            "frequencies": self.frequencies_,
            "fitted_on_rows": self.fitted_on_rows,
        }

    def save(self, path: str | Path) -> Path:
        return save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CategoricalEncoder:
        return cls(
            one_hot_columns=list(payload.get("one_hot_columns", [])),
            frequency_columns=list(payload.get("frequency_columns", [])),
            categories_={k: list(v) for k, v in payload.get("categories", {}).items()},
            frequencies_={
                k: {str(kk): float(vv) for kk, vv in v.items()}
                for k, v in payload.get("frequencies", {}).items()
            },
            fitted_on_rows=int(payload.get("fitted_on_rows", 0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> CategoricalEncoder:
        return cls.from_dict(load_json(path))

    # ------------------------------------------------------------------ сервис
    def output_columns(self) -> list[str]:
        """Имена колонок, которые добавит :meth:`transform`."""
        cols: list[str] = []
        for col in self.one_hot_columns:
            cols += [f"{col}_{str(c).replace(' ', '_')}" for c in self.categories_.get(col, [])]
        cols += [f"{col}_freq" for col in self.frequency_columns]
        return cols


__all__ = ["UNSEEN_CATEGORY", "CategoricalEncoder"]
