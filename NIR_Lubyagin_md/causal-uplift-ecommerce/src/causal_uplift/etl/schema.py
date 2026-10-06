"""Схемы и контракты данных (ETL-1, ETL-6).

Схема задаётся явно, а не выводится pandas'ом: это и защита от «тихого»
изменения типов при смене источника (синтетика ↔ RetailHero), и источник
правил валидации — строки, не прошедшие проверку, уходят в карантин.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Описание схемы таблицы
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Контракт одной колонки."""

    name: str
    dtype: str
    nullable: bool = False
    # предикат валидности значения: True = строка валидна
    check: Callable[[pd.Series], pd.Series] | None = None
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Контракт таблицы: колонки, ключ, даты."""

    name: str
    columns: tuple[ColumnSpec, ...]
    key: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    description: str = ""

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if not c.nullable)

    def spec(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        msg = f"колонка {name!r} отсутствует в схеме {self.name!r}"
        raise KeyError(msg)

    def pandas_dtypes(self) -> dict[str, str]:
        """dtypes для ``read_csv`` (без datetime — они парсятся отдельно)."""
        return {
            c.name: c.dtype
            for c in self.columns
            if c.name not in self.datetime_columns and c.dtype != "datetime64[ns]"
        }

    # ------------------------------------------------------------------ проверки
    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.column_names if c not in df.columns]

    def validate_rows(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Разделить строки на валидные и карантинные (ETL-1).

        Returns:
            ``(valid, quarantined)``; в карантинном кадре добавлена колонка
            ``_quarantine_reason`` с причиной отбраковки.
        """
        if df.empty:
            return df, df.assign(_quarantine_reason=pd.Series(dtype="object"))

        reasons = pd.Series("", index=df.index, dtype="object")

        def add_reason(mask: pd.Series, text: str) -> None:
            if mask.any():
                reasons.loc[mask & (reasons == "")] = text

        for col in self.columns:
            if col.name not in df.columns:
                continue
            series = df[col.name]
            if not col.nullable:
                add_reason(series.isna(), f"пустое значение в {col.name}")
            if col.check is not None:
                bad = ~col.check(series) & series.notna()
                add_reason(bad, f"значение вне допустимого диапазона в {col.name}")

        bad_mask = reasons != ""
        valid = df.loc[~bad_mask]
        quarantined = df.loc[bad_mask].assign(_quarantine_reason=reasons.loc[bad_mask])
        return valid, quarantined


# ---------------------------------------------------------------------------
# Предикаты
# ---------------------------------------------------------------------------


def _non_negative(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce") >= 0


def _positive(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce") > 0


def _binary(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").isin([0, 1])


def _plausible_age(s: pd.Series) -> pd.Series:
    age = pd.to_numeric(s, errors="coerce")
    return (age >= 14) & (age <= 100)


def _known_gender(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().isin(["F", "M", "U"])


def _non_empty_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.len() > 0


# ---------------------------------------------------------------------------
# Схемы сырых таблиц RetailHero (SPEC §5.1)
# ---------------------------------------------------------------------------

CLIENTS_SCHEMA = TableSchema(
    name="clients",
    description="Профиль клиента: даты вступления в программу лояльности и соц-дем",
    key=("client_id",),
    datetime_columns=("first_issue_date", "first_redeem_date"),
    columns=(
        ColumnSpec(
            "client_id", "string", check=_non_empty_str, description="идентификатор клиента"
        ),
        ColumnSpec("first_issue_date", "datetime64[ns]", description="выдача карты лояльности"),
        ColumnSpec(
            "first_redeem_date",
            "datetime64[ns]",
            nullable=True,
            description="первое списание баллов; NaT = баллы не использовались",
        ),
        ColumnSpec("age", "float32", nullable=True, check=_plausible_age, description="возраст"),
        ColumnSpec("gender", "string", nullable=True, check=_known_gender, description="пол F/M/U"),
    ),
)

UPLIFT_TRAIN_SCHEMA = TableSchema(
    name="uplift_train",
    description="Разметка эксперимента: флаг воздействия T и целевой отклик Y",
    key=("client_id",),
    columns=(
        ColumnSpec("client_id", "string", check=_non_empty_str),
        ColumnSpec("treatment_flg", "int8", check=_binary, description="T ∈ {0,1}"),
        ColumnSpec("target", "int8", check=_binary, description="Y ∈ {0,1}"),
    ),
)

PURCHASES_SCHEMA = TableSchema(
    name="purchases",
    description="Товарные строки транзакций (самая крупная таблица, читается чанками)",
    key=("client_id", "transaction_id", "product_id"),
    datetime_columns=("transaction_datetime",),
    columns=(
        ColumnSpec("client_id", "string", check=_non_empty_str),
        ColumnSpec("transaction_id", "string", check=_non_empty_str),
        ColumnSpec("transaction_datetime", "datetime64[ns]"),
        ColumnSpec("regular_points_received", "float32", check=_non_negative),
        ColumnSpec("express_points_received", "float32", check=_non_negative),
        ColumnSpec("regular_points_spent", "float32", description="значения ≤ 0 (списание)"),
        ColumnSpec("express_points_spent", "float32", description="значения ≤ 0 (списание)"),
        ColumnSpec("purchase_sum", "float32", check=_positive, description="сумма чека"),
        ColumnSpec("store_id", "string", check=_non_empty_str),
        ColumnSpec("product_id", "string", check=_non_empty_str),
        ColumnSpec("product_quantity", "float32", check=_positive),
        ColumnSpec("trn_sum_from_iss", "float32", nullable=True, check=_non_negative),
        ColumnSpec(
            "trn_sum_from_red",
            "float32",
            nullable=True,
            check=_non_negative,
            description="часть чека, оплаченная баллами",
        ),
    ),
)

PRODUCTS_SCHEMA = TableSchema(
    name="products",
    description="Товарный справочник: иерархия категорий и атрибуты",
    key=("product_id",),
    columns=(
        ColumnSpec("product_id", "string", check=_non_empty_str),
        ColumnSpec("level_1", "string", nullable=True),
        ColumnSpec("level_2", "string", nullable=True),
        ColumnSpec("level_3", "string", nullable=True),
        ColumnSpec("level_4", "string", nullable=True),
        ColumnSpec("segment_id", "string", nullable=True),
        ColumnSpec("brand_id", "string", nullable=True),
        ColumnSpec("vendor_id", "string", nullable=True),
        ColumnSpec("netto", "float32", nullable=True, check=_non_negative),
        ColumnSpec("is_own_trademark", "int8", nullable=True, check=_binary),
        ColumnSpec("is_alcohol", "int8", nullable=True, check=_binary),
    ),
)

RAW_SCHEMAS: Mapping[str, TableSchema] = {
    "clients": CLIENTS_SCHEMA,
    "uplift_train": UPLIFT_TRAIN_SCHEMA,
    "purchases": PURCHASES_SCHEMA,
    "products": PRODUCTS_SCHEMA,
}

# ---------------------------------------------------------------------------
# Контракт витрины (ETL-6)
# ---------------------------------------------------------------------------

#: Колонки витрины, не являющиеся ковариатами
FEATURE_STORE_RESERVED: tuple[str, ...] = (
    "client_id",
    "treatment",
    "target",
    "revenue",
    "split",
    "true_tau",  # только для синтетики: ground truth для sanity-теста §11
    "true_mu0",
)


@dataclass
class ContractViolation:
    """Нарушение контракта витрины."""

    rule: str
    detail: str


@dataclass
class FeatureStoreContract:
    """Проверка строгого контракта ``feature_store.parquet`` (ETL-6)."""

    min_covariates: int = 30
    outcome: str = "target"

    required_columns: tuple[str, ...] = field(default=("client_id", "treatment", "target"))

    def covariates(self, df: pd.DataFrame) -> list[str]:
        """Колонки ковариат ``X``: всё, кроме служебных."""
        return [c for c in df.columns if c not in FEATURE_STORE_RESERVED]

    def validate(self, df: pd.DataFrame) -> list[ContractViolation]:
        violations: list[ContractViolation] = []

        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            violations.append(
                ContractViolation("обязательные колонки", f"отсутствуют: {', '.join(missing)}")
            )

        if self.outcome == "revenue" and "revenue" not in df.columns:
            violations.append(
                ContractViolation("исход", "outcome=revenue, но колонка revenue отсутствует")
            )

        covariates = self.covariates(df)
        if len(covariates) < self.min_covariates:
            violations.append(
                ContractViolation(
                    "число ковариат",
                    f"{len(covariates)} < {self.min_covariates} (критерий приёмки 3)",
                )
            )

        if "treatment" in df.columns:
            uniq = set(pd.unique(df["treatment"].dropna()))
            if not uniq <= {0, 1}:
                violations.append(
                    ContractViolation(
                        "treatment", f"ожидались значения 0/1, получено {sorted(uniq)}"
                    )
                )
            if df["treatment"].isna().any():
                violations.append(ContractViolation("treatment", "присутствуют NaN"))

        if "target" in df.columns:
            if df["target"].isna().any():
                violations.append(ContractViolation("target", "присутствуют NaN"))
            uniq_y = set(pd.unique(df["target"].dropna()))
            if not uniq_y <= {0, 1}:
                violations.append(
                    ContractViolation(
                        "target", f"ожидались значения 0/1, получено {sorted(uniq_y)}"
                    )
                )

        if covariates:
            numeric = df[covariates].select_dtypes(include=[np.number])
            non_numeric = sorted(set(covariates) - set(numeric.columns))
            if non_numeric:
                violations.append(
                    ContractViolation(
                        "типы ковариат",
                        f"нечисловые колонки в X: {', '.join(non_numeric[:10])}",
                    )
                )
            nan_cols = [c for c in numeric.columns if numeric[c].isna().any()]
            if nan_cols:
                violations.append(
                    ContractViolation(
                        "NaN в X", f"колонки с пропусками: {', '.join(nan_cols[:10])}"
                    )
                )
            inf_cols = [c for c in numeric.columns if np.isinf(numeric[c].to_numpy()).any()]
            if inf_cols:
                violations.append(
                    ContractViolation("inf в X", f"колонки с inf: {', '.join(inf_cols[:10])}")
                )

        if "client_id" in df.columns and df["client_id"].duplicated().any():
            dup = int(df["client_id"].duplicated().sum())
            violations.append(ContractViolation("уникальность", f"дубли client_id: {dup}"))

        return violations

    def assert_valid(self, df: pd.DataFrame) -> None:
        violations = self.validate(df)
        if violations:
            details = "\n".join(f"  - [{v.rule}] {v.detail}" for v in violations)
            msg = f"витрина нарушает контракт ETL-6:\n{details}"
            raise ValueError(msg)


def cast_to_schema(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Привести типы кадра к схеме; незаполнимые значения → NaN/NaT."""
    out = df.copy()
    for col in schema.columns:
        if col.name not in out.columns:
            continue
        if col.name in schema.datetime_columns or col.dtype == "datetime64[ns]":
            out[col.name] = pd.to_datetime(out[col.name], errors="coerce")
        elif col.dtype.startswith(("float", "int")):
            numeric = pd.to_numeric(out[col.name], errors="coerce")
            if col.dtype.startswith("int"):
                # int не держит NaN — безопасно кастуем только полностью заполненные колонки
                out[col.name] = (
                    numeric.astype(col.dtype)
                    if not numeric.isna().any()
                    else numeric.astype("float32")
                )
            else:
                out[col.name] = numeric.astype(col.dtype)
        else:
            out[col.name] = out[col.name].astype(col.dtype)
    return out


def describe_schema(schema: TableSchema) -> list[str]:
    """Markdown-описание схемы (для docs/reports)."""
    lines = [
        f"### `{schema.name}`",
        "",
        schema.description,
        "",
        "| Колонка | Тип | Nullable | Смысл |",
        "|---|---|---|---|",
    ]
    for col in schema.columns:
        lines.append(
            f"| `{col.name}` | `{col.dtype}` | {'да' if col.nullable else 'нет'} | "
            f"{col.description or '—'} |"
        )
    if schema.key:
        lines += ["", f"Ключ: `{', '.join(schema.key)}`"]
    return lines


def schema_summary() -> dict[str, Any]:
    """Снимок всех схем — попадает в журнал прогона для отслеживания контрактов."""
    return {
        name: {
            "columns": list(schema.column_names),
            "key": list(schema.key),
            "datetime_columns": list(schema.datetime_columns),
        }
        for name, schema in RAW_SCHEMAS.items()
    }


def covariate_columns(df: pd.DataFrame, extra_exclude: Sequence[str] = ()) -> list[str]:
    """Список колонок-ковариат витрины (служебные исключены)."""
    excluded = set(FEATURE_STORE_RESERVED) | set(extra_exclude)
    return [c for c in df.columns if c not in excluded]


__all__ = [
    "CLIENTS_SCHEMA",
    "FEATURE_STORE_RESERVED",
    "PRODUCTS_SCHEMA",
    "PURCHASES_SCHEMA",
    "RAW_SCHEMAS",
    "UPLIFT_TRAIN_SCHEMA",
    "ColumnSpec",
    "ContractViolation",
    "FeatureStoreContract",
    "TableSchema",
    "cast_to_schema",
    "covariate_columns",
    "describe_schema",
    "schema_summary",
]
