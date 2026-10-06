"""Общие утилиты ввода-вывода и воспроизводимости.

Здесь только то, что нужно всем слоям: детерминизм (ML-1), атомарная запись
артефактов и единый формат JSON/Markdown-выгрузок для ВКР.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Воспроизводимость (ML-1)
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> np.random.Generator:
    """Зафиксировать все источники случайности и вернуть именованный generator.

    Глобальные сиды нужны сторонним библиотекам (LightGBM, sklearn, dowhy),
    а прикладной код пользуется возвращаемым ``Generator`` — он
    воспроизводим независимо от порядка вызовов в других модулях.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — требуется для sklearn/lightgbm legacy-путей
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def utc_now_iso() -> str:
    """Текущее время в ISO-8601 (UTC) — для журнала прогонов."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp | datetime):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    msg = f"не сериализуется в JSON: {type(obj).__name__}"
    raise TypeError(msg)


def save_json(payload: Mapping[str, Any] | Sequence[Any], path: str | Path) -> Path:
    """Атомарно записать JSON (UTF-8, с отступами, без ASCII-эскейпов)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
        fh.write("\n")
    tmp.replace(target)
    return target


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Parquet / CSV
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, engine="pyarrow", index=index, compression="snappy")
    return target


def load_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", columns=list(columns) if columns else None)


def append_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Дописать строки в CSV (используется для карантина невалидных строк, ETL-1)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = not target.exists()
    df.to_csv(target, mode="a", header=header, index=False, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Markdown-отчёты (выгрузки для ВКР, W13)
# ---------------------------------------------------------------------------
def write_markdown(lines: Sequence[str], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    target.write_text(text, encoding="utf-8")
    return target


def df_to_markdown(df: pd.DataFrame, *, floatfmt: str = "{:.4f}", index: bool = False) -> str:
    """Markdown-таблица без зависимости от tabulate."""
    view = df.reset_index() if index else df
    formatted = view.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda v: "—" if pd.isna(v) else floatfmt.format(v)  # noqa: B023
            )
        else:
            formatted[col] = formatted[col].astype(str)
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    sep = "|" + "|".join("---" for _ in formatted.columns) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def fmt_float(value: float | None, digits: int = 4) -> str:
    """Форматирование числа для отчётов: None/NaN → тире."""
    if value is None:
        return "—"
    fvalue = float(value)
    if not np.isfinite(fvalue):
        return "—"
    return f"{fvalue:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


__all__ = [
    "append_csv",
    "df_to_markdown",
    "fmt_float",
    "fmt_pct",
    "load_json",
    "load_parquet",
    "save_json",
    "save_parquet",
    "set_global_seed",
    "utc_now_iso",
    "write_markdown",
]
