"""Загрузка сырых таблиц с явной схемой и карантином (ETL-1, ETL-7).

Два режима чтения:

* небольшие справочники (``clients``, ``products``, ``uplift_train``) читаются
  целиком с приведением типов и валидацией;
* ``purchases`` — только итератором по чанкам: конвейер не должен требовать
  загрузки всех транзакций в RAM (ETL-7).

Невалидные строки не отбрасываются молча: они уходят в карантинный CSV с
указанием причины (ETL-1), и их число попадает в журнал прогона.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from causal_uplift.config import Config
from causal_uplift.etl.schema import RAW_SCHEMAS, TableSchema, cast_to_schema
from causal_uplift.io_utils import append_csv
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Расширения, в которых ищется каждая сырая таблица (ETL-1: CSV или Parquet)
_SUPPORTED_SUFFIXES = (".parquet", ".csv", ".csv.gz", ".csv.zip", ".tsv")


class RawDataNotFoundError(FileNotFoundError):
    """Сырая таблица отсутствует в ``data/raw``."""


@dataclass
class LoadStats:
    """Статистика загрузки одной таблицы — попадает в журнал и EDA-отчёт."""

    table: str
    rows_read: int = 0
    rows_valid: int = 0
    rows_quarantined: int = 0
    rows_duplicated: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def merge(self, other: LoadStats) -> None:
        self.rows_read += other.rows_read
        self.rows_valid += other.rows_valid
        self.rows_quarantined += other.rows_quarantined
        self.rows_duplicated += other.rows_duplicated
        for reason, count in other.reasons.items():
            self.reasons[reason] = self.reasons.get(reason, 0) + count

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "rows_quarantined": self.rows_quarantined,
            "rows_duplicated": self.rows_duplicated,
            "quarantine_reasons": dict(sorted(self.reasons.items(), key=lambda kv: -kv[1])),
        }


# ---------------------------------------------------------------------------
# Поиск файлов
# ---------------------------------------------------------------------------


def find_raw_file(cfg: Config, table: str) -> Path:
    """Найти файл таблицы в ``data/raw`` по любому поддерживаемому расширению."""
    raw_dir = cfg.raw_dir
    for suffix in _SUPPORTED_SUFFIXES:
        candidate = raw_dir / f"{table}{suffix}"
        if candidate.is_file():
            return candidate
    msg = (
        f"таблица {table!r} не найдена в {raw_dir}. Ожидался один из файлов: "
        + ", ".join(f"{table}{s}" for s in _SUPPORTED_SUFFIXES)
        + ". Для синтетического источника выполните `make data`; для RetailHero "
        "распакуйте архив датасета в data/raw."
    )
    raise RawDataNotFoundError(msg)


def available_tables(cfg: Config) -> dict[str, Path]:
    """Какие из известных таблиц реально лежат в ``data/raw``."""
    found: dict[str, Path] = {}
    for table in RAW_SCHEMAS:
        try:
            found[table] = find_raw_file(cfg, table)
        except RawDataNotFoundError:
            continue
    return found


def check_retailhero_presence(cfg: Config) -> list[str]:
    """Вернуть список отсутствующих обязательных таблиц RetailHero."""
    required = ("clients", "purchases", "products", "uplift_train")
    present = available_tables(cfg)
    return [t for t in required if t not in present]


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------


def _read_whole(path: Path, schema: TableSchema, columns: Sequence[str] | None) -> pd.DataFrame:
    usecols = list(columns) if columns else None
    if path.suffix == ".parquet":
        return pd.read_parquet(path, engine="pyarrow", columns=usecols)
    sep = "\t" if path.suffix == ".tsv" else ","
    # dtype=str на чтении + явный каст: иначе pandas молча угадает типы
    # по-разному для разных источников (риск «тихого» расхождения схем).
    return pd.read_csv(
        path,
        sep=sep,
        usecols=usecols,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "nan", "NaN", "None", "null"],
        low_memory=False,
    )


def _quarantine(cfg: Config, table: str, bad: pd.DataFrame) -> dict[str, int]:
    """Сложить отбракованные строки в карантинный файл, вернуть счётчик причин."""
    if bad.empty:
        return {}
    reasons = bad["_quarantine_reason"].value_counts().to_dict()
    payload = bad.assign(_table=table)
    append_csv(payload, cfg.resolve(cfg.data.quarantine_path))
    logger.warning(
        "таблица %s: %d строк в карантине (%s)",
        table,
        len(bad),
        "; ".join(f"{k}: {v}" for k, v in list(reasons.items())[:3]),
    )
    return {str(k): int(v) for k, v in reasons.items()}


def load_table(
    cfg: Config,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, LoadStats]:
    """Прочитать небольшую таблицу целиком, привести типы и провалидировать."""
    schema = RAW_SCHEMAS[table]
    path = find_raw_file(cfg, table)
    raw = _read_whole(path, schema, columns)

    stats = LoadStats(table=table, rows_read=len(raw))
    missing = [c for c in (columns or schema.column_names) if c not in raw.columns]
    if missing:
        msg = f"в {path} отсутствуют колонки схемы {schema.name}: {', '.join(missing)}"
        raise ValueError(msg)

    typed = cast_to_schema(raw, schema)
    if not validate:
        stats.rows_valid = len(typed)
        return typed, stats

    valid, bad = schema.validate_rows(typed)
    stats.reasons = _quarantine(cfg, table, bad)
    stats.rows_quarantined = len(bad)

    if schema.key and all(k in valid.columns for k in schema.key):
        before = len(valid)
        valid = valid.drop_duplicates(subset=list(schema.key), keep="first")
        stats.rows_duplicated = before - len(valid)

    stats.rows_valid = len(valid)
    logger.info(
        "загружено %s: %d строк (валидных %d, карантин %d, дублей %d) из %s",
        table,
        stats.rows_read,
        stats.rows_valid,
        stats.rows_quarantined,
        stats.rows_duplicated,
        path.name,
    )
    return valid.reset_index(drop=True), stats


def iter_purchases(
    cfg: Config,
    *,
    columns: Sequence[str] | None = None,
    validate: bool = True,
) -> Iterator[tuple[pd.DataFrame, LoadStats]]:
    """Итератор чанков таблицы ``purchases`` (ETL-7).

    Yields:
        Кортежи ``(чанк, статистика чанка)``. Типы приведены по схеме,
        невалидные строки уже отправлены в карантин, дубли внутри чанка сняты
        (межчанковая дедупликация — в :mod:`causal_uplift.etl.cleaning`).
    """
    schema = RAW_SCHEMAS["purchases"]
    path = find_raw_file(cfg, "purchases")
    usecols = list(columns) if columns else None

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = parquet_file.iter_batches(batch_size=cfg.data.chunk_size, columns=usecols)
        raw_chunks: Iterator[pd.DataFrame] = (b.to_pandas() for b in batches)
    else:
        sep = "\t" if path.suffix == ".tsv" else ","
        raw_chunks = pd.read_csv(
            path,
            sep=sep,
            usecols=usecols,
            dtype=str,
            keep_default_na=True,
            na_values=["", "NA", "nan", "NaN", "None", "null"],
            chunksize=cfg.data.chunk_size,
        )

    for chunk_no, raw in enumerate(raw_chunks):
        stats = LoadStats(table="purchases", rows_read=len(raw))
        typed = cast_to_schema(raw, schema)

        if validate:
            valid, bad = schema.validate_rows(typed)
            stats.reasons = _quarantine(cfg, "purchases", bad)
            stats.rows_quarantined = len(bad)
        else:
            valid = typed

        if all(k in valid.columns for k in schema.key):
            before = len(valid)
            valid = valid.drop_duplicates(subset=list(schema.key), keep="first")
            stats.rows_duplicated = before - len(valid)

        stats.rows_valid = len(valid)
        logger.debug(
            "purchases чанк %d: %d → %d строк (карантин %d, дубли %d)",
            chunk_no,
            stats.rows_read,
            stats.rows_valid,
            stats.rows_quarantined,
            stats.rows_duplicated,
        )
        yield valid.reset_index(drop=True), stats


def scan_datetime_bounds(cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Границы времени наблюдения по ``purchases``, одна колонка, чанками.

    Нужно, чтобы окно наблюдения (ETL-3) определялось по данным, а не по
    системной дате: дата последнего события — конец окна и для синтетики,
    и для RetailHero.
    """
    lo: pd.Timestamp | None = None
    hi: pd.Timestamp | None = None
    for chunk, _ in iter_purchases(cfg, columns=["transaction_datetime"], validate=False):
        series = pd.to_datetime(chunk["transaction_datetime"], errors="coerce").dropna()
        if series.empty:
            continue
        chunk_lo, chunk_hi = series.min(), series.max()
        lo = chunk_lo if lo is None else min(lo, chunk_lo)
        hi = chunk_hi if hi is None else max(hi, chunk_hi)
    if lo is None or hi is None:
        msg = "не удалось определить границы transaction_datetime: нет валидных дат"
        raise ValueError(msg)
    logger.info("окно данных purchases: %s … %s", lo.date(), hi.date())
    return lo, hi


__all__ = [
    "LoadStats",
    "RawDataNotFoundError",
    "available_tables",
    "check_retailhero_presence",
    "find_raw_file",
    "iter_purchases",
    "load_table",
    "scan_datetime_bounds",
]
