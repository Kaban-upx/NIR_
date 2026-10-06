"""Фаза A ETL: товарные строки → транзакционный уровень (ETL-7).

``purchases`` — единственная таблица, которая не помещается в память целиком,
поэтому она обрабатывается строго потоком чанков. Внутри одного прохода
считаются:

* агрегаты уровня чека (число позиций, доля СТМ и алкоголя в корзине);
* накопители уникальных пар «клиент × категория/магазин/бренд» — из них
  витрина получает ассортиментные признаки без повторного чтения логов;
* частоты пар «клиент × магазин» и «клиент × бренд» — для определения
  любимого магазина/бренда (высококардинальные признаки, ETL-5).

Результат — ``data/interim/transactions.parquet``: на порядок компактнее
сырых логов, и уже с него считаются клиентские признаки (фаза B, W4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from causal_uplift.config import Config
from causal_uplift.etl.cleaning import (
    CrossChunkDeduplicator,
    ObservationWindow,
    build_window,
    filter_observation_window,
)
from causal_uplift.etl.loaders import LoadStats, iter_purchases, load_table, scan_datetime_bounds
from causal_uplift.etl.schema import PURCHASES_SCHEMA
from causal_uplift.io_utils import load_json, save_json, save_parquet
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Окно наблюдения фиксируется на диске: признаки, причинный слой и дашборд
#: обязаны видеть одни и те же границы (ETL-3)
WINDOW_ARTIFACT = "window.json"

#: Уровни товарной иерархии, по которым считается разнообразие корзины
CATEGORY_LEVELS: tuple[str, ...] = ("level_1", "level_2", "level_3", "level_4")

#: Правила слияния частичных агрегатов чека (чек может разъехаться по чанкам)
_MERGE_FIRST = ("transaction_datetime", "store_id")
_MERGE_MAX = (
    "purchase_sum",
    "trn_sum_from_iss",
    "trn_sum_from_red",
    "regular_points_received",
    "express_points_received",
)
_MERGE_MIN = ("regular_points_spent", "express_points_spent")  # списания отрицательны
_MERGE_SUM = (
    "n_items",
    "quantity_total",
    "n_unique_products",
    "own_trademark_items",
    "alcohol_items",
    "netto_total",
)


class UniquePairsAccumulator:
    """Накопитель уникальных пар (клиент × значение) поверх потока чанков."""

    def __init__(self, value_column: str, *, key_column: str = "client_id") -> None:
        self.key_column = key_column
        self.value_column = value_column
        self._pairs: pd.DataFrame | None = None

    def update(self, df: pd.DataFrame) -> None:
        if self.value_column not in df.columns or df.empty:
            return
        pairs = df[[self.key_column, self.value_column]].dropna().drop_duplicates()
        if pairs.empty:
            return
        self._pairs = (
            pairs
            if self._pairs is None
            else pd.concat([self._pairs, pairs], ignore_index=True).drop_duplicates()
        )

    def counts(self, name: str) -> pd.Series:
        """Число уникальных значений на клиента."""
        if self._pairs is None or self._pairs.empty:
            return pd.Series(dtype="int32", name=name)
        out = self._pairs.groupby(self.key_column, observed=True)[self.value_column].nunique()
        out.name = name
        return out.astype("int32")


class PairCountAccumulator:
    """Накопитель частот пар (клиент × значение) — для «любимого» значения."""
    def __init__(self, value_column: str, *, key_column: str = "client_id") -> None:
        self.key_column = key_column
        self.value_column = value_column
        self._counts: pd.DataFrame | None = None

    def update(self, df: pd.DataFrame) -> None:
        if self.value_column not in df.columns or df.empty:
            return
        counts = (
            df.groupby([self.key_column, self.value_column], observed=True)
            .size()
            .reset_index(name="_n")
        )
        if self._counts is None:
            self._counts = counts
        else:
            self._counts = (
                pd.concat([self._counts, counts], ignore_index=True)
                .groupby([self.key_column, self.value_column], observed=True, as_index=False)["_n"]
                .sum()
            )

    def argmax(self, name: str) -> pd.Series:
        """Самое частое значение на клиента (ties → лексикографически первое)."""
        if self._counts is None or self._counts.empty:
            return pd.Series(dtype="object", name=name)
        ordered = self._counts.sort_values(
            [self.key_column, "_n", self.value_column], ascending=[True, False, True]
        )
        out = ordered.groupby(self.key_column, observed=True)[self.value_column].first()
        out.name = name
        return out


class OutcomeRevenueAccumulator:
    """Выручка клиента в окне измерения исхода (для ``data.outcome: revenue``).

    Эти транзакции **не** участвуют в признаках: они лежат за
    ``window.feature_end`` и были бы прямой утечкой целевой переменной.
    Накапливаются пары (клиент, чек) с суммой чека, потому что ``purchase_sum``
    дублируется по товарным строкам и чек может разъехаться по чанкам.
    """

    def __init__(self) -> None:
        self._checks: pd.DataFrame | None = None

    def update(self, chunk: pd.DataFrame, window: ObservationWindow) -> int:
        if chunk.empty:
            return 0
        ts = pd.to_datetime(chunk["transaction_datetime"], errors="coerce")
        mask = (ts >= window.feature_end) & (ts < window.outcome_end)
        if not mask.any():
            return 0
        part = (
            chunk.loc[mask, ["client_id", "transaction_id", "purchase_sum"]]
            .groupby(["client_id", "transaction_id"], observed=True, as_index=False)["purchase_sum"]
            .max()
        )
        self._checks = (
            part
            if self._checks is None
            else pd.concat([self._checks, part], ignore_index=True)
            .groupby(["client_id", "transaction_id"], observed=True, as_index=False)["purchase_sum"]
            .max()
        )
        return int(mask.sum())

    def revenue(self) -> pd.Series:
        if self._checks is None or self._checks.empty:
            empty = pd.Series(dtype="float32", name="revenue")
            empty.index.name = "client_id"
            return empty
        out = self._checks.groupby("client_id", observed=True)["purchase_sum"].sum()
        out.name = "revenue"
        out.index.name = "client_id"
        return out.astype("float32")


@dataclass
class AggregationResult:
    """Выход фазы A: транзакционный кадр и накопленные ассортиментные агрегаты."""

    transactions_path: Path
    n_transactions: int
    n_rows_read: int
    n_rows_in_window: int
    load_stats: LoadStats
    dedup_stats: dict[str, int]
    rows_outside_window: int
    assortment: pd.DataFrame = field(default_factory=pd.DataFrame)
    outcome_revenue: pd.Series = field(default_factory=lambda: pd.Series(dtype="float32"))

    def as_dict(self) -> dict[str, object]:
        return {
            "transactions_path": str(self.transactions_path),
            "n_transactions": self.n_transactions,
            "n_purchase_rows_read": self.n_rows_read,
            "n_purchase_rows_in_window": self.n_rows_in_window,
            "rows_outside_window": self.rows_outside_window,
            "load": self.load_stats.as_dict(),
            "dedup": self.dedup_stats,
        }


def _load_product_attributes(cfg: Config) -> pd.DataFrame:
    """Товарный справочник в компактном виде для join'а с чанками."""
    products, _ = load_table(cfg, "products")
    keep = ["product_id", *CATEGORY_LEVELS, "brand_id", "is_own_trademark", "is_alcohol", "netto"]
    available = [c for c in keep if c in products.columns]
    out = products[available].copy()
    for flag in ("is_own_trademark", "is_alcohol"):
        if flag in out.columns:
            out[flag] = pd.to_numeric(out[flag], errors="coerce").fillna(0).astype("int8")
    if "netto" in out.columns:
        out["netto"] = pd.to_numeric(out["netto"], errors="coerce").fillna(0.0).astype("float32")
    return out.drop_duplicates(subset=["product_id"])


def _aggregate_chunk_to_transactions(chunk: pd.DataFrame) -> pd.DataFrame:
    """Свернуть товарные строки чанка до уровня чека."""
    agg_spec: dict[str, tuple[str, str]] = {
        "transaction_datetime": ("transaction_datetime", "first"),
        "store_id": ("store_id", "first"),
        # purchase_sum дублируется по строкам чека → max, а не sum
        "purchase_sum": ("purchase_sum", "max"),
        "n_items": ("product_id", "size"),
        "n_unique_products": ("product_id", "nunique"),
        "quantity_total": ("product_quantity", "sum"),
    }
    for col, how in (
        ("regular_points_received", "max"),
        ("express_points_received", "max"),
        ("regular_points_spent", "min"),
        ("express_points_spent", "min"),
        ("trn_sum_from_iss", "max"),
        ("trn_sum_from_red", "max"),
    ):
        if col in chunk.columns:
            agg_spec[col] = (col, how)
    for col in ("own_trademark_items", "alcohol_items", "netto_total"):
        if col in chunk.columns:
            agg_spec[col] = (col, "sum")

    return (
        chunk.groupby(["client_id", "transaction_id"], observed=True, sort=False)
        .agg(**agg_spec)  # type: ignore[arg-type]
        .reset_index()
    )


def _merge_partial_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Склеить части одного чека, разъехавшиеся по границам чанков.

    ``n_unique_products`` при таком склеивании может быть завышен (одинаковый
    товар в двух частях чека посчитается дважды) — это принятая погрешность:
    альтернатива требует полной сортировки логов по ключу чека.
    """
    duplicated_keys = df.duplicated(subset=["client_id", "transaction_id"], keep=False)
    if not duplicated_keys.any():
        return df

    logger.info(
        "склейка %d частичных записей чеков, разъехавшихся по границам чанков",
        int(duplicated_keys.sum()),
    )
    spec: dict[str, str] = {}
    for col in df.columns:
        if col in ("client_id", "transaction_id"):
            continue
        if col in _MERGE_FIRST:
            spec[col] = "first"
        elif col in _MERGE_MAX:
            spec[col] = "max"
        elif col in _MERGE_MIN:
            spec[col] = "min"
        elif col in _MERGE_SUM:
            spec[col] = "sum"
        else:
            spec[col] = "first"
    return df.groupby(["client_id", "transaction_id"], observed=True, as_index=False).agg(spec)


def aggregate_purchases(
    cfg: Config,
    window: ObservationWindow,
    *,
    category_levels: Sequence[str] = CATEGORY_LEVELS,
) -> AggregationResult:
    """Пройти ``purchases`` потоком и собрать транзакционный кадр (фаза A).

    Args:
        cfg: конфигурация конвейера.
        window: окно наблюдения признаков (ETL-3).
        category_levels: уровни товарной иерархии для ассортиментных признаков.

    Returns:
        :class:`AggregationResult` с путём к ``transactions.parquet``.
    """
    products = _load_product_attributes(cfg)
    product_index = products.set_index("product_id")

    dedup = CrossChunkDeduplicator(PURCHASES_SCHEMA.key)
    total_stats = LoadStats(table="purchases")
    rows_outside_window = 0
    rows_in_window = 0

    level_acc = {lvl: UniquePairsAccumulator(lvl) for lvl in category_levels if lvl in products}
    store_unique = UniquePairsAccumulator("store_id")
    brand_unique = UniquePairsAccumulator("brand_id")
    store_counts = PairCountAccumulator("store_id")
    brand_counts = PairCountAccumulator("brand_id")
    outcome_revenue = OutcomeRevenueAccumulator()

    partials: list[pd.DataFrame] = []

    for chunk, stats in iter_purchases(cfg):
        total_stats.merge(stats)
        if chunk.empty:
            continue

        # Выручка окна исхода собирается до фильтра признаков: это целевая
        # переменная при outcome=revenue, а не ковариата.
        outcome_revenue.update(chunk, window)

        chunk, dropped = filter_observation_window(chunk, window)
        rows_outside_window += dropped
        if chunk.empty:
            continue

        chunk = dedup.apply(chunk)
        if chunk.empty:
            continue
        rows_in_window += len(chunk)

        # Товарные атрибуты подтягиваются через индекс — дешевле полноценного merge
        joined = chunk.join(product_index, on="product_id", how="left")
        if "is_own_trademark" in joined.columns:
            joined["own_trademark_items"] = joined["is_own_trademark"].fillna(0).astype("int8")
        if "is_alcohol" in joined.columns:
            joined["alcohol_items"] = joined["is_alcohol"].fillna(0).astype("int8")
        if "netto" in joined.columns:
            joined["netto_total"] = (
                joined["netto"].fillna(0.0).astype("float32")
                * joined["product_quantity"].fillna(0.0).astype("float32")
            )

        for lvl, acc in level_acc.items():
            acc.update(joined)
        store_unique.update(joined)
        brand_unique.update(joined)
        store_counts.update(joined)
        brand_counts.update(joined)

        partials.append(_aggregate_chunk_to_transactions(joined))

    if not partials:
        msg = (
            "после фильтрации по окну наблюдения не осталось ни одной транзакции — "
            f"проверьте data.observation_window_days ({cfg.data.observation_window_days}) "
            f"и окно {window.as_dict()}"
        )
        raise ValueError(msg)

    transactions = _merge_partial_transactions(pd.concat(partials, ignore_index=True))
    transactions["transaction_datetime"] = pd.to_datetime(transactions["transaction_datetime"])
    transactions = transactions.sort_values(["client_id", "transaction_datetime"]).reset_index(
        drop=True
    )

    assortment = pd.concat(
        [
            *(acc.counts(f"n_unique_{lvl}") for lvl, acc in level_acc.items()),
            store_unique.counts("n_unique_stores"),
            brand_unique.counts("n_unique_brands"),
            store_counts.argmax("favorite_store_id"),
            brand_counts.argmax("favorite_brand_id"),
        ],
        axis=1,
    )
    assortment.index.name = "client_id"

    path = cfg.interim_dir / "transactions.parquet"
    save_parquet(transactions, path)
    assortment_path = cfg.interim_dir / "assortment.parquet"
    save_parquet(assortment.reset_index(), assortment_path)

    revenue = outcome_revenue.revenue()
    save_parquet(revenue.reset_index(), cfg.interim_dir / "outcome_revenue.parquet")

    logger.info(
        "фаза A: %d товарных строк → %d транзакций (%d вне окна, %d дублей), %s",
        total_stats.rows_read,
        len(transactions),
        rows_outside_window,
        dedup.dropped,
        path.name,
    )

    return AggregationResult(
        transactions_path=path,
        n_transactions=len(transactions),
        n_rows_read=total_stats.rows_read,
        n_rows_in_window=rows_in_window,
        load_stats=total_stats,
        dedup_stats=dedup.stats,
        rows_outside_window=rows_outside_window,
        assortment=assortment,
        outcome_revenue=revenue,
    )


def load_transactions(cfg: Config, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Прочитать транзакционный кадр фазы A."""
    path = cfg.interim_dir / "transactions.parquet"
    if not path.is_file():
        msg = f"не найден {path}: сначала выполните `make etl`"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path, engine="pyarrow", columns=list(columns) if columns else None)


def load_assortment(cfg: Config) -> pd.DataFrame:
    """Прочитать ассортиментные агрегаты фазы A (индекс — ``client_id``)."""
    path = cfg.interim_dir / "assortment.parquet"
    if not path.is_file():
        msg = f"не найден {path}: сначала выполните `make etl`"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(path, engine="pyarrow")
    return df.set_index("client_id")


def resolve_window(cfg: Config, *, refresh: bool = False) -> ObservationWindow:
    """Окно наблюдения: прочитать с диска либо вычислить и зафиксировать.

    Границы окна зависят от даты последнего события в ``purchases`` — то есть
    от однопроходного сканирования логов. Пересчитывать их на каждом этапе
    нельзя: если данные изменились, признаки и причинный слой получили бы
    разные окна, а сравнение метрик §11 между прогонами потеряло бы смысл.
    """
    path = cfg.artifact(WINDOW_ARTIFACT)
    if path.is_file() and not refresh:
        payload = load_json(path)
        if int(payload.get("observation_window_days", -1)) == cfg.data.observation_window_days and (
            int(payload.get("outcome_window_days", -1)) == cfg.data.outcome_window_days
        ):
            return ObservationWindow(
                feature_start=pd.Timestamp(payload["feature_start"]),
                feature_end=pd.Timestamp(payload["feature_end"]),
                outcome_end=pd.Timestamp(payload["outcome_end"]),
            )
        logger.info("параметры окна в конфигурации изменились — пересчитываю границы")

    _, data_end = scan_datetime_bounds(cfg)
    window = build_window(
        data_end,
        observation_window_days=cfg.data.observation_window_days,
        outcome_window_days=cfg.data.outcome_window_days,
    )
    save_json(window.as_dict(), path)
    logger.info("окно наблюдения: %s", window.as_dict())
    return window


def load_outcome_revenue(cfg: Config) -> pd.Series:
    """Выручка клиентов в окне исхода (пусто, если окно исхода не содержит чеков)."""
    path = cfg.interim_dir / "outcome_revenue.parquet"
    if not path.is_file():
        empty = pd.Series(dtype="float32", name="revenue")
        empty.index.name = "client_id"
        return empty
    df = pd.read_parquet(path, engine="pyarrow")
    if df.empty:
        empty = pd.Series(dtype="float32", name="revenue")
        empty.index.name = "client_id"
        return empty
    return df.set_index("client_id")["revenue"].astype("float32")


def transaction_count_estimate(cfg: Config) -> int:
    """Число транзакций в interim-кадре (без загрузки всех колонок)."""
    return len(load_transactions(cfg, columns=["transaction_id"]))


__all__ = [
    "CATEGORY_LEVELS",
    "WINDOW_ARTIFACT",
    "AggregationResult",
    "OutcomeRevenueAccumulator",
    "PairCountAccumulator",
    "UniquePairsAccumulator",
    "aggregate_purchases",
    "load_assortment",
    "load_outcome_revenue",
    "load_transactions",
    "resolve_window",
    "transaction_count_estimate",
]
