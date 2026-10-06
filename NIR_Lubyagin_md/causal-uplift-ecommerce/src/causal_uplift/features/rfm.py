"""RFM-признаки: Recency, Frequency, Monetary (SPEC §5.3, W4).

Считаются по транзакционному кадру фазы A. Все окна отсчитываются от конца
окна признаков (``window.feature_end``), а не от системной даты — иначе
прогоны были бы невоспроизводимы, а признаки могли бы заглянуть в окно
измерения исхода.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.etl.cleaning import ObservationWindow
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Окна (в днях) для частотных и монетарных признаков
SHORT_WINDOWS: tuple[int, ...] = (7, 30, 90)


def _days_from(end: pd.Timestamp, series: pd.Series) -> pd.Series:
    return (end - series).dt.total_seconds() / 86_400.0


def build_rfm_features(
    transactions: pd.DataFrame,
    window: ObservationWindow,
    *,
    short_windows: tuple[int, ...] = SHORT_WINDOWS,
) -> pd.DataFrame:
    """Собрать RFM-блок витрины по клиенту.

    Args:
        transactions: кадр уровня чека — выход
            :func:`~causal_uplift.etl.aggregate.aggregate_purchases`.
        window: окно наблюдения; ``feature_end`` — точка отсчёта recency.
        short_windows: короткие окна в днях для частоты и оборота.

    Returns:
        Кадр, индексированный ``client_id``.
    """
    if transactions.empty:
        msg = "транзакционный кадр пуст — RFM-признаки не вычислимы"
        raise ValueError(msg)

    tx = transactions.copy()
    tx["transaction_datetime"] = pd.to_datetime(tx["transaction_datetime"])
    tx["_date"] = tx["transaction_datetime"].dt.normalize()

    grouped = tx.groupby("client_id", observed=True)

    # --- Recency -------------------------------------------------------
    last_purchase = grouped["transaction_datetime"].max()
    first_purchase = grouped["transaction_datetime"].min()
    features = pd.DataFrame(
        {
            "days_since_last_purchase": _days_from(window.feature_end, last_purchase),
            "days_since_first_purchase": _days_from(window.feature_end, first_purchase),
        }
    )
    features["purchase_history_span_days"] = (
        features["days_since_first_purchase"] - features["days_since_last_purchase"]
    )

    # --- Frequency -----------------------------------------------------
    features["n_transactions_total"] = grouped.size().astype("float32")
    features["n_active_days"] = grouped["_date"].nunique().astype("float32")

    for days in short_windows:
        cutoff = window.feature_end - pd.Timedelta(days=days)
        recent = tx.loc[tx["transaction_datetime"] >= cutoff]
        counts = recent.groupby("client_id", observed=True).size()
        features[f"n_transactions_{days}d"] = counts.reindex(features.index).fillna(0.0)
        sums = recent.groupby("client_id", observed=True)["purchase_sum"].sum()
        features[f"spend_{days}d"] = sums.reindex(features.index).fillna(0.0)

    # Частота сессий за 30 дней (признак из §5.3) и средний интервал между покупками
    features["sessions_per_30d"] = features["n_transactions_30d"] / 30.0
    span = features["purchase_history_span_days"].clip(lower=1.0)
    features["avg_days_between_purchases"] = span / features["n_transactions_total"].clip(lower=1.0)

    # Разброс интервалов между покупками — индикатор нерегулярности
    tx_sorted = tx.sort_values(["client_id", "transaction_datetime"])
    deltas = (
        tx_sorted.groupby("client_id", observed=True)["transaction_datetime"]
        .diff()
        .dt.total_seconds()
        / 86_400.0
    )
    tx_sorted = tx_sorted.assign(_delta=deltas)
    features["std_days_between_purchases"] = (
        tx_sorted.groupby("client_id", observed=True)["_delta"]
        .std()
        .reindex(features.index)
        .fillna(0.0)
    )

    # --- Monetary ------------------------------------------------------
    basket = grouped["purchase_sum"]
    features["total_spend"] = basket.sum()
    features["avg_basket"] = basket.mean()
    features["median_basket"] = basket.median()
    features["std_basket"] = basket.std().fillna(0.0)
    features["max_basket"] = basket.max()
    features["min_basket"] = basket.min()
    features["avg_items_per_basket"] = grouped["n_items"].mean()
    features["total_quantity"] = grouped["quantity_total"].sum()
    features["avg_unique_products_per_basket"] = grouped["n_unique_products"].mean()

    features.index.name = "client_id"
    numeric = features.select_dtypes(include=[np.number]).columns
    features[numeric] = features[numeric].astype("float32")
    logger.info("RFM-блок: %d признаков на %d клиентов", features.shape[1], features.shape[0])
    return features


__all__ = ["SHORT_WINDOWS", "build_rfm_features"]
