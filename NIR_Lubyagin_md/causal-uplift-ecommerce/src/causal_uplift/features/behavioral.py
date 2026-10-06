"""Поведенческие признаки: лояльность, ассортимент, тренды, соц-дем (SPEC §5.3).

Дополняют RFM-блок до требуемых ≥ 30 ковариат. Блок трендов отвечает за
признаки затухания активности — именно они в синтетическом генераторе связаны
с истинным ``τ(x)``, и на реальных данных это та же содержательная гипотеза:
уходящего клиента купон возвращает, а активного — нет.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.etl.cleaning import ObservationWindow
from causal_uplift.etl.synthetic import REFERENCE_DATE
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Границы возрастных бинов (SPEC §5.3: «возрастные бины»)
AGE_BINS: tuple[float, ...] = (0, 25, 35, 45, 55, 65, 200)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, *, fill: float = 0.0) -> pd.Series:
    denom = denominator.replace(0.0, np.nan)
    return (numerator / denom).fillna(fill)


def build_loyalty_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Признаки программы лояльности: начисления, списания, доля оплаты баллами."""
    tx = transactions
    grouped = tx.groupby("client_id", observed=True)

    out = pd.DataFrame(index=grouped.size().index)
    out.index.name = "client_id"

    for col, name in (
        ("regular_points_received", "regular_points_received_sum"),
        ("express_points_received", "express_points_received_sum"),
    ):
        out[name] = grouped[col].sum() if col in tx.columns else 0.0

    # Списания записаны отрицательными — работаем с модулем
    for col, name in (
        ("regular_points_spent", "regular_points_spent_sum"),
        ("express_points_spent", "express_points_spent_sum"),
    ):
        out[name] = grouped[col].sum().abs() if col in tx.columns else 0.0

    total_spend = grouped["purchase_sum"].sum()
    if "trn_sum_from_red" in tx.columns:
        redeemed = grouped["trn_sum_from_red"].sum()
        # Доля покупок, оплаченная баллами (§5.3: trn_sum_from_red / purchase_sum)
        out["redeem_amount_share"] = _safe_ratio(redeemed, total_spend)
        out["redeem_amount_total"] = redeemed
        with_redeem = (
            tx.assign(_has_redeem=(tx["trn_sum_from_red"] > 0).astype("float32"))
            .groupby("client_id", observed=True)["_has_redeem"]
            .mean()
        )
        out["share_transactions_with_redeem"] = with_redeem
    else:
        out["redeem_amount_share"] = 0.0
        out["redeem_amount_total"] = 0.0
        out["share_transactions_with_redeem"] = 0.0

    n_tx = grouped.size().astype("float32")
    out["points_spent_per_transaction"] = _safe_ratio(
        out["regular_points_spent_sum"] + out["express_points_spent_sum"], n_tx
    )
    out["points_balance_proxy"] = (
        out["regular_points_received_sum"] + out["express_points_received_sum"]
    ) - (out["regular_points_spent_sum"] + out["express_points_spent_sum"])

    return out.astype("float32")


def build_assortment_features(
    transactions: pd.DataFrame, assortment: pd.DataFrame
) -> pd.DataFrame:
    """Признаки структуры корзины: категории, СТМ, алкоголь, магазины.

    Числа уникальных категорий/магазинов/брендов приходят из накопителей фазы A
    (их нельзя посчитать по транзакционному кадру — товарный уровень там уже
    свёрнут), остальное считается здесь.
    """
    grouped = transactions.groupby("client_id", observed=True)
    out = pd.DataFrame(index=grouped.size().index)
    out.index.name = "client_id"

    n_items = grouped["n_items"].sum()
    if "own_trademark_items" in transactions.columns:
        out["own_trademark_share"] = _safe_ratio(grouped["own_trademark_items"].sum(), n_items)
    else:
        out["own_trademark_share"] = 0.0
    if "alcohol_items" in transactions.columns:
        out["alcohol_share"] = _safe_ratio(grouped["alcohol_items"].sum(), n_items)
    else:
        out["alcohol_share"] = 0.0
    if "netto_total" in transactions.columns:
        out["avg_netto_per_transaction"] = grouped["netto_total"].mean()
    else:
        out["avg_netto_per_transaction"] = 0.0

    numeric_cols = [c for c in assortment.columns if c.startswith("n_unique_")]
    joined = out.join(assortment[numeric_cols], how="left")
    joined[numeric_cols] = joined[numeric_cols].fillna(0.0)

    # Разнообразие корзины: уникальных категорий верхнего уровня на транзакцию
    if "n_unique_level_1" in joined.columns:
        joined["categories_per_transaction"] = _safe_ratio(
            joined["n_unique_level_1"], grouped.size().astype("float32")
        )
    if {"n_unique_level_4", "n_unique_level_1"} <= set(joined.columns):
        joined["assortment_depth_ratio"] = _safe_ratio(
            joined["n_unique_level_4"], joined["n_unique_level_1"]
        )

    result = joined.astype("float32")

    # Любимый магазин и любимый бренд остаются строками: это высококардинальные
    # категории из features.high_cardinality, и в числа их переводит частотное
    # кодирование на train-фолде (ETL-5). Привести их к float здесь — значит
    # либо потерять признак, либо выдумать порядок на идентификаторах.
    for col in ("favorite_store_id", "favorite_brand_id"):
        if col in assortment.columns:
            result[col] = assortment[col].reindex(result.index).astype("string")

    return result


def build_trend_features(
    transactions: pd.DataFrame,
    rfm: pd.DataFrame,
    window: ObservationWindow,
) -> pd.DataFrame:
    """Признаки трендов: отношение активности 30д/90д и слоуп суммы покупок.

    Слоуп считается через замкнутую формулу OLS по агрегатам (Σx, Σy, Σxy, Σx²),
    а не ``apply`` с регрессией на группу: на сотнях тысяч клиентов это
    разница между секундами и десятками минут.
    """
    tx = transactions.copy()
    tx["transaction_datetime"] = pd.to_datetime(tx["transaction_datetime"])
    # x — дни от начала окна наблюдения
    tx["_x"] = (tx["transaction_datetime"] - window.feature_start).dt.total_seconds() / 86_400.0
    tx["_y"] = tx["purchase_sum"].astype("float64")
    tx["_xy"] = tx["_x"] * tx["_y"]
    tx["_xx"] = tx["_x"] * tx["_x"]

    agg = tx.groupby("client_id", observed=True).agg(
        n=("_x", "size"),
        sum_x=("_x", "sum"),
        sum_y=("_y", "sum"),
        sum_xy=("_xy", "sum"),
        sum_xx=("_xx", "sum"),
    )
    denom = agg["n"] * agg["sum_xx"] - agg["sum_x"] ** 2
    slope = (agg["n"] * agg["sum_xy"] - agg["sum_x"] * agg["sum_y"]) / denom.replace(0.0, np.nan)

    out = pd.DataFrame(index=agg.index)
    out.index.name = "client_id"
    out["spend_slope_per_day"] = slope.fillna(0.0)

    # Отношения коротких окон к длинным — признак затухания (§5.3)
    for short, long in ((30, 90), (7, 30)):
        c_short = rfm.get(f"n_transactions_{short}d")
        c_long = rfm.get(f"n_transactions_{long}d")
        if c_short is not None and c_long is not None:
            out[f"activity_ratio_{short}_{long}"] = _safe_ratio(
                c_short.reindex(out.index), c_long.reindex(out.index)
            )
        s_short = rfm.get(f"spend_{short}d")
        s_long = rfm.get(f"spend_{long}d")
        if s_short is not None and s_long is not None:
            out[f"spend_ratio_{short}_{long}"] = _safe_ratio(
                s_short.reindex(out.index), s_long.reindex(out.index)
            )

    # Средний чек последних 30 дней против среднего за всё окно
    cutoff = window.feature_end - pd.Timedelta(days=30)
    recent_basket = (
        tx.loc[tx["transaction_datetime"] >= cutoff]
        .groupby("client_id", observed=True)["purchase_sum"]
        .mean()
        .reindex(out.index)
    )
    overall_basket = rfm["avg_basket"].reindex(out.index)
    out["basket_trend_30d"] = _safe_ratio(recent_basket.fillna(0.0), overall_basket, fill=0.0)

    # Доля покупок во второй половине окна — простой индикатор направления тренда
    midpoint = window.feature_start + (window.feature_end - window.feature_start) / 2
    second_half = (
        tx.assign(_late=(tx["transaction_datetime"] >= midpoint).astype("float32"))
        .groupby("client_id", observed=True)["_late"]
        .mean()
        .reindex(out.index)
    )
    out["share_purchases_second_half"] = second_half.fillna(0.0)

    return out.astype("float32")


def build_socdem_features(
    clients: pd.DataFrame,
    window: ObservationWindow,
    *,
    age_bins: tuple[float, ...] = AGE_BINS,
) -> pd.DataFrame:
    """Соц-дем и даты программы лояльности.

    Пропуски не затираются молча: для ``age`` и ``first_redeem_date``
    добавляются флаги-индикаторы — профиль пропусков фиксируется в EDA (W3),
    а модель получает сам факт отсутствия значения как признак.
    """
    df = clients.set_index("client_id")
    out = pd.DataFrame(index=df.index)
    out.index.name = "client_id"

    age = pd.to_numeric(df.get("age"), errors="coerce")
    out["age_missing"] = age.isna().astype("float32")
    median_age = float(age.median()) if age.notna().any() else 40.0
    age_filled = age.fillna(median_age)
    out["age"] = age_filled.astype("float32")
    out["age_bin"] = (
        pd.cut(age_filled, bins=list(age_bins), labels=False, include_lowest=True)
        .astype("float32")
        .fillna(0.0)
    )

    issue = pd.to_datetime(df.get("first_issue_date"), errors="coerce")
    out["days_since_first_issue"] = (
        (window.feature_end - issue).dt.total_seconds() / 86_400.0
    ).astype("float32")
    out["days_since_first_issue"] = out["days_since_first_issue"].fillna(
        float(out["days_since_first_issue"].median(skipna=True) or 0.0)
    )

    redeem = pd.to_datetime(df.get("first_redeem_date"), errors="coerce")
    out["has_redeemed_ever"] = redeem.notna().astype("float32")
    days_since_redeem = (window.feature_end - redeem).dt.total_seconds() / 86_400.0
    # Для «никогда не списывал» подставляем давность выдачи карты: смысл признака —
    # «как давно клиент пользовался баллами», и отсутствие события это максимум.
    out["days_since_first_redeem"] = (
        days_since_redeem.fillna(out["days_since_first_issue"]).astype("float32")
    )
    out["redeem_delay_days"] = (
        ((redeem - issue).dt.total_seconds() / 86_400.0).fillna(-1.0).astype("float32")
    )

    # Сезонность вступления в программу: отдельный узел DAG (§6.1)
    issue_filled = issue.fillna(REFERENCE_DATE)
    out["first_issue_month"] = issue_filled.dt.month.astype("float32")
    out["first_issue_quarter"] = issue_filled.dt.quarter.astype("float32")

    # gender остаётся категориальным: one-hot выполняется в encoding.py (ETL-5)
    out["gender"] = df.get("gender", pd.Series(index=df.index, dtype="object")).astype("string")
    return out


def build_behavioral_block(
    transactions: pd.DataFrame,
    assortment: pd.DataFrame,
    clients: pd.DataFrame,
    rfm: pd.DataFrame,
    window: ObservationWindow,
) -> pd.DataFrame:
    """Собрать блоки лояльности, ассортимента, трендов и соц-дема в один кадр."""
    loyalty = build_loyalty_features(transactions)
    assort = build_assortment_features(transactions, assortment)
    trends = build_trend_features(transactions, rfm, window)
    socdem = build_socdem_features(clients, window)

    merged = loyalty.join(assort, how="outer").join(trends, how="outer")
    merged = merged.join(socdem, how="outer")
    logger.info(
        "поведенческий блок: лояльность %d + ассортимент %d + тренды %d + соц-дем %d признаков",
        loyalty.shape[1],
        assort.shape[1],
        trends.shape[1],
        socdem.shape[1],
    )
    return merged


__all__ = [
    "AGE_BINS",
    "build_assortment_features",
    "build_behavioral_block",
    "build_loyalty_features",
    "build_socdem_features",
    "build_trend_features",
]
