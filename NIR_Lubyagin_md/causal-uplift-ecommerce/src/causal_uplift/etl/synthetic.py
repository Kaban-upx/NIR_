"""Генератор синтетического датасета RetailHero-совместимой схемы (W2).

Зачем он нужен, даже когда есть реальные данные:

* **Sanity-тест §11.** Нужен датасет с *аналитически известным* ``τ(x)``, иначе
  отрицательный результат на реальных данных невозможно разложить на «эффекта
  нет в данных» и «модель сломана» (риск §14).
* **Воспроизводимость.** RetailHero лежит за авторизацией; генератор даёт
  детерминированный вход для CI и для сквозных прогонов на чистой машине.

Генерируются те же четыре таблицы, что в публичном датасете (SPEC §5.1), с теми
же именами колонок и семантикой, включая характерные особенности:
``purchase_sum`` дублируется по всем товарным строкам одного чека, списания
баллов отрицательны, часть дат отсутствует. Ниже слоёв ETL никакой разницы
между синтетикой и реальными CSV нет.

Причинная структура соответствует DAG §6.1: латентные характеристики клиента
порождают и наблюдаемые ковариаты, и propensity ``e(x)``, и исход ``Y`` —
то есть конфаундинг реальный, а не декоративный, и наблюдаемая разность
``Δ_corr`` смещена относительно ``ATE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from causal_uplift.config import Config
from causal_uplift.io_utils import save_parquet
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Конец окна наблюдения. Зафиксирован, чтобы прогоны были воспроизводимы
#: (ML-1) и не зависели от системной даты. Соответствует хронологии RetailHero.
REFERENCE_DATE = pd.Timestamp("2019-03-01")

#: Файл с истинными значениями τ(x) и μ₀(x). Префикс `_` — не часть схемы
#: RetailHero: ETL его не читает, он доступен только тестам и sanity-отчёту.
TRUTH_FILENAME = "_synthetic_truth.parquet"

_LEVEL_SIZES = (12, 48, 180, 520)  # мощности иерархии level_1..level_4


@dataclass
class LatentClients:
    """Латентные характеристики клиентов — причина и ковариат, и T, и Y."""

    client_ids: np.ndarray
    log_lambda: np.ndarray  # логарифм месячной частоты покупок
    activity: np.ndarray  # стандартизованная активность a
    spend: np.ndarray  # ожидаемый средний чек
    spend_z: np.ndarray  # стандартизованный log(spend)
    loyalty: np.ndarray  # склонность пользоваться баллами ∈ [0,1]
    variety: np.ndarray  # разнообразие корзины ∈ [0,1]
    trend: np.ndarray  # тренд активности (>0 — рост, <0 — затухание)
    basket_size: np.ndarray  # среднее число товарных строк в чеке
    age: np.ndarray
    gender: np.ndarray
    favorite_store: np.ndarray
    favorite_products: np.ndarray  # (n_clients, 5)
    first_issue_offset: np.ndarray  # дней до REFERENCE_DATE
    redeem_flag: np.ndarray

    @property
    def n(self) -> int:
        return int(self.client_ids.size)


@dataclass
class SyntheticDataset:
    """Результат генерации: пути к файлам, ground truth и сводка."""

    paths: dict[str, Path]
    truth: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Латентный слой
# ---------------------------------------------------------------------------


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _sample_latents(cfg: Config, rng: np.random.Generator) -> LatentClients:
    syn = cfg.data.synthetic
    n = syn.n_clients
    months = cfg.data.observation_window_days / 30.0

    # Месячная частота покупок: lognormal с заданным средним числом транзакций
    sigma_lambda = 0.6
    mean_lambda = syn.avg_transactions_per_client / months
    log_lambda = rng.normal(np.log(mean_lambda) - sigma_lambda**2 / 2, sigma_lambda, size=n)

    # Средний чек
    log_spend = rng.normal(np.log(600.0) - 0.125, 0.5, size=n)
    spend = np.exp(log_spend)

    loyalty = rng.beta(2.0, 3.0, size=n)
    variety = rng.beta(2.0, 2.0, size=n)
    trend = np.clip(rng.normal(0.0, 0.5, size=n), -1.5, 1.5)
    basket_size = 1.0 + rng.gamma(shape=2.0, scale=1.1, size=n)

    age = np.clip(rng.normal(42.0, 14.0, size=n), 18, 90).round().astype(np.int16)
    gender = rng.choice(np.array(["F", "M", "U"]), size=n, p=[0.60, 0.35, 0.05])

    favorite_store = rng.integers(0, syn.stores_count, size=n)
    favorite_products = rng.integers(0, syn.products_catalog_size, size=(n, 5))

    # Дата выдачи карты: от 40 до 1500 дней до конца окна наблюдения
    first_issue_offset = rng.integers(40, 1500, size=n)
    # Списание баллов — только у части клиентов, вероятность растёт с лояльностью
    redeem_flag = rng.uniform(size=n) < (0.25 + 0.55 * loyalty)

    client_ids = np.array([f"c{idx:07d}" for idx in range(n)], dtype=object)

    return LatentClients(
        client_ids=client_ids,
        log_lambda=log_lambda,
        activity=_standardize(log_lambda),
        spend=spend,
        spend_z=_standardize(log_spend),
        loyalty=loyalty,
        variety=variety,
        trend=trend,
        basket_size=basket_size,
        age=age,
        gender=gender,
        favorite_store=favorite_store,
        favorite_products=favorite_products,
        first_issue_offset=first_issue_offset,
        redeem_flag=redeem_flag,
    )


# ---------------------------------------------------------------------------
# Причинный слой: propensity, μ₀, τ, потенциальные исходы
# ---------------------------------------------------------------------------


def _causal_layer(
    cfg: Config, lat: LatentClients, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Назначение воздействия и генерация потенциальных исходов.

    * ``e(x)`` зависит от ковариат — это и есть backdoor-путь ``X → T``
      (сила регулируется ``confounding_strength``);
    * ``μ₀(x)`` — базовая вероятность отклика без купона;
    * ``τ(x)`` — истинный эффект: максимален у «средних по активности»
      лояльных клиентов (*Persuadables*), около нуля у сверхактивных
      (*Sure Things*) и пассивных (*Lost Causes*), отрицателен у заданной
      доли (*Sleeping Dogs*).
    """
    syn = cfg.data.synthetic
    k = syn.confounding_strength

    # --- X → T ---------------------------------------------------------
    propensity_logit = k * (
        0.80 * lat.activity + 0.45 * lat.spend_z - 0.50 * (lat.loyalty - 0.4) + 0.35 * lat.trend
    )
    propensity = np.clip(expit(propensity_logit), 0.02, 0.98)
    treatment = (rng.uniform(size=lat.n) < propensity).astype(np.int8)

    # --- X → Y (базовый отклик μ₀) -------------------------------------
    mu0_logit = -1.35 + 1.05 * lat.activity + 0.35 * lat.spend_z + 0.30 * lat.trend
    mu0 = np.clip(expit(mu0_logit), 0.01, 0.95)

    # --- истинный эффект τ(x) ------------------------------------------
    # Высокая активность снижает эффект (покупка произошла бы и так),
    # лояльность и разнообразие — повышают, затухание активности — повышает
    # (клиента ещё можно вернуть).
    raw_tau = (
        1.00 * (lat.loyalty - 0.40)
        + 0.55 * (lat.variety - 0.50)
        - 0.75 * np.clip(lat.activity, 0.0, None)
        - 0.35 * lat.trend
    )
    tau_z = _standardize(raw_tau)
    # Сдвиг подобран так, чтобы доля τ(x) < 0 равнялась sleeping_dogs_share
    shift = -float(norm.ppf(max(syn.sleeping_dogs_share, 1e-6)))
    tau_nominal = syn.uplift_scale * (tau_z + shift)

    # Эффект, фактически реализуемый в вероятности: p₁ ограничена [0.01, 0.99]
    p1 = np.clip(mu0 + tau_nominal, 0.01, 0.99)
    tau_true = p1 - mu0

    # --- потенциальные исходы ------------------------------------------
    # Общий uniform для Y(0) и Y(1): нет «defiers», знак эффекта согласован
    # с τ(x), что и требуется для sanity-проверки восстановления τ.
    u = rng.uniform(size=lat.n)
    y0 = (u < mu0).astype(np.int8)
    y1 = (u < p1).astype(np.int8)
    target = np.where(treatment == 1, y1, y0).astype(np.int8)

    # Выручка: ненулевая только при покупке
    revenue = target * lat.spend * np.exp(rng.normal(0.0, 0.3, size=lat.n))

    return {
        "propensity": propensity,
        "treatment": treatment,
        "mu0": mu0,
        "tau_nominal": tau_nominal,
        "tau_true": tau_true,
        "y0": y0,
        "y1": y1,
        "target": target,
        "revenue": revenue.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Справочники
# ---------------------------------------------------------------------------


def _build_products(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    size = cfg.data.synthetic.products_catalog_size
    idx = np.arange(size)
    levels = {}
    for level_no, cardinality in enumerate(_LEVEL_SIZES, start=1):
        # Иерархия: level_1 грубее level_4, вложенность детерминирована индексом
        bucket = (idx * (level_no + 1)) % cardinality
        levels[f"level_{level_no}"] = np.array(
            [f"l{level_no}_{b:03d}" for b in bucket], dtype=object
        )

    return pd.DataFrame(
        {
            "product_id": [f"p{i:06d}" for i in idx],
            **levels,
            "segment_id": [f"s{i % 80:02d}" for i in idx],
            "brand_id": [f"b{i % 300:03d}" for i in idx],
            "vendor_id": [f"v{i % 120:03d}" for i in idx],
            "netto": np.round(rng.gamma(2.0, 0.35, size=size), 3).astype(np.float32),
            "is_own_trademark": (rng.uniform(size=size) < 0.18).astype(np.int8),
            "is_alcohol": (rng.uniform(size=size) < 0.07).astype(np.int8),
        }
    )


def _build_clients(cfg: Config, lat: LatentClients) -> pd.DataFrame:
    first_issue = REFERENCE_DATE - pd.to_timedelta(lat.first_issue_offset, unit="D")
    # Первое списание баллов — позже выдачи карты, не позже конца окна
    redeem_gap = np.round(lat.first_issue_offset * (0.2 + 0.6 * lat.loyalty)).astype(int)
    first_redeem = first_issue + pd.to_timedelta(redeem_gap, unit="D")
    first_redeem = pd.Series(first_redeem).where(pd.Series(lat.redeem_flag), pd.NaT)
    first_redeem = first_redeem.where(first_redeem <= REFERENCE_DATE, pd.NaT)

    # Реалистичные пропуски в соц-деме (профиль пропусков фиксируется в EDA, W3)
    age = pd.Series(lat.age, dtype="float32")
    gender = pd.Series(lat.gender, dtype="string")
    missing_age = np.arange(lat.n) % 37 == 0
    missing_gender = np.arange(lat.n) % 53 == 0
    age[missing_age] = np.nan
    gender[missing_gender] = pd.NA

    return pd.DataFrame(
        {
            "client_id": lat.client_ids,
            "first_issue_date": first_issue,
            "first_redeem_date": first_redeem.to_numpy(),
            "age": age,
            "gender": gender,
        }
    )


# ---------------------------------------------------------------------------
# Транзакции
# ---------------------------------------------------------------------------


def _trend_sample(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Обратное преобразование для плотности ∝ exp(k·s) на s ∈ [0, 1].

    ``k > 0`` — транзакции смещены к концу окна (рост активности),
    ``k < 0`` — к началу (затухание). Нужно, чтобы признаки-тренды
    (активность 30д / 90д) действительно несли сигнал.
    """
    out = np.empty_like(u)
    flat = np.abs(k) < 1e-6
    out[flat] = u[flat]
    kk = k[~flat]
    out[~flat] = np.log1p(u[~flat] * (np.expm1(kk))) / kk
    return out


def _generate_purchase_block(
    cfg: Config,
    lat: LatentClients,
    block: slice,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Товарные строки транзакций для блока клиентов (векторизованно)."""
    syn = cfg.data.synthetic
    window_days = cfg.data.observation_window_days
    months = window_days / 30.0

    idx = np.arange(lat.n)[block]
    if idx.size == 0:
        return pd.DataFrame()

    lam = np.exp(lat.log_lambda[idx]) * months
    n_tx = rng.poisson(lam)
    total_tx = int(n_tx.sum())
    if total_tx == 0:
        return pd.DataFrame()

    tx_client = np.repeat(idx, n_tx)

    # Даты транзакций внутри окна наблюдения, с учётом тренда клиента
    u = rng.uniform(size=total_tx)
    s = _trend_sample(u, lat.trend[tx_client] * 2.0)
    day_offset = (s * (window_days - 1)).astype(int)
    seconds = rng.integers(8 * 3600, 22 * 3600, size=total_tx)
    tx_datetime = (
        REFERENCE_DATE
        - pd.to_timedelta(window_days, unit="D")
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(seconds, unit="s")
    )

    # Сумма чека: вокруг индивидуального среднего
    tx_sum = np.round(
        lat.spend[tx_client] * np.exp(rng.normal(0.0, 0.40, size=total_tx)), 2
    ).astype(np.float32)
    tx_sum = np.maximum(tx_sum, 10.0)

    # Магазин: обычно любимый
    use_fav_store = rng.uniform(size=total_tx) < 0.75
    tx_store = np.where(
        use_fav_store,
        lat.favorite_store[tx_client],
        rng.integers(0, syn.stores_count, size=total_tx),
    )

    # Баллы: начисление ~1% от чека; списание — у лояльных клиентов
    regular_received = np.round(tx_sum * 0.01, 2).astype(np.float32)
    express_received = np.where(
        rng.uniform(size=total_tx) < 0.12, np.round(tx_sum * 0.02, 2), 0.0
    ).astype(np.float32)
    redeem_now = rng.uniform(size=total_tx) < (0.10 + 0.45 * lat.loyalty[tx_client])
    redeem_share = np.where(redeem_now, rng.uniform(0.05, 0.45, size=total_tx), 0.0)
    trn_from_red = np.round(tx_sum * redeem_share, 2).astype(np.float32)
    trn_from_iss = np.round(tx_sum - trn_from_red, 2).astype(np.float32)
    # В RetailHero списания записаны отрицательными числами
    regular_spent = -np.round(trn_from_red * 0.8, 2).astype(np.float32)
    express_spent = -np.round(trn_from_red * 0.2, 2).astype(np.float32)

    # Товарные строки чека
    n_items = 1 + rng.poisson(np.maximum(lat.basket_size[tx_client] - 1.0, 0.1))
    total_rows = int(n_items.sum())
    row_tx = np.repeat(np.arange(total_tx), n_items)
    row_client = tx_client[row_tx]

    take_favorite = rng.uniform(size=total_rows) < (1.0 - lat.variety[row_client])
    fav_slot = rng.integers(0, lat.favorite_products.shape[1], size=total_rows)
    product_idx = np.where(
        take_favorite,
        lat.favorite_products[row_client, fav_slot],
        rng.integers(0, syn.products_catalog_size, size=total_rows),
    )
    quantity = np.round(1 + rng.poisson(0.6, size=total_rows), 0).astype(np.float32)

    # Идентификаторы транзакций уникальны глобально: смещаем на индекс блока
    tx_global_id = np.arange(total_tx) + int(idx[0]) * 1000

    return pd.DataFrame(
        {
            "client_id": lat.client_ids[row_client],
            "transaction_id": np.array([f"t{i:09d}" for i in tx_global_id], dtype=object)[row_tx],
            "transaction_datetime": tx_datetime.to_numpy()[row_tx],
            "regular_points_received": regular_received[row_tx],
            "express_points_received": express_received[row_tx],
            "regular_points_spent": regular_spent[row_tx],
            "express_points_spent": express_spent[row_tx],
            # Как в RetailHero: сумма чека дублируется по всем его строкам
            "purchase_sum": tx_sum[row_tx],
            "store_id": np.array([f"st{i:04d}" for i in range(syn.stores_count)], dtype=object)[
                tx_store[row_tx]
            ],
            "product_id": np.array(
                [f"p{i:06d}" for i in range(syn.products_catalog_size)], dtype=object
            )[product_idx],
            "product_quantity": quantity,
            "trn_sum_from_iss": trn_from_iss[row_tx],
            "trn_sum_from_red": trn_from_red[row_tx],
        }
    )


# ---------------------------------------------------------------------------
# Публичный вход
# ---------------------------------------------------------------------------


def generate_raw_dataset(cfg: Config, *, block_size: int = 5_000) -> SyntheticDataset:
    """Сгенерировать сырые CSV в ``data/raw`` и ground truth для sanity-тестов.

    Args:
        cfg: конфигурация конвейера (секция ``data.synthetic``).
        block_size: число клиентов в блоке генерации транзакций — ограничивает
            пик потребления памяти (та же логика, что ETL-7).

    Returns:
        Пути к созданным файлам, таблица истинных ``τ(x)``/``μ₀(x)`` и сводка.
    """
    rng = np.random.default_rng(cfg.seed)
    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "генерация синтетики: %d клиентов, окно наблюдения %d дн., сид %d",
        cfg.data.synthetic.n_clients,
        cfg.data.observation_window_days,
        cfg.seed,
    )

    lat = _sample_latents(cfg, rng)
    causal = _causal_layer(cfg, lat, rng)

    clients = _build_clients(cfg, lat)
    products = _build_products(cfg, rng)
    uplift_train = pd.DataFrame(
        {
            "client_id": lat.client_ids,
            "treatment_flg": causal["treatment"],
            "target": causal["target"],
        }
    )

    clients_path = raw_dir / "clients.csv"
    products_path = raw_dir / "products.csv"
    uplift_path = raw_dir / "uplift_train.csv"
    purchases_path = raw_dir / "purchases.csv"

    clients.to_csv(clients_path, index=False, encoding="utf-8")
    products.to_csv(products_path, index=False, encoding="utf-8")
    uplift_train.to_csv(uplift_path, index=False, encoding="utf-8")

    # Транзакции пишутся блоками: полный кадр в память не собирается
    if purchases_path.exists():
        purchases_path.unlink()
    total_rows = 0
    total_tx = 0
    for start in range(0, lat.n, block_size):
        block = slice(start, min(start + block_size, lat.n))
        chunk = _generate_purchase_block(cfg, lat, block, rng)
        if chunk.empty:
            continue
        chunk.to_csv(
            purchases_path,
            mode="a",
            header=not purchases_path.exists() or total_rows == 0,
            index=False,
            encoding="utf-8",
        )
        total_rows += len(chunk)
        total_tx += int(chunk["transaction_id"].nunique())
    logger.info("purchases.csv: %d товарных строк, ~%d транзакций", total_rows, total_tx)

    truth = pd.DataFrame(
        {
            "client_id": lat.client_ids,
            "true_propensity": causal["propensity"].astype(np.float32),
            "true_mu0": causal["mu0"].astype(np.float32),
            "true_tau": causal["tau_true"].astype(np.float32),
            "true_y0": causal["y0"],
            "true_y1": causal["y1"],
            "revenue": causal["revenue"],
        }
    )
    truth_path = raw_dir / TRUTH_FILENAME
    save_parquet(truth, truth_path)

    treatment = causal["treatment"]
    target = causal["target"]
    tau = causal["tau_true"]
    delta_corr = float(target[treatment == 1].mean() - target[treatment == 0].mean())
    summary = {
        "n_clients": lat.n,
        "n_purchase_rows": total_rows,
        "treatment_share": float(treatment.mean()),
        "target_rate": float(target.mean()),
        "true_ate": float(tau.mean()),
        "observed_delta_corr": delta_corr,
        "confounding_bias": delta_corr - float(tau.mean()),
        "sleeping_dogs_share": float((tau < 0).mean()),
        "reference_date": str(REFERENCE_DATE.date()),
        "seed": cfg.seed,
    }
    logger.info(
        "истинный ATE=%.4f, наблюдаемая Δ_corr=%.4f (смещение конфаундеров %.4f)",
        summary["true_ate"],
        summary["observed_delta_corr"],
        summary["confounding_bias"],
    )

    return SyntheticDataset(
        paths={
            "clients": clients_path,
            "products": products_path,
            "uplift_train": uplift_path,
            "purchases": purchases_path,
            "truth": truth_path,
        },
        truth=truth,
        summary=summary,
    )


def load_truth(cfg: Config) -> pd.DataFrame | None:
    """Прочитать ground truth синтетики, если он есть (иначе ``None``)."""
    path = cfg.raw_dir / TRUTH_FILENAME
    if not path.is_file():
        return None
    return pd.read_parquet(path)


__all__ = [
    "REFERENCE_DATE",
    "TRUTH_FILENAME",
    "LatentClients",
    "SyntheticDataset",
    "generate_raw_dataset",
    "load_truth",
]
