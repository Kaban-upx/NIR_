"""Сквозные тесты слоёв 1–2 (SPEC §11, критерии приёмки 1–4 и 7).

Здесь проверяется не отдельная функция, а то, что конвейер целиком даёт
обещанный результат: витрина удовлетворяет контракту ETL-6, граф действительно
определяет множество корректировки, а скорректированный ATE ближе к истинному
эффекту синтетики, чем наблюдаемая разность ``Δ_corr``. Последнее — главная
проверка гипотезы §1.3: если она не выполняется, причинный слой не делает того,
ради чего существует.

Все тесты помечены ``integration`` и ``slow``: витрина собирается один раз на
сессию (фикстура ``layer1``), причинный слой и аудит — по одному разу на модуль.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_uplift.causal.identify import (
    ATE_ARTIFACT,
    ATE_REPORT,
    ESTIMAND_ARTIFACT,
    CausalAnalysis,
    run_causal_analysis,
)
from causal_uplift.causal.refute import (
    REFUTATION_ARTIFACT,
    REFUTATION_REPORT,
    RefutationAudit,
    run_refutation,
)
from causal_uplift.config import Config
from causal_uplift.eda import EDA_ARTIFACT, EDA_REPORT, split_balance_table
from causal_uplift.etl.aggregate import WINDOW_ARTIFACT
from causal_uplift.etl.schema import FeatureStoreContract
from causal_uplift.features.build import (
    CLIP_ARTIFACT,
    ENCODER_ARTIFACT,
    FEATURE_META_ARTIFACT,
    SCALER_ARTIFACT,
    feature_store_covariates,
    load_feature_store,
)
from causal_uplift.io_utils import load_json
from causal_uplift.pipeline.run import run_stage
from causal_uplift.splitting import SPLIT_COLUMN, SPLIT_LABELS

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="session")
def store(layer1: Config) -> pd.DataFrame:
    return load_feature_store(layer1)


@pytest.fixture(scope="session")
def causal(layer1: Config) -> CausalAnalysis:
    """Причинный слой на **всей** витрине.

    ``causal.estimation_sample`` в тестовом профиле намеренно мал (прогон
    быстрый), но проверка «ATE ближе к истине, чем Δ_corr» — утверждение о
    качестве оценки, и прореживать выборку под ней значит проверять шум.
    """
    cfg = layer1.model_copy(deep=True)
    cfg.causal.estimation_sample = len(load_feature_store(cfg))
    return run_causal_analysis(cfg)


@pytest.fixture(scope="session")
def audit(layer1: Config) -> RefutationAudit:
    return run_refutation(layer1)


# =========================================================================
# Слой 1: ETL, витрина, EDA
# =========================================================================


def test_layer1_writes_every_declared_artifact(layer1: Config) -> None:
    """Критерий приёмки 1: этапы оставляют после себя воспроизводимые файлы."""
    expected = [
        layer1.feature_store_path,
        layer1.interim_dir / "transactions.parquet",
        layer1.interim_dir / "assortment.parquet",
        layer1.interim_dir / "outcome_revenue.parquet",
        layer1.artifact(WINDOW_ARTIFACT),
        layer1.artifact(SCALER_ARTIFACT),
        layer1.artifact(ENCODER_ARTIFACT),
        layer1.artifact(CLIP_ARTIFACT),
        layer1.artifact(FEATURE_META_ARTIFACT),
        layer1.artifact(EDA_ARTIFACT),
        layer1.report(EDA_REPORT),
        layer1.report("corr_matrix.png"),
        layer1.report("corr_matrix.csv"),
        layer1.report("dropped_features.json"),
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    assert missing == []


def test_feature_store_satisfies_etl6_contract(layer1: Config, store: pd.DataFrame) -> None:
    """ETL-6 / критерий приёмки 3: ≥ 30 ковариат и ни одного нарушения контракта."""
    covariates = feature_store_covariates(store)
    assert len(covariates) >= layer1.features.min_covariates

    contract = FeatureStoreContract(
        min_covariates=layer1.features.min_covariates, outcome=layer1.data.outcome
    )
    contract.assert_valid(store.drop(columns=[SPLIT_COLUMN]))


def test_feature_meta_describes_the_actual_frame(layer1: Config, store: pd.DataFrame) -> None:
    """Паспорт витрины обязан совпадать с самой витриной, иначе он вводит в заблуждение."""
    meta = load_json(layer1.artifact(FEATURE_META_ARTIFACT))
    assert meta["covariates"] == feature_store_covariates(store)
    assert meta["n_covariates"] == len(feature_store_covariates(store))
    assert meta["n_clients"] == len(store)
    assert meta["outcome"] == layer1.data.outcome


def test_covariates_contain_no_missing_or_infinite_values(store: pd.DataFrame) -> None:
    """Пропуск или ``inf`` в ``X`` уронил бы propensity-модель уже внутри DoWhy."""
    values = store[feature_store_covariates(store)].to_numpy(dtype="float64")
    assert np.isfinite(values).all()


def test_zscore_normalisation_is_fitted_on_train(layer1: Config, store: pd.DataFrame) -> None:
    """ETL-4: на train средние ≈ 0 и std ≈ 1; на valid/test — нет (и это правильно)."""
    covariates = feature_store_covariates(store)
    train = store.loc[store[SPLIT_COLUMN] == "train", covariates]
    assert train["age"].mean() == pytest.approx(0.0, abs=1e-5)
    assert train["age"].std(ddof=0) == pytest.approx(1.0, abs=1e-4)

    scaler = load_json(layer1.artifact(SCALER_ARTIFACT))
    assert scaler["type"] == "zscore"
    assert set(scaler["columns"]) == set(covariates)


def test_split_labels_cover_every_client_exactly_once(store: pd.DataFrame) -> None:
    assert store[SPLIT_COLUMN].notna().all()
    assert set(store[SPLIT_COLUMN].unique()) == set(SPLIT_LABELS)
    assert store["client_id"].is_unique


def test_split_preserves_treatment_and_target_shares(layer1: Config, store: pd.DataFrame) -> None:
    """ML-2: доли ``T=1`` и отклика сопоставимы между частями сплита."""
    table = split_balance_table(store, outcome=layer1.data.outcome)
    overall_t = float(store["treatment"].mean())
    overall_y = float(store[layer1.data.outcome].mean())
    for _, row in table.iterrows():
        assert row["share_treated"] == pytest.approx(overall_t, abs=0.04)
        assert row[f"{layer1.data.outcome}_rate"] == pytest.approx(overall_y, abs=0.05)


def test_eda_report_states_the_actual_group_ratio(layer1: Config) -> None:
    """Критерий приёмки 2: фактическое соотношение групп зафиксировано в отчёте."""
    text = layer1.report(EDA_REPORT).read_text(encoding="utf-8")
    assert "критерий приёмки 2" in text
    assert "T=1 : T=0" in text
    assert "Δ_corr" in text
    # Отчёт обязан называть обе величины, иначе смещение не обсуждаемо
    assert "Окно наблюдения" in text


def test_eda_artifact_reports_both_groups(layer1: Config) -> None:
    payload = load_json(layer1.artifact(EDA_ARTIFACT))
    balance = payload["balance"]
    assert balance["n_treated"] > 0
    assert balance["n_control"] > 0
    assert balance["n_total"] == balance["n_treated"] + balance["n_control"]
    assert 0.0 < balance["share_treated"] < 1.0
    assert payload["n_covariates"] >= layer1.features.min_covariates


def test_run_stage_records_a_run_journal(layer1: Config) -> None:
    """ML-5: прогон этапа оставляет запись в журнале с метриками."""
    run_stage(layer1, "eda", journal=True)

    index_path = layer1.resolve(layer1.tracking.runs_dir) / "index.jsonl"
    entries = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    last = entries[-1]
    assert last["stage"] == "eda"
    assert last["status"] == "success"
    # _flatten_metrics разворачивает вложенные словари в плоские скаляры
    assert "balance.delta_corr" in last["metrics"]
    assert (layer1.resolve(layer1.tracking.runs_dir) / f"{last['run_id']}.json").is_file()


# =========================================================================
# Слой 2: идентификация и оценка эффекта
# =========================================================================


def test_identification_yields_a_nonempty_backdoor_set(
    causal: CausalAnalysis, store: pd.DataFrame
) -> None:
    """CAU-1: эффект идентифицирован, множество корректировки непустое."""
    assert causal.backdoor_variables
    assert set(causal.backdoor_variables) <= set(feature_store_covariates(store))


def test_seasonality_features_stay_out_of_the_backdoor_set(causal: CausalAnalysis) -> None:
    """Граф не декорация: сезонность влияет только на ``Y`` и в корректировку не входит.

    Если бы код подставлял в множество корректировки все колонки витрины
    подряд, этот тест упал бы — и именно он отличает реальное использование
    критерия заднего входа от «посчитаем регрессию на всём, что есть».
    """
    seasonal = causal.graph.groups.get("seasonality", [])
    assert seasonal, "в витрине не осталось сезонных признаков — проверять нечего"
    assert set(seasonal) == set(causal.graph.outcome_only)
    assert not set(seasonal) & set(causal.backdoor_variables)


def test_both_estimators_converge_and_agree_in_sign(causal: CausalAnalysis) -> None:
    """CAU-3: минимум две оценки, и обе показывают эффект одного знака."""
    assert len(causal.estimates) >= 2
    failed = [e.method for e in causal.estimates if e.failed is not None]
    assert failed == []

    signs = {e.ate > 0 for e in causal.estimates}
    assert len(signs) == 1, "оценщики расходятся в знаке эффекта — результату доверять нельзя"
    assert causal.spread >= 0.0
    assert causal.as_dict()["estimator_spread"] == pytest.approx(causal.spread, abs=1e-6)


def test_confounding_bias_is_delta_minus_ate(causal: CausalAnalysis) -> None:
    """CAU-2: смещение конфаундинга — это результат, а не промежуточная величина."""
    assert causal.confounding_bias == pytest.approx(causal.delta_corr - causal.ate, abs=1e-9)
    payload = causal.as_dict()
    assert payload["confounding_bias"] == pytest.approx(
        payload["delta_corr"] - payload["ate"], abs=1e-5
    )


def test_adjustment_moves_the_estimate_towards_the_true_effect(causal: CausalAnalysis) -> None:
    """Главная проверка §1.3 на синтетике с известным ``τ(x)``.

    Генератор строит конфаундинг так, что ``Δ_corr`` завышает эффект: в
    treatment-группу чаще попадают активные клиенты с высоким базовым откликом.
    Корректировка по backdoor-множеству обязана уменьшить ошибку — иначе весь
    причинный слой не выполняет своего назначения.
    """
    assert causal.true_ate is not None, "ground truth синтетики не присоединён к витрине"
    err_adjusted = abs(causal.ate - causal.true_ate)
    err_observed = abs(causal.delta_corr - causal.true_ate)
    assert err_adjusted < err_observed, (
        f"ATE={causal.ate:+.4f} не ближе к истине {causal.true_ate:+.4f}, "
        f"чем Δ_corr={causal.delta_corr:+.4f}"
    )


def test_ate_report_shows_both_quantities(layer1: Config) -> None:
    """CAU-4: отчёт называет и наблюдаемую разность, и скорректированный эффект."""
    text = layer1.report(ATE_REPORT).read_text(encoding="utf-8")
    for fragment in (
        "Δ_corr",
        "измеренное смещение конфаундинга",
        "Backdoor-множество",
        "Estimand",
        "ненаблюдаемых",
    ):
        assert fragment in text, fragment


def test_estimand_artifact_records_identification(layer1: Config, causal: CausalAnalysis) -> None:
    payload = load_json(layer1.artifact(ESTIMAND_ARTIFACT))
    assert payload["identification"] == layer1.causal.identification
    assert payload["backdoor_variables"] == causal.backdoor_variables
    assert payload["n_backdoor_variables"] == len(causal.backdoor_variables)
    assert payload["treatment"] == "treatment"
    assert payload["outcome"] == layer1.data.outcome
    assert payload["estimand_text"].strip()


def test_ate_artifact_mirrors_the_analysis(layer1: Config, causal: CausalAnalysis) -> None:
    payload = load_json(layer1.artifact(ATE_ARTIFACT))
    assert payload["primary_method"] == layer1.causal.estimators[0]
    assert payload["ate"] == pytest.approx(round(causal.ate, 6), abs=1e-6)
    assert payload["n_treated"] + payload["n_control"] == payload["n_rows"]


def test_expanded_graph_artifact_is_readable_back(layer1: Config, causal: CausalAnalysis) -> None:
    """Развёрнутый DAG — проверяемый артефакт, а не внутреннее состояние прогона."""
    path = layer1.resolve(layer1.causal.expanded_graph_path)
    restored = nx.read_gml(path)
    assert set(restored.nodes()) == set(causal.graph.graph.nodes())
    assert restored.has_edge("treatment", layer1.data.outcome)
    assert nx.is_directed_acyclic_graph(restored)


# =========================================================================
# Слой 2: аудит причинных допущений
# =========================================================================


def test_refutation_grades_every_configured_test(layer1: Config, audit: RefutationAudit) -> None:
    """Все заявленные тесты выполнены и получили вердикт (критерий приёмки 7)."""
    assert [o.test for o in audit.outcomes] == list(layer1.refutation.tests)
    assert len(audit.graded) == len(layer1.refutation.tests)
    for outcome in audit.outcomes:
        assert outcome.verdict in {"PASS", "FAIL", "ERROR"}
    assert audit.n_passed <= len(audit.graded)


def test_refutation_artifacts_are_written(layer1: Config, audit: RefutationAudit) -> None:
    payload = load_json(layer1.artifact(REFUTATION_ARTIFACT))
    assert payload["n_tests"] == len(audit.graded)
    assert payload["n_passed"] == audit.n_passed
    assert payload["all_passed"] == audit.all_passed

    text = layer1.report(REFUTATION_REPORT).read_text(encoding="utf-8")
    assert "Плацебо-воздействие" in text
    # Отчёт обязан честно оговаривать границы метода
    assert "Ни один из них не доказывает отсутствие" in text


def test_placebo_effect_is_small_relative_to_the_estimate(audit: RefutationAudit) -> None:
    """Содержательная проверка: на перестановленном ``T`` эффекта быть не должно."""
    placebo = next(o for o in audit.outcomes if o.test == "placebo_treatment_refuter")
    assert placebo.failed is None, placebo.failed
    assert placebo.new_effect is not None
    assert abs(placebo.new_effect) < abs(audit.original)
