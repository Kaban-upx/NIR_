"""Тесты логики аудита причинности: вердикты PASS/FAIL (W9, SPEC §6.4).

DoWhy здесь подменён заглушкой: проверяется не то, как библиотека тасует
данные, а наша логика — относительные пороги, трактовка нулевого исходного
эффекта и, главное, что **не выполненный** тест не превращается в
подтверждение причинности. Сквозной прогон refuters с настоящей моделью
живёт в ``test_pipeline_layers.py`` и помечен ``slow``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from causal_uplift.causal.refute import (
    MIN_DENOMINATOR,
    TEST_TITLES,
    RefutationAudit,
    RefutationOutcome,
    outcomes_table,
    render_report,
    run_test,
)
from causal_uplift.config import Config


class _FakeEstimate:
    """Минимальный аналог ``CausalEstimate``: refuter читает только ``value``."""

    def __init__(self, value: float) -> None:
        self.value = value


class _FakeResult:
    """Результат refuter-а DoWhy в том виде, в котором его разбирает ``_extract``."""

    def __init__(
        self,
        new_effect: Any = None,
        *,
        p_value: Any = None,
        text: str = "sensitivity analysis without a point estimate",
    ) -> None:
        self.new_effect = new_effect
        self.refutation_result = None if p_value is None else {"p_value": p_value}
        self._text = text

    def __str__(self) -> str:
        return self._text


class _FakeModel:
    """Подмена ``CausalModel``: запоминает параметры вызова и возвращает заготовку."""

    def __init__(self, result: Any = None, *, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def refute_estimate(
        self,
        estimand: Any,
        estimate: Any,
        *,
        method_name: str,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.calls.append((method_name, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result


def _run(cfg: Config, test: str, original: float, new_effect: Any, **kw: Any) -> RefutationOutcome:
    model = _FakeModel(_FakeResult(new_effect, **kw))
    return run_test(model, None, _FakeEstimate(original), test, cfg)


# ------------------------------------------------------------------ вердикты


def test_unchanged_estimate_passes_random_common_cause(cfg: Config) -> None:
    outcome = _run(cfg, "random_common_cause", 0.1, 0.1)
    assert outcome.verdict == "PASS"
    assert outcome.relative == pytest.approx(0.0)
    assert outcome.threshold == cfg.refutation.thresholds.random_common_cause_rel_tol


def test_shifted_estimate_fails_random_common_cause(cfg: Config) -> None:
    """Сдвиг на 30 % при пороге 15 %: добавление шумовой переменной не должно влиять."""
    outcome = _run(cfg, "random_common_cause", 0.1, 0.13)
    assert outcome.verdict == "FAIL"
    assert outcome.relative == pytest.approx(0.3, abs=1e-9)


def test_placebo_effect_near_zero_passes(cfg: Config) -> None:
    outcome = _run(cfg, "placebo_treatment_refuter", 0.1, 0.005)
    assert outcome.verdict == "PASS"
    # Для плацебо сравнивается сам эффект, а не его отклонение от исходного
    assert outcome.relative == pytest.approx(0.05, abs=1e-9)


def test_placebo_effect_equal_to_original_fails(cfg: Config) -> None:
    outcome = _run(cfg, "placebo_treatment_refuter", 0.1, 0.1)
    assert outcome.verdict == "FAIL"
    assert outcome.relative == pytest.approx(1.0)


def test_data_subset_deviation_is_graded_by_its_own_threshold(cfg: Config) -> None:
    outcome = _run(cfg, "data_subset_refuter", 0.2, 0.21)
    assert outcome.threshold == cfg.refutation.thresholds.data_subset_rel_tol
    assert outcome.verdict == "PASS"


# ------------------------------------------------------- параметры refuter-ов


def test_every_refuter_gets_seed_and_simulation_count(cfg: Config) -> None:
    """ML-1: без сида вердикт менялся бы от прогона к прогону."""
    for test in ("random_common_cause", "placebo_treatment_refuter", "data_subset_refuter"):
        model = _FakeModel(_FakeResult(0.0))
        run_test(model, None, _FakeEstimate(0.1), test, cfg)
        (method, kwargs) = model.calls[0]
        assert method == test
        assert kwargs["random_seed"] == cfg.seed
        assert kwargs["num_simulations"] == cfg.refutation.num_simulations


def test_placebo_uses_permutation_not_a_fresh_random_vector(cfg: Config) -> None:
    model = _FakeModel(_FakeResult(0.0))
    run_test(model, None, _FakeEstimate(0.1), "placebo_treatment_refuter", cfg)
    assert model.calls[0][1]["placebo_type"] == "permute"


def test_subset_fraction_comes_from_configuration(cfg: Config) -> None:
    model = _FakeModel(_FakeResult(0.1))
    run_test(model, None, _FakeEstimate(0.1), "data_subset_refuter", cfg)
    assert model.calls[0][1]["subset_fraction"] == cfg.refutation.subset_fraction
    assert "placebo_type" not in model.calls[0][1]


# --------------------------------------------------------- разбор результата


def test_list_valued_new_effect_is_averaged(cfg: Config) -> None:
    """Часть refuter-ов возвращает вектор по симуляциям, а не скаляр."""
    outcome = _run(cfg, "random_common_cause", 0.1, [0.09, 0.11])
    assert outcome.new_effect == pytest.approx(0.1)


def test_p_value_is_extracted_as_a_reference_value(cfg: Config) -> None:
    outcome = _run(cfg, "random_common_cause", 0.1, 0.1, p_value=[0.42])
    assert outcome.p_value == pytest.approx(0.42)
    # Решение строится на пороге, а не на p-value — вердикт остаётся PASS
    assert outcome.verdict == "PASS"


def test_result_without_point_estimate_stays_reference_only(cfg: Config) -> None:
    outcome = _run(cfg, "add_unobserved_common_cause", 0.1, None, text="k=1.5 → ATE 0.08")
    assert outcome.verdict == "—"
    assert outcome.passed is None
    assert "k=1.5" in outcome.detail

    audit = RefutationAudit(outcomes=[outcome], original=0.1, method="m", n_rows=10, n_simulations=3)
    assert audit.graded == []
    assert audit.all_passed is False


def test_failed_refuter_is_an_error_not_a_pass(cfg: Config) -> None:
    """Невыполненная проверка не может считаться подтверждением причинности."""
    model = _FakeModel(raises=RuntimeError("propensity-модель не сошлась"))
    outcome = run_test(model, None, _FakeEstimate(0.1), "random_common_cause", cfg)

    assert outcome.verdict == "ERROR"
    assert outcome.failed is not None
    assert "не сошлась" in outcome.failed

    audit = RefutationAudit(outcomes=[outcome], original=0.1, method="m", n_rows=10, n_simulations=3)
    assert audit.graded == [outcome]
    assert audit.n_passed == 0
    assert audit.all_passed is False


def test_near_zero_original_effect_does_not_explode(cfg: Config) -> None:
    """При ATE ≈ 0 относительное отклонение считается к минимальному знаменателю."""
    outcome = _run(cfg, "random_common_cause", 0.0, 0.0)
    assert outcome.relative == pytest.approx(0.0)
    assert "близок к нулю" in outcome.detail
    assert f"{MIN_DENOMINATOR:g}" in outcome.detail


# ------------------------------------------------------------- сводка аудита


def _outcome(test: str, *, passed: bool) -> RefutationOutcome:
    return RefutationOutcome(
        test=test,
        original=0.1,
        new_effect=0.1 if passed else 0.2,
        relative=0.0 if passed else 1.0,
        threshold=0.15,
        passed=passed,
    )


def test_audit_requires_every_graded_test_to_pass() -> None:
    audit = RefutationAudit(
        outcomes=[
            _outcome("random_common_cause", passed=True),
            _outcome("data_subset_refuter", passed=False),
        ],
        original=0.1,
        method="propensity_score_weighting",
        n_rows=1000,
        n_simulations=3,
    )
    assert audit.n_passed == 1
    assert len(audit.graded) == 2
    assert audit.all_passed is False


def test_audit_with_all_tests_passing_is_a_pass() -> None:
    audit = RefutationAudit(
        outcomes=[
            _outcome("random_common_cause", passed=True),
            _outcome("placebo_treatment_refuter", passed=True),
        ],
        original=0.1,
        method="propensity_score_weighting",
        n_rows=1000,
        n_simulations=3,
    )
    assert audit.all_passed is True
    assert json.dumps(audit.as_dict(), ensure_ascii=False)


def test_outcomes_table_has_a_titled_row_per_test() -> None:
    audit = RefutationAudit(
        outcomes=[
            _outcome("random_common_cause", passed=True),
            _outcome("placebo_treatment_refuter", passed=False),
        ],
        original=0.1,
        method="m",
        n_rows=10,
        n_simulations=3,
    )
    table = outcomes_table(audit)
    assert len(table) == 2
    assert table["вердикт"].tolist() == ["PASS", "FAIL"]
    assert TEST_TITLES["random_common_cause"] in table["тест"].tolist()


def test_report_headline_reflects_the_overall_verdict(cfg: Config) -> None:
    passing = RefutationAudit(
        outcomes=[_outcome("random_common_cause", passed=True)],
        original=0.1,
        method="m",
        n_rows=10,
        n_simulations=3,
    )
    text = "\n".join(render_report(passing, cfg))
    assert "Все тесты пройдены" in text
    assert "Ни один из них не доказывает отсутствие" in text

    failing = RefutationAudit(
        outcomes=[
            _outcome("random_common_cause", passed=True),
            _outcome("placebo_treatment_refuter", passed=False),
        ],
        original=0.1,
        method="m",
        n_rows=10,
        n_simulations=3,
    )
    text = "\n".join(render_report(failing, cfg))
    assert "Пройдено 1 из 2" in text
    # Провал плацебо-теста обязан быть объяснён словами, а не только числом
    assert "утечк" in text


def test_report_explains_a_failed_test_run(cfg: Config) -> None:
    broken = RefutationOutcome(
        test="data_subset_refuter", original=0.1, failed="LinAlgError: singular matrix"
    )
    audit = RefutationAudit(
        outcomes=[broken], original=0.1, method="m", n_rows=10, n_simulations=3
    )
    text = "\n".join(render_report(audit, cfg))
    assert "Тест не выполнен" in text
    assert "singular matrix" in text
    assert "не считается подтверждением" in text
