"""Аудит причинных допущений: refutation-тесты DoWhy (W9, SPEC §6.4).

Модуль отвечает на вопрос, который нельзя решить метриками качества: не
является ли полученный ATE артефактом неверного графа или случайности. Три
обязательных теста и их логика вердикта:

* **random_common_cause** — в данные добавляется независимая случайная
  переменная как общая причина. Корректно идентифицированный эффект не должен
  измениться: если изменился, множество корректировки неустойчиво.
* **placebo_treatment_refuter** — воздействие заменяется перестановкой
  (``placebo_type="permute"``). Эффект обязан стать ≈ 0: ненулевой
  плацебо-эффект означает, что оценщик улавливает структуру данных, а не
  воздействие.
* **data_subset_refuter** — оценка повторяется на случайной подвыборке.
  Существенное расхождение говорит, что результат держится на отдельных
  наблюдениях.

Вердикты PASS/FAIL считаются по относительным порогам из
``refutation.thresholds``: p-value DoWhy приводится как справочная величина,
но решение на нём не строится — его знак интерпретируется по-разному для
разных тестов и менялся между версиями библиотеки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from causal_uplift.causal.dag import build_expanded_graph
from causal_uplift.causal.identify import (
    build_model,
    estimate_single,
    estimation_frame,
    identify,
)
from causal_uplift.config import Config
from causal_uplift.features.build import feature_store_covariates, load_feature_store
from causal_uplift.io_utils import df_to_markdown, fmt_float, save_json, write_markdown
from causal_uplift.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

REFUTATION_ARTIFACT = "refutation.json"
REFUTATION_REPORT = "refutation.md"

#: Минимальный знаменатель для относительных отклонений: при ATE ≈ 0 деление
#: на него превращает любой шум в «провал», что неинформативно
MIN_DENOMINATOR = 1e-6

#: Человекочитаемые названия тестов для отчёта
TEST_TITLES: dict[str, str] = {
    "random_common_cause": "Добавление случайной общей причины",
    "placebo_treatment_refuter": "Плацебо-воздействие (перестановка T)",
    "data_subset_refuter": "Оценка на подвыборке данных",
    "add_unobserved_common_cause": "Чувствительность к ненаблюдаемому конфаундеру",
}

#: Что именно проверяет вердикт каждого теста
TEST_EXPECTATION: dict[str, str] = {
    "random_common_cause": "оценка не меняется",
    "placebo_treatment_refuter": "эффект обращается в нуль",
    "data_subset_refuter": "оценка не меняется",
    "add_unobserved_common_cause": "справочный тест, вердикт не выносится",
}


@dataclass
class RefutationOutcome:
    """Результат одного refutation-теста."""

    test: str
    original: float
    new_effect: float | None = None
    relative: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    p_value: float | None = None
    detail: str = ""
    failed: str | None = None

    @property
    def verdict(self) -> str:
        if self.failed is not None:
            return "ERROR"
        if self.passed is None:
            return "—"
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, object]:
        return {
            "test": self.test,
            "verdict": self.verdict,
            "original_effect": round(self.original, 6),
            "new_effect": None if self.new_effect is None else round(self.new_effect, 6),
            "relative_deviation": None if self.relative is None else round(self.relative, 6),
            "threshold": self.threshold,
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "detail": self.detail,
            "failed": self.failed,
        }


@dataclass
class RefutationAudit:
    """Сводный результат аудита причинности."""

    outcomes: list[RefutationOutcome]
    original: float
    method: str
    n_rows: int
    n_simulations: int
    report_path: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def graded(self) -> list[RefutationOutcome]:
        """Тесты, по которым выносится вердикт (справочные исключены)."""
        return [o for o in self.outcomes if o.passed is not None or o.failed is not None]

    @property
    def n_passed(self) -> int:
        return sum(1 for o in self.graded if o.passed)

    @property
    def all_passed(self) -> bool:
        """Все вердиктные тесты пройдены.

        Ошибка выполнения теста — не PASS: невыполненная проверка не может
        считаться подтверждением, иначе падение refuter-а молча превращалось бы
        в «причинность подтверждена».
        """
        return bool(self.graded) and all(o.passed for o in self.graded)

    def as_dict(self) -> dict[str, object]:
        return {
            "original_effect": round(self.original, 6),
            "method": self.method,
            "n_rows": self.n_rows,
            "n_simulations": self.n_simulations,
            "n_tests": len(self.graded),
            "n_passed": self.n_passed,
            "all_passed": self.all_passed,
            "tests": [o.as_dict() for o in self.outcomes],
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
        }


# ---------------------------------------------------------------------------
# Запуск отдельного теста
# ---------------------------------------------------------------------------


def _refuter_kwargs(test: str, cfg: Config) -> dict[str, Any]:
    """Параметры refuter-а из конфигурации (магических чисел в коде нет)."""
    kwargs: dict[str, Any] = {
        "num_simulations": cfg.refutation.num_simulations,
        # ML-1: refuters сами тасуют данные, без сида вердикт менялся бы от прогона
        "random_seed": cfg.seed,
    }
    if test == "placebo_treatment_refuter":
        # Перестановка сохраняет маргинальное распределение T, в отличие от
        # генерации нового случайного вектора: проверяется именно связь T с Y.
        kwargs["placebo_type"] = "permute"
    if test == "data_subset_refuter":
        kwargs["subset_fraction"] = cfg.refutation.subset_fraction
    return kwargs


def _grade(test: str, original: float, new_effect: float, cfg: Config) -> tuple[float, float, bool]:
    """Относительное отклонение, порог и вердикт теста."""
    thresholds = cfg.refutation.thresholds
    denom = max(abs(original), MIN_DENOMINATOR)
    if test == "placebo_treatment_refuter":
        # Ожидание — нуль, поэтому сравнивается сам новый эффект, а не его
        # отклонение от исходной оценки.
        relative = abs(new_effect) / denom
        threshold = thresholds.placebo_abs_rel_tol
    elif test == "random_common_cause":
        relative = abs(new_effect - original) / denom
        threshold = thresholds.random_common_cause_rel_tol
    else:
        relative = abs(new_effect - original) / denom
        threshold = thresholds.data_subset_rel_tol
    return relative, threshold, relative <= threshold


def _extract(result: Any) -> tuple[float | None, float | None]:
    """Вытащить из результата DoWhy новый эффект и p-value."""
    new_effect = getattr(result, "new_effect", None)
    if isinstance(new_effect, list | tuple):
        new_effect = sum(float(v) for v in new_effect) / len(new_effect)
    payload = getattr(result, "refutation_result", None)
    p_value = payload.get("p_value") if isinstance(payload, dict) else None
    if isinstance(p_value, list | tuple):
        p_value = float(p_value[0])
    return (
        None if new_effect is None else float(new_effect),
        None if p_value is None else float(p_value),
    )


def run_test(
    model: Any, estimand: Any, estimate: Any, test: str, cfg: Config
) -> RefutationOutcome:
    """Выполнить один refutation-тест и вынести вердикт."""
    original = float(estimate.value)
    with log_stage(logger, f"refutation: {test}"):
        try:
            result = model.refute_estimate(
                estimand,
                estimate,
                method_name=test,
                show_progress_bar=False,
                **_refuter_kwargs(test, cfg),
            )
        except Exception as exc:  # noqa: BLE001 — причина уходит в отчёт как ERROR
            logger.warning("тест %s не выполнен: %s", test, exc)
            return RefutationOutcome(test=test, original=original, failed=str(exc))

    new_effect, p_value = _extract(result)
    if new_effect is None:
        # Тест такого типа (например sensitivity-анализ) не возвращает
        # сопоставимую точечную оценку — оставляем как справочный.
        return RefutationOutcome(
            test=test,
            original=original,
            p_value=p_value,
            detail=str(result).replace("\n", " ")[:300],
        )

    relative, threshold, passed = _grade(test, original, new_effect, cfg)
    detail = ""
    if abs(original) <= MIN_DENOMINATOR:
        detail = (
            "исходный эффект близок к нулю: относительное отклонение считалось "
            f"к минимальному знаменателю {MIN_DENOMINATOR:g}"
        )
    logger.info(
        "%s: %s → %+.6f (отклонение %.4f при пороге %.2f) — %s",
        test,
        f"{original:+.6f}",
        new_effect,
        relative,
        threshold,
        "PASS" if passed else "FAIL",
    )
    return RefutationOutcome(
        test=test,
        original=original,
        new_effect=new_effect,
        relative=relative,
        threshold=threshold,
        passed=passed,
        p_value=p_value,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


def outcomes_table(audit: RefutationAudit) -> pd.DataFrame:
    rows = []
    for o in audit.outcomes:
        rows.append(
            {
                "тест": TEST_TITLES.get(o.test, o.test),
                "ожидание": TEST_EXPECTATION.get(o.test, "—"),
                "исходный эффект": o.original,
                "эффект после теста": float("nan") if o.new_effect is None else o.new_effect,
                "отклонение": float("nan") if o.relative is None else o.relative,
                "порог": float("nan") if o.threshold is None else o.threshold,
                "p-value": float("nan") if o.p_value is None else o.p_value,
                "вердикт": o.verdict,
            }
        )
    return pd.DataFrame(rows)


def render_report(audit: RefutationAudit, cfg: Config) -> list[str]:
    """Markdown ``reports/refutation.md`` (критерий приёмки 7)."""
    headline = (
        "**Все тесты пройдены (PASS).** Причинное заключение устойчиво к проведённым "
        "проверкам."
        if audit.all_passed
        else f"**Пройдено {audit.n_passed} из {len(audit.graded)}.** Ниже разобрано, "
        "какие допущения не подтвердились."
    )

    lines = [
        "# Аудит причинных допущений (refutation-тесты)",
        "",
        headline,
        "",
        f"Проверяемая оценка: `{audit.method}`, ATE = {fmt_float(audit.original, 6)}; "
        f"выборка {audit.n_rows} строк, симуляций на тест — {audit.n_simulations}.",
        "",
        "## 1. Результаты",
        "",
        df_to_markdown(outcomes_table(audit), floatfmt="{:+.6f}"),
        "",
        "Колонка «отклонение» — относительная величина: для плацебо это "
        "`|эффект| / |ATE|`, для остальных тестов `|новый − исходный| / |ATE|`. "
        "Вердикт — сравнение с порогом из `refutation.thresholds`.",
        "",
        "## 2. Как читать каждый тест",
        "",
    ]

    for o in audit.outcomes:
        lines += [
            f"### {TEST_TITLES.get(o.test, o.test)} — {o.verdict}",
            "",
        ]
        if o.failed is not None:
            lines += [
                f"Тест не выполнен: `{o.failed}`.",
                "",
                "Невыполненная проверка не считается подтверждением: итоговый вердикт",
                "аудита в этом случае отрицательный.",
                "",
            ]
            continue
        if o.passed is None:
            lines += [
                "Справочный тест: точечная оценка, сопоставимая с исходной, не возвращается,",
                "поэтому вердикт PASS/FAIL не выносится.",
                "",
                f"Результат: {o.detail or '—'}",
                "",
            ]
            continue

        lines += [
            f"* исходная оценка: {fmt_float(o.original, 6)}",
            f"* оценка после вмешательства теста: {fmt_float(o.new_effect, 6)}",
            f"* относительное отклонение: {fmt_float(o.relative, 4)} при пороге "
            f"{fmt_float(o.threshold, 2)}",
            f"* p-value DoWhy (справочно): {fmt_float(o.p_value, 4)}",
            "",
            _interpretation(o),
            "",
        ]
        if o.detail:
            lines += [f"Примечание: {o.detail}.", ""]

    lines += [
        "## 3. Что эти тесты не проверяют",
        "",
        "Ни один из них не доказывает отсутствие **ненаблюдаемого** конфаундера:",
        "это допущение принципиально непроверяемо по данным. Тесты показывают лишь,",
        "что оценка устойчива к добавлению независимого шума, к разрушению связи",
        "`T → Y` и к удалению части наблюдений. Поэтому результат аудита —",
        "необходимое, но не достаточное условие причинной интерпретации.",
        "",
        f"Пороги и число симуляций заданы в `configs/config.yaml` → `refutation`; "
        f"машинный вид результата — `{cfg.artifacts_dir.name}/{REFUTATION_ARTIFACT}`.",
    ]
    return lines


def _interpretation(o: RefutationOutcome) -> str:
    """Вывод словами — без него таблица чисел не является результатом."""
    if o.test == "placebo_treatment_refuter":
        return (
            "Перестановка воздействия обратила эффект практически в нуль — оценщик "
            "реагирует на воздействие, а не на структуру данных."
            if o.passed
            else "Плацебо-воздействие дало ненулевой эффект: оценщик улавливает "
            "регулярность, не связанную с воздействием. Это прямое указание на "
            "ошибку в множестве корректировки или на утечку в признаках."
        )
    if o.test == "random_common_cause":
        return (
            "Добавление независимой случайной общей причины не изменило оценку — "
            "множество корректировки устойчиво к появлению лишней переменной."
            if o.passed
            else "Оценка сдвинулась после добавления независимой переменной: оценщик "
            "чувствителен к составу ковариат, и доверять точечному значению ATE нельзя."
        )
    return (
        "Удаление части наблюдений не изменило оценку — результат не держится на "
        "отдельных наблюдениях."
        if o.passed
        else "Оценка существенно изменилась на подвыборке: результат неустойчив, "
        "вероятны влиятельные наблюдения или нехватка данных в одной из групп."
    )


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_refutation(cfg: Config, *, write: bool = True) -> RefutationAudit:
    """Полный аудит причинности (Э6, W9).

    Идентификация и оценка повторяются здесь заново на отдельной подвыборке
    ``refutation.sample_size``: каждый тест прогоняет оценщик десятки раз, и
    сравнивать его результат нужно с базовой оценкой на **тех же** данных, а не
    с ATE, посчитанным на другой выборке.
    """
    df = load_feature_store(cfg)
    covariates = feature_store_covariates(df)

    graph = build_expanded_graph(cfg, covariates, save=False)
    frame, _ = estimation_frame(cfg, df, covariates, limit=cfg.refutation.sample_size)

    with log_stage(logger, "базовая оценка для аудита"):
        model = build_model(frame, graph)
        estimand = identify(model)
        method = cfg.causal.estimators[0]
        estimate = estimate_single(
            model,
            estimand,
            method,
            confidence=cfg.evaluation.bootstrap.confidence,
            n_simulations=cfg.causal.n_simulations,
            propensity_clip=cfg.model.xlearner.propensity_clip,
        )

    tests = list(cfg.refutation.tests)
    if cfg.refutation.add_unobserved_common_cause:
        tests.append("add_unobserved_common_cause")

    outcomes = [run_test(model, estimand, estimate, test, cfg) for test in tests]

    audit = RefutationAudit(
        outcomes=outcomes,
        original=float(estimate.value),
        method=method,
        n_rows=len(frame),
        n_simulations=cfg.refutation.num_simulations,
    )
    logger.info(
        "аудит причинности: пройдено %d из %d тестов (итог %s)",
        audit.n_passed,
        len(audit.graded),
        "PASS" if audit.all_passed else "FAIL",
    )

    if write:
        audit.artifacts["refutation"] = save_json(
            audit.as_dict(), cfg.artifact(REFUTATION_ARTIFACT)
        )
        audit.report_path = write_markdown(
            render_report(audit, cfg), cfg.report(REFUTATION_REPORT)
        )
        audit.artifacts["report"] = audit.report_path

    return audit


__all__ = [
    "MIN_DENOMINATOR",
    "REFUTATION_ARTIFACT",
    "REFUTATION_REPORT",
    "TEST_EXPECTATION",
    "TEST_TITLES",
    "RefutationAudit",
    "RefutationOutcome",
    "outcomes_table",
    "render_report",
    "run_refutation",
    "run_test",
]
