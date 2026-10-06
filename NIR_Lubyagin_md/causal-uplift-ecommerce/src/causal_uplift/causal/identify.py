"""Идентификация и оценка причинного эффекта через DoWhy (W5, SPEC §6.2–6.3).

Слой реализует четыре требования:

* **CAU-1** — идентификация строго через ``identify_effect``; отсутствие
  backdoor-оценки останавливает конвейер. Это не формальность: без
  выполненного критерия заднего входа любая дальнейшая оценка ATE не имеет
  причинной интерпретации, и продолжать расчёт — значит выдавать корреляцию
  за эффект.
* **CAU-2** — в отчёт выводятся обе величины: наблюдаемая разность отклика
  ``Δ_corr`` и скорректированный ATE. Их разность и есть измеренное смещение
  конфаундинга — ключевой результат §1.3 НИР.
* **CAU-3** — ATE оценивается минимум двумя методами (взвешивание по
  propensity и линейная регрессия); расхождение оценок логируется.
* **CAU-4** — текстовый вид estimand сохраняется в
  ``artifacts/estimand.json``, сводка — в ``reports/ate_summary.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_uplift.causal.dag import (
    ExpandedGraph,
    build_expanded_graph,
    group_table,
    validate_against_frame,
)
from causal_uplift.config import Config
from causal_uplift.features.build import feature_store_covariates, load_feature_store
from causal_uplift.io_utils import df_to_markdown, fmt_float, save_json, write_markdown
from causal_uplift.logging_setup import get_logger, log_stage

logger = get_logger(__name__)

ESTIMAND_ARTIFACT = "estimand.json"
ATE_ARTIFACT = "ate.json"
ATE_REPORT = "ate_summary.md"

#: Соответствие коротких имён из конфигурации полным именам методов DoWhy
ESTIMATOR_METHODS: dict[str, str] = {
    "propensity_score_weighting": "backdoor.propensity_score_weighting",
    "propensity_score_matching": "backdoor.propensity_score_matching",
    "propensity_score_stratification": "backdoor.propensity_score_stratification",
    "linear_regression": "backdoor.linear_regression",
}


class IdentificationError(RuntimeError):
    """Эффект не идентифицируется по критерию заднего входа (CAU-1)."""


@dataclass
class EstimateResult:
    """Оценка ATE одним методом."""

    method: str
    ate: float
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    failed: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "ate": None if self.failed else round(self.ate, 6),
            "stderr": None if self.stderr is None else round(self.stderr, 6),
            "ci_low": None if self.ci_low is None else round(self.ci_low, 6),
            "ci_high": None if self.ci_high is None else round(self.ci_high, 6),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "params": self.params,
            "failed": self.failed,
        }


@dataclass
class CausalAnalysis:
    """Полный результат причинного слоя: estimand + оценки + смещение."""

    delta_corr: float
    estimates: list[EstimateResult]
    primary_method: str
    estimand_text: str
    backdoor_variables: list[str]
    graph: ExpandedGraph
    n_rows: int
    n_treated: int
    n_control: int
    response_treated: float
    response_control: float
    true_ate: float | None = None
    report_path: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def primary(self) -> EstimateResult:
        for est in self.estimates:
            if est.method == self.primary_method and est.failed is None:
                return est
        successful = [e for e in self.estimates if e.failed is None]
        if not successful:
            msg = "ни один оценщик ATE не завершился успешно"
            raise RuntimeError(msg)
        return successful[0]

    @property
    def ate(self) -> float:
        return self.primary.ate

    @property
    def confounding_bias(self) -> float:
        """``Δ_corr − ATE`` — та самая величина смещения отбора (CAU-2)."""
        return self.delta_corr - self.ate

    @property
    def spread(self) -> float:
        """Разброс оценок между методами — индикатор устойчивости (CAU-3)."""
        values = [e.ate for e in self.estimates if e.failed is None]
        return float(max(values) - min(values)) if len(values) > 1 else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "delta_corr": round(self.delta_corr, 6),
            "ate": round(self.ate, 6),
            "confounding_bias": round(self.confounding_bias, 6),
            "primary_method": self.primary_method,
            "estimator_spread": round(self.spread, 6),
            "estimates": [e.as_dict() for e in self.estimates],
            "n_rows": self.n_rows,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "response_rate_treated": round(self.response_treated, 6),
            "response_rate_control": round(self.response_control, 6),
            "n_backdoor_variables": len(self.backdoor_variables),
            "true_ate": None if self.true_ate is None else round(self.true_ate, 6),
            "graph": self.graph.summary(),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
        }


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------


def estimation_frame(
    cfg: Config,
    df: pd.DataFrame,
    covariates: list[str],
    *,
    limit: int | None | str = "config",
) -> tuple[pd.DataFrame, int]:
    """Подвыборка для DoWhy: стратифицированная по ``(T, Y)``.

    DoWhy не чанкуется, а линейная регрессия на 30+ ковариатах и миллионе
    строк не укладывается в бюджет ML-6. Поэтому оценка считается на
    подвыборке ``causal.estimation_sample``, но стратифицированной — иначе при
    дефицитной treatment-группе случайная подвыборка легко теряет большую
    часть ``T=1``, и propensity-веса взрываются.

    Args:
        limit: ``"config"`` — взять ``causal.estimation_sample``; ``None`` —
            не прореживать; число — явный размер (refutation-слой берёт меньше,
            поскольку каждый тест прогоняет оценку десятки раз).
    """
    outcome = cfg.data.outcome
    needed = ["treatment", outcome, *covariates]
    frame = df[needed].copy()
    size = cfg.causal.estimation_sample if limit == "config" else limit
    if size is None or len(frame) <= size:
        return frame, len(frame)
    assert isinstance(size, int)  # noqa: S101 — отсечено проверкой выше

    strata = frame["treatment"].astype(int).astype(str) + "_" + (
        frame[outcome] > frame[outcome].median()
    ).astype(int).astype(str)
    rng = np.random.default_rng(cfg.seed)
    keep: list[np.ndarray] = []
    counts = strata.value_counts()
    for name, count in counts.items():
        idx = np.asarray(frame.index[strata == name])
        quota = max(1, int(round(size * count / len(frame))))
        quota = min(quota, idx.size)
        keep.append(rng.choice(idx, size=quota, replace=False))
    sample = frame.loc[np.concatenate(keep)].sort_index()
    logger.info(
        "подвыборка для DoWhy: %d из %d строк (доля T=1 %.4f → %.4f)",
        len(sample),
        len(frame),
        float(frame["treatment"].mean()),
        float(sample["treatment"].mean()),
    )
    return sample, len(frame)


def observed_delta(df: pd.DataFrame, *, outcome: str) -> tuple[float, float, float]:
    """Наблюдаемая разность отклика ``Δ_corr`` и отклики по группам."""
    treated = df.loc[df["treatment"] == 1, outcome]
    control = df.loc[df["treatment"] == 0, outcome]
    if treated.empty or control.empty:
        msg = "одна из групп пуста: Δ_corr не определена"
        raise ValueError(msg)
    resp_t = float(treated.mean())
    resp_c = float(control.mean())
    return resp_t - resp_c, resp_t, resp_c


# ---------------------------------------------------------------------------
# DoWhy
# ---------------------------------------------------------------------------


def build_model(frame: pd.DataFrame, graph: ExpandedGraph) -> Any:
    """Создать ``dowhy.CausalModel`` по развёрнутому DAG.

    Сначала граф передаётся объектом ``networkx`` — это точный путь без
    сериализации. Старые сборки DoWhy принимают только строку, поэтому
    предусмотрен откат на однострочный GML: падать из-за формата графа, когда
    сам граф корректен, незачем.
    """
    from dowhy import CausalModel

    try:
        return CausalModel(
            data=frame,
            treatment=graph.treatment,
            outcome=graph.outcome,
            graph=graph.graph,
        )
    except Exception as exc:  # noqa: BLE001 — различаем только «формат не принят»
        logger.debug("CausalModel не принял networkx-граф (%s), пробуем GML-строку", exc)
        return CausalModel(
            data=frame,
            treatment=graph.treatment,
            outcome=graph.outcome,
            graph=graph.to_gml(),
        )


def identify(model: Any, *, proceed_when_unidentifiable: bool = True) -> Any:
    """Идентифицировать эффект; отсутствие backdoor-оценки — остановка (CAU-1).

    ``proceed_when_unidentifiable=True`` относится к ненаблюдаемым
    конфаундерам: DoWhy иначе задаёт интерактивный вопрос, что несовместимо с
    ``make all``. Предположение об отсутствии ненаблюдаемых конфаундеров
    зафиксировано как допущение и проверяется refutation-тестами (§6.4).
    """
    estimand = model.identify_effect(proceed_when_unidentifiable=proceed_when_unidentifiable)
    estimands = getattr(estimand, "estimands", None) or {}
    if not estimands.get("backdoor"):
        msg = (
            "эффект не идентифицируется по критерию заднего входа: DoWhy не вернул "
            "backdoor-estimand. Конвейер остановлен — дальнейшая оценка ATE не имела бы "
            "причинной интерпретации. Проверьте configs/dag.gml: есть ли рёбра X → T и X → Y."
        )
        raise IdentificationError(msg)

    variables = backdoor_variables(estimand)
    logger.info("идентификация выполнена: backdoor-множество из %d переменных", len(variables))
    return estimand


def backdoor_variables(estimand: Any) -> list[str]:
    """Переменные backdoor-множества найденного estimand.

    У DoWhy это поле меняло форму между версиями (список → словарь
    «идентификатор набора → переменные»), поэтому читаем через публичный
    геттер, а словарь разбираем только как откат.
    """
    getter = getattr(estimand, "get_backdoor_variables", None)
    if callable(getter):
        try:
            return sorted(getter() or [])
        except Exception:  # noqa: BLE001 — переходим к разбору словаря
            logger.debug("get_backdoor_variables() недоступен, читаем backdoor_variables")

    raw = getattr(estimand, "backdoor_variables", None)
    if isinstance(raw, dict):
        key = getattr(estimand, "default_backdoor_id", None)
        values = raw.get(key) if key in raw else next(iter(raw.values()), [])
        return sorted(values or [])
    return sorted(raw or [])


def _method_params(method: str, cfg_like: dict[str, Any]) -> dict[str, Any]:
    """Параметры оценщика: число бутстрэп-итераций и клиппинг propensity."""
    params: dict[str, Any] = {
        "num_simulations": cfg_like["n_simulations"],
        "num_null_simulations": cfg_like["n_simulations"],
        "confidence_level": cfg_like["confidence"],
    }
    if "propensity" in method:
        # Риск §14: при дефицитной treatment-группе крайние propensity дают
        # веса 1/p порядка сотен, и одна строка начинает определять ATE.
        lo, hi = cfg_like["propensity_clip"]
        params["min_ps_score"] = lo
        params["max_ps_score"] = hi
    return params


def estimate_ate(
    model: Any,
    estimand: Any,
    *,
    methods: list[str],
    confidence: float = 0.95,
    n_simulations: int = 50,
    propensity_clip: tuple[float, float] = (0.01, 0.99),
) -> list[EstimateResult]:
    """Оценить ATE каждым из методов (CAU-3).

    Падение одного оценщика не прерывает слой: причина фиксируется в
    результате, и отчёт честно показывает, какой метод не сошёлся. Полная
    остановка оправдана только если не сработал ни один.
    """
    knobs = {
        "n_simulations": n_simulations,
        "confidence": confidence,
        "propensity_clip": propensity_clip,
    }
    results: list[EstimateResult] = []
    for short in methods:
        full = ESTIMATOR_METHODS.get(short, short if "." in short else f"backdoor.{short}")
        params = _method_params(full, knobs)
        with log_stage(logger, f"оценка ATE: {full}"):
            estimate = _run_estimator(model, estimand, full, params)
            if isinstance(estimate, str):
                logger.warning("оценщик %s не сошёлся: %s", full, estimate)
                results.append(EstimateResult(method=short, ate=float("nan"), failed=estimate))
                continue

            results.append(
                EstimateResult(
                    method=short,
                    ate=float(estimate.value),
                    stderr=_safe_stderr(estimate),
                    **_safe_interval(estimate, confidence),
                    p_value=_safe_pvalue(estimate),
                    params=params,
                )
            )
            logger.info("%s: ATE = %+.6f", full, float(estimate.value))

    if all(r.failed is not None for r in results):
        reasons = "; ".join(f"{r.method}: {r.failed}" for r in results)
        msg = f"ни один оценщик ATE не завершился успешно ({reasons})"
        raise RuntimeError(msg)
    return results


def estimate_single(
    model: Any,
    estimand: Any,
    method: str,
    *,
    confidence: float = 0.95,
    n_simulations: int = 50,
    propensity_clip: tuple[float, float] = (0.01, 0.99),
) -> Any:
    """Вернуть «сырой» объект оценки DoWhy одним методом.

    Нужен refutation-слою: ``refute_estimate`` принимает именно объект
    ``CausalEstimate``, а не наш ``EstimateResult``.
    """
    full = ESTIMATOR_METHODS.get(method, method if "." in method else f"backdoor.{method}")
    params = _method_params(
        full,
        {
            "n_simulations": n_simulations,
            "confidence": confidence,
            "propensity_clip": propensity_clip,
        },
    )
    result = _run_estimator(model, estimand, full, params)
    if isinstance(result, str):
        msg = f"оценщик {full} не сошёлся: {result}"
        raise RuntimeError(msg)
    return result


def _run_estimator(model: Any, estimand: Any, method: str, params: dict[str, Any]) -> Any:
    """Вызвать ``estimate_effect``; при ошибке вернуть её текст строкой.

    Второй проход без ``method_params`` — страховка от того, что конкретная
    сборка DoWhy не знает одного из переданных ключей: лучше получить оценку с
    параметрами по умолчанию, чем пустую строку в отчёте.
    """
    for attempt_params in (params, None):
        try:
            return model.estimate_effect(
                estimand,
                method_name=method,
                test_significance=True,
                confidence_intervals=True,
                method_params=attempt_params,
            )
        except TypeError as exc:
            if attempt_params is None:
                return str(exc)
            logger.warning(
                "%s не принял параметры %s (%s) — повтор с настройками по умолчанию",
                method,
                sorted(params),
                exc,
            )
        except Exception as exc:  # noqa: BLE001 — причина уходит в отчёт
            return str(exc)
    return "оценщик не вернул результат"


def _safe_stderr(estimate: Any) -> float | None:
    try:
        value = estimate.get_standard_error()
    except Exception:  # noqa: BLE001 — не все оценщики DoWhy это поддерживают
        return None
    return _scalar(value)


def _safe_interval(estimate: Any, confidence: float) -> dict[str, float | None]:
    try:
        interval = estimate.get_confidence_intervals(confidence_level=confidence)
    except Exception:  # noqa: BLE001
        return {"ci_low": None, "ci_high": None}
    flat = np.ravel(np.asarray(interval, dtype="float64"))
    if flat.size < 2:
        return {"ci_low": None, "ci_high": None}
    return {"ci_low": float(flat[0]), "ci_high": float(flat[-1])}


def _safe_pvalue(estimate: Any) -> float | None:
    significance = getattr(estimate, "test_stat_significance", None)
    if callable(significance):
        try:
            payload = significance()
        except Exception:  # noqa: BLE001
            return None
        return _scalar(payload.get("p_value") if isinstance(payload, dict) else payload)
    payload = getattr(estimate, "significance_test", None)
    if isinstance(payload, dict):
        return _scalar(payload.get("p_value"))
    return None


def _scalar(value: Any) -> float | None:
    """Привести результат DoWhy (скаляр, список или массив) к одному числу."""
    if value is None:
        return None
    flat = np.ravel(np.asarray(value, dtype="float64"))
    return float(flat[0]) if flat.size else None


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


def estimates_table(analysis: CausalAnalysis) -> pd.DataFrame:
    rows = []
    for est in analysis.estimates:
        rows.append(
            {
                "метод": ESTIMATOR_METHODS.get(est.method, est.method),
                "ATE": float("nan") if est.failed else est.ate,
                "95% ДИ": "—"
                if est.ci_low is None
                else f"[{est.ci_low:+.4f}; {est.ci_high:+.4f}]",
                "p-value": float("nan") if est.p_value is None else est.p_value,
                "статус": "не сошёлся" if est.failed else "ок",
            }
        )
    return pd.DataFrame(rows)


def render_report(analysis: CausalAnalysis, cfg: Config) -> list[str]:
    """Markdown ``reports/ate_summary.md`` (CAU-4)."""
    bias = analysis.confounding_bias
    rel = abs(bias) / abs(analysis.ate) if analysis.ate else float("nan")

    lines = [
        "# Причинный слой: идентификация и оценка ATE",
        "",
        f"Исход: `{cfg.data.outcome}`; идентификация: критерий заднего входа "
        f"(Pearl backdoor); основной оценщик: `{analysis.primary_method}`.",
        "",
        "## 1. Наблюдаемая разность против скорректированного эффекта",
        "",
        "| Величина | Значение | Смысл |",
        "|---|---|---|",
        f"| `Δ_corr` | {fmt_float(analysis.delta_corr, 6)} | наблюдаемая разность отклика "
        "между группами — **смещена** отбором |",
        f"| `ATE` | {fmt_float(analysis.ate, 6)} | эффект после корректировки на "
        "backdoor-множество |",
        f"| `Δ_corr − ATE` | {fmt_float(bias, 6)} | **измеренное смещение конфаундинга** |",
        f"| относительное смещение | {fmt_float(rel * 100, 2)} % | доля `Δ_corr`, "
        "объясняемая отбором, а не воздействием |",
        "",
        f"Отклик при `T=1`: {fmt_float(analysis.response_treated, 6)}; "
        f"при `T=0`: {fmt_float(analysis.response_control, 6)} "
        f"({analysis.n_treated} против {analysis.n_control} клиентов).",
        "",
    ]

    if analysis.true_ate is not None:
        err = analysis.ate - analysis.true_ate
        lines += [
            "### Сверка с ground truth (синтетические данные)",
            "",
            f"* истинный `ATE` генератора: {fmt_float(analysis.true_ate, 6)}",
            f"* оценка причинного слоя: {fmt_float(analysis.ate, 6)} "
            f"(ошибка {fmt_float(err, 6)})",
            f"* наблюдаемая `Δ_corr`: {fmt_float(analysis.delta_corr, 6)} "
            f"(ошибка {fmt_float(analysis.delta_corr - analysis.true_ate, 6)})",
            "",
            "Сверка возможна только на синтетике: на реальных данных истинный эффект",
            "ненаблюдаем, и единственная доступная проверка — refutation-тесты §6.4.",
            "",
        ]

    lines += [
        "## 2. Оценки ATE разными методами (CAU-3)",
        "",
        df_to_markdown(estimates_table(analysis), floatfmt="{:+.6f}"),
        "",
        f"Разброс между сошедшимися оценщиками: {fmt_float(analysis.spread, 6)}.",
        "",
        "Согласие двух независимых подходов (взвешивание по propensity и регрессионная",
        "корректировка) — слабое, но необходимое свидетельство устойчивости: расхождение",
        "означало бы, что результат держится на конкретной функциональной форме, а не",
        "на структуре графа.",
        "",
        "## 3. Backdoor-множество",
        "",
        f"Переменных в множестве корректировки: **{len(analysis.backdoor_variables)}**.",
        "",
    ]
    if analysis.graph.outcome_only:
        lines += [
            "Признаки, влияющие только на исход и потому **не** входящие в множество "
            "корректировки: "
            + ", ".join(f"`{f}`" for f in analysis.graph.outcome_only)
            + ".",
            "",
            "Это проверка того, что граф действительно используется: включение таких",
            "переменных в корректировку не сделало бы оценку смещённой, но и не требуется",
            "критерием заднего входа — DoWhy определяет множество по структуре DAG, а не",
            "по списку «всех доступных колонок».",
            "",
        ]

    lines += [
        "## 4. Структура DAG",
        "",
        *group_table(analysis.graph),
        "",
        f"Развёрнутый граф: `{cfg.causal.expanded_graph_path}` "
        f"({analysis.graph.summary()['n_nodes']} узлов, "
        f"{analysis.graph.summary()['n_edges']} рёбер).",
        "",
        "## 5. Estimand",
        "",
        "```",
        analysis.estimand_text.strip(),
        "```",
        "",
        f"Полный текст и backdoor-множество: `{cfg.artifacts_dir.name}/{ESTIMAND_ARTIFACT}`.",
        "",
        "## 6. Оговорки",
        "",
        f"* оценка считалась на подвыборке {analysis.n_rows} строк "
        f"(`causal.estimation_sample`), стратифицированной по `(T, Y)`;",
        "* выполнено допущение об отсутствии **ненаблюдаемых** конфаундеров: оно не",
        "  проверяемо данными и тестируется косвенно в `reports/refutation.md`;",
        "* `Δ_corr` и `ATE` сопоставимы только при одном и том же определении исхода",
        f"  (здесь — `{cfg.data.outcome}` в окне измерения исхода).",
    ]
    return lines


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_causal_analysis(cfg: Config, *, write: bool = True) -> CausalAnalysis:
    """Полный причинный слой: DAG → идентификация → ATE → отчёт (Э3)."""
    df = load_feature_store(cfg)
    covariates = feature_store_covariates(df)
    outcome = cfg.data.outcome

    with log_stage(logger, "разворот DAG"):
        graph = build_expanded_graph(cfg, covariates, save=write)

    frame, n_full = estimation_frame(cfg, df, covariates)
    delta, resp_t, resp_c = observed_delta(frame, outcome=outcome)

    with log_stage(logger, "идентификация эффекта (CAU-1)"):
        # Проверка до DoWhy: его сообщение о пропавшем узле графа куда менее
        # внятное, чем наше, а причина всегда одна — рассинхрон витрины и DAG.
        validate_against_frame(graph, frame.columns)
        model = build_model(frame, graph)
        estimand = identify(model)
        variables = backdoor_variables(estimand)

    estimates = estimate_ate(
        model,
        estimand,
        methods=cfg.causal.estimators,
        confidence=cfg.evaluation.bootstrap.confidence,
        n_simulations=cfg.causal.n_simulations,
        propensity_clip=cfg.model.xlearner.propensity_clip,
    )

    true_ate = None
    if "true_tau" in df.columns:
        true_ate = float(df["true_tau"].mean())

    analysis = CausalAnalysis(
        delta_corr=delta,
        estimates=estimates,
        primary_method=cfg.causal.estimators[0],
        estimand_text=str(estimand),
        backdoor_variables=variables,
        graph=graph,
        n_rows=len(frame),
        n_treated=int((frame["treatment"] == 1).sum()),
        n_control=int((frame["treatment"] == 0).sum()),
        response_treated=resp_t,
        response_control=resp_c,
        true_ate=true_ate,
    )

    logger.info(
        "Δ_corr = %+.6f, ATE = %+.6f, смещение конфаундинга = %+.6f "
        "(подвыборка %d из %d строк)",
        analysis.delta_corr,
        analysis.ate,
        analysis.confounding_bias,
        len(frame),
        n_full,
    )

    if write:
        analysis.artifacts["estimand"] = save_json(
            {
                "estimand_text": analysis.estimand_text,
                "identification": cfg.causal.identification,
                "backdoor_variables": variables,
                "n_backdoor_variables": len(variables),
                "treatment": graph.treatment,
                "outcome": graph.outcome,
                "graph": graph.summary(),
            },
            cfg.artifact(ESTIMAND_ARTIFACT),
        )
        analysis.artifacts["ate"] = save_json(analysis.as_dict(), cfg.artifact(ATE_ARTIFACT))
        analysis.report_path = write_markdown(
            render_report(analysis, cfg), cfg.report(ATE_REPORT)
        )
        analysis.artifacts["report"] = analysis.report_path
        analysis.artifacts["graph"] = cfg.resolve(cfg.causal.expanded_graph_path)

    return analysis


__all__ = [
    "ATE_ARTIFACT",
    "ATE_REPORT",
    "ESTIMAND_ARTIFACT",
    "ESTIMATOR_METHODS",
    "CausalAnalysis",
    "EstimateResult",
    "IdentificationError",
    "backdoor_variables",
    "build_model",
    "estimate_ate",
    "estimate_single",
    "estimation_frame",
    "identify",
    "observed_delta",
    "render_report",
    "run_causal_analysis",
]
