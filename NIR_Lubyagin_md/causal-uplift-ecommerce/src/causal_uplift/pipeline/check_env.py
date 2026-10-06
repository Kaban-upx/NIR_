"""Gate этапа Э0: проверка совместимости окружения (W1, риск §14).

Запускается первым (`make env` → `make check-env`) и до прикладного кода:
связка ``dowhy`` + ``econml`` + ``numpy`` под Python 3.13 — главный риск
проекта, поэтому проверка не декларативная, а выполняет короткий сквозной
сценарий (обучение LightGBM, построение CausalModel, вызов X-Learner).

Отчёт пишется в ``reports/env_check.md``; код возврата ≠ 0 означает, что
окружение собрано неработоспособно.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Минимально требуемые версии (нижние границы из pyproject)
REQUIRED: dict[str, str] = {
    "numpy": "2.0",
    "pandas": "2.2",
    "pyarrow": "16.0",
    "scipy": "1.13",
    "sklearn": "1.5",
    "statsmodels": "0.14",
    "lightgbm": "4.5",
    "dowhy": "0.12",
    "econml": "0.15",
    "networkx": "3.3",
    "pydantic": "2.8",
    "yaml": "6.0",
    "matplotlib": "3.8",
    "streamlit": "1.37",
}

# Соответствие «имя модуля → имя дистрибутива» там, где они расходятся
DIST_NAMES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    version: str = ""


@dataclass
class EnvReport:
    python_version: str
    platform: str
    imports: list[CheckResult] = field(default_factory=list)
    smoke: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in [*self.imports, *self.smoke])


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_imports() -> list[CheckResult]:
    """Импортировать каждую зависимость и сверить версию с нижней границей."""
    results: list[CheckResult] = []
    for module_name, min_version in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — нужен диагноз, а не падение
            results.append(
                CheckResult(
                    name=DIST_NAMES.get(module_name, module_name),
                    ok=False,
                    detail=f"импорт провалился: {type(exc).__name__}: {exc}",
                )
            )
            continue
        version = str(getattr(module, "__version__", "") or "")
        if not version:
            try:
                from importlib.metadata import version as dist_version

                version = dist_version(DIST_NAMES.get(module_name, module_name))
            except Exception:  # noqa: BLE001
                version = "unknown"
        ok = version == "unknown" or _version_tuple(version) >= _version_tuple(min_version)
        results.append(
            CheckResult(
                name=DIST_NAMES.get(module_name, module_name),
                ok=ok,
                version=version,
                detail="" if ok else f"требуется ≥ {min_version}",
            )
        )
    return results


def _smoke_lightgbm() -> str:
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] + rng.normal(scale=0.3, size=400) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1, random_state=0)
    model.fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    return f"LGBMClassifier обучен, mean(p)={proba.mean():.3f}"


def _smoke_dowhy() -> str:
    import numpy as np
    import pandas as pd
    from dowhy import CausalModel

    rng = np.random.default_rng(1)
    n = 800
    x0 = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-x0))).astype(int)
    y = (0.3 * x0 + 0.5 * t + rng.normal(scale=0.2, size=n) > 0).astype(int)
    df = pd.DataFrame({"x0": x0, "treatment": t, "outcome": y})
    graph = (
        'graph [directed 1 node [id "x0" label "x0"] node [id "treatment" label "treatment"] '
        'node [id "outcome" label "outcome"] edge [source "x0" target "treatment"] '
        'edge [source "x0" target "outcome"] edge [source "treatment" target "outcome"]]'
    )
    model = CausalModel(data=df, treatment="treatment", outcome="outcome", graph=graph)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    if not getattr(estimand, "estimands", {}).get("backdoor"):
        msg = "identify_effect не вернул backdoor-estimand"
        raise RuntimeError(msg)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    return f"backdoor-estimand получен, ATE(linear_regression)={float(estimate.value):+.4f}"


def _smoke_econml() -> str:
    import numpy as np
    from econml.metalearners import XLearner
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(size=(n, 4))
    t = (rng.uniform(size=n) < 0.5).astype(int)
    y = 0.4 * x[:, 0] + t * (0.3 + 0.2 * x[:, 1]) + rng.normal(scale=0.1, size=n)
    learner = XLearner(models=LGBMRegressor(n_estimators=20, verbose=-1, random_state=0))
    learner.fit(y, t, X=x)
    tau = learner.effect(x)
    return f"econml.XLearner обучен, mean(tau)={float(np.mean(tau)):+.4f}"


def _smoke_pipeline_imports() -> str:
    """Собственный пакет должен импортироваться (установлен в editable-режиме)."""
    from causal_uplift import __version__
    from causal_uplift.config import Config

    _ = Config()  # дефолты валидны
    return f"causal_uplift {__version__} импортируется, дефолтная конфигурация валидна"


SMOKE_CHECKS: dict[str, tuple[str, object]] = {
    "lightgbm": ("обучение LGBMClassifier", _smoke_lightgbm),
    "dowhy": ("CausalModel + identify_effect(backdoor) + estimate_effect", _smoke_dowhy),
    "econml": ("econml.metalearners.XLearner на LightGBM", _smoke_econml),
    "causal_uplift": ("импорт собственного пакета и валидация конфига", _smoke_pipeline_imports),
}


def run_smoke_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, (title, fn) in SMOKE_CHECKS.items():
        try:
            detail = fn()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=f"{name}: {title}",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}\n```\n{traceback.format_exc()}```",
                )
            )
        else:
            results.append(CheckResult(name=f"{name}: {title}", ok=True, detail=str(detail)))
    return results


def build_report() -> EnvReport:
    return EnvReport(
        python_version=platform.python_version(),
        platform=platform.platform(),
        imports=check_imports(),
        smoke=run_smoke_checks(),
    )


def render_markdown(report: EnvReport) -> list[str]:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        "# Проверка окружения (gate этапа Э0)",
        "",
        f"**Вердикт: {status}**",
        "",
        f"- Python: `{report.python_version}`",
        f"- Платформа: `{report.platform}`",
        "",
        "## Версии зависимостей",
        "",
        "| Пакет | Версия | Статус | Комментарий |",
        "|---|---|---|---|",
    ]
    for r in report.imports:
        mark = "OK" if r.ok else "FAIL"
        lines.append(f"| {r.name} | `{r.version or '—'}` | {mark} | {r.detail or '—'} |")

    lines += [
        "",
        "## Функциональные проверки связки",
        "",
    ]
    for r in report.smoke:
        mark = "OK" if r.ok else "FAIL"
        lines.append(f"- **{mark}** — {r.name}")
        if r.detail:
            lines.append(f"  - {r.detail}")

    if not report.ok:
        lines += [
            "",
            "> Окружение неработоспособно. Предусмотренное спецификацией смягчение "
            "(SPEC §14): откат на Python 3.12 с фиксацией отступления в "
            "`docs/architecture.md`.",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения конвейера (Э0)")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/env_check.md"),
        help="путь к markdown-отчёту",
    )
    args = parser.parse_args(argv)

    report = build_report()
    lines = render_markdown(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nотчёт: {args.report}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
