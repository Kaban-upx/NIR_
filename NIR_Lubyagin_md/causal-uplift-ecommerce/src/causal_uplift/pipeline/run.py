"""CLI-оркестратор конвейера (SPEC §8, §12).

Единственная точка входа для всех этапов: ``make`` вызывает именно её, так что
последовательность шагов и их параметры всегда совпадают между ручным
запуском и ``make all``.

Каждый этап — отдельная функция ``stage_*``, зарегистрированная в ``STAGES``.
Этап обязан быть идемпотентным: повторный запуск с тем же сидом и той же
конфигурацией должен давать тот же результат (критерий приёмки 10).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from causal_uplift.config import Config
from causal_uplift.io_utils import set_global_seed
from causal_uplift.logging_setup import get_logger, log_stage, setup_logging
from causal_uplift.tracking import RunLogger

logger = get_logger(__name__)

StageFn = Callable[[Config], dict[str, Any]]


# ---------------------------------------------------------------------------
# Этапы
# ---------------------------------------------------------------------------


def stage_data(cfg: Config) -> dict[str, Any]:
    """Э1: подготовить сырые данные в ``data/raw``.

    Для ``data.source: synthetic`` генерирует детерминированный датасет
    RetailHero-совместимой схемы; для ``retailhero`` только проверяет наличие
    файлов — реальные архивы распаковывает пользователь.
    """
    from causal_uplift.etl.loaders import available_tables, check_retailhero_presence
    from causal_uplift.etl.synthetic import generate_raw_dataset

    cfg.ensure_dirs()
    if cfg.data.source == "synthetic":
        dataset = generate_raw_dataset(cfg)
        return {
            "paths": {k: str(v) for k, v in dataset.paths.items()},
            **dataset.summary,
        }

    missing = check_retailhero_presence(cfg)
    if missing:
        msg = (
            f"в {cfg.raw_dir} не найдены таблицы: {', '.join(missing)}. "
            "Положите CSV RetailHero (clients, purchases, products, uplift_train) "
            "в этот каталог либо переключите data.source на synthetic."
        )
        raise FileNotFoundError(msg)
    tables = available_tables(cfg)
    logger.info("найдены таблицы RetailHero: %s", ", ".join(sorted(tables)))
    return {"paths": {k: str(v) for k, v in tables.items()}}


def stage_etl(cfg: Config) -> dict[str, Any]:
    """Э1: фаза A — товарные строки → транзакционный уровень (ETL-7)."""
    from causal_uplift.etl.aggregate import aggregate_purchases, resolve_window

    window = resolve_window(cfg, refresh=True)
    result = aggregate_purchases(cfg, window)
    return {"window": window.as_dict(), **result.as_dict()}


def stage_features(cfg: Config) -> dict[str, Any]:
    """Э2: фазы B и C — витрина признаков (W4)."""
    from causal_uplift.features.build import build_feature_store

    result = build_feature_store(cfg)
    return result.as_dict()


def stage_eda(cfg: Config) -> dict[str, Any]:
    """Э1/W3: разведочный анализ и ``reports/eda_summary.md``."""
    from causal_uplift.eda import run_eda

    result = run_eda(cfg)
    return result.as_dict()


def stage_causal(cfg: Config) -> dict[str, Any]:
    """Э3: причинный слой — DAG, идентификация, ATE (W5)."""
    from causal_uplift.causal.identify import run_causal_analysis

    result = run_causal_analysis(cfg)
    return result.as_dict()


def stage_refute(cfg: Config) -> dict[str, Any]:
    """Э6: аудит причинных допущений — refutation-тесты (W9)."""
    from causal_uplift.causal.refute import run_refutation

    audit = run_refutation(cfg)
    return audit.as_dict()


STAGES: dict[str, StageFn] = {
    "data": stage_data,
    "etl": stage_etl,
    "features": stage_features,
    "eda": stage_eda,
    "causal": stage_causal,
    "refute": stage_refute,
}

#: Порядок этапов для ``run all``
ALL_ORDER: tuple[str, ...] = ("data", "etl", "features", "eda", "causal", "refute")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-uplift",
        description="ИТ-конвейер причинно-следственного анализа и uplift-моделирования",
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="этап конвейера (all — полный цикл в порядке %s)" % ", ".join(ALL_ORDER),
    )
    parser.add_argument("--config", default="configs/config.yaml", help="путь к конфигурации")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="FILE.yaml",
        help="дополнительный YAML поверх основного (можно повторять)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="key.path=value",
        help="точечное переопределение параметра (можно повторять)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="переопределить seed (по умолчанию из конфигурации)"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="уровень логирования",
    )
    parser.add_argument(
        "--no-journal",
        action="store_true",
        help="не писать журнал прогона в artifacts/runs (для тестов)",
    )
    return parser


def run_stage(cfg: Config, name: str, *, journal: bool = True) -> dict[str, Any]:
    """Выполнить один этап с журналированием (ML-5)."""
    fn = STAGES[name]
    if not journal:
        with log_stage(logger, f"этап {name}"):
            return fn(cfg)

    with RunLogger(cfg, name) as run:
        run.log_params(stage=name, config_source=str(cfg.project_root))
        with log_stage(logger, f"этап {name}"):
            result = fn(cfg)
        run.log_metrics(**_flatten_metrics(result))
        run.note(f"этап {name} завершён")
    return result


def _flatten_metrics(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Разложить вложенный результат этапа в плоские числовые метрики.

    В журнал идут только скаляры: вложенные словари и списки путей раздувают
    ``index.jsonl`` и мешают сравнивать прогоны.
    """
    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.update(_flatten_metrics(value, f"{prefix}{key}."))
    elif isinstance(payload, (int, float, bool)):
        out[prefix.rstrip(".")] = payload
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config.load(args.config, overlays=args.overlay, overrides=args.overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.ensure_dirs()

    setup_logging(
        args.log_level or cfg.logging.level,
        cfg.resolve(cfg.logging.file) if cfg.logging.file else None,
    )
    set_global_seed(cfg.seed)

    stages = ALL_ORDER if args.stage == "all" else (args.stage,)
    logger.info("конвейер: %s (сид %d, источник %s)", " → ".join(stages), cfg.seed, cfg.data.source)

    for name in stages:
        try:
            run_stage(cfg, name, journal=not args.no_journal)
        except Exception:
            logger.exception("этап %s завершился ошибкой", name)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
