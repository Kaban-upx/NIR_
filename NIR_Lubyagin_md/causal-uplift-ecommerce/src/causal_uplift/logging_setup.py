"""Настройка логирования конвейера.

Один вход: :func:`setup_logging`. Пишет и в stderr, и в файл из конфига —
прогон должен оставлять след на диске (``artifacts/pipeline.log``), иначе
воспроизводимость по §12.10 нечем подтвердить.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-34s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: str = "INFO", log_file: Path | None = None, *, force: bool = False
) -> None:
    """Сконфигурировать корневой логгер (идемпотентно).

    Args:
        level: уровень логирования.
        log_file: путь к файлу журнала; каталог создаётся при необходимости.
        force: пересобрать обработчики, даже если логирование уже настроено.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Шумные сторонние логгеры
    for noisy in ("matplotlib", "PIL", "numexpr", "py4j", "dowhy", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Логгер модуля (без побочной настройки корневого)."""
    return logging.getLogger(name)


@contextmanager
def log_stage(logger: logging.Logger, title: str) -> Iterator[None]:
    """Залогировать начало/конец этапа и его длительность (бюджет ML-6)."""
    logger.info("▶ %s", title)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("✖ %s — прервано с ошибкой", title)
        raise
    else:
        logger.info("✔ %s — %.1f с", title, time.perf_counter() - started)


__all__ = ["get_logger", "log_stage", "setup_logging"]
