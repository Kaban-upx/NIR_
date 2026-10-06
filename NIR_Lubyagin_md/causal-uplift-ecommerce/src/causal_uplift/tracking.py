"""Локальный трекинг прогонов (ML-5).

MLflow в scope не входит как обязательный (SPEC §9 допускает JSON-журнал),
поэтому реализован journal-логгер: один файл на прогон в ``artifacts/runs/``
плюс сводный ``artifacts/runs/index.jsonl`` для быстрых сравнений.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Self

from causal_uplift.config import Config
from causal_uplift.io_utils import save_json, utc_now_iso
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)


class RunLogger:
    """Журнал одного прогона: параметры + метрики + ссылки на артефакты."""

    def __init__(self, cfg: Config, stage: str, *, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.stage = stage
        self.runs_dir = cfg.resolve(cfg.tracking.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self._make_run_id(stage)
        self._started = time.perf_counter()
        self._payload: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": stage,
            "started_at": utc_now_iso(),
            "seed": cfg.seed,
            "treatment_share": cfg.data.treatment_share,
            "data_source": cfg.data.source,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "config": cfg.dump(),
            "params": {},
            "metrics": {},
            "artifacts": [],
            "notes": [],
        }

    # ------------------------------------------------------------------ API
    def log_params(self, **params: Any) -> None:
        self._payload["params"].update(params)

    def log_metrics(self, **metrics: Any) -> None:
        self._payload["metrics"].update(metrics)

    def log_metric_dict(self, prefix: str, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            self._payload["metrics"][f"{prefix}.{key}"] = value

    def log_artifact(self, path: str | Path, description: str = "") -> None:
        try:
            rel = str(Path(path).resolve().relative_to(self.cfg.project_root))
        except ValueError:
            rel = str(path)
        self._payload["artifacts"].append({"path": rel, "description": description})

    def note(self, message: str) -> None:
        self._payload["notes"].append(message)

    # ------------------------------------------------------------------ финализация
    def finish(self, status: str = "success") -> Path:
        self._payload["status"] = status
        self._payload["finished_at"] = utc_now_iso()
        self._payload["duration_sec"] = round(time.perf_counter() - self._started, 3)

        run_path = self.runs_dir / f"{self.run_id}.json"
        save_json(self._payload, run_path)

        index = self.runs_dir / "index.jsonl"
        summary = {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": status,
            "finished_at": self._payload["finished_at"],
            "duration_sec": self._payload["duration_sec"],
            "treatment_share": self._payload["treatment_share"],
            "metrics": self._payload["metrics"],
        }
        with index.open("a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

        logger.info(
            "журнал прогона: %s (%s, %.1f с)",
            run_path,
            status,
            self._payload["duration_sec"],
        )
        return run_path

    # ------------------------------------------------------------------ контекст
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        self.finish("failed" if exc_type is not None else "success")

    # ------------------------------------------------------------------ служебное
    @staticmethod
    def _make_run_id(stage: str) -> str:
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{stage}"


__all__ = ["RunLogger"]
