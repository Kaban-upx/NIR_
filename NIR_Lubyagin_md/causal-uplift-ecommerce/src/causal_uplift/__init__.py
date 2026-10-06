"""Пакет конвейера причинно-следственного анализа и Uplift-моделирования.

Слои (SPEC §4) связаны строго через типизированные артефакты на диске:

* ``etl``          — слой 1: сырые логи → ``data/interim``;
* ``features``     — слой 1: витрина признаков → ``feature_store.parquet``;
* ``causal``       — слой 2: DAG, backdoor-идентификация, ATE, refuters;
* ``models``       — слой 3: X-Learner + LightGBM и baseline-модели;
* ``evaluation``   — слой 4: AUUC/Qini/uplift@k, кривые, отчёты;
* ``segmentation`` — слой 4: четыре квадранта uplift-сегментации;
* ``economics``    — слой 4: экономия бюджета, инкрементальная выручка, ROI;
* ``pipeline``     — оркестрация этапов и эксперименты;
* ``app``          — Streamlit-дашборд на готовых артефактах.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
