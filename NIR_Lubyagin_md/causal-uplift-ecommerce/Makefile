# ИТ-конвейер причинно-следственного анализа и Uplift-моделирования
# Все цели идемпотентны и запускаются независимо: слои связаны артефактами на диске (SPEC §4).
#
#   make env        — создать .venv (Python 3.13) и установить зависимости из lock-файла
#   make check-env  — проверить совместимость dowhy/econml/lightgbm (gate этапа Э0)
#   make data       — подготовить сырые таблицы в data/raw (синтетика или проверка RetailHero)
#   make etl        — сырые логи → data/interim (чанковая обработка)
#   make features   — витрина признаков → data/processed/feature_store.parquet
#   make eda        — reports/eda_summary.md
#   make causal     — DAG, backdoor-идентификация, ATE → artifacts/estimand.json
#   make train      — X-Learner + LightGBM → artifacts/model.pkl, uplift_scores.parquet
#   make eval       — AUUC/Qini/uplift@k + baselines → reports/metrics.md
#   make refute     — refutation-тесты → reports/refutation.md
#   make segment    — 4 квадранта + экономика → artifacts/segments.parquet
#   make experiment — полный эксперимент в двух режимах дисбаланса
#   make report     — свести выгрузки для ВКР в reports/
#   make app        — Streamlit-дашборд
#   make all        — полный цикл (критерий приёмки 1)

SHELL       := /bin/bash
PYTHON_BIN  ?= python3.13
VENV        ?= .venv
PY          := $(VENV)/bin/python
UV          ?= uv
CONFIG      ?= configs/config.yaml
RUN         := $(PY) -m causal_uplift.pipeline.run --config $(CONFIG)
# Переопределение любого параметра конфига из командной строки:
#   make train OVERRIDES="--set model.xlearner.cross_fitting_folds=1"
OVERRIDES   ?=

.DEFAULT_GOAL := help
.PHONY: help env lock check-env data etl features eda causal train eval refute segment \
        experiment report app all test test-fast lint format typecheck clean clean-all dirs

help:
	@grep -E '^#   make ' $(MAKEFILE_LIST) | sed 's/^#   //'

# ------------------------------------------------------------------ окружение (W1)
env: dirs
	@command -v $(UV) >/dev/null 2>&1 || { echo "нужен uv: https://docs.astral.sh/uv/"; exit 1; }
	$(UV) venv --python $(PYTHON_BIN) $(VENV)
	@if [ -f requirements.lock ]; then \
		echo ">>> установка из requirements.lock (воспроизводимое окружение)"; \
		$(UV) pip install --python $(PY) -r requirements.lock; \
	else \
		echo ">>> requirements.lock отсутствует — установка из pyproject (затем выполните make lock)"; \
		$(UV) pip install --python $(PY) -e ".[dev,notebooks]"; \
	fi
	$(UV) pip install --python $(PY) --no-deps -e .
	@$(MAKE) check-env

lock:
	$(UV) pip compile pyproject.toml --extra dev --extra notebooks \
		--python-version 3.13 -o requirements.lock

# Gate этапа Э0 (риск §14: несовместимость версий под Python 3.13)
check-env:
	$(PY) -m causal_uplift.pipeline.check_env --report reports/env_check.md

dirs:
	@mkdir -p data/raw data/interim data/processed artifacts artifacts/runs reports docs
	@touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep \
	       artifacts/.gitkeep reports/.gitkeep

# ------------------------------------------------------------------ слой 1 (W2–W4)
data: dirs
	$(RUN) data $(OVERRIDES)

etl: dirs
	$(RUN) etl $(OVERRIDES)

features: dirs
	$(RUN) features $(OVERRIDES)

eda: dirs
	$(RUN) eda $(OVERRIDES)

# ------------------------------------------------------------------ слой 2 (W5, W9)
causal: dirs
	$(RUN) causal $(OVERRIDES)

refute: dirs
	$(RUN) refute $(OVERRIDES)

# ------------------------------------------------------------------ слой 3 (W6, W7)
train: dirs
	$(RUN) train $(OVERRIDES)

# ------------------------------------------------------------------ слой 4 (W8, W10, W11)
eval: dirs
	$(RUN) eval $(OVERRIDES)

segment: dirs
	$(RUN) segment $(OVERRIDES)

experiment: dirs
	$(PY) -m causal_uplift.pipeline.experiment --config $(CONFIG) $(OVERRIDES)

report: dirs
	$(RUN) report $(OVERRIDES)

app:
	$(VENV)/bin/streamlit run src/causal_uplift/app/dashboard.py -- --config $(CONFIG)

# Полный цикл: окружение → данные → ... → артефакты для дашборда (критерий приёмки 1)
all: dirs
	$(RUN) all $(OVERRIDES)

# ------------------------------------------------------------------ качество (W12)
test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck:
	$(PY) -m mypy

clean:
	rm -rf data/interim/* data/processed/* artifacts/runs/* .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

clean-all: clean
	rm -rf data/raw/* artifacts/* reports/*
	@$(MAKE) dirs
