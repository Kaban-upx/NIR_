# ИТ-конвейер причинно-следственного анализа и Uplift-моделирования для e-commerce

НИР, МАИ, Институт №8 (2026). Конвейер отвечает на вопрос, который обычная
модель отклика решить не может: **кого из клиентов рассылка действительно
изменит**, а не кто купит и так.

Техническое задание — [`../SPEC.md`](../SPEC.md); исходная НИР —
[`../NIR_Lubyagin.md`](../NIR_Lubyagin.md).

## Состояние реализации

| Слой | Содержание | Статус |
|---|---|---|
| 1 — данные и признаки | ETL, витрина ≥30 ковариат, EDA | **готов** |
| 2 — причинный | DAG, backdoor-идентификация, ATE, refutation-аудит | **готов** |
| 3 — uplift-ядро | X-Learner + LightGBM, cross-fitting | отложен |
| 4 — выводы | AUUC/Qini, baselines, сегментация, экономика, Streamlit | отложен |

Каталоги `models/`, `evaluation/`, `segmentation/`, `economics/`, `app/`
созданы, но пока пусты — это точки расширения под слои 3–4, а не забытый код.

## Быстрый старт

```bash
make env        # .venv на Python 3.13 + зависимости (нужен uv) и gate совместимости
make data       # синтетические таблицы RetailHero-схемы в data/raw
make etl        # товарные строки → транзакции, data/interim
make features   # витрина признаков → data/processed/feature_store.parquet
make eda        # reports/eda_summary.md
make causal     # DAG, backdoor-estimand, ATE → reports/ate_summary.md
make refute     # refutation-тесты → reports/refutation.md
```

Или всё сразу: `make all` (этапы `data → etl → features → eda → causal → refute`).
`make help` печатает список целей.

Любой параметр переопределяется без правки YAML:

```bash
make causal OVERRIDES="--set causal.estimation_sample=20000"
make all     OVERRIDES="--set data.treatment_share=0.15"
```

## Данные

`data.source` в `configs/config.yaml` выбирает источник:

* `synthetic` (по умолчанию) — детерминированный генератор таблиц
  RetailHero-схемы с **известным** истинным `τ(x)`. Нужен для sanity-проверок:
  оценку ATE можно сравнить с истинным значением, чего на реальных данных не
  бывает никогда.
* `retailhero` — реальные CSV (`clients`, `purchases`, `products`,
  `uplift_train`) из `data/raw`. Схема и имена колонок совпадают с
  синтетическими, поэтому переключение источника не требует правок кода.

## Что получается на выходе

| Артефакт | Смысл |
|---|---|
| `data/processed/feature_store.parquet` | витрина: `client_id`, `treatment`, `target`, `split`, ≥30 Z-нормированных ковариат |
| `artifacts/window.json` | границы окна признаков и окна исхода — одни и те же для всех слоёв |
| `artifacts/scaler.json`, `encoders.json`, `clip_bounds.json` | параметры препроцессинга, обученные **только на train** |
| `artifacts/dag_expanded.gml` | DAG на уровне фактических колонок |
| `artifacts/estimand.json` | backdoor-estimand и множество корректировки |
| `reports/eda_summary.md` | фактическое соотношение `T=1 : T=0`, пропуски, выбросы, карантин |
| `reports/ate_summary.md` | `Δ_corr`, скорректированный ATE и **измеренное смещение конфаундинга** |
| `reports/refutation.md` | вердикты PASS/FAIL по трём refutation-тестам |
| `artifacts/runs/` | журнал прогонов: параметры, метрики, длительность (ML-5) |

## Почему устроено именно так

* **Окна жёстко разделены.** Признаки считаются на `[feature_start, feature_end)`,
  исход — на `[feature_end, outcome_end)`. Границы фиксируются один раз в
  `artifacts/window.json`: пересчёт на каждом этапе означал бы, что причинный
  слой видит другое окно, чем слой признаков.
* **Препроцессинг обучается только на train.** Z-score, категориальные
  кодировки, границы зажатия выбросов и фильтр мультиколлинеарности — всё
  фитится на train-фолде, колонка `split` пишется в витрину. Иначе тестовая
  часть протекает в обучение, а метрики становятся недостижимо хорошими.
* **Идентификация — жёсткий gate.** Если DoWhy не вернул backdoor-estimand,
  конвейер останавливается с ошибкой. Продолжать расчёт нельзя: полученное
  число не имело бы причинной интерпретации.
* **`Δ_corr` и ATE выводятся вместе.** Их разность — не диагностика, а
  результат: она численно показывает, какая часть «эффекта рассылки» на самом
  деле объясняется тем, кому её отправляли.
* **Структура DAG — в конфигурации, не в коде.** Шаблон на уровне групп живёт
  в `configs/dag.gml`, карта «колонка → группа» — в
  `configs/feature_groups.yaml`. Добавление признака не требует правок
  причинного слоя. `seasonality` намеренно влияет только на `Y`, поэтому DoWhy
  сам исключает её из множества корректировки — это проверка, что граф
  действительно используется.

## Воспроизводимость

Один `seed` (`configs/config.yaml`) управляет генерацией данных, сплитами,
подвыборками для DoWhy и refuters. Два независимых прогона с одинаковой
конфигурацией дают совпадающие артефакты; это проверяется тестом
`tests/test_reproducibility.py`.

## Разработка

```bash
make test        # pytest
make test-fast   # без медленных (DoWhy/refuters) тестов
make lint        # ruff check + format --check
make typecheck   # mypy
```

## Структура

```
configs/      config.yaml (единственный источник параметров), dag.gml, feature_groups.yaml
src/causal_uplift/
  etl/        загрузка, валидация схем, карантин, агрегация чеков
  features/   RFM, поведенческие блоки, кодировки, Z-score, фильтр корреляций, сборка витрины
  causal/     dag.py (шаблон → развёрнутый DAG), identify.py (estimand + ATE), refute.py (аудит)
  pipeline/   run.py (CLI-оркестратор), check_env.py (gate совместимости версий)
  eda.py      разведочный анализ и отчёт
docs/         architecture.md
tests/        unit, property, интеграционные, тесты на утечки
```
