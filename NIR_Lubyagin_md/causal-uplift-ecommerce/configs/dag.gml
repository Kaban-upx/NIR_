graph [
  directed 1
  comment "Базовый DAG причинных связей (SPEC §6.1, рис. 1 отчёта о НИР).
    Хранится декларативно: код не хардкодит структуру, а читает этот файл.
    Узлы-группы (socdem, behavior, loyalty, assortment, trend, seasonality)
    разворачиваются в фактические колонки витрины модулем causal/dag.py:
    каждый признак наследует все исходящие рёбра своей группы.
    Чтобы добавить/убрать группу ковариат или ввести сезонность как
    отдельный узел — правится только этот файл."
  node [
    id 0
    label "socdem"
    kind "covariate_group"
    description "Соц-дем: age, gender, возрастные бины"
  ]
  node [
    id 1
    label "behavior"
    kind "covariate_group"
    description "Кликстрим/активность: recency, frequency, monetary, число сессий"
  ]
  node [
    id 2
    label "loyalty"
    kind "covariate_group"
    description "Баллы regular/express, доля покупок с оплатой баллами"
  ]
  node [
    id 3
    label "assortment"
    kind "covariate_group"
    description "Структура корзины: категории, СТМ, алкоголь, магазины"
  ]
  node [
    id 4
    label "trend"
    kind "covariate_group"
    description "Тренды активности: 30д/90д, слоуп суммы покупок"
  ]
  node [
    id 5
    label "seasonality"
    kind "covariate_group"
    description "Сезонность: месяц/квартал первой активности. Влияет только на Y."
  ]
  node [
    id 6
    label "treatment"
    kind "treatment"
    description "T: флаг отправки купона"
  ]
  node [
    id 7
    label "outcome"
    kind "outcome"
    description "Y: покупка в окне outcome_window_days (или revenue)"
  ]

  # X -> T : ковариаты влияют на вероятность попадания в воздействие (конфаундинг)
  edge [ source 0 target 6 ]
  edge [ source 1 target 6 ]
  edge [ source 2 target 6 ]
  edge [ source 3 target 6 ]
  edge [ source 4 target 6 ]

  # X -> Y : те же ковариаты влияют на исход
  edge [ source 0 target 7 ]
  edge [ source 1 target 7 ]
  edge [ source 2 target 7 ]
  edge [ source 3 target 7 ]
  edge [ source 4 target 7 ]
  edge [ source 5 target 7 ]

  # T -> Y : изучаемый причинный эффект
  edge [ source 6 target 7 ]
]
