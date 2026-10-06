"""Граф причинных связей: чтение шаблона и разворот в признаки витрины (W5).

SPEC §6.1 требует, чтобы структура DAG задавалась декларативно. Поэтому
шаблон хранится в ``configs/dag.gml`` на уровне **групп** ковариат
(``socdem``, ``behavior``, ``loyalty``, ``assortment``, ``trend``,
``seasonality``), а DoWhy нужен граф на уровне фактических колонок.

Этот модуль выполняет разворот: каждый признак становится отдельным узлом и
наследует все исходящие рёбра своей группы. Благодаря этому разница между
группами сохраняется и видна в идентификации: ``seasonality`` влияет только
на ``Y``, не входит в backdoor-множество, и DoWhy это обнаруживает сам —
структура графа не «подсказывается» коду списком ковариат.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import yaml

from causal_uplift.config import Config
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)

#: Виды узлов шаблона
KIND_GROUP = "covariate_group"
KIND_TREATMENT = "treatment"
KIND_OUTCOME = "outcome"

#: Имена узлов воздействия и исхода в шаблоне
TEMPLATE_TREATMENT = "treatment"
TEMPLATE_OUTCOME = "outcome"


@dataclass
class FeatureGroups:
    """Карта «колонка витрины → узел-группа DAG» из YAML."""

    patterns: dict[str, list[re.Pattern[str]]] = field(default_factory=dict)
    fallback_group: str = "behavior"

    @classmethod
    def load(cls, path: str | Path) -> FeatureGroups:
        target = Path(path)
        if not target.is_file():
            msg = f"не найдена карта групп признаков: {target}"
            raise FileNotFoundError(msg)
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        groups = payload.get("groups") or {}
        if not groups:
            msg = f"в {target} не задана секция groups"
            raise ValueError(msg)
        compiled = {
            group: [re.compile(p) for p in patterns] for group, patterns in groups.items()
        }
        return cls(patterns=compiled, fallback_group=str(payload.get("fallback_group", "behavior")))

    def group_of(self, feature: str) -> str | None:
        """Первая группа, чей шаблон полностью совпадает с именем признака."""
        for group, patterns in self.patterns.items():
            if any(p.fullmatch(feature) for p in patterns):
                return group
        return None

    def assign(self, features: Sequence[str]) -> tuple[dict[str, str], list[str]]:
        """Разложить признаки по группам.

        Returns:
            Отображение «признак → группа» и список признаков, попавших в
            ``fallback_group`` (они же — повод дописать шаблон).
        """
        assignment: dict[str, str] = {}
        unmatched: list[str] = []
        for feature in features:
            group = self.group_of(feature)
            if group is None:
                group = self.fallback_group
                unmatched.append(feature)
            assignment[feature] = group
        if unmatched:
            logger.warning(
                "%d признаков не распознаны картой групп и отнесены к %r: %s — "
                "проверьте configs/feature_groups.yaml",
                len(unmatched),
                self.fallback_group,
                ", ".join(unmatched[:12]),
            )
        return assignment, unmatched


# ---------------------------------------------------------------------------
# Шаблон графа
# ---------------------------------------------------------------------------


def load_template(path: str | Path) -> nx.DiGraph:
    """Прочитать шаблонный DAG из GML."""
    target = Path(path)
    if not target.is_file():
        msg = f"не найден файл графа: {target}"
        raise FileNotFoundError(msg)
    graph = nx.read_gml(target)
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        msg = f"граф {target} не является DAG: найден цикл {cycle}"
        raise ValueError(msg)
    for required in (TEMPLATE_TREATMENT, TEMPLATE_OUTCOME):
        if required not in graph:
            msg = f"в графе {target} отсутствует обязательный узел {required!r}"
            raise ValueError(msg)
    if not graph.has_edge(TEMPLATE_TREATMENT, TEMPLATE_OUTCOME):
        msg = (
            f"в графе {target} нет ребра {TEMPLATE_TREATMENT} → {TEMPLATE_OUTCOME}: "
            "изучаемый эффект не задан"
        )
        raise ValueError(msg)
    return graph


def template_groups(graph: nx.DiGraph) -> list[str]:
    """Узлы-группы ковариат шаблона."""
    return [
        node for node, data in graph.nodes(data=True) if data.get("kind", KIND_GROUP) == KIND_GROUP
    ]


@dataclass
class ExpandedGraph:
    """Развёрнутый DAG на уровне колонок витрины."""

    graph: nx.DiGraph
    treatment: str
    outcome: str
    assignment: dict[str, str]
    unmatched: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def confounders(self) -> list[str]:
        """Признаки с рёбрами и в ``T``, и в ``Y`` — кандидаты backdoor-множества."""
        return [
            f
            for f in self.assignment
            if self.graph.has_edge(f, self.treatment) and self.graph.has_edge(f, self.outcome)
        ]

    @property
    def outcome_only(self) -> list[str]:
        """Признаки, влияющие только на исход (например сезонность)."""
        return [
            f
            for f in self.assignment
            if self.graph.has_edge(f, self.outcome)
            and not self.graph.has_edge(f, self.treatment)
        ]

    def to_gml(self) -> str:
        """GML-представление для ``dowhy.CausalModel(graph=...)``.

        Строки склеиваются пробелом, а не переводом строки: DoWhy распознаёт
        формат переданной строки регулярным выражением без флага ``DOTALL``, и
        многострочный GML он не опознаёт. В файл (``save``) граф пишется
        обычным многострочным GML — там читаемость важнее.
        """
        return " ".join(line.strip() for line in nx.generate_gml(self.graph))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gml(self.graph, target)
        logger.info(
            "развёрнутый DAG: %s (%d узлов, %d рёбер)",
            target,
            len(self.graph),
            self.graph.number_of_edges(),
        )
        return target

    def summary(self) -> dict[str, object]:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "n_nodes": len(self.graph),
            "n_edges": self.graph.number_of_edges(),
            "n_covariates": len(self.assignment),
            "group_sizes": {g: len(v) for g, v in self.groups.items()},
            "n_confounders": len(self.confounders),
            "outcome_only_features": self.outcome_only,
            "unmatched_features": self.unmatched,
        }


def expand_graph(
    template: nx.DiGraph,
    covariates: Sequence[str],
    groups: FeatureGroups,
    *,
    treatment: str = "treatment",
    outcome: str = "target",
) -> ExpandedGraph:
    """Развернуть узлы-группы шаблона в фактические колонки витрины.

    Каждый признак наследует исходящие рёбра своей группы. Рёбра между
    группами (если появятся в шаблоне) разворачиваются в полный двудольный
    набор — это консервативно: лишнее ребро внутри backdoor-множества не
    делает оценку смещённой, а пропущенное сделало бы.
    """
    if not covariates:
        msg = "список ковариат пуст: нечего разворачивать"
        raise ValueError(msg)

    assignment, unmatched = groups.assign(covariates)
    known_groups = set(template_groups(template))
    unknown = sorted({g for g in assignment.values() if g not in known_groups})
    if unknown:
        msg = (
            f"карта групп ссылается на узлы, отсутствующие в шаблоне графа: {', '.join(unknown)}; "
            f"узлы шаблона: {', '.join(sorted(known_groups))}"
        )
        raise ValueError(msg)

    by_group: dict[str, list[str]] = {g: [] for g in known_groups}
    for feature, group in assignment.items():
        by_group[group].append(feature)

    empty = sorted(g for g, members in by_group.items() if not members)
    if empty:
        logger.warning(
            "узлы-группы без признаков (будут исключены из графа): %s", ", ".join(empty)
        )

    out = nx.DiGraph()
    out.add_node(treatment, kind=KIND_TREATMENT)
    out.add_node(outcome, kind=KIND_OUTCOME)
    for feature, group in assignment.items():
        out.add_node(feature, kind="covariate", group=group)

    rename = {TEMPLATE_TREATMENT: treatment, TEMPLATE_OUTCOME: outcome}

    def targets(node: str) -> Iterable[str]:
        """Во что разворачивается узел шаблона."""
        if node in rename:
            return [rename[node]]
        return by_group.get(node, [])

    for src, dst in template.edges():
        for s in targets(src):
            for d in targets(dst):
                if s != d:
                    out.add_edge(s, d)

    if not out.has_edge(treatment, outcome):
        msg = "в развёрнутом графе потерялось ребро T → Y"
        raise ValueError(msg)
    if not nx.is_directed_acyclic_graph(out):
        msg = "развёрнутый граф содержит цикл — проверьте рёбра между группами в шаблоне"
        raise ValueError(msg)

    expanded = ExpandedGraph(
        graph=out,
        treatment=treatment,
        outcome=outcome,
        assignment=assignment,
        unmatched=unmatched,
        groups={g: sorted(v) for g, v in by_group.items() if v},
    )
    logger.info(
        "DAG развёрнут: %d ковариат → %d узлов, %d рёбер; конфаундеров %d, только на Y %d",
        len(assignment),
        len(out),
        out.number_of_edges(),
        len(expanded.confounders),
        len(expanded.outcome_only),
    )
    return expanded


def build_expanded_graph(
    cfg: Config, covariates: Sequence[str], *, save: bool = True
) -> ExpandedGraph:
    """Прочитать шаблон и карту групп из конфигурации и развернуть DAG."""
    template = load_template(cfg.resolve(cfg.causal.graph))
    groups = FeatureGroups.load(cfg.resolve(cfg.causal.feature_groups))
    expanded = expand_graph(
        template,
        covariates,
        groups,
        treatment="treatment",
        outcome=cfg.data.outcome,
    )
    if save:
        expanded.save(cfg.resolve(cfg.causal.expanded_graph_path))
    return expanded


def describe_template(graph: nx.DiGraph) -> list[str]:
    """Markdown-описание шаблонного DAG (для docs/architecture.md и отчётов)."""
    lines = [
        "| Узел | Вид | Смысл | Исходящие рёбра |",
        "|---|---|---|---|",
    ]
    for node, data in graph.nodes(data=True):
        successors = ", ".join(f"`{s}`" for s in sorted(graph.successors(node))) or "—"
        lines.append(
            f"| `{node}` | {data.get('kind', KIND_GROUP)} | "
            f"{data.get('description', '—')} | {successors} |"
        )
    return lines


def group_table(expanded: ExpandedGraph) -> list[str]:
    """Markdown-таблица «группа → признаки» для отчёта по причинному слою."""
    lines = ["| Группа | Признаков | Влияет на T | Влияет на Y | Состав |", "|---|---|---|---|---|"]
    for group, members in sorted(expanded.groups.items()):
        probe = members[0]
        to_t = "да" if expanded.graph.has_edge(probe, expanded.treatment) else "нет"
        to_y = "да" if expanded.graph.has_edge(probe, expanded.outcome) else "нет"
        preview = ", ".join(f"`{m}`" for m in members[:6])
        if len(members) > 6:
            preview += f", … (+{len(members) - 6})"
        lines.append(f"| `{group}` | {len(members)} | {to_t} | {to_y} | {preview} |")
    return lines


def validate_against_frame(expanded: ExpandedGraph, columns: Iterable[str]) -> None:
    """Убедиться, что все узлы графа присутствуют в витрине."""
    available = set(columns)
    missing = sorted(set(expanded.graph.nodes()) - available)
    if missing:
        msg = (
            "узлы графа отсутствуют в витрине: "
            f"{', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}"
        )
        raise ValueError(msg)


def covariates_by_group(expanded: ExpandedGraph) -> Mapping[str, list[str]]:
    return expanded.groups


__all__ = [
    "KIND_GROUP",
    "KIND_OUTCOME",
    "KIND_TREATMENT",
    "ExpandedGraph",
    "FeatureGroups",
    "build_expanded_graph",
    "covariates_by_group",
    "describe_template",
    "expand_graph",
    "group_table",
    "load_template",
    "template_groups",
    "validate_against_frame",
]
