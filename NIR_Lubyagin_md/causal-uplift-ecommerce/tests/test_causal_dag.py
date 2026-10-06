"""Тесты причинного графа: карта групп и разворот шаблона в колонки (W5, §6.1).

Ключевое проверяемое свойство — структура графа действительно различает
группы: ``seasonality`` влияет только на ``Y`` и потому не попадает в
backdoor-множество. Если бы код подставлял в скорректированное множество все
ковариаты подряд, граф был бы декорацией, а тест §11 этого не заметил бы.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pytest

from causal_uplift.causal.dag import (
    KIND_OUTCOME,
    KIND_TREATMENT,
    FeatureGroups,
    build_expanded_graph,
    describe_template,
    expand_graph,
    group_table,
    load_template,
    template_groups,
    validate_against_frame,
)
from causal_uplift.config import Config

#: Небольшой, но представительный набор: по признаку на каждую группу шаблона
SAMPLE_COVARIATES = (
    "age",  # socdem
    "total_spend",  # behavior
    "regular_points_received",  # loyalty
    "n_unique_level_1",  # assortment
    "activity_trend_30d",  # trend
    "first_issue_month",  # seasonality
)


@pytest.fixture
def groups(cfg: Config) -> FeatureGroups:
    return FeatureGroups.load(cfg.resolve(cfg.causal.feature_groups))


@pytest.fixture
def template(cfg: Config) -> nx.DiGraph:
    return load_template(cfg.resolve(cfg.causal.graph))


# ------------------------------------------------------------- карта групп


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("age", "socdem"),
        ("age_bin", "socdem"),
        ("gender_F", "socdem"),
        ("total_spend", "behavior"),
        ("days_since_last_purchase", "behavior"),
        ("n_transactions_90d", "behavior"),
        ("regular_points_received", "loyalty"),
        ("share_transactions_with_redeem", "loyalty"),
        ("days_since_first_issue", "loyalty"),
        ("n_unique_level_1", "assortment"),
        ("own_trademark_share", "assortment"),
        ("favorite_store_id_freq", "assortment"),
        ("activity_trend_30d", "trend"),
        ("spend_slope_per_day", "trend"),
        ("activity_ratio_30_90", "trend"),
        ("spend_ratio_7_30", "trend"),
        ("first_issue_month", "seasonality"),
        ("first_issue_quarter", "seasonality"),
    ],
)
def test_feature_assigned_to_expected_group(
    groups: FeatureGroups, feature: str, expected: str
) -> None:
    assert groups.group_of(feature) == expected


def test_more_specific_group_wins_over_behavior(groups: FeatureGroups) -> None:
    """``spend_ratio_7_30`` — тренд, а не поведение: порядок групп в YAML значим."""
    assert groups.group_of("spend_ratio_7_30") == "trend"
    assert groups.group_of("spend_90d") == "behavior"


def test_unknown_feature_goes_to_fallback_and_is_reported(groups: FeatureGroups) -> None:
    assignment, unmatched = groups.assign(["age", "непонятный_признак"])
    assert unmatched == ["непонятный_признак"]
    assert assignment["непонятный_признак"] == groups.fallback_group
    assert assignment["age"] == "socdem"


def test_group_of_returns_none_for_unknown(groups: FeatureGroups) -> None:
    assert groups.group_of("непонятный_признак") is None


def test_missing_group_map_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="карта групп"):
        FeatureGroups.load(tmp_path / "нет.yaml")


def test_group_map_without_groups_section_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "feature_groups.yaml"
    path.write_text("fallback_group: behavior\n", encoding="utf-8")
    with pytest.raises(ValueError, match="groups"):
        FeatureGroups.load(path)


# ------------------------------------------------------------- шаблон графа


def test_template_has_required_nodes_and_effect_edge(template: nx.DiGraph) -> None:
    assert {"treatment", "outcome"} <= set(template.nodes())
    assert template.has_edge("treatment", "outcome")
    assert nx.is_directed_acyclic_graph(template)


def test_template_groups_are_the_six_covariate_nodes(template: nx.DiGraph) -> None:
    assert set(template_groups(template)) == {
        "socdem",
        "behavior",
        "loyalty",
        "assortment",
        "trend",
        "seasonality",
    }


def test_seasonality_affects_outcome_only_in_template(template: nx.DiGraph) -> None:
    assert template.has_edge("seasonality", "outcome")
    assert not template.has_edge("seasonality", "treatment")


def test_missing_template_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="файл графа"):
        load_template(tmp_path / "нет.gml")


def test_template_without_effect_edge_is_rejected(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_edge("socdem", "treatment")
    graph.add_edge("socdem", "outcome")
    path = tmp_path / "broken.gml"
    nx.write_gml(graph, path)
    with pytest.raises(ValueError, match="изучаемый эффект не задан"):
        load_template(path)


def test_template_without_treatment_node_is_rejected(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_edge("socdem", "outcome")
    path = tmp_path / "no_treatment.gml"
    nx.write_gml(graph, path)
    with pytest.raises(ValueError, match="treatment"):
        load_template(path)


def test_cyclic_template_is_rejected(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_edge("treatment", "outcome")
    graph.add_edge("outcome", "treatment")
    path = tmp_path / "cycle.gml"
    nx.write_gml(graph, path)
    with pytest.raises(ValueError, match="не является DAG"):
        load_template(path)


def test_describe_template_renders_markdown_table(template: nx.DiGraph) -> None:
    lines = describe_template(template)
    assert lines[0].startswith("| Узел |")
    assert any("`seasonality`" in line for line in lines)
    assert len(lines) == len(template) + 2  # заголовок и разделитель


# ------------------------------------------------------------- разворот графа


def test_expanded_graph_contains_every_covariate(
    template: nx.DiGraph, groups: FeatureGroups
) -> None:
    expanded = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target")
    assert set(expanded.graph.nodes()) == {*SAMPLE_COVARIATES, "treatment", "target"}
    assert expanded.graph.nodes["treatment"]["kind"] == KIND_TREATMENT
    assert expanded.graph.nodes["target"]["kind"] == KIND_OUTCOME
    assert expanded.graph.has_edge("treatment", "target")
    assert expanded.unmatched == []


def test_seasonality_is_outcome_only_after_expansion(
    template: nx.DiGraph, groups: FeatureGroups
) -> None:
    """Главное свойство: сезонность не конфаундер, и это видно из графа."""
    expanded = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target")
    assert expanded.outcome_only == ["first_issue_month"]
    assert "first_issue_month" not in expanded.confounders
    assert set(expanded.confounders) == {
        "age",
        "total_spend",
        "regular_points_received",
        "n_unique_level_1",
        "activity_trend_30d",
    }


def test_covariate_nodes_keep_their_group(template: nx.DiGraph, groups: FeatureGroups) -> None:
    expanded = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target")
    assert expanded.graph.nodes["age"]["group"] == "socdem"
    assert expanded.groups["seasonality"] == ["first_issue_month"]


def test_group_without_features_is_skipped(template: nx.DiGraph, groups: FeatureGroups) -> None:
    expanded = expand_graph(template, ["age", "first_issue_month"], groups, outcome="target")
    assert set(expanded.groups) == {"socdem", "seasonality"}
    assert expanded.summary()["group_sizes"] == {"socdem": 1, "seasonality": 1}


def test_to_gml_is_single_line(template: nx.DiGraph, groups: FeatureGroups) -> None:
    """DoWhy распознаёт GML регуляркой без DOTALL — многострочный он не видит."""
    gml = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target").to_gml()
    assert "\n" not in gml
    assert gml.startswith("graph [")
    assert "treatment" in gml


def test_saved_graph_is_readable_back(
    template: nx.DiGraph, groups: FeatureGroups, tmp_path: Path
) -> None:
    expanded = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target")
    path = expanded.save(tmp_path / "artifacts" / "dag_expanded.gml")
    restored = nx.read_gml(path)
    assert set(restored.nodes()) == set(expanded.graph.nodes())
    assert restored.number_of_edges() == expanded.graph.number_of_edges()


def test_summary_reports_counts(template: nx.DiGraph, groups: FeatureGroups) -> None:
    summary = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target").summary()
    assert summary["treatment"] == "treatment"
    assert summary["outcome"] == "target"
    assert summary["n_covariates"] == len(SAMPLE_COVARIATES)
    assert summary["n_confounders"] == 5
    assert summary["outcome_only_features"] == ["first_issue_month"]
    assert summary["unmatched_features"] == []


def test_empty_covariate_list_is_rejected(template: nx.DiGraph, groups: FeatureGroups) -> None:
    with pytest.raises(ValueError, match="список ковариат пуст"):
        expand_graph(template, [], groups, outcome="target")


def test_group_absent_from_template_is_rejected(template: nx.DiGraph) -> None:
    """Карта групп и шаблон графа должны согласовываться — иначе молчаливая потеря X."""
    broken = FeatureGroups(patterns={"такой_группы_нет": [re.compile(".*")]})
    with pytest.raises(ValueError, match="отсутствующие в шаблоне"):
        expand_graph(template, ["age"], broken, outcome="target")


def test_validate_against_frame_detects_missing_column(
    template: nx.DiGraph, groups: FeatureGroups
) -> None:
    expanded = expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target")
    validate_against_frame(expanded, [*SAMPLE_COVARIATES, "treatment", "target", "client_id"])
    with pytest.raises(ValueError, match="отсутствуют в витрине"):
        validate_against_frame(expanded, ["treatment", "target"])


def test_group_table_marks_seasonality_as_outcome_only(
    template: nx.DiGraph, groups: FeatureGroups
) -> None:
    lines = group_table(expand_graph(template, SAMPLE_COVARIATES, groups, outcome="target"))
    seasonality_row = next(line for line in lines if "`seasonality`" in line)
    # Колонки: группа | признаков | влияет на T | влияет на Y | состав
    cells = [c.strip() for c in seasonality_row.strip("|").split("|")]
    assert cells[2] == "нет"
    assert cells[3] == "да"


def test_build_expanded_graph_saves_artifact(cfg: Config) -> None:
    expanded = build_expanded_graph(cfg, list(SAMPLE_COVARIATES))
    assert cfg.resolve(cfg.causal.expanded_graph_path).is_file()
    assert expanded.outcome == cfg.data.outcome
