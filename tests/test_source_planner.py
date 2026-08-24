import pytest

from cortheon.source_planner import (
    SourcePlanner,
    SourcePlanningConstraints,
    build_research_source_profiles,
    default_source_planner,
)


def test_software_topic_selects_code_search() -> None:
    plan = SourcePlanner().plan(
        "build a REST API in Python",
        build_research_source_profiles([], search_provider_name=None, seed_url_count=0),
        SourcePlanningConstraints(0, 0, 2, 0, None),
    )
    assert any(item.name == "github_repositories" and item.selected for item in plan)
    assert all(item.planner == "heuristic" for item in plan)


def test_default_planner_is_always_deterministic() -> None:
    assert isinstance(default_source_planner(), SourcePlanner)
    assert isinstance(default_source_planner("auto"), SourcePlanner)
    assert isinstance(default_source_planner("heuristic"), SourcePlanner)
    with pytest.raises(ValueError, match=r"auto.*heuristic"):
        default_source_planner("model")
