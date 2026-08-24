"""Architecture contract for repository-only automatic evidence acquisition."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest

import cortheon.auto_evidence as facade
import cortheon.knowledge_pool as knowledge_pool
import cortheon.slash as slash

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "auto_evidence.py"
CORE = ROOT / "src" / "cortheon" / "auto_evidence_core"
EXPECTED_CONSUMERS = {
    "src/cortheon/knowledge_pool.py",
    "src/cortheon/slash.py",
}
CORE_OWNERS = {
    "acquisition.py": {
        "run",
        "run_agent",
        "repo_agent",
        "package_agent",
        "api_agent",
        "research_agent",
    },
    "classification.py": {
        "unique_tags",
        "package_evidence_tags",
        "api_evidence_tags",
        "research_evidence_tags",
        "research_agent_satisfied",
        "research_report_has_substance",
        "technology_report_has_substance",
        "architecture_report_has_substance",
        "grounded_claim_count",
        "extract_api_target",
        "named_technology_candidates",
        "normalize_name",
        "research_topic",
    },
    "details.py": {
        "api_source_details",
        "recommendation_details",
        "package_source_details",
        "research_details",
        "research_summary",
        "auto_notes",
    },
}
PUBLIC_SIGNATURES = {
    "EvidenceAgentRun": "(agent: 'str', missing_evidence: 'str', status: 'str', produced_tags: 'list[str]', summary: 'str', details: 'dict[str, Any]' = <factory>, errors: 'list[str]' = <factory>) -> None",
    "AutoDecisionReport": "(task: 'str', proposed_action: 'str | None', initial_decision: 'DecisionReport', final_decision: 'DecisionReport', evidence_tags: 'list[str]', agent_runs: 'list[EvidenceAgentRun]', notes: 'list[str]') -> None",
    "AutoEvidenceLimits": "(max_search_results: 'int' = 5, max_scholarly_results: 'int' = 8, max_github_results: 'int' = 3, max_trial_results: 'int' = 5, max_follow_up_queries: 'int' = 2, max_adaptive_queries: 'int' = 1, max_artifact_inspections: 'int' = 2, max_pages: 'int' = 10, max_depth: 'int' = 1) -> None",
    "EvidenceAcquisitionLoop": "(engine: 'CortheonEngine', *, research_engine: 'ResearchEngine | None' = None, source_planner_strategy: 'str | None' = 'auto', repo_path: 'str | None' = None) -> 'None'",
    "unique_tags": "(tags: 'list[str]') -> 'list[str]'",
    "package_evidence_tags": "(report: 'RecommendationReport', proposed_action: 'str | None') -> 'list[str]'",
    "api_evidence_tags": "(report: 'ApiEvidenceReport') -> 'list[str]'",
    "api_source_details": "(report: 'ApiEvidenceReport') -> 'list[dict[str, Any]]'",
    "research_evidence_tags": "(report: 'ResearchReport', *, technology_choice: 'bool') -> 'list[str]'",
    "research_agent_satisfied": "(missing: 'str', tags: 'list[str]') -> 'bool'",
    "research_report_has_substance": "(report: 'ResearchReport') -> 'bool'",
    "technology_report_has_substance": "(report: 'ResearchReport') -> 'bool'",
    "architecture_report_has_substance": "(report: 'ResearchReport') -> 'bool'",
    "grounded_claim_count": "(report: 'ResearchReport') -> 'int'",
    "extract_api_target": "(text: 'str') -> 'tuple[str, str] | None'",
    "named_technology_candidates": "(text: 'str') -> 'set[str]'",
    "normalize_name": "(value: 'str') -> 'str'",
    "research_topic": "(task: 'str', proposed_action: 'str | None', context: 'str | None') -> 'str'",
    "recommendation_details": "(report: 'RecommendationReport') -> 'dict[str, Any]'",
    "package_source_details": "(report: 'RecommendationReport') -> 'list[dict[str, Any]]'",
    "research_details": "(report: 'ResearchReport') -> 'dict[str, Any]'",
    "research_summary": "(report: 'ResearchReport', tags: 'list[str]') -> 'str'",
    "auto_notes": "(initial: 'DecisionReport', final: 'DecisionReport', runs: 'list[EvidenceAgentRun]') -> 'list[str]'",
}
METHOD_SIGNATURES = {
    "run": "(self, task: 'str', *, proposed_action: 'str | None' = None, context: 'str | None' = None, evidence: 'list[str] | None' = None, limits: 'AutoEvidenceLimits | None' = None) -> 'AutoDecisionReport'",
    "run_agent": "(self, missing: 'str', *, task: 'str', proposed_action: 'str | None', context: 'str | None', limits: 'AutoEvidenceLimits') -> 'EvidenceAgentRun'",
    "repo_agent": "(self, missing: 'str') -> 'EvidenceAgentRun'",
    "package_agent": "(self, task: 'str', proposed_action: 'str | None', context: 'str | None', limits: 'AutoEvidenceLimits') -> 'EvidenceAgentRun'",
    "api_agent": "(self, task: 'str', proposed_action: 'str | None', context: 'str | None') -> 'EvidenceAgentRun'",
    "research_agent": "(self, missing: 'str', topic: 'str', limits: 'AutoEvidenceLimits', *, technology_choice: 'bool') -> 'EvidenceAgentRun'",
}


def _imports(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        (isinstance(node, ast.ImportFrom) and node.module == module)
        or (isinstance(node, ast.Import) and any(alias.name == module for alias in node.names))
        for node in ast.walk(tree)
    )


def _core_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("cortheon.auto_evidence_core.")
    }


def test_facade_core_and_architecture_test_stay_below_file_limit() -> None:
    for path in [FACADE, *sorted(CORE.glob("*.py")), Path(__file__)]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_core_ownership_is_explicit_and_does_not_reenter_facade() -> None:
    for filename, expected in CORE_OWNERS.items():
        path = CORE / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        actual = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert actual == expected
        assert not _imports(path, "cortheon.auto_evidence")


def test_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"auto_evidence_core cycle through {name}"
        if name in visited:
            return
        active.add(name)
        for dependency in graph.get(name, set()):
            visit(dependency, active, visited)
        active.remove(name)
        visited.add(name)

    visited: set[str] = set()
    for module in graph:
        visit(module, set(), visited)


def test_direct_source_consumers_are_explicit_and_share_facade_identities() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and _imports(path, "cortheon.auto_evidence")
    }
    assert consumers == EXPECTED_CONSUMERS
    assert knowledge_pool.EvidenceAcquisitionLoop is facade.EvidenceAcquisitionLoop
    assert knowledge_pool.AutoEvidenceLimits is facade.AutoEvidenceLimits
    assert slash.EvidenceAcquisitionLoop is facade.EvidenceAcquisitionLoop
    assert slash.AutoEvidenceLimits is facade.AutoEvidenceLimits


def test_auto_evidence_is_repository_only() -> None:
    assert '"auto_evidence"' not in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "src/cortheon/auto_evidence.py" not in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")


def test_public_signatures_type_hints_and_module_identities_are_stable() -> None:
    for name, expected in PUBLIC_SIGNATURES.items():
        value = getattr(facade, name)
        assert str(inspect.signature(value)) == expected
        assert value.__module__ == "cortheon.auto_evidence"
        hinted = value.__init__ if name == "EvidenceAcquisitionLoop" else value
        assert get_type_hints(hinted)
    for name, expected in METHOD_SIGNATURES.items():
        method = getattr(facade.EvidenceAcquisitionLoop, name)
        assert str(inspect.signature(method)) == expected
        assert method.__module__ == "cortheon.auto_evidence"
        assert get_type_hints(method)


def test_acquisition_uses_late_bound_facade_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = RuntimeError("patched facade helper")

    def patched_unique_tags(_tags: list[str]) -> list[str]:
        raise marker

    monkeypatch.setattr(facade, "unique_tags", patched_unique_tags)
    loop = facade.EvidenceAcquisitionLoop(cast(Any, SimpleNamespace()))
    with pytest.raises(RuntimeError) as raised:
        loop.run("task")
    assert raised.value is marker


def test_helper_wrappers_keep_late_bound_facade_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = [{"source_type": "patched"}]
    report = SimpleNamespace(
        winner="winner",
        profile="profile",
        candidates=[],
        notes=[],
    )
    monkeypatch.setattr(facade, "package_source_details", lambda _report: sentinel)
    assert facade.recommendation_details(cast(Any, report))["sources"] is sentinel
