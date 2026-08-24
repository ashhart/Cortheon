"""Architecture contract for repository-only pooled evidence synthesis."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest

import cortheon.knowledge_pool as facade
import cortheon.slash as slash

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "knowledge_pool.py"
CORE = ROOT / "src" / "cortheon" / "knowledge_pool_core"
CORE_OWNERS = {
    "pooler.py": {"run", "research"},
    "limits.py": {"merge_limits"},
    "reports.py": {"build_blocked_report", "build_report"},
    "answering.py": {
        "pool_topic",
        "couple_verdict_to_answer_status",
        "answer_status",
        "answer_confidence",
        "best_approach",
        "key_steps",
        "evidence_gaps",
        "synthesis_gaps",
    },
    "sources.py": {
        "top_sources",
        "agent_pooled_sources",
        "agent_winner",
        "dedupe_sources",
        "claims_by_source",
        "string_or_none",
        "float_or_none",
        "source_plan_summary",
        "discovery_counts",
        "compact_text",
        "unique_notes",
    },
}
PUBLIC_SIGNATURES = {
    "PooledSource": "(title: 'str | None', url: 'str', source_type: 'str', authority_score: 'float | None', relevance_score: 'float | None', summary: 'str | None', derived_claims: 'list[str]' = <factory>) -> None",
    "SourcePlanSummary": "(name: 'str', source_type: 'str', selected: 'bool', reason: 'str', trust_tier: 'str', budget: 'int | None', observed_count: 'int | None' = None) -> None",
    "KnowledgePoolReport": "(task: 'str', proposed_action: 'str | None', generated_at: 'datetime', answer_status: 'str', verdict: 'str', confidence: 'float', best_supported_approach: 'str', key_steps: 'list[str]', evidence_tags: 'list[str]', decision: 'AutoDecisionReport', source_summaries: 'list[PooledSource]', source_plan: 'list[SourcePlanSummary]', discovery_counts: 'dict[str, int]', synthesis_status: 'str | None', evidence_gaps: 'list[str]', notes: 'list[str]', errors: 'list[str]') -> None",
    "KnowledgePooler": "(engine: 'CortheonEngine', *, research_engine: 'ResearchEngine | None' = None, source_planner_strategy: 'str | None' = 'auto') -> 'None'",
    "_merge_limits": "(limits: 'AutoEvidenceLimits', plan: 'TaskResearchPlan') -> 'AutoEvidenceLimits'",
    "build_blocked_report": "(task: 'str', proposed_action: 'str | None', decision: 'AutoDecisionReport') -> 'KnowledgePoolReport'",
    "build_report": "(task: 'str', proposed_action: 'str | None', decision: 'AutoDecisionReport', research_report: 'ResearchReport', research_tags: 'list[str]') -> 'KnowledgePoolReport'",
    "pool_topic": "(task: 'str', proposed_action: 'str | None', context: 'str | None') -> 'str'",
    "couple_verdict_to_answer_status": "(verdict: 'str', status: 'str') -> 'tuple[str, list[str]]'",
    "answer_status": "(decision: 'AutoDecisionReport', research_report: 'ResearchReport', sources: 'list[PooledSource]') -> 'str'",
    "answer_confidence": "(decision: 'AutoDecisionReport', research_report: 'ResearchReport') -> 'float'",
    "best_approach": "(decision: 'AutoDecisionReport', research_report: 'ResearchReport') -> 'str'",
    "key_steps": "(decision: 'AutoDecisionReport', research_report: 'ResearchReport') -> 'list[str]'",
    "evidence_gaps": "(decision: 'AutoDecisionReport', research_report: 'ResearchReport') -> 'list[str]'",
    "synthesis_gaps": "(report: 'ResearchReport') -> 'list[str]'",
    "top_sources": "(report: 'ResearchReport', limit: 'int' = 6) -> 'list[PooledSource]'",
    "agent_pooled_sources": "(decision: 'AutoDecisionReport') -> 'list[PooledSource]'",
    "agent_winner": "(decision: 'AutoDecisionReport') -> 'str | None'",
    "dedupe_sources": "(sources: 'list[PooledSource]') -> 'list[PooledSource]'",
    "claims_by_source": "(report: 'ResearchReport') -> 'dict[str, list[str]]'",
    "string_or_none": "(value: 'object') -> 'str | None'",
    "float_or_none": "(value: 'object') -> 'float | None'",
    "source_plan_summary": "(report: 'ResearchReport') -> 'list[SourcePlanSummary]'",
    "discovery_counts": "(report: 'ResearchReport') -> 'dict[str, int]'",
    "compact_text": "(text: 'str', limit: 'int' = 360) -> 'str | None'",
    "unique_notes": "(items: 'list[str]') -> 'list[str]'",
}
METHOD_SIGNATURES = {
    "run": "(self, task: 'str', *, proposed_action: 'str | None' = None, context: 'str | None' = None, evidence: 'list[str] | None' = None, limits: 'AutoEvidenceLimits | None' = None) -> 'KnowledgePoolReport'",
    "_research": "(self, task: 'str', proposed_action: 'str | None', context: 'str | None', limits: 'AutoEvidenceLimits') -> 'ResearchReport'",
}
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+cortheon\.knowledge_pool\s+import\b|import\s+cortheon\.knowledge_pool\b)",
    re.MULTILINE,
)


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
        and node.module.startswith("cortheon.knowledge_pool_core.")
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
        assert not _imports(path, "cortheon.knowledge_pool")


def test_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"knowledge_pool_core cycle through {name}"
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


def test_direct_source_consumer_is_explicit_and_shares_facade_identity() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and IMPORT_PATTERN.search(path.read_text(encoding="utf-8"))
    }
    assert consumers == {"src/cortheon/slash.py"}
    assert slash.KnowledgePooler is facade.KnowledgePooler


def test_knowledge_pool_is_repository_only() -> None:
    assert '"knowledge_pool"' not in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "src/cortheon/knowledge_pool.py" not in (ROOT / "MANIFEST.in").read_text(
        encoding="utf-8"
    )


def test_public_signatures_type_hints_and_module_identities_are_stable() -> None:
    for name, expected in PUBLIC_SIGNATURES.items():
        value = getattr(facade, name)
        assert str(inspect.signature(value)) == expected
        assert value.__module__ == "cortheon.knowledge_pool"
        hinted = value.__init__ if name == "KnowledgePooler" else value
        assert get_type_hints(hinted)
    for name, expected in METHOD_SIGNATURES.items():
        method = getattr(facade.KnowledgePooler, name)
        assert str(inspect.signature(method)) == expected
        assert method.__module__ == "cortheon.knowledge_pool"
        assert get_type_hints(method)


def test_pooler_uses_late_bound_facade_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = RuntimeError("patched unique tags")

    def patched_unique_tags(_items: list[str]) -> list[str]:
        raise marker

    monkeypatch.setattr(facade, "unique_tags", patched_unique_tags)
    pool = facade.KnowledgePooler(cast(Any, SimpleNamespace()))
    with pytest.raises(RuntimeError) as raised:
        pool.run("task")
    assert raised.value is marker


def test_source_workflows_use_late_bound_sanitizer(monkeypatch: pytest.MonkeyPatch) -> None:
    cleaned = SimpleNamespace(clean_text="sanitized")
    monkeypatch.setattr(facade, "scan_text", lambda _text: cleaned)
    page = SimpleNamespace(
        final_url="https://example.test",
        url="https://example.test",
        title="title",
        source_type="web",
        authority_score=0.5,
        text="unsafe",
    )
    report = SimpleNamespace(
        claims=[],
        crawled_pages=[page],
        scholarly_works=[],
        artifacts=[],
        search_results=[],
    )
    assert facade.top_sources(cast(Any, report))[0].summary == "sanitized"
