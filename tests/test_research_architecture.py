from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest import mock

import cortheon.auto_evidence as auto_evidence
import cortheon.knowledge_pool as knowledge_pool
import cortheon.research as facade
import cortheon.slash as slash

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "cortheon" / "research_core"
EXPECTED_CORE_FILES = {
    "__init__.py",
    "_compat.py",
    "discovery.py",
    "engine.py",
    "evidence.py",
    "gaps.py",
    "notes.py",
}
ORIGINAL_DEFINITIONS = {
    "ResearchEngine",
    "artifact_assessment_evidence",
    "artifact_evidence",
    "artifact_mix",
    "artifact_notes",
    "build_gap_closures",
    "count_pass_seeds",
    "coverage_notes",
    "dedupe",
    "gap_closure_evidence",
    "gap_kind",
    "gap_metric_improved",
    "grounding_evidence",
    "limit_discovered_artifacts",
    "lineage_evidence",
    "merge_scholarly_works",
    "merge_search_results",
    "mission_plan_evidence",
    "mission_plan_notes",
    "per_query_limit",
    "research_notes",
    "scholarly_source_profiles",
    "source_coverage_evidence",
    "source_mix",
    "synthesis_evidence",
    "trial_registry_source_profiles",
}


def _definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _core_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prefix = "cortheon.research_core."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def test_facade_and_every_research_file_stay_below_cap() -> None:
    source = ROOT / "src" / "cortheon" / "research.py"
    authored = [source, *CORE.glob("*.py")]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in CORE.glob("*.py")} == EXPECTED_CORE_FILES
    assert counts["research.py"] <= 250
    assert all(count <= 500 for count in counts.values()), counts


def test_original_definitions_have_one_core_owner_and_stable_module_identity() -> None:
    owners = {
        name: [path.name for path in CORE.glob("*.py") if name in _definitions(path)]
        for name in ORIGINAL_DEFINITIONS
    }

    assert all(len(paths) == 1 for paths in owners.values()), owners
    assert set(facade.__all__) >= ORIGINAL_DEFINITIONS
    for name in ORIGINAL_DEFINITIONS:
        public = getattr(facade, name)
        assert public.__module__ == "cortheon.research"
    assert facade.ResearchEngine.research.__module__ == "cortheon.research"
    assert facade.ResearchEngine._run_discovery_queries.__module__ == "cortheon.research"


def test_research_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"research_core import cycle through {name}"
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


def test_public_engine_signatures_remain_stable() -> None:
    assert str(inspect.signature(facade.ResearchEngine)) == (
        "(search_provider: 'SearchProvider | None' = None, scholarly_discovery: "
        "'CompositeScholarlyDiscovery | None' = None, github_discovery: "
        "'GitHubRepositorySearch | None' = None, trial_discovery: "
        "'ClinicalTrialsGovDiscovery | None' = None, crawler: 'WebCrawler | None' = None, "
        "ledger: 'EvidenceLedger | None' = None, source_planner: 'SourcePlanner | None' = None, "
        "source_planner_strategy: 'str | None' = None) -> 'None'"
    )
    assert str(inspect.signature(facade.build_gap_closures)).startswith(
        "(adaptive_queries: 'list[ResearchQuery]'"
    )


def test_direct_consumers_receive_the_same_engine_class() -> None:
    for consumer in (
        auto_evidence,
        knowledge_pool,
        slash,
    ):
        assert consumer.ResearchEngine is facade.ResearchEngine


def test_facade_constructor_patches_drive_moved_engine() -> None:
    sentinels = {
        "search_provider": object(),
        "scholarly_discovery": object(),
        "github_discovery": object(),
        "trial_discovery": object(),
        "crawler": object(),
        "ledger": object(),
        "source_planner": object(),
    }
    patches = (
        ("ConfiguredSearchProvider", "search_provider"),
        ("CompositeScholarlyDiscovery", "scholarly_discovery"),
        ("GitHubRepositorySearch", "github_discovery"),
        ("ClinicalTrialsGovDiscovery", "trial_discovery"),
        ("WebCrawler", "crawler"),
        ("EvidenceLedger", "ledger"),
        ("default_source_planner", "source_planner"),
    )
    with (
        mock.patch.object(facade, patches[0][0], return_value=sentinels[patches[0][1]]),
        mock.patch.object(facade, patches[1][0], return_value=sentinels[patches[1][1]]),
        mock.patch.object(facade, patches[2][0], return_value=sentinels[patches[2][1]]),
        mock.patch.object(facade, patches[3][0], return_value=sentinels[patches[3][1]]),
        mock.patch.object(facade, patches[4][0], return_value=sentinels[patches[4][1]]),
        mock.patch.object(facade, patches[5][0], return_value=sentinels[patches[5][1]]),
        mock.patch.object(facade, patches[6][0], return_value=sentinels[patches[6][1]]),
    ):
        engine = facade.ResearchEngine()

    for attribute, expected in sentinels.items():
        assert getattr(engine, attribute) is expected


def test_facade_search_patch_drives_discovery_pass() -> None:
    engine = object.__new__(facade.ResearchEngine)
    engine.search_provider = object()
    engine.scholarly_discovery = object()
    engine.github_discovery = object()
    engine.trial_discovery = object()
    query = facade.ResearchQuery(query="current fact", purpose="test", source="test")
    with mock.patch.object(facade, "search_with_errors", return_value=([], [], [])) as search:
        result = engine._run_discovery_queries(
            [query],
            scholarly_limit=0,
            search_limit=1,
            github_limit=0,
            trial_limit=0,
            scholarly_connectors=[],
        )

    assert search.call_count == 1
    assert len(result[-1]) == 1


def test_research_core_remains_repository_only() -> None:
    for config in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        assert "research_core" not in (ROOT / config).read_text(encoding="utf-8")
