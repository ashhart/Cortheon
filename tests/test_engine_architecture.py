"""Architecture contract for the repository-only evidence engine."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest

import cortheon.engine as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "engine.py"
CORE = ROOT / "src" / "cortheon" / "engine_core"
EXPECTED_CONSUMERS = {
    "src/cortheon/auto_evidence.py",
    "src/cortheon/freshness_daemon.py",
    "src/cortheon/knowledge_pool.py",
    "src/cortheon/option_ranker.py",
    "src/cortheon/parity_benchmark_core/casepack.py",
    "src/cortheon/parity_benchmark_core/cli.py",
    "src/cortheon/slash.py",
}
CORE_OWNERS = {
    "packages.py": {"inspect_package", "compare", "recommend"},
    "api.py": {"retrieve_api_evidence", "check_generated_code", "fetch_docs", "diff_api"},
    "repository.py": {"verify_patch", "scan_repository", "check_repo_fit"},
    "recommendations.py": {
        "build_recommendation",
        "ranking_to_recommendation",
        "merge_examples",
    },
}
PUBLIC_SIGNATURES = {
    "CortheonEngine": "(pypi: 'PyPIConnector | None' = None, osv: 'OSVConnector | None' = None, github: 'GitHubConnector | None' = None, docs: 'DocumentationConnector | None' = None, api_extractor: 'ApiEvidenceExtractor | None' = None, docs_reader: 'DocsSiteReader | None' = None, ledger: 'EvidenceLedger | None' = None) -> 'None'",
    "ranking_to_recommendation": "(task: 'str', ranking: 'OptionRankingReport') -> 'RecommendationReport'",
    "merge_examples": "(*groups: 'list[str]', limit: 'int' = 4) -> 'list[str]'",
}
METHOD_SIGNATURES = {
    "inspect_package": "(self, package: 'str', *, task_text: 'str | None' = None, candidate: 'Candidate | None' = None, run_install: 'bool' = False, run_examples: 'bool' = False, sandbox: 'bool' = False, write_report: 'bool' = True) -> 'PackageReport'",
    "compare": "(self, packages: 'list[str]', *, task_text: 'str | None' = None, run_install: 'bool' = False, write_report: 'bool' = True) -> 'RecommendationReport'",
    "recommend": "(self, task: 'str', *, run_install: 'bool' = False, write_report: 'bool' = True) -> 'RecommendationReport'",
    "retrieve_api_evidence": "(self, package: 'str', query: 'str', *, include_docs: 'bool' = False, write_report: 'bool' = True) -> 'ApiEvidenceReport'",
    "check_generated_code": "(self, package: 'str', code: 'str', *, write_report: 'bool' = True) -> 'CodeUsageReport'",
    "fetch_docs": "(self, package: 'str', *, version: 'str | None' = None, max_pages: 'int' = 4, write_report: 'bool' = True) -> 'DocsSiteReport'",
    "diff_api": "(self, package: 'str', old_version: 'str', new_version: 'str', *, write_report: 'bool' = True) -> 'ApiDiffReport'",
    "verify_patch": "(self, repo_path: 'str', patch_text: 'str', *, test_command: 'str | None' = None, run_baseline: 'bool' = True, test_isolation: 'str' = 'host', sandbox_image: 'str' = 'python:3.12-slim-bookworm', write_report: 'bool' = True) -> 'PatchReport'",
    "scan_repo": "(self, path: 'str' = '.', *, write_report: 'bool' = True) -> 'RepoReport'",
    "check_repo_fit": "(self, package: 'str', repo_path: 'str' = '.', *, write_report: 'bool' = True) -> 'RepoFitReport'",
    "_recommendation": "(self, *, task: 'str', profile: 'str | None', candidates: 'list[PackageReport]', notes: 'list[str]') -> 'RecommendationReport'",
}
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+cortheon\.engine\s+import\b|import\s+cortheon\.engine\b)",
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
        and node.module.startswith("cortheon.engine_core.")
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
        assert not _imports(path, "cortheon.engine")


def test_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"engine_core cycle through {name}"
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


def test_direct_source_consumers_are_explicit() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and IMPORT_PATTERN.search(path.read_text(encoding="utf-8"))
    }
    assert consumers == EXPECTED_CONSUMERS


def test_engine_is_repository_only() -> None:
    assert '"engine"' not in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "src/cortheon/engine.py" not in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")


def test_public_signatures_type_hints_and_module_identities_are_stable() -> None:
    for name, expected in PUBLIC_SIGNATURES.items():
        value = getattr(facade, name)
        assert str(inspect.signature(value)) == expected
        assert value.__module__ == "cortheon.engine"
        hinted = value.__init__ if name == "CortheonEngine" else value
        assert get_type_hints(hinted)
    for name, expected in METHOD_SIGNATURES.items():
        method = getattr(facade.CortheonEngine, name)
        assert str(inspect.signature(method)) == expected
        assert method.__module__ == "cortheon.engine"
        code_usage_report = facade._code_usage_models()[1]
        assert get_type_hints(
            method,
            globalns={**vars(facade), "CodeUsageReport": code_usage_report},
        )


def test_constructor_keeps_late_bound_facade_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinels = {name: object() for name in ("pypi", "osv", "github", "docs", "api", "reader")}
    ledger = SimpleNamespace(base_dir=tmp_path)
    monkeypatch.setattr(facade, "PyPIConnector", lambda: sentinels["pypi"])
    monkeypatch.setattr(facade, "OSVConnector", lambda: sentinels["osv"])
    monkeypatch.setattr(facade, "GitHubConnector", lambda: sentinels["github"])
    monkeypatch.setattr(facade, "DocumentationConnector", lambda: sentinels["docs"])
    monkeypatch.setattr(facade, "EvidenceLedger", lambda: ledger)
    monkeypatch.setattr(facade, "ApiEvidenceExtractor", lambda **_kwargs: sentinels["api"])
    monkeypatch.setattr(facade, "DocsSiteReader", lambda: sentinels["reader"])
    engine = facade.CortheonEngine()
    assert engine.pypi is sentinels["pypi"]
    assert engine.osv is sentinels["osv"]
    assert engine.github is sentinels["github"]
    assert engine.docs is sentinels["docs"]
    assert engine.ledger is ledger
    assert engine.api_extractor is sentinels["api"]
    assert engine.docs_reader is sentinels["reader"]


def test_moved_workflows_use_late_bound_facade_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = SimpleNamespace(root="sentinel")
    monkeypatch.setattr(facade, "scan_repo", lambda _path: expected)
    engine = facade.CortheonEngine.__new__(facade.CortheonEngine)
    engine.ledger = cast(Any, SimpleNamespace())
    assert engine.scan_repo("repo", write_report=False) is expected
