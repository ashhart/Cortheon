"""Architecture contracts for repository-only report value types."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import pytest

import cortheon.claims as claims
import cortheon.decision as decision
import cortheon.models as models
import cortheon.repo_scanner as repo_scanner
import cortheon.scoring as scoring
import cortheon.search as search
from cortheon.models_core.base import to_jsonable as moved_to_jsonable

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "src" / "cortheon" / "models.py"
CORE = ROOT / "src" / "cortheon" / "models_core"
CORE_FILES = {
    "__init__.py",
    "_compat.py",
    "base.py",
    "discovery.py",
    "package.py",
    "repository.py",
    "research.py",
}
ORIGINAL_DEFINITIONS = {
    "ApiDiffReport",
    "ApiEvidenceReport",
    "ApiSymbol",
    "ApiSymbolChange",
    "ClaimCluster",
    "CodeUsageFinding",
    "CodeUsageReport",
    "ContradictionGroup",
    "CrawledPage",
    "DecisionCheck",
    "DecisionReport",
    "DistributionArtifact",
    "DocsExample",
    "DocsPage",
    "DocsSiteReport",
    "DocumentationReport",
    "Evidence",
    "EvidenceStatus",
    "ExampleRunResult",
    "GitHubRepoReport",
    "PackageMetadata",
    "PackageReport",
    "PatchReport",
    "PatchTestRun",
    "RecommendationReport",
    "RepoDependency",
    "RepoFitReport",
    "RepoReport",
    "ResearchArtifact",
    "ResearchArtifactAssessment",
    "ResearchClaim",
    "ResearchCoverageItem",
    "ResearchDiscoveryPass",
    "ResearchGapClosure",
    "ResearchQuery",
    "ResearchReport",
    "ResearchSourceDecision",
    "ResearchSynthesis",
    "ScholarlyWork",
    "ScoreBreakdown",
    "SearchResult",
    "SourceLineage",
    "SupportLevel",
    "VerificationResult",
    "VulnerabilityReport",
    "parse_datetime",
    "to_jsonable",
    "utc_now",
}
SIGNATURE_DIGEST = "f3cef801a036342baf057377feb4f991f416691ddd792d5966b99be44af03c81"
FIELD_DIGEST = "30fb3de5c6d5fbe4c114c21374f20aa4b463ccab8535dcbf561a6ad55e2b9dfe"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _core_imports(path: Path) -> set[str]:
    prefix = "cortheon.models_core."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0]
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def _field_contract() -> list[str]:
    contract: list[str] = []
    for name in sorted(ORIGINAL_DEFINITIONS):
        value = getattr(models, name)
        if not isinstance(value, type) or not dataclasses.is_dataclass(value):
            continue
        for item in dataclasses.fields(value):
            default = "<MISSING>" if item.default is dataclasses.MISSING else repr(item.default)
            factory = (
                "<MISSING>"
                if item.default_factory is dataclasses.MISSING
                else getattr(item.default_factory, "__name__", repr(item.default_factory))
            )
            contract.append(f"{name}:{item.name}:{item.type}:{default}:{factory}")
    return contract


def _signature_contract() -> list[str]:
    contract: list[str] = []
    for name in sorted(ORIGINAL_DEFINITIONS):
        value = getattr(models, name)
        if isinstance(value, type) and issubclass(value, Enum):
            members = ",".join(f"{member.name}={member.value!r}" for member in value)
            contract.append(f"{name}:enum:{members}")
        else:
            contract.append(f"{name}:{inspect.signature(value)}")
    return contract


def test_facade_and_core_membership_stay_below_file_cap() -> None:
    paths = sorted(CORE.glob("*.py"))
    authored = [FACADE, *paths, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in paths} == CORE_FILES
    assert counts[FACADE.name] <= 175
    assert all(count <= 500 for count in counts.values()), counts


def test_original_definitions_have_one_core_owner() -> None:
    paths = sorted(CORE.glob("*.py"))
    owners = {
        name: [path.name for path in paths if name in _definitions(path)]
        for name in ORIGINAL_DEFINITIONS
    }
    all_definitions = set().union(*(_definitions(path) for path in paths))

    assert all(len(files) == 1 for files in owners.values()), owners
    assert all_definitions == ORIGINAL_DEFINITIONS | {"facade"}


def test_core_import_graph_is_acyclic() -> None:
    paths = sorted(CORE.glob("*.py"))
    graph = {path.stem: _core_imports(path) for path in paths}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"models import cycle through {name}"
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


def test_signatures_fields_slots_and_module_identities_are_stable() -> None:
    signatures = "\n".join(_signature_contract())
    fields = "\n".join(_field_contract())

    assert hashlib.sha256(signatures.encode()).hexdigest() == SIGNATURE_DIGEST
    assert hashlib.sha256(fields.encode()).hexdigest() == FIELD_DIGEST
    for name in ORIGINAL_DEFINITIONS:
        value = getattr(models, name)
        assert value.__module__ == "cortheon.models"
        if isinstance(value, type) and dataclasses.is_dataclass(value):
            assert isinstance(value.__slots__, tuple)  # pyright: ignore[reportAttributeAccessIssue]
            for member in vars(value).values():
                accessors = (
                    (member.fget, member.fset, member.fdel)
                    if isinstance(member, property)
                    else (member,)
                )
                for accessor in accessors:
                    if callable(accessor) and getattr(accessor, "__module__", "").startswith(
                        "cortheon.models_core"
                    ):
                        raise AssertionError(f"moved identity leaked for {name}")


def test_only_facade_imports_models_core_from_repository_source() -> None:
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path not in CORE.glob("*.py")
        and any(
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cortheon.models_core")
            for node in ast.walk(_tree(path))
        )
    }
    assert importers == {"src/cortheon/models.py"}


def test_representative_consumers_receive_identical_objects() -> None:
    assert claims.ResearchClaim is models.ResearchClaim
    assert decision.DecisionCheck is models.DecisionCheck
    assert repo_scanner.RepoReport is models.RepoReport
    assert scoring.PackageReport is models.PackageReport
    assert search.Evidence is models.Evidence


def test_facade_patches_drive_moved_helpers_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = datetime(2024, 1, 2, tzinfo=UTC)
    evidence = models.Evidence(
        "claim",
        "test",
        None,
        retrieved_at=fixed,
        expires_at=fixed + timedelta(days=1),
    )
    monkeypatch.setattr(models, "utc_now", lambda: fixed + timedelta(days=2))
    assert evidence.refresh_status() is models.EvidenceStatus.STALE

    monkeypatch.setattr(models, "to_jsonable", lambda value: f"patched:{value}")
    assert moved_to_jsonable([1, 2]) == ["patched:1", "patched:2"]


def test_facade_defines_no_second_implementation_owner() -> None:
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in _tree(FACADE).body
    )
