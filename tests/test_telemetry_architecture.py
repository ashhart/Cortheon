"""Architecture contracts for the split telemetry implementation."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

import cortheon.telemetry as telemetry

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "src" / "cortheon" / "telemetry.py"
CORE = ROOT / "src" / "cortheon" / "telemetry_core"
CORE_FILES = {"__init__.py", "_compat.py", "metrics.py", "outcomes.py"}
ORIGINAL_DEFINITIONS = {
    "ProxyMetrics",
    "_float",
    "_int",
    "_outcome",
    "_tenant_snapshot",
    "_update_tenant_stats",
    "agent_completion_outcome",
    "agent_inconclusive_outcome",
    "decision_outcome",
    "enforcement_outcome",
    "labeled_error_kind",
    "patch_outcome",
    "verification_audit",
}
PUBLIC_DEFINITIONS = ORIGINAL_DEFINITIONS - {
    "_float",
    "_int",
    "_outcome",
    "_tenant_snapshot",
    "_update_tenant_stats",
}
PROXY_METHODS = {"__init__", "_rotate_sink", "observe", "snapshot"}
SIGNATURES = {
    "ProxyMetrics": "(path: 'Path | str | None' = None, *, max_file_bytes: 'int' = 67108864, retained_files: 'int' = 4) -> 'None'",
    "agent_completion_outcome": "(*, required_checks: 'tuple[str, ...]' = ('contract_checked', 'evidence_cited'), passed_checks: 'tuple[str, ...]' = ('contract_checked', 'evidence_cited'), evidence_kinds: 'tuple[str, ...]' = ('tool_observation', 'citation'), evidence_count: 'int' = 2) -> 'dict[str, Any]'",
    "agent_inconclusive_outcome": "(reason: 'str') -> 'dict[str, Any]'",
    "decision_outcome": "(decision: 'dict[str, Any]') -> 'dict[str, Any]'",
    "enforcement_outcome": "(meta: 'dict[str, Any]') -> 'dict[str, Any]'",
    "labeled_error_kind": "(outcome: 'dict[str, Any]', expected_verdict: 'str') -> 'str | None'",
    "patch_outcome": "(verdict: 'str') -> 'dict[str, Any]'",
    "verification_audit": "(outcome: 'dict[str, Any]') -> 'dict[str, Any]'",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _core_imports(path: Path) -> set[str]:
    prefix = "cortheon.telemetry_core."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0]
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def test_facade_and_core_membership_stay_below_file_cap() -> None:
    paths = sorted(CORE.glob("*.py"))
    authored = [FACADE, *paths, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in paths} == CORE_FILES
    assert counts[FACADE.name] <= 100
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
        assert name not in active, f"telemetry import cycle through {name}"
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


def test_public_signatures_and_module_identities_are_stable() -> None:
    for name in ORIGINAL_DEFINITIONS:
        value = getattr(telemetry, name)
        assert value.__module__ == "cortheon.telemetry"
    for name, expected in SIGNATURES.items():
        assert str(inspect.signature(getattr(telemetry, name))) == expected
    for name in PROXY_METHODS:
        assert getattr(telemetry.ProxyMetrics, name).__module__ == "cortheon.telemetry"


def test_outcome_facade_patch_point_drives_moved_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"patched": True}

    def patched_outcome(**_values: Any) -> dict[str, bool]:
        return sentinel

    monkeypatch.setattr(telemetry, "_outcome", patched_outcome)
    assert telemetry.enforcement_outcome({"verdict": "allow"}) is sentinel
    assert telemetry.patch_outcome("allow") is sentinel


def test_metrics_facade_patch_points_drive_moved_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_calls: list[tuple[str, float]] = []

    def patched_audit(_outcome: dict[str, Any]) -> dict[str, Any]:
        return {
            "contract_present": False,
            "supported_verified": False,
            "unsupported_verified_claim": False,
            "evidence_count": 0,
            "missing_checks": [],
            "evidence_kinds": [],
        }

    def patched_tenant(
        _stats: dict[str, dict[str, Any]],
        tenant_id: str,
        **values: Any,
    ) -> None:
        tenant_calls.append((tenant_id, values["total_latency"]))

    monkeypatch.setattr(telemetry, "_float", lambda _value: 17.0)
    monkeypatch.setattr(telemetry, "verification_audit", patched_audit)
    monkeypatch.setattr(telemetry, "_update_tenant_stats", patched_tenant)
    metrics = telemetry.ProxyMetrics()
    metrics.observe({"outcome": {}, "timing_ms": {"request_total": "ignored"}})

    assert metrics.snapshot()["latency_ms"] == {"average": 17.0, "maximum": 17.0}
    assert tenant_calls == [("default", 17.0)]


def test_facade_defines_no_second_implementation_owner() -> None:
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in _tree(FACADE).body
    )
