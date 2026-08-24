"""Architecture guards for the cognitive runtime decomposition.

The cognitive runtime must not grow back into a god file: the facade stays
small, every implementation module in ``cognitive_core`` stays focused, each
top-level definition has exactly one owner, and the compatibility surface
keeps resolving.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cortheon.cognitive_runtime as cognitive_runtime
from cortheon.cognitive_core.runtime import CognitiveRuntime

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/cognitive_runtime.py"
CORE_DIR = ROOT / "src/cortheon/cognitive_core"


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_facade_stays_small() -> None:
    assert len(FACADE.read_text(encoding="utf-8").splitlines()) <= 250


def test_every_core_module_stays_focused() -> None:
    modules = sorted(CORE_DIR.glob("*.py"))
    assert modules, "cognitive_core package must exist"
    for module in modules:
        line_count = len(module.read_text(encoding="utf-8").splitlines())
        assert line_count <= 500, f"{module.name} has {line_count} lines"


def test_single_owner_for_top_level_definitions() -> None:
    owners: dict[str, Path] = {}
    for path in [FACADE, *sorted(CORE_DIR.glob("*.py"))]:
        for name in _top_level_definitions(path):
            if name == "__all__":
                continue
            assert name not in owners, (
                f"{name} is defined in both {owners[name].name} and {path.name}"
            )
            owners[name] = path


def test_public_compatibility_surface_resolves() -> None:
    from cortheon.cognitive_runtime import (  # noqa: F401
        CognitiveRuntimeError,
        EffortProfile,
        EvidenceRequest,
        Investigation,
        InvestigationNotFound,
        Observation,
        StrictnessProfile,
        _diff_establishes_change,
        _infer_deliverable,
        _requests_change,
    )

    for name in (
        "CognitiveRuntime",
        "CognitiveRuntimeError",
        "InvestigationNotFound",
        "Observation",
        "EvidenceRequest",
        "Hypothesis",
        "EffortProfile",
        "StrictnessProfile",
        "EFFORT_PROFILES",
        "STRICTNESS_PROFILES",
        "_session_graph",
        "_infer_deliverable",
        "_requests_change",
        "_diff_establishes_change",
        "_validate_host_observation_batch",
        "_claim_verification_profiles",
        "_semantic_join_analysis",
    ):
        assert hasattr(cognitive_runtime, name), name

    for method in (
        "start",
        "step",
        "observe",
        "retract",
        "challenge",
        "verify",
        "complete",
        "finish",
        "heartbeat",
        "note_failed_submission",
        "describe_sessions",
        "close_evidence",
    ):
        assert callable(getattr(CognitiveRuntime, method)), method
