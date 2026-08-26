from __future__ import annotations

import ast
from pathlib import Path

import cortheon.operator_lift as lift

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/cortheon/operator_lift"
MEMBERS = {
    "__init__.py",
    "case_bank.py",
    "case_builders.py",
    "cases_derivation.py",
    "cases_discrimination.py",
    "cases_hypothesis.py",
    "cases_revision.py",
    "cases_stopping.py",
    "contrasts.py",
    "execution_models.py",
    "execution_release.py",
    "execution_release_verify.py",
    "execution_report.py",
    "execution_runner.py",
    "execution_schedule.py",
    "execution_storage.py",
    "cli.py",
    "models.py",
    "oracles.py",
    "preregister.py",
    "replay.py",
    "replay_responder.py",
    "report.py",
    "sealing.py",
    "statistics.py",
}
EXECUTION_MEMBERS = {
    "cli.py",
    "execution_models.py",
    "execution_release.py",
    "execution_release_verify.py",
    "execution_report.py",
    "execution_runner.py",
    "execution_schedule.py",
    "execution_storage.py",
}
PROHIBITED = {
    "cortheon.benchmark_core",
    "cortheon.qualification_core",
    "cortheon.pi_core",
    "cortheon.opencode_core",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _operator_lift_imports(path: Path) -> set[str]:
    dependencies: set[str] = set()
    prefix = "cortheon.operator_lift."
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
            dependencies.add(node.module.removeprefix(prefix).split(".", 1)[0] + ".py")
    return dependencies


def test_repository_only_package_has_explicit_membership_and_line_cap() -> None:
    assert {path.name for path in SOURCE.glob("*.py")} == MEMBERS
    authored = [*(SOURCE / name for name in MEMBERS), *ROOT.glob("tests/test_operator_lift_*.py")]
    assert authored
    for path in authored:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_instrument_does_not_couple_to_moving_execution_or_adapter_internals() -> None:
    for path in (SOURCE / name for name in MEMBERS - EXECUTION_MEMBERS):
        imports = {
            node.module
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(
            imported == prohibited or imported.startswith(prohibited + ".")
            for imported in imports
            for prohibited in PROHIBITED
        ), path


def test_execution_lane_uses_generic_evaluator_and_never_native_adapter_modules() -> None:
    for path in (SOURCE / name for name in EXECUTION_MEMBERS):
        imports = {
            node.module
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(
            imported.startswith("cortheon.pi_core") or imported.startswith("cortheon.opencode_core")
            for imported in imports
        ), path


def test_internal_import_graph_is_acyclic() -> None:
    graph = {name: _operator_lift_imports(SOURCE / name) & MEMBERS for name in MEMBERS}

    def visit(name: str, trail: tuple[str, ...]) -> None:
        assert name not in trail, " -> ".join((*trail, name))
        for dependency in graph[name]:
            visit(dependency, (*trail, name))

    for member in MEMBERS:
        visit(member, ())


def test_public_surface_is_small_and_makes_no_parity_claim() -> None:
    assert lift.__all__ == [
        "OPERATORS",
        "ConditionBinding",
        "LiftCase",
        "LiftManifest",
        "LiftSubmission",
        "LiftThresholds",
        "OracleResult",
        "build_lift_report",
        "build_manifest",
        "design_sha256",
        "development_cases",
        "public_case",
        "score_and_pair",
    ]
    source = (SOURCE / "__init__.py").read_text(encoding="utf-8").casefold()
    assert "does not make" in source
    assert "frontier-parity claim" in source
