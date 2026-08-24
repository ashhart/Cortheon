from __future__ import annotations

import ast
from pathlib import Path

import cortheon.threat_model as threat_model

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/cortheon/threat_model"
MEMBERS = {
    "__init__.py",
    "catalog.py",
    "catalog_release.py",
    "models.py",
    "promotion.py",
    "report.py",
    "validation.py",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _internal_imports(path: Path) -> set[str]:
    prefix = "cortheon.threat_model."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0] + ".py"
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def test_package_membership_and_authored_files_stay_below_cap() -> None:
    assert {path.name for path in SOURCE.glob("*.py")} == MEMBERS
    authored = [
        *(SOURCE / member for member in MEMBERS),
        *ROOT.glob("tests/test_threat_model_*.py"),
    ]
    assert authored
    for path in authored:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_internal_import_graph_is_acyclic() -> None:
    graph = {member: _internal_imports(SOURCE / member) & MEMBERS for member in MEMBERS}

    def visit(member: str, trail: tuple[str, ...]) -> None:
        assert member not in trail, " -> ".join((*trail, member))
        for dependency in graph[member]:
            visit(dependency, (*trail, member))

    for member in MEMBERS:
        visit(member, ())


def test_repository_gate_does_not_import_runtime_or_measurement_internals() -> None:
    forbidden = (
        "cortheon.qualification",
        "cortheon.benchmark",
        "cortheon.pi_core",
        "cortheon.opencode",
    )
    for member in MEMBERS:
        modules = {
            node.module
            for node in ast.walk(_tree(SOURCE / member))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(module.startswith(forbidden) for module in modules)


def test_public_surface_is_small_and_does_not_claim_security() -> None:
    assert threat_model.__all__ == [
        "THREAT_MANIFEST",
        "ReviewerSignoff",
        "build_report_bytes",
        "evaluate_promotion",
        "report_sha256",
        "validate_threat_model",
    ]
    assert "no claim of deployment security" in (SOURCE / "__init__.py").read_text()
