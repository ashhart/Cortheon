from __future__ import annotations

import ast
from pathlib import Path

import cortheon.power_analysis as power_analysis

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/cortheon/power_analysis"
MEMBERS = {
    "__init__.py",
    "models.py",
    "pilot.py",
    "planner.py",
    "report.py",
    "sealing.py",
    "sensitivity.py",
    "sequential.py",
    "statistics.py",
    "taxonomy.py",
    "validation.py",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_package_is_repository_only_focused_and_below_the_line_cap() -> None:
    assert {path.name for path in SOURCE.glob("*.py")} == MEMBERS
    authored = [*(SOURCE / name for name in MEMBERS), *ROOT.glob("tests/test_power_analysis_*.py")]
    assert authored
    for path in authored:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_package_does_not_import_runtime_or_measurement_implementations() -> None:
    forbidden = (
        "cortheon.benchmark",
        "cortheon.qualification",
        "cortheon.parity",
        "cortheon.pi_core",
        "cortheon.opencode",
    )
    for name in MEMBERS:
        imports = {
            node.module
            for node in ast.walk(_tree(SOURCE / name))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(module.startswith(forbidden) for module in imports)


def test_internal_import_graph_is_acyclic() -> None:
    prefix = "cortheon.power_analysis."
    graph: dict[str, set[str]] = {}
    for name in MEMBERS:
        graph[name] = {
            node.module.removeprefix(prefix).split(".", 1)[0] + ".py"
            for node in ast.walk(_tree(SOURCE / name))
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
        } & MEMBERS

    def visit(name: str, trail: tuple[str, ...]) -> None:
        assert name not in trail, " -> ".join((*trail, name))
        for dependency in graph[name]:
            visit(dependency, (*trail, name))

    for name in MEMBERS:
        visit(name, ())


def test_public_surface_is_exact() -> None:
    assert power_analysis.__all__ == [
        "CampaignManifest",
        "ObservedContrast",
        "PilotArtifact",
        "PilotPair",
        "ResourceAssumptions",
        "build_power_plan",
        "build_power_report",
        "power_plan_sha256",
        "sensitivity_rows",
        "sequential_decision",
        "validate_campaign_manifest",
    ]
