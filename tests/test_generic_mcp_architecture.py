"""Size, ownership, and acyclic-boundary pins for generic MCP evaluation."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
CORE = ROOT / "src/cortheon/benchmark_core"
GENERIC = tuple(sorted(CORE.glob("generic_mcp_*.py")))
INTEGRATION = (
    CORE / "cli.py",
    CORE / "cli_generic.py",
    CORE / "execution_provenance.py",
    CORE / "models.py",
    CORE / "outcomes.py",
    CORE / "runner_hosts.py",
    CORE / "runner_local.py",
    CORE / "run_support.py",
    CORE / "scaling_identity.py",
    CORE / "transport_outcomes.py",
)


def _imports(path: Path) -> set[str]:
    selected: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        prefix = "cortheon.benchmark_core.generic_mcp_"
        if node.module.startswith(prefix):
            selected.add(node.module.rsplit(".", 1)[-1])
    return selected


def test_generic_evaluator_and_touched_integration_files_stay_below_cap() -> None:
    assert GENERIC
    oversized = {
        path.relative_to(ROOT).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in (*GENERIC, *INTEGRATION)
        if len(path.read_text(encoding="utf-8").splitlines()) > 500
    }
    assert oversized == {}


def test_generic_evaluator_static_import_graph_is_acyclic() -> None:
    names = {path.stem for path in GENERIC}
    graph = {path.stem: _imports(path) & names for path in GENERIC}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise AssertionError(f"generic MCP import cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for module in graph:
        visit(module)
