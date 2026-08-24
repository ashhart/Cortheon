"""Architecture contract for the mechanically split cognitive benchmark tests."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cognitive_benchmark_cases_coverage as coverage
import cognitive_benchmark_cases_delivery as delivery
import cognitive_benchmark_cases_execution as execution
import cognitive_benchmark_cases_measurement as measurement
import cognitive_benchmark_cases_reasoning as reasoning
import test_cognitive_benchmark as facade

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "tests" / "test_cognitive_benchmark.py"
CASE_FILES = (
    ROOT / "tests" / "cognitive_benchmark_cases_execution.py",
    ROOT / "tests" / "cognitive_benchmark_cases_delivery.py",
    ROOT / "tests" / "cognitive_benchmark_cases_reasoning.py",
    ROOT / "tests" / "cognitive_benchmark_cases_coverage.py",
    ROOT / "tests" / "cognitive_benchmark_cases_measurement.py",
)
CASE_MODULES = {
    "cognitive_benchmark_cases_execution.py": execution,
    "cognitive_benchmark_cases_delivery.py": delivery,
    "cognitive_benchmark_cases_reasoning.py": reasoning,
    "cognitive_benchmark_cases_coverage.py": coverage,
    "cognitive_benchmark_cases_measurement.py": measurement,
}
AST_BODY_DIGEST = "c6a29d24b50691f5b9943cff98c4d871679ba0630a8256a8a8961106c2148424"
COLLECTION_DIGEST = "ee0df0ec08c7f6a08e7b215e0296c0784f8de9183afa5dae071099cbd05a0c3a"
ORIGINAL_IMPORT_DIGEST = "2137ae8ec748f9d7c5ab59ebe79e5f95a2353c5c5d7d03f1cf1fd65fb18bece1"
HELPERS = {"_pi_end_event", "_pi_start_event", "_process_capture", "_scaling_report"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(path: Path) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    return [
        node
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_ast_dump(node: ast.AST) -> str:
    try:
        dumped = ast.dump(node, include_attributes=False, **{"show_empty": True})
    except TypeError:
        dumped = ast.dump(node, include_attributes=False)
    return dumped.replace(", type_params=[]", "")


def test_split_membership_and_every_authored_file_stay_below_cap() -> None:
    discovered = tuple(sorted((ROOT / "tests").glob("cognitive_benchmark_cases_*.py")))
    assert {path.name for path in discovered} == {path.name for path in CASE_FILES}

    paths = [FACADE, *CASE_FILES, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in paths}
    assert all(count <= 500 for count in counts.values()), counts


def test_every_original_definition_has_one_owner_and_exact_ast_body() -> None:
    definitions = [node for path in CASE_FILES for node in _definitions(path)]
    owners: dict[str, list[str]] = {}
    for path in CASE_FILES:
        for node in _definitions(path):
            owners.setdefault(node.name, []).append(path.name)

    bodies = [(node.name, _stable_ast_dump(node)) for node in definitions]
    assert len(definitions) == 68
    assert all(len(paths) == 1 for paths in owners.values()), owners
    assert _digest(bodies) == AST_BODY_DIGEST


def test_facade_is_implementation_free_and_reexports_every_definition() -> None:
    assert _definitions(FACADE) == []
    owned_names = {node.name for path in CASE_FILES for node in _definitions(path)}
    imported_names = {
        alias.asname or alias.name
        for node in _tree(FACADE).body
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("cognitive_benchmark_cases_")
        for alias in node.names
    }
    assert imported_names == owned_names
    assert {name for name in owned_names if name.startswith("test_")} == {
        name for name in vars(facade) if name.startswith("test_")
    }


def test_original_import_alias_seams_are_preserved() -> None:
    original_imports = [
        _stable_ast_dump(node)
        for node in _tree(FACADE).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cognitive_benchmark_cases_")
        )
    ]
    assert len(original_imports) == 9
    assert _digest(original_imports) == ORIGINAL_IMPORT_DIGEST

    for helper in HELPERS:
        owner = next(
            module
            for filename, module in CASE_MODULES.items()
            if helper in {node.name for node in _definitions(ROOT / "tests" / filename)}
        )
        assert getattr(facade, helper) is getattr(owner, helper)
        assert getattr(facade, helper).__module__ == "test_cognitive_benchmark"


def test_case_module_import_graph_is_acyclic() -> None:
    module_names = {path.stem for path in CASE_FILES}
    graph = {
        path.stem: {
            node.module
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.ImportFrom) and node.module in module_names
        }
        for path in CASE_FILES
    }

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"cognitive benchmark test cycle through {name}"
        if name in visited:
            return
        active.add(name)
        for dependency in graph[name]:
            visit(dependency, active, visited)
        active.remove(name)
        visited.add(name)

    visited: set[str] = set()
    for module in graph:
        visit(module, set(), visited)


def test_pytest_collection_node_ids_and_order_are_exact() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(FACADE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    node_ids = [
        line.replace(str(ROOT) + "/", "")
        for line in result.stdout.splitlines()
        if line.startswith((str(FACADE), "tests/test_cognitive_benchmark.py::"))
    ]
    assert len(node_ids) == 64
    assert hashlib.sha256(("\n".join(node_ids) + "\n").encode()).hexdigest() == (COLLECTION_DIGEST)
