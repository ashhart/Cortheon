"""Inventory and ownership guards for the split cognitive HTTP test suite."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import cognitive_http_cases_common as common
import test_cognitive_http as suite

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
FACADE = TESTS / "test_cognitive_http.py"
CASE_FILES = {
    "cognitive_http_cases_common.py",
    "cognitive_http_cases_hooks.py",
    "cognitive_http_cases_reasoning.py",
    "cognitive_http_cases_security.py",
    "cognitive_http_cases_status.py",
    "cognitive_http_cases_transport.py",
}
LEGACY_TEST_COUNT = 15
LEGACY_NAME_DIGEST = "c5e20bb33051ff17c3c37670b76dfba88367d12e50743439e29524f41154e851"
HELPER_OWNERS = {
    "post": "cognitive_http_cases_common.py",
    "running_server": "cognitive_http_cases_common.py",
}


def _case_paths() -> list[Path]:
    return sorted(TESTS.glob("cognitive_http_cases_*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _case_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name for alias in node.names if alias.name.startswith("cognitive_http_cases_")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cognitive_http_cases_")
        ):
            imports.add(node.module)
    return imports


def _owned_functions() -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for path in _case_paths():
        for node in _tree(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owners.setdefault(node.name, []).append(path.name)
    return owners


def test_http_case_files_have_exact_membership_and_stay_below_cap() -> None:
    paths = _case_paths()
    authored = [FACADE, *paths, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in paths} == CASE_FILES
    assert counts[FACADE.name] <= 100
    assert all(count <= 500 for count in counts.values()), counts


def test_original_pytest_function_names_and_count_are_exact() -> None:
    names = {
        name
        for name, value in vars(suite).items()
        if name.startswith("test_") and inspect.isfunction(value)
    }
    digest = hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()

    assert len(names) == LEGACY_TEST_COUNT
    assert digest == LEGACY_NAME_DIGEST


def test_every_test_body_and_helper_has_one_owner() -> None:
    owners = _owned_functions()
    tests = {
        name
        for name, value in vars(suite).items()
        if name.startswith("test_") and inspect.isfunction(value)
    }

    assert {name for name in owners if name.startswith("test_")} == tests
    assert all(len(owners[name]) == 1 for name in tests), owners
    for helper, expected_owner in HELPER_OWNERS.items():
        assert owners[helper] == [expected_owner]


def test_public_helpers_remain_reachable_through_the_facade() -> None:
    assert suite.post is common.post
    assert suite.running_server is common.running_server


def test_case_module_import_graph_is_acyclic() -> None:
    paths = _case_paths()
    names = {path.stem for path in paths}
    graph = {
        path.stem: {dependency for dependency in _case_imports(path) if dependency in names}
        for path in paths
    }

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"HTTP test import cycle through {name}"
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


def test_facade_only_reexports_tests_and_helpers() -> None:
    tree = _tree(FACADE)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )
    imported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported == {
        "post",
        "running_server",
        *{
            name
            for name, value in vars(suite).items()
            if name.startswith("test_") and inspect.isfunction(value)
        },
    }
