"""Inventory guard for the split cognitive runtime test suite."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

import test_cognitive_runtime as suite

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "tests" / "test_cognitive_runtime.py"
CASE_FILES = {
    "cognitive_runtime_cases_change_budget.py",
    "cognitive_runtime_cases_claim_verification.py",
    "cognitive_runtime_cases_common.py",
    "cognitive_runtime_cases_completion.py",
    "cognitive_runtime_cases_discovery.py",
    "cognitive_runtime_cases_evidence.py",
    "cognitive_runtime_cases_lifecycle.py",
    "cognitive_runtime_cases_profiles.py",
    "cognitive_runtime_cases_reasoning.py",
    "cognitive_runtime_cases_recovery.py",
    "cognitive_runtime_cases_research.py",
    "cognitive_runtime_cases_semantic_relations.py",
    "cognitive_runtime_cases_semantic_rules.py",
    "cognitive_runtime_cases_terminal.py",
}
TEST_CASES = (
    "CognitiveRuntimeTests",
    "WaiverAndRetractionTests",
    "StrictnessProfileTests",
    "ToolCallBudgetTests",
    "ResearchReframeTests",
    "ConciseChangeBudgetTests",
    "ClaimVerificationEngineTests",
)
STANDALONE_TESTS = (
    "test_read_only_goal_naming_change_flavored_paths_stays_read_only",
    "test_move_and_copy_goals_request_change",
    "test_host_hook_diff_receipt_establishes_change_when_observed",
)
LEGACY_TEST_COUNT = 91
LEGACY_NAME_DIGEST = "f838833331e9670400fb384d3b6f6f20585a0a084adcfbeaebe5140cdd9fe37d"
HELPER_OWNERS = {
    "FakeClock": "cognitive_runtime_cases_common.py",
    "_complete": "cognitive_runtime_cases_change_budget.py",
    "_ready_code_task": "cognitive_runtime_cases_lifecycle.py",
    "_session_with_change": "cognitive_runtime_cases_change_budget.py",
    "_start_code_task": "cognitive_runtime_cases_lifecycle.py",
    "_web_observation": "cognitive_runtime_cases_profiles.py",
}


def _case_paths() -> list[Path]:
    return sorted((ROOT / "tests").glob("cognitive_runtime_cases_*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _defined_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(_tree(path))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _case_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("cognitive_runtime_cases_")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cognitive_runtime_cases_")
        ):
            imports.add(node.module)
    return imports


def test_runtime_test_files_have_exact_membership_and_stay_below_cap() -> None:
    paths = _case_paths()
    counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in [FACADE, *paths, Path(__file__)]
    }

    assert {path.name for path in paths} == CASE_FILES
    assert counts[FACADE.name] <= 250
    assert all(count <= 500 for count in counts.values()), counts


def test_original_pytest_collection_names_and_count_are_exact() -> None:
    loader = unittest.defaultTestLoader
    names = {
        f"{case_name}::{method_name}"
        for case_name in TEST_CASES
        for method_name in loader.getTestCaseNames(getattr(suite, case_name))
    }
    names.update(STANDALONE_TESTS)
    digest = hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()

    assert len(names) == LEGACY_TEST_COUNT
    assert digest == LEGACY_NAME_DIGEST


def test_every_test_body_has_one_case_module_owner() -> None:
    owners: dict[str, list[str]] = {}
    for path in _case_paths():
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                owners.setdefault(node.name, []).append(path.name)

    inherited_names = {
        method_name
        for case_name in TEST_CASES
        for method_name in unittest.defaultTestLoader.getTestCaseNames(getattr(suite, case_name))
    }
    assert set(owners) == inherited_names
    assert all(len(paths) == 1 for paths in owners.values()), owners


def test_helpers_have_one_coherent_owner_and_remain_reachable() -> None:
    definitions = {path.name: _defined_names(path) for path in _case_paths()}
    for helper, expected_owner in HELPER_OWNERS.items():
        actual = [name for name, names in definitions.items() if helper in names]
        assert actual == [expected_owner]

    assert suite.CognitiveRuntimeTests._start_code_task
    assert suite.CognitiveRuntimeTests._ready_code_task
    assert suite.StrictnessProfileTests._web_observation
    assert suite.ConciseChangeBudgetTests._session_with_change
    assert suite.ConciseChangeBudgetTests._complete
    assert suite.FakeClock


def test_case_module_import_graph_is_acyclic() -> None:
    paths = _case_paths()
    graph = {
        path.stem: {
            dependency
            for dependency in _case_imports(path)
            if dependency in {item.stem for item in paths}
        }
        for path in paths
    }

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"runtime test import cycle through {name}"
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


def test_facade_contains_no_moved_test_implementation() -> None:
    tree = _tree(FACADE)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert set(classes) == set(TEST_CASES)
    for node in classes.values():
        assert len(node.body) == 1
        assert isinstance(node.body[0], ast.Pass)

    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions == set(STANDALONE_TESTS)
