"""Inventory guard for the split cognitive adversarial test suite."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

import cognitive_adversarial_cases_common as common
import test_cognitive_adversarial as suite

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "tests" / "test_cognitive_adversarial.py"
CASE_FILES = {
    "cognitive_adversarial_cases_atomicity.py",
    "cognitive_adversarial_cases_common.py",
    "cognitive_adversarial_cases_completion_evidence.py",
    "cognitive_adversarial_cases_completion_hostile.py",
    "cognitive_adversarial_cases_completion_integrity.py",
    "cognitive_adversarial_cases_concurrency.py",
    "cognitive_adversarial_cases_host_receipts.py",
    "cognitive_adversarial_cases_protocol.py",
    "cognitive_adversarial_cases_randomized.py",
}
TEST_CASES = (
    "CognitiveAtomicityTests",
    "CognitiveCompletionContractTests",
    "CognitiveHostReceiptHardeningTests",
    "CognitiveConcurrencyTests",
    "CognitiveProtocolFuzzTests",
    "CognitiveRandomizedStateTests",
)
LEGACY_TEST_COUNT = 25
LEGACY_NAME_DIGEST = "18f7a49b48ad262bffc4253035c8ef8cb78f539468107ce5ba3620bbc9abb162"
HELPER_OWNERS = {
    "_code_session": "cognitive_adversarial_cases_common.py",
    "hypothesis": "cognitive_adversarial_cases_common.py",
}


def _case_paths() -> list[Path]:
    return sorted((ROOT / "tests").glob("cognitive_adversarial_cases_*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _case_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("cognitive_adversarial_cases_")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cognitive_adversarial_cases_")
        ):
            imports.add(node.module)
    return imports


def test_adversarial_files_have_exact_membership_and_stay_below_cap() -> None:
    paths = _case_paths()
    authored = [FACADE, *paths, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in paths} == CASE_FILES
    assert counts[FACADE.name] <= 100
    assert all(count <= 500 for count in counts.values()), counts


def test_original_pytest_collection_names_and_count_are_exact() -> None:
    names = {
        f"{case_name}::{method_name}"
        for case_name in TEST_CASES
        for method_name in unittest.defaultTestLoader.getTestCaseNames(getattr(suite, case_name))
    }
    digest = hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()

    assert len(names) == LEGACY_TEST_COUNT
    assert digest == LEGACY_NAME_DIGEST


def test_every_hostile_test_body_and_helper_has_one_owner() -> None:
    test_owners: dict[str, list[str]] = {}
    helper_owners: dict[str, list[str]] = {}
    for path in _case_paths():
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    test_owners.setdefault(node.name, []).append(path.name)
                elif node.name in HELPER_OWNERS:
                    helper_owners.setdefault(node.name, []).append(path.name)

    inherited_tests = {
        name
        for case_name in TEST_CASES
        for name in unittest.defaultTestLoader.getTestCaseNames(getattr(suite, case_name))
    }
    assert set(test_owners) == inherited_tests
    assert all(len(paths) == 1 for paths in test_owners.values()), test_owners
    for helper, expected_owner in HELPER_OWNERS.items():
        assert helper_owners[helper] == [expected_owner]


def test_public_hypothesis_helper_and_completion_fixture_remain_reachable() -> None:
    assert suite.hypothesis is common.hypothesis
    assert (
        suite.CognitiveCompletionContractTests._code_session is common.CompletionCase._code_session
    )


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
        assert name not in active, f"adversarial test import cycle through {name}"
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


def test_facade_contains_only_empty_test_wrappers() -> None:
    tree = _tree(FACADE)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert set(classes) == set(TEST_CASES)
    for node in classes.values():
        assert len(node.body) == 1
        assert isinstance(node.body[0], ast.Pass)
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
