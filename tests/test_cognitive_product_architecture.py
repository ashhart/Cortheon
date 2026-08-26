"""Inventory guard for the split cognitive product test suite."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import cognitive_product_cases_opencode_session as opencode_session
import cognitive_product_cases_pi_adapter as pi_adapter
import test_cognitive_product as suite

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "tests" / "test_cognitive_product.py"
CASE_FILES = {
    "cognitive_product_cases_host_install.py",
    "cognitive_product_cases_installer_safety.py",
    "cognitive_product_cases_opencode_completion.py",
    "cognitive_product_cases_opencode_documents.py",
    "cognitive_product_cases_opencode_install.py",
    "cognitive_product_cases_opencode_release.py",
    "cognitive_product_cases_opencode_repair_execution.py",
    "cognitive_product_cases_opencode_repair_guards.py",
    "cognitive_product_cases_opencode_session.py",
    "cognitive_product_cases_pi_adapter.py",
    "cognitive_product_cases_pi_install.py",
    "cognitive_product_cases_runtime_operations.py",
}
HELPER_OWNERS = {
    "_opencode_adapter_corpus": "cognitive_product_cases_opencode_session.py",
    "_pi_module_graph_source": "cognitive_product_cases_pi_adapter.py",
}
LEGACY_NODE_COUNT = 50
LEGACY_NODE_DIGEST = "bdcb4a64547747d59ff35ac9b8f910f13cc7ba3f34d78e44da1d919ee9985450"
PARAMETERIZED_TEST = "test_opencode_bounded_repair_fails_closed_and_preserves_files"
PARAMETER_IDS = ("failing_test", "symlink_target")


def _case_paths() -> list[Path]:
    return sorted((ROOT / "tests").glob("cognitive_product_cases_*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_functions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_product_test_files_have_exact_membership_and_stay_below_cap() -> None:
    paths = _case_paths()
    authored = [FACADE, *paths, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in paths} == CASE_FILES
    assert counts[FACADE.name] <= 100
    assert all(count <= 500 for count in counts.values()), counts


def test_original_pytest_collection_names_and_count_are_exact() -> None:
    function_names = {
        name for name, value in vars(suite).items() if name.startswith("test_") and callable(value)
    }
    nodes = function_names - {PARAMETERIZED_TEST}
    nodes.update(f"{PARAMETERIZED_TEST}[{parameter}]" for parameter in PARAMETER_IDS)
    digest = hashlib.sha256("\n".join(sorted(nodes)).encode("utf-8")).hexdigest()

    assert len(nodes) == LEGACY_NODE_COUNT
    assert digest == LEGACY_NODE_DIGEST


def test_every_test_and_helper_body_has_one_case_module_owner() -> None:
    owners: dict[str, list[str]] = {}
    for path in _case_paths():
        for name in _top_level_functions(path):
            owners.setdefault(name, []).append(path.name)

    facade_tests = {
        name for name, value in vars(suite).items() if name.startswith("test_") and callable(value)
    }
    assert set(owners) == facade_tests | set(HELPER_OWNERS)
    assert all(len(paths) == 1 for paths in owners.values()), owners
    for helper, expected_owner in HELPER_OWNERS.items():
        assert owners[helper] == [expected_owner]


def test_helpers_remain_available_through_the_original_module() -> None:
    assert suite._opencode_adapter_corpus is opencode_session._opencode_adapter_corpus
    assert suite._pi_module_graph_source is pi_adapter._pi_module_graph_source


def test_parameterization_contract_is_unchanged() -> None:
    function = getattr(suite, PARAMETERIZED_TEST)
    marks = [mark for mark in function.pytestmark if mark.name == "parametrize"]

    assert len(marks) == 1
    assert marks[0].args == ("scenario", list(PARAMETER_IDS))


def test_case_modules_do_not_import_each_other() -> None:
    for path in _case_paths():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("cognitive_product_cases_") for alias in node.names
                )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("cognitive_product_cases_")


def test_facade_contains_no_moved_test_implementation() -> None:
    tree = _tree(FACADE)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )
