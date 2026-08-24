"""Architecture contracts for the split GitHub connector."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import typing
from pathlib import Path
from typing import Any, cast

import pytest

import cortheon.artifacts as artifacts
import cortheon.connectors.github as github
import cortheon.engine as engine
import cortheon.research as research
import cortheon.research_core.engine as research_engine

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "src" / "cortheon" / "connectors" / "github.py"
CORE = ROOT / "src" / "cortheon" / "connectors" / "github_core"
CORE_FILES = {
    "__init__.py",
    "_compat.py",
    "artifacts.py",
    "client.py",
    "constants.py",
    "normalization.py",
    "transport.py",
}
ORIGINAL_DEFINITIONS = {
    "GitHubConnector",
    "GitHubRepositorySearch",
    "_int_or_none",
    "adjusted_repository_confidence",
    "find_github_url",
    "github_headers",
    "implementation_signals",
    "int_or_zero",
    "language_metadata",
    "normalize_github_url",
    "normalize_readme",
    "parse_owner_repo",
    "readme_metadata",
    "readme_text",
    "repository_artifact_confidence",
    "repository_health_score",
    "repository_item_to_artifact",
    "repository_metadata",
    "repository_query_terms",
    "repository_relevance",
    "repository_search_url",
    "root_content_metadata",
    "safe_get_json",
    "split_metadata_csv",
}
ORIGINAL_CONSTANTS = {"GITHUB_RE", "REPOSITORY_SEARCH_STOPWORDS"}
CONTRACT_DIGEST = "9a7ae5f5e108d73e57891e769c15e2c93265b244daf2677a21fe9d16cbb59c3b"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assigned_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _core_imports(path: Path) -> set[str]:
    prefix = "cortheon.connectors.github_core."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0]
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def _api_contract() -> str:
    contract: list[tuple[str, str, dict[str, str]]] = []
    for name in sorted(ORIGINAL_DEFINITIONS):
        value = getattr(github, name)
        contract.append(
            (
                name,
                str(inspect.signature(value)),
                {key: str(hint) for key, hint in sorted(typing.get_type_hints(value).items())},
            )
        )
        if isinstance(value, type):
            for member_name, member in sorted(vars(value).items()):
                if inspect.isfunction(member):
                    contract.append(
                        (
                            f"{name}.{member_name}",
                            str(inspect.signature(member)),
                            {
                                key: str(hint)
                                for key, hint in sorted(typing.get_type_hints(member).items())
                            },
                        )
                    )
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def test_facade_and_core_membership_stay_below_file_cap() -> None:
    paths = sorted(CORE.glob("*.py"))
    authored = [FACADE, *paths, Path(__file__)]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in paths} == CORE_FILES
    assert counts[FACADE.name] <= 150
    assert all(count <= 500 for count in counts.values()), counts


def test_original_definitions_and_constants_have_one_core_owner() -> None:
    paths = sorted(CORE.glob("*.py"))
    definition_owners = {
        name: [path.name for path in paths if name in _definitions(path)]
        for name in ORIGINAL_DEFINITIONS
    }
    constant_owners = {
        name: [path.name for path in paths if name in _assigned_names(path)]
        for name in ORIGINAL_CONSTANTS
    }
    all_definitions = set().union(*(_definitions(path) for path in paths))

    assert all(len(files) == 1 for files in definition_owners.values()), definition_owners
    assert all(len(files) == 1 for files in constant_owners.values()), constant_owners
    assert all_definitions == ORIGINAL_DEFINITIONS | {"facade"}


def test_core_import_graph_is_acyclic() -> None:
    paths = sorted(CORE.glob("*.py"))
    graph = {path.stem: _core_imports(path) for path in paths}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"GitHub connector import cycle through {name}"
        if name in visited:
            return
        active.add(name)
        for dependency in graph.get(name, set()):
            visit(dependency, active, visited)
        active.remove(name)
        visited.add(name)

    visited: set[str] = set()
    for module in graph:
        visit(module, set(), visited)


def test_signatures_type_hints_and_module_identities_are_stable() -> None:
    assert hashlib.sha256(_api_contract().encode()).hexdigest() == CONTRACT_DIGEST
    for name in ORIGINAL_DEFINITIONS:
        value = getattr(github, name)
        assert value.__module__ == "cortheon.connectors.github"
        if isinstance(value, type):
            for member in vars(value).values():
                if inspect.isfunction(member):
                    assert member.__module__ == "cortheon.connectors.github"


def test_only_facade_imports_github_core_from_repository_source() -> None:
    core_paths = set(CORE.glob("*.py"))
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path not in core_paths
        and any(
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cortheon.connectors.github_core")
            for node in ast.walk(_tree(path))
        )
    }
    assert importers == {"src/cortheon/connectors/github.py"}


def test_direct_consumers_receive_identical_objects() -> None:
    assert artifacts.parse_owner_repo is github.parse_owner_repo
    assert engine.GitHubConnector is github.GitHubConnector
    assert research.GitHubRepositorySearch is github.GitHubRepositorySearch
    assert research_engine.GitHubRepositorySearch is github.GitHubRepositorySearch


class _SearchClient:
    def get_json(self, _url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        del headers
        return {
            "items": [
                {
                    "full_name": "example/project",
                    "html_url": "https://github.com/example/project",
                }
            ]
        }


def test_facade_patches_drive_moved_search_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(github, "repository_item_to_artifact", lambda *_args: sentinel)
    client = cast(Any, _SearchClient())
    artifacts_found, _, _ = github.GitHubRepositorySearch(client).search("demo", 1)
    assert artifacts_found == [sentinel]

    monkeypatch.setattr(github, "normalize_github_url", lambda _url: "patched")
    assert github.find_github_url({"Source": "https://github.com/example/project"}) == "patched"


def test_facade_defines_no_second_implementation_owner() -> None:
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in _tree(FACADE).body
    )
