"""Architecture contracts for the split documentation reader."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import typing
from datetime import UTC, datetime
from pathlib import Path

import pytest

import cortheon.docs_reader as docs_reader
import cortheon.engine as engine

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "src" / "cortheon" / "docs_reader.py"
CORE = ROOT / "src" / "cortheon" / "docs_reader_core"
CORE_FILES = {
    "__init__.py",
    "_compat.py",
    "constants.py",
    "discovery.py",
    "extraction.py",
    "parser.py",
    "reader.py",
}
ORIGINAL_DEFINITIONS = {
    "DocsHtmlParser",
    "DocsSiteReader",
    "detect_version_in_url",
    "error_docs_page",
    "fetch_robots",
    "find_symbol_mention",
    "looks_like_docs",
    "prefer_versioned_link",
    "raw_github_url",
    "resolve_docs_url",
    "runnable_examples",
    "select_guide_links",
    "version_variants",
    "versioned_docs_candidates",
}
ORIGINAL_CONSTANTS = {
    "API_GUIDE_KEYWORDS",
    "CHANGELOG_LABELS",
    "DOCS_LABELS",
    "GUIDE_KEYWORDS",
    "MAX_CHANGELOG_HEAD_CHARS",
    "MAX_CODE_BLOCK_CHARS",
    "MAX_CODE_BLOCKS_PER_PAGE",
    "MAX_PAGE_TEXT_CHARS",
    "MAX_VERSION_PROBES",
}
CONTRACT_DIGEST = "4174ba198d52647747df051ef7c9d09af870057670c1f095a12388d17f1ee77b"


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
    prefix = "cortheon.docs_reader_core."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0]
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def _api_contract() -> str:
    contract: list[tuple[str, str, dict[str, str]]] = []
    for name in sorted(ORIGINAL_DEFINITIONS):
        value = getattr(docs_reader, name)
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
        assert name not in active, f"docs-reader import cycle through {name}"
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


def test_signatures_type_hints_defaults_and_module_identities_are_stable() -> None:
    assert hashlib.sha256(_api_contract().encode()).hexdigest() == CONTRACT_DIGEST
    for name in ORIGINAL_DEFINITIONS:
        value = getattr(docs_reader, name)
        assert value.__module__ == "cortheon.docs_reader"
        if isinstance(value, type):
            for member in vars(value).values():
                if inspect.isfunction(member):
                    assert member.__module__ == "cortheon.docs_reader"
    read_default = (
        inspect.signature(docs_reader.DocsSiteReader.read).parameters["guide_keywords"].default
    )
    select_default = (
        inspect.signature(docs_reader.select_guide_links).parameters["keywords"].default
    )
    assert read_default is docs_reader.GUIDE_KEYWORDS
    assert select_default is docs_reader.GUIDE_KEYWORDS


def test_only_facade_imports_docs_reader_core_from_repository_source() -> None:
    core_paths = set(CORE.glob("*.py"))
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path not in core_paths
        and any(
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("cortheon.docs_reader_core")
            for node in ast.walk(_tree(path))
        )
    }
    assert importers == {"src/cortheon/docs_reader.py"}


def test_direct_engine_consumer_receives_identical_objects() -> None:
    assert engine.DocsSiteReader is docs_reader.DocsSiteReader
    assert engine.find_symbol_mention is docs_reader.find_symbol_mention
    assert engine.API_GUIDE_KEYWORDS is docs_reader.API_GUIDE_KEYWORDS


def test_facade_patches_drive_reader_parser_discovery_and_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = datetime(2024, 1, 2, tzinfo=UTC)
    monkeypatch.setattr(docs_reader, "resolve_docs_url", lambda _urls: None)
    monkeypatch.setattr(docs_reader, "pick_url", lambda _urls, _labels: None)
    monkeypatch.setattr(docs_reader, "utc_now", lambda: fixed)
    report = docs_reader.DocsSiteReader(obey_robots=False).read(
        docs_reader.PackageMetadata(
            "demo", "1", None, None, None, {}, [], [], None, 0, [], "source"
        )
    )
    assert report.generated_at == fixed
    assert report.docs_url is None

    monkeypatch.setattr(
        docs_reader,
        "version_variants",
        lambda _version: [("patched", "exact")],
    )
    assert docs_reader.detect_version_in_url("https://docs.example/patched/", "1") == ("exact")

    monkeypatch.setattr(docs_reader, "normalize_space", lambda _value: "patched")
    parser = docs_reader.DocsHtmlParser("https://docs.example/")
    parser.feed("<h1>Heading</h1>")
    parser.close()
    assert parser.headings == ["patched"]


def test_facade_defines_no_second_implementation_owner() -> None:
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in _tree(FACADE).body
    )
