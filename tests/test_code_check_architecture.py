"""Architecture contract for source-backed generated-code verification."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest

import cortheon.code_check as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "code_check.py"
CORE = ROOT / "src" / "cortheon" / "code_check_core"
EXPECTED_CONSUMERS = {
    "src/cortheon/engine.py",
}
CORE_OWNERS = {
    "checking.py": {"check_api_usage", "check_keywords"},
    "symbols.py": {
        "called_short_name",
        "callable_param_index",
        "parse_signature_params",
        "param_annotations",
        "composition_bridge",
        "deprecation_index",
        "suggest_replacements",
        "is_legacy_path",
        "known_attribute_names",
    },
    "syntax.py": {
        "extract_code_blocks",
        "apply_deprecation_renames",
        "names_bound_to_package",
        "binding_pairs",
        "attribute_root",
    },
}
SIGNATURES = {
    "ApiUsageFinding": "(package: 'str', attribute: 'str', line: 'int', reason: 'str', kind: 'str' = 'unknown_symbol') -> None",
    "CodeCheckReport": "(package: 'str', parsed: 'bool', checked_calls: 'int', findings: 'list[ApiUsageFinding]', notes: 'list[str]') -> None",
    "extract_code_blocks": "(text: 'str') -> 'list[str]'",
    "check_api_usage": "(code: 'str', package: 'str', symbols: 'list[ApiSymbol]') -> 'CodeCheckReport'",
    "_check_keywords": "(package: 'str', short: 'str', call: 'ast.Call', param_index: 'dict[str, list[tuple[set[str], bool]]]', symbols: 'list[ApiSymbol]') -> 'list[ApiUsageFinding]'",
    "_called_short_name": "(func: 'ast.expr', bound: 'set[str]', known: 'set[str]') -> 'str | None'",
    "_callable_param_index": "(symbols: 'list[ApiSymbol]') -> 'dict[str, list[tuple[set[str], bool]]]'",
    "_parse_signature_params": "(signature: 'str | None') -> 'tuple[set[str], bool] | None'",
    "_param_annotations": "(signature: 'str | None') -> 'dict[str, str]'",
    "composition_bridge": "(short: 'str', keyword_name: 'str', accepters: 'list[str]', symbols: 'list[ApiSymbol]') -> 'str | None'",
    "_deprecation_index": "(symbols: 'list[ApiSymbol]') -> 'tuple[set[str], set[str]]'",
    "suggest_replacements": "(deprecated_name: 'str', live_names: 'set[str]', limit: 'int' = 3) -> 'list[str]'",
    "is_legacy_path": "(qualname: 'str') -> 'bool'",
    "_Renamer": "(mapping: 'dict[str, str]') -> 'None'",
    "apply_deprecation_renames": "(code: 'str', symbols: 'list[ApiSymbol]') -> 'tuple[str, dict[str, str]]'",
    "_known_attribute_names": "(symbols: 'list[ApiSymbol]') -> 'set[str]'",
    "_names_bound_to_package": "(tree: 'ast.AST', package: 'str') -> 'set[str]'",
    "_binding_pairs": "(tree: 'ast.AST') -> 'list[tuple[str, ast.AST]]'",
    "_attribute_root": "(node: 'ast.AST') -> 'str | None'",
}
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+cortheon\.code_check\s+import\b|import\s+cortheon\.code_check\b)",
    re.MULTILINE,
)


def _imports(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        (isinstance(node, ast.ImportFrom) and node.module == module)
        or (isinstance(node, ast.Import) and any(alias.name == module for alias in node.names))
        for node in ast.walk(tree)
    )


def _core_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("cortheon.code_check_core.")
    }


def test_facade_core_and_architecture_test_stay_below_file_limit() -> None:
    for path in [FACADE, *sorted(CORE.glob("*.py")), Path(__file__)]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_core_ownership_is_explicit_and_does_not_reenter_facade() -> None:
    for filename, expected in CORE_OWNERS.items():
        path = CORE / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        actual = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert actual == expected
        assert not _imports(path, "cortheon.code_check")


def test_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"code_check_core cycle through {name}"
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


def test_direct_source_consumers_are_explicit() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and IMPORT_PATTERN.search(path.read_text(encoding="utf-8"))
    }
    assert consumers == EXPECTED_CONSUMERS


def test_code_check_is_repository_only() -> None:
    assert '"code_check"' not in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "src/cortheon/code_check.py" not in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")


def test_signatures_type_hints_and_module_identities_are_stable() -> None:
    for name, expected in SIGNATURES.items():
        value = getattr(facade, name)
        assert str(inspect.signature(value)) == expected
        assert value.__module__ == "cortheon.code_check"
        hinted = value.__init__ if name == "_Renamer" else value
        assert get_type_hints(hinted)


def test_checker_uses_late_bound_facade_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = RuntimeError("patched symbols")

    def patched_known(_symbols: list[Any]) -> set[str]:
        raise marker

    monkeypatch.setattr(facade, "_known_attribute_names", patched_known)
    with pytest.raises(RuntimeError) as raised:
        facade.check_api_usage("import pkg\npkg.call()", "pkg", cast(Any, []))
    assert raised.value is marker


def test_rewriter_uses_late_bound_facade_renamer(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = RuntimeError("patched renamer")

    class PatchedRenamer:
        def __init__(self, _mapping: dict[str, str]) -> None:
            raise marker

    monkeypatch.setattr(facade, "_deprecation_index", lambda _symbols: ({"old"}, {"new"}))
    monkeypatch.setattr(facade, "suggest_replacements", lambda *_args: ["new"])
    monkeypatch.setattr(facade, "_Renamer", PatchedRenamer)
    with pytest.raises(RuntimeError) as raised:
        facade.apply_deprecation_renames("old()", cast(Any, []))
    assert raised.value is marker
