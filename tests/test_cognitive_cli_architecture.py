"""Structural and compatibility contract for the split cognitive CLI."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import cortheon.cognitive_cli as cli

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/cognitive_cli.py"
CORE = ROOT / "src/cortheon/cognitive_cli_core"
MEMBERS = {
    "__init__.py",
    "conformance.py",
    "diagnostics.py",
    "dispatch.py",
    "operations.py",
    "parser.py",
}
OWNERS = {
    "asset_paths": "operations.py",
    "build_parser": "parser.py",
    "configure": "operations.py",
    "doctor": "diagnostics.py",
    "host_conformance": "conformance.py",
    "install": "operations.py",
    "main": "dispatch.py",
    "mcp": "operations.py",
    "runtime_health": "diagnostics.py",
    "runtime_results": "diagnostics.py",
    "serve": "operations.py",
    "uninstall": "operations.py",
}
PUBLIC_SIGNATURES = {
    "build_parser": "() -> 'argparse.ArgumentParser'",
    "main": "(argv: 'list[str] | None' = None) -> 'int'",
    "_serve": "(args: 'argparse.Namespace') -> 'int'",
    "_mcp": "(args: 'argparse.Namespace') -> 'int'",
    "doctor": (
        "(runtime_url: 'str' = 'http://127.0.0.1:8743', *, token: 'str' = '', "
        "require_runtime: 'bool' = False, hosts: 'list[str] | tuple[str, ...]' = (), "
        "scope: 'str' = 'user', project_dir: 'str | None' = None) -> 'dict[str, Any]'"
    ),
    "_runtime_health": "(url: 'str', *, token: 'str') -> 'dict[str, Any]'",
    "runtime_results": (
        "(runtime_url: 'str' = 'http://127.0.0.1:8743', *, token: 'str' = '') -> 'dict[str, Any]'"
    ),
    "host_conformance": (
        "(runtime_url: 'str' = 'http://127.0.0.1:8743', *, token: 'str' = '', "
        "hosts: 'list[str] | tuple[str, ...]' = (), timeout_seconds: 'float' = 15.0) "
        "-> 'dict[str, Any]'"
    ),
    "_asset_paths": "() -> 'dict[str, str]'",
    "_install": "(args: 'argparse.Namespace') -> 'list[Any]'",
    "_uninstall": "(args: 'argparse.Namespace') -> 'list[Any]'",
    "_configure": "(args: 'argparse.Namespace') -> 'list[Any]'",
}
COMMANDS = (
    "serve",
    "mcp",
    "doctor",
    "conformance",
    "results",
    "install",
    "uninstall",
    "configure",
    "capabilities",
    "paths",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _core_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module:
            prefix = "cortheon.cognitive_cli_core."
            if node.module.startswith(prefix):
                imports.add(node.module.removeprefix(prefix).split(".", 1)[0] + ".py")
    return imports


def test_core_membership_and_line_cap_are_explicit() -> None:
    assert {path.name for path in CORE.glob("*.py")} == MEMBERS
    for path in [FACADE, *(CORE / name for name in MEMBERS)]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_each_core_definition_has_one_owner() -> None:
    observed: dict[str, list[str]] = {name: [] for name in OWNERS}
    for path in (CORE / name for name in MEMBERS):
        for node in _tree(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in observed:
                observed[node.name].append(path.name)
    assert observed == {name: [owner] for name, owner in OWNERS.items()}


def test_core_import_graph_is_acyclic() -> None:
    graph = {name: _core_imports(CORE / name) & MEMBERS for name in MEMBERS}

    def visit(name: str, trail: tuple[str, ...]) -> None:
        assert name not in trail, " -> ".join((*trail, name))
        for dependency in graph[name]:
            visit(dependency, (*trail, name))

    for member in MEMBERS:
        visit(member, ())


def test_public_signatures_and_module_identities_are_stable() -> None:
    assert {
        name: str(inspect.signature(getattr(cli, name))) for name in PUBLIC_SIGNATURES
    } == PUBLIC_SIGNATURES
    assert all(
        getattr(cli, name).__module__ == "cortheon.cognitive_cli" for name in PUBLIC_SIGNATURES
    )


def test_supported_command_surface_is_exact_and_demo_is_not_advertised() -> None:
    parser = cli.build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    choices = subparsers.choices
    assert choices is not None
    assert tuple(choices) == COMMANDS
    assert "demo" not in parser.format_help()


def test_facade_wrappers_retain_late_bound_patch_seams(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "protocol_capabilities", lambda: {"patched": True})
    assert cli.main(["capabilities"]) == 0
    assert '"patched": true' in capsys.readouterr().out

    monkeypatch.setattr(cli, "_asset_paths", lambda: {"patched": "/tmp/asset"})
    assert cli.main(["paths"]) == 0
    assert '"patched": "/tmp/asset"' in capsys.readouterr().out
