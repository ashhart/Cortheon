"""Architecture guards for the cognitive MCP decomposition.

The MCP module must not grow back into a god file: the facade stays a thin
re-export surface, every implementation module in ``cognitive_mcp_core`` and
every split test module stays focused, each top-level definition and each
``CortheonMcpServer`` method has exactly one implementation owner, the
dependency graph stays one-directional and acyclic, and the compatibility
surface keeps resolving to identical objects.
"""

from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

import pytest

import cortheon.cognitive_mcp as mcp

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/cognitive_mcp.py"
CORE_DIR = ROOT / "src/cortheon/cognitive_mcp_core"
TEST_MODULES = sorted(
    [
        *Path(__file__).parent.glob("test_cognitive_mcp_*.py"),
        Path(__file__).parent / "cognitive_mcp_helpers.py",
        Path(__file__).parent / "cognitive_mcp_packaging_support.py",
    ]
)
LINE_CAP = 500
PACKAGE = "cortheon.cognitive_mcp_core"

# Every top-level definition of the original 1,000-line cognitive_mcp.py.
ORIGINAL_DEFINITIONS = {
    "CortheonMcpServer",
    "HOST_EVIDENCE_PREFIX",
    "HOST_RECEIPT_OUTCOMES",
    "JSONRPC_INTERNAL_ERROR",
    "JSONRPC_INVALID_PARAMS",
    "JSONRPC_INVALID_REQUEST",
    "JSONRPC_METHOD_NOT_FOUND",
    "JSONRPC_PARSE_ERROR",
    "MAX_MESSAGE_CHARS",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOLS",
    "_coerce_json_array",
    "_compact_payload",
    "_error",
    "_observations_with_host_receipts",
    "_optional_object_list",
    "_optional_string",
    "_optional_string_list",
    "_required_object_list",
    "_required_string",
    "_required_string_list",
    "_result",
    "_tool_result",
    "_write",
    "main",
    "serve",
    "tool_definitions",
}
# Names the original module bound by importing them; callers could always
# reach them through ``cortheon.cognitive_mcp``, so the facade still does.
ORIGINAL_IMPORT_BINDINGS = {
    "Any",
    "CognitiveRuntime",
    "CognitiveRuntimeError",
    "TextIO",
    "__version__",
    "annotations",
    "argparse",
    "contextlib",
    "json",
    "protocol_capabilities",
    "sys",
}
ORIGINAL_SURFACE = ORIGINAL_DEFINITIONS | ORIGINAL_IMPORT_BINDINGS

# The one name the split adds beyond the original surface: the installer for
# the facade patch bridge. The facade deletes it once it has run, so the
# historical attribute set is unchanged - see the guard below and
# tests/test_cognitive_mcp_compat.py for what the bridge preserves.
BRIDGE_DEFINITIONS = {"install_facade_patch_bridge"}

# Methods of the original monolithic CortheonMcpServer.
ORIGINAL_SERVER_METHODS = {
    "__init__",
    "_call_tool",
    "_initialize",
    "_malformed_observe",
    "handle",
}

# Declared layering, lowest first. Each module may only import from strictly
# earlier ones, which makes the graph one-directional and acyclic by
# construction rather than by inspection.
# ``compat`` sits at the bottom: it imports no sibling and is reached only
# from the facade.
LAYERS = ["compat", "protocol", "arguments", "tools", "server", "stdio"]

# The lean packaging allowlist that must ship every one of these modules.
ALLOWLIST_NAME = "COGNITIVE_MCP_CORE_MODULES"


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _core_modules() -> list[Path]:
    modules = sorted(CORE_DIR.glob("*.py"))
    assert modules, "cognitive_mcp_core package must exist"
    return modules


def _implementation_modules() -> list[Path]:
    return [path for path in _core_modules() if path.stem != "__init__"]


def _core_sources() -> list[tuple[str, str]]:
    return [(path.name, path.read_text(encoding="utf-8")) for path in _core_modules()]


def _package_imports(path: Path) -> set[str]:
    """Sibling ``cognitive_mcp_core`` modules this module imports from."""

    edges: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.level == 0, f"{path.name} uses a relative import"
            if node.module.startswith(f"{PACKAGE}."):
                edges.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE}."):
                    edges.add(alias.name.split(".")[-1])
    return edges


def _is_contract_stub(item: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    # A contract stub is a body consisting solely of ``raise NotImplementedError``
    # (with or without parentheses); it owns nothing.
    if len(item.body) != 1 or not isinstance(item.body[0], ast.Raise):
        return False
    exc = item.body[0].exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def _concrete_method_owners(
    sources: list[tuple[str, str]], class_names: set[str]
) -> dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Collect the single concrete owner of each method across ALL modules.

    A second module defining a concrete (non-stub) body for a method that is
    already owned somewhere else fails loudly instead of silently winning.
    """
    owners: dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for module_name, source in sources:
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef) or node.name not in class_names:
                continue
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _is_contract_stub(item):
                    continue
                assert item.name not in owners, (
                    f"{item.name} has concrete owners in both "
                    f"{owners[item.name][0]} and {module_name}"
                )
                owners[item.name] = (module_name, item)
    return owners


def test_facade_is_a_small_stable_surface() -> None:
    assert len(FACADE.read_text(encoding="utf-8").splitlines()) <= 150
    assert not _top_level_definitions(FACADE), "facade must not own definitions"


def test_every_edited_module_stays_under_the_line_cap() -> None:
    for module in [FACADE, *_core_modules(), *TEST_MODULES]:
        line_count = len(module.read_text(encoding="utf-8").splitlines())
        assert line_count <= LINE_CAP, f"{module.name} has {line_count} lines"


def test_the_pre_split_god_files_are_gone() -> None:
    assert not (ROOT / "tests/test_cognitive_mcp.py").exists()
    assert len(_implementation_modules()) >= 5
    assert len(TEST_MODULES) >= 5


def test_single_owner_for_top_level_definitions() -> None:
    owners: dict[str, Path] = {}
    for path in [FACADE, *_core_modules()]:
        for name in _top_level_definitions(path):
            assert name not in owners, (
                f"{name} is defined in both {owners[name].name} and {path.name}"
            )
            owners[name] = path


def test_every_original_definition_is_owned_exactly_once() -> None:
    corpus: set[str] = set()
    for path in _core_modules():
        corpus |= _top_level_definitions(path)
    expected = ORIGINAL_DEFINITIONS | BRIDGE_DEFINITIONS
    assert corpus == expected, (
        f"missing: {sorted(expected - corpus)}, unexpected: {sorted(corpus - expected)}"
    )


def test_every_original_method_has_one_implementation_owner() -> None:
    owners = _concrete_method_owners(_core_sources(), {"CortheonMcpServer"})
    assert set(owners) == ORIGINAL_SERVER_METHODS
    assert {module for module, _ in owners.values()} == {"server.py"}


def test_duplicate_concrete_owner_in_a_second_module_is_rejected() -> None:
    # Mutation-style guard: a per-file scan merged with ``dict.update`` would
    # silently overwrite a duplicate concrete body defined in another module.
    # Stubs raising NotImplementedError only must never count as owners.
    module_a = "class CortheonMcpServer:\n    def handle(self, message):\n        return 'a'\n"
    stub_only = (
        "class CortheonMcpServer:\n    def handle(self, message):\n"
        "        raise NotImplementedError()\n"
    )
    duplicate = "class CortheonMcpServer:\n    def handle(self, message):\n        return 'dup'\n"
    owners = _concrete_method_owners(
        [("server.py", module_a), ("stdio.py", stub_only)], {"CortheonMcpServer"}
    )
    assert owners["handle"][0] == "server.py"
    with pytest.raises(AssertionError, match="concrete owners in both"):
        _concrete_method_owners(
            [("server.py", module_a), ("stdio.py", duplicate)], {"CortheonMcpServer"}
        )


def test_facade_exposes_exactly_the_original_surface() -> None:
    attrs = {name for name in vars(mcp) if not name.startswith("__")}
    attrs |= {"annotations", "__version__"} & set(vars(mcp))
    assert attrs == ORIGINAL_SURFACE, (
        f"missing: {sorted(ORIGINAL_SURFACE - attrs)}, extra: {sorted(attrs - ORIGINAL_SURFACE)}"
    )


def test_facade_keeps_the_original_star_import_surface() -> None:
    # The pre-split module declared no ``__all__``; adding one here would
    # silently narrow ``from cortheon.cognitive_mcp import *``.
    assert not hasattr(mcp, "__all__")
    exported = {name for name in vars(mcp) if not name.startswith("_")}
    assert exported == {name for name in ORIGINAL_SURFACE if not name.startswith("_")}


def test_facade_reexports_are_identical_to_core_definitions() -> None:
    from cortheon import __version__
    from cortheon.cognitive_mcp_core import arguments, protocol, server, stdio, tools
    from cortheon.cognitive_protocol import protocol_capabilities
    from cortheon.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeError

    identity_pairs = [
        (mcp.PROTOCOL_VERSION, protocol.PROTOCOL_VERSION),
        (mcp.SUPPORTED_PROTOCOLS, protocol.SUPPORTED_PROTOCOLS),
        (mcp.JSONRPC_PARSE_ERROR, protocol.JSONRPC_PARSE_ERROR),
        (mcp.JSONRPC_INVALID_REQUEST, protocol.JSONRPC_INVALID_REQUEST),
        (mcp.JSONRPC_METHOD_NOT_FOUND, protocol.JSONRPC_METHOD_NOT_FOUND),
        (mcp.JSONRPC_INVALID_PARAMS, protocol.JSONRPC_INVALID_PARAMS),
        (mcp.JSONRPC_INTERNAL_ERROR, protocol.JSONRPC_INTERNAL_ERROR),
        (mcp.MAX_MESSAGE_CHARS, protocol.MAX_MESSAGE_CHARS),
        (mcp.HOST_EVIDENCE_PREFIX, protocol.HOST_EVIDENCE_PREFIX),
        (mcp.HOST_RECEIPT_OUTCOMES, protocol.HOST_RECEIPT_OUTCOMES),
        (mcp._result, protocol._result),
        (mcp._error, protocol._error),
        (mcp._tool_result, protocol._tool_result),
        (mcp._required_string, arguments._required_string),
        (mcp._optional_string, arguments._optional_string),
        (mcp._optional_string_list, arguments._optional_string_list),
        (mcp._required_string_list, arguments._required_string_list),
        (mcp._required_object_list, arguments._required_object_list),
        (mcp._optional_object_list, arguments._optional_object_list),
        (mcp._coerce_json_array, arguments._coerce_json_array),
        (mcp._observations_with_host_receipts, arguments._observations_with_host_receipts),
        (mcp.tool_definitions, tools.tool_definitions),
        (mcp.CortheonMcpServer, server.CortheonMcpServer),
        (mcp._compact_payload, server._compact_payload),
        (mcp.serve, stdio.serve),
        (mcp._write, stdio._write),
        (mcp.main, stdio.main),
        (mcp.CognitiveRuntime, CognitiveRuntime),
        (mcp.CognitiveRuntimeError, CognitiveRuntimeError),
        (mcp.protocol_capabilities, protocol_capabilities),
        (mcp.__version__, __version__),
    ]
    assert len(identity_pairs) == len(ORIGINAL_DEFINITIONS) + 4
    for facade_object, core_object in identity_pairs:
        assert facade_object is core_object


def test_the_patch_bridge_installs_without_widening_the_surface() -> None:
    """The bridge is installed through a name the facade then deletes.

    The pre-split module resolved its lookups through its own globals, so
    patching it changed what ran. The facade keeps that seam by taking a
    module type whose assignments reach the owning modules; the installer's
    name is deleted afterwards so ``vars()``, ``dir()``, and star imports
    stay exactly what they were. What the seam actually preserves is proved
    end to end in tests/test_cognitive_mcp_compat.py.
    """

    source = FACADE.read_text(encoding="utf-8")
    assert "install_facade_patch_bridge(sys.modules[__name__])" in source
    assert "del install_facade_patch_bridge" in source

    assert "install_facade_patch_bridge" not in vars(mcp)
    assert isinstance(mcp, types.ModuleType)
    assert type(mcp) is not types.ModuleType, "the facade must carry the bridge type"
    # dir() must still be exactly the module dict: the bridge type adds
    # no class attribute that a star import or completion would surface.
    assert dir(mcp) == sorted(vars(mcp))


def test_dependency_graph_is_one_directional_and_acyclic() -> None:
    assert {path.stem for path in _implementation_modules()} == set(LAYERS)
    rank = {name: index for index, name in enumerate(LAYERS)}
    for path in _implementation_modules():
        for target in _package_imports(path):
            assert rank[target] < rank[path.stem], (
                f"{path.stem} imports {target}, which is not a lower layer"
            )
    # Nothing inside the package may import the facade back.
    for name, source in _core_sources():
        assert "cortheon.cognitive_mcp " not in source
        assert "from cortheon import cognitive_mcp" not in source, name


def test_no_star_imports_anywhere_in_the_split() -> None:
    for path in [FACADE, *_core_modules(), *TEST_MODULES]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name != "*" for alias in node.names), path.name


def test_every_core_module_is_importable_and_reachable_from_the_facade() -> None:
    for module_path in _implementation_modules():
        module = importlib.import_module(f"{PACKAGE}.{module_path.stem}")
        assert module.__name__.endswith(module_path.stem)

    reached: set[str] = set()
    frontier = _package_imports(FACADE)
    while frontier:
        name = frontier.pop()
        if name in reached:
            continue
        reached.add(name)
        frontier |= _package_imports(CORE_DIR / f"{name}.py")
    assert reached == {path.stem for path in _implementation_modules()}, sorted(reached)

    loaded = {
        value.__name__.split(".")[-1]
        for value in vars(importlib.import_module(PACKAGE)).values()
        if isinstance(value, types.ModuleType) and value.__name__.startswith(PACKAGE)
    }
    assert loaded <= {path.stem for path in _implementation_modules()}


def test_lean_packaging_allowlist_ships_every_core_module() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    node = next(
        assignment
        for assignment in ast.parse(setup_source).body
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == ALLOWLIST_NAME
            for target in assignment.targets
        )
    )
    # ``frozenset({...})`` is a call, so evaluate the literal set it wraps.
    call = node.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "frozenset", f"{ALLOWLIST_NAME} must stay a frozenset"
    allowlisted = set(ast.literal_eval(call.args[0]))
    assert allowlisted == {path.stem for path in _core_modules()}
    assert "recursive-include src/cortheon/cognitive_mcp_core *.py" in (
        ROOT / "MANIFEST.in"
    ).read_text(encoding="utf-8")
