"""Source compaction for built artifacts."""

from __future__ import annotations

import ast
from pathlib import Path


def compact_host_adapter(path: Path) -> None:
    """Remove layout-only bytes from shipped JavaScript and TypeScript."""

    lines: list[str] = []
    in_block_comment = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            in_block_comment = "*/" not in stripped
            continue
        if stripped and not stripped.startswith("//"):
            lines.append(stripped)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_host_source(path: Path) -> None:
    """Remove comments/layout while retaining indentation for later bundling."""

    lines: list[str] = []
    in_block_comment = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            in_block_comment = "*/" not in stripped
            continue
        if stripped and not stripped.startswith("//"):
            lines.append(line.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def drop_typescript_declarations(path: Path) -> None:
    """Drop type-only declarations from the built Pi bundle."""

    kept: list[str] = []
    block_depth = 0
    type_alias = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if block_depth:
            block_depth += line.count("{") - line.count("}")
            continue
        if type_alias:
            type_alias = not stripped.rstrip().endswith(";")
            continue
        if stripped.startswith("interface "):
            block_depth = line.count("{") - line.count("}")
            continue
        if stripped.startswith("type "):
            type_alias = not stripped.rstrip().endswith(";")
            continue
        kept.append(line)
    if block_depth or type_alias:
        raise ValueError("unterminated TypeScript declaration")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


_DEFINITION = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


class _DropLocalAnnotations(ast.NodeTransformer):
    """Remove annotations that Python never exposes at runtime."""

    def __init__(self) -> None:
        self._scopes: list[str] = []

    def _visit_scope(self, node: ast.AST, scope: str) -> ast.AST:
        self._scopes.append(scope)
        rewritten = self.generic_visit(node)
        self._scopes.pop()
        return rewritten

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._visit_scope(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_scope(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_scope(node, "function")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if self._scopes[-1:] == ["function"] and node.value is not None:
            return ast.copy_location(
                ast.Assign(targets=[node.target], value=node.value),
                node,
            )
        return node


def _drop_private_metadata(
    tree: ast.Module,
    *,
    private_module: bool,
    all_callables: bool,
) -> None:
    """Drop metadata outside the supported import surface."""

    if private_module and ast.get_docstring(tree, clean=False) is not None:
        tree.body.pop(0)
    for node in ast.walk(tree):
        if not isinstance(node, _DEFINITION):
            continue
        private = node.name.startswith("_") and not (
            node.name.startswith("__") and node.name.endswith("__")
        )
        if not all_callables and not private:
            continue
        if ast.get_docstring(node, clean=False) is not None:
            node.body.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            arguments = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            for argument in arguments:
                argument.annotation = None


def compact_python_module(
    path: Path,
    *,
    strip_private_metadata: bool = False,
    private_module: bool = False,
    strip_all_callable_metadata: bool = False,
) -> None:
    """Compact one built module while preserving its supported API."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = _DropLocalAnnotations().visit(tree)
    if strip_private_metadata:
        _drop_private_metadata(
            tree,
            private_module=private_module,
            all_callables=strip_all_callable_metadata,
        )
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
