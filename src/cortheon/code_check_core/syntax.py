"""Code extraction, package binding, and deterministic AST rewrites."""

from __future__ import annotations

import ast
from types import ModuleType
from typing import Any


def extract_code_blocks(bindings: ModuleType, text: str) -> list[str]:
    blocks = [block.strip() for block in bindings.FENCED.findall(text) if block.strip()]
    if blocks:
        return blocks
    try:
        ast.parse(text)
    except SyntaxError:
        return []
    return [text.strip()]


def apply_deprecation_renames(
    bindings: ModuleType,
    code: str,
    symbols: list[Any],
) -> tuple[str, dict[str, str]]:
    deprecated, live = bindings._deprecation_index(symbols)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, {}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in deprecated:
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in deprecated:
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            used.update(alias.name for alias in node.names if alias.name in deprecated)
    mapping: dict[str, str] = {}
    for name in sorted(used):
        replacements = bindings.suggest_replacements(name, live)
        if replacements:
            mapping[name] = replacements[0]
    if not mapping:
        return code, {}
    new_tree = bindings._Renamer(mapping).visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree), mapping
    except Exception:
        return code, {}


def names_bound_to_package(bindings: ModuleType, tree: ast.AST, package: str) -> set[str]:
    top = package.split(".")[0].replace("-", "_").lower()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0].replace("-", "_").lower() == top:
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0].replace("-", "_").lower() == top:
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Attribute) and bindings._attribute_root(node) == top:
            bound.add(top)
    for _ in range(4):
        before = len(bound)
        for target, value in bindings._binding_pairs(tree):
            call = value.func if isinstance(value, ast.Call) else value
            root = (
                bindings._attribute_root(call)
                if isinstance(call, ast.Attribute)
                else (call.id if isinstance(call, ast.Name) else None)
            )
            if root in bound:
                bound.add(target)
        if len(bound) == before:
            break
    return bound


def binding_pairs(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    pairs: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            pairs.append((node.targets[0].id, node.value))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            pairs.extend(
                (item.optional_vars.id, item.context_expr)
                for item in node.items
                if isinstance(item.optional_vars, ast.Name)
            )
    return pairs


def attribute_root(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None
