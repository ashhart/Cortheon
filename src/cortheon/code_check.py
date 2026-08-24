"""Verify generated calls against source-derived package symbol tables."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from types import ModuleType

from cortheon.code_check_core import checking
from cortheon.code_check_core import symbols as symbol_helpers
from cortheon.code_check_core import syntax as syntax_helpers
from cortheon.models import ApiSymbol

FENCED = re.compile(r"```(?:python|py)?\s*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
LEGACY_SEGMENT = re.compile(r"^(v\d+|deprecated|legacy|compat|_compat)$", re.IGNORECASE)


@dataclass(slots=True)
class ApiUsageFinding:
    package: str
    attribute: str
    line: int
    reason: str
    kind: str = "unknown_symbol"


@dataclass(slots=True)
class CodeCheckReport:
    package: str
    parsed: bool
    checked_calls: int
    findings: list[ApiUsageFinding]
    notes: list[str]

    @property
    def verdict(self) -> str:
        if not self.parsed:
            return "block"
        return "block" if self.findings else "allow"


def _bindings() -> ModuleType:
    return sys.modules[__name__]


def extract_code_blocks(text: str) -> list[str]:
    return syntax_helpers.extract_code_blocks(_bindings(), text)


def check_api_usage(code: str, package: str, symbols: list[ApiSymbol]) -> CodeCheckReport:
    """Flag calls that the package's current source-derived API does not support."""
    return checking.check_api_usage(_bindings(), code, package, symbols)


def _check_keywords(
    package: str,
    short: str,
    call: ast.Call,
    param_index: dict[str, list[tuple[set[str], bool]]],
    symbols: list[ApiSymbol],
) -> list[ApiUsageFinding]:
    return checking.check_keywords(
        _bindings(),
        package,
        short,
        call,
        param_index,
        symbols,
    )


def _called_short_name(func: ast.expr, bound: set[str], known: set[str]) -> str | None:
    return symbol_helpers.called_short_name(_bindings(), func, bound, known)


def _callable_param_index(
    symbols: list[ApiSymbol],
) -> dict[str, list[tuple[set[str], bool]]]:
    return symbol_helpers.callable_param_index(_bindings(), symbols)


def _parse_signature_params(signature: str | None) -> tuple[set[str], bool] | None:
    return symbol_helpers.parse_signature_params(signature)


def _param_annotations(signature: str | None) -> dict[str, str]:
    return symbol_helpers.param_annotations(signature)


def composition_bridge(
    short: str,
    keyword_name: str,
    accepters: list[str],
    symbols: list[ApiSymbol],
) -> str | None:
    """Return a source-derived constructor composition for a rejected keyword."""
    return symbol_helpers.composition_bridge(
        _bindings(),
        short,
        keyword_name,
        accepters,
        symbols,
    )


def _deprecation_index(symbols: list[ApiSymbol]) -> tuple[set[str], set[str]]:
    return symbol_helpers.deprecation_index(_bindings(), symbols)


def suggest_replacements(
    deprecated_name: str,
    live_names: set[str],
    limit: int = 3,
) -> list[str]:
    """Return live symbols that plausibly replace a deprecated name."""
    return symbol_helpers.suggest_replacements(deprecated_name, live_names, limit)


def is_legacy_path(qualname: str) -> bool:
    """Return whether a symbol lives under a legacy or compatibility namespace."""
    return symbol_helpers.is_legacy_path(_bindings(), qualname)


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        if node.attr in self.mapping:
            node.attr = self.mapping[node.attr]
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        for alias in node.names:
            if alias.name in self.mapping and alias.asname is None:
                alias.name = self.mapping[alias.name]
        return node


def apply_deprecation_renames(
    code: str,
    symbols: list[ApiSymbol],
) -> tuple[str, dict[str, str]]:
    """Rename deprecated symbols to source-derived live replacements."""
    return syntax_helpers.apply_deprecation_renames(_bindings(), code, symbols)


def _known_attribute_names(symbols: list[ApiSymbol]) -> set[str]:
    return symbol_helpers.known_attribute_names(symbols)


def _names_bound_to_package(tree: ast.AST, package: str) -> set[str]:
    return syntax_helpers.names_bound_to_package(_bindings(), tree, package)


def _binding_pairs(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    return syntax_helpers.binding_pairs(tree)


def _attribute_root(node: ast.AST) -> str | None:
    return syntax_helpers.attribute_root(node)
