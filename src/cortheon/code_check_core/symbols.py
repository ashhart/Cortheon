"""Symbol-table indexing, deprecation, and composition inference."""

from __future__ import annotations

import ast
from types import ModuleType
from typing import Any


def called_short_name(
    bindings: ModuleType,
    func: ast.expr,
    bound: set[str],
    known: set[str],
) -> str | None:
    if isinstance(func, ast.Attribute):
        root = bindings._attribute_root(func)
        if root in bound and not func.attr.startswith("_"):
            return func.attr
        if isinstance(func.value, ast.Call):
            inner = bindings._called_short_name(func.value.func, bound, known)
            if inner is not None and not func.attr.startswith("_"):
                return func.attr
        return None
    if isinstance(func, ast.Name) and func.id in bound and func.id in known:
        return func.id
    return None


def callable_param_index(
    bindings: ModuleType,
    symbols: list[Any],
) -> dict[str, list[tuple[set[str], bool]]]:
    parsed: dict[str, tuple[set[str], bool]] = {}
    classes: set[str] = set()
    for symbol in symbols:
        if symbol.kind == "class":
            classes.add(symbol.qualname)
        params = bindings._parse_signature_params(symbol.signature)
        if params is not None:
            parsed[symbol.qualname] = params

    index: dict[str, list[tuple[set[str], bool]]] = {}
    for qualname, params in parsed.items():
        short = qualname.split(".")[-1]
        if short == "__init__":
            owner = qualname.rsplit(".", 1)[0]
            if owner in classes:
                index.setdefault(owner.split(".")[-1], []).append(params)
            continue
        index.setdefault(short, []).append(params)
    return index


def parse_signature_params(signature: str | None) -> tuple[set[str], bool] | None:
    if not signature:
        return None
    try:
        tree = ast.parse(f"def {signature}: pass")
        function = tree.body[0]
    except (SyntaxError, ValueError, IndexError):
        return None
    if not isinstance(function, ast.FunctionDef):
        return None
    args = function.args
    names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
    return names, args.kwarg is not None


def param_annotations(signature: str | None) -> dict[str, str]:
    if not signature:
        return {}
    try:
        tree = ast.parse(f"def {signature}: pass")
        function = tree.body[0]
    except (SyntaxError, ValueError, IndexError):
        return {}
    if not isinstance(function, ast.FunctionDef):
        return {}
    args = function.args
    output: dict[str, str] = {}
    for argument in args.posonlyargs + args.args + args.kwonlyargs:
        if argument.annotation is not None:
            try:
                output[argument.arg] = ast.unparse(argument.annotation)
            except Exception:
                continue
    return output


def composition_bridge(
    bindings: ModuleType,
    short: str,
    keyword_name: str,
    accepters: list[str],
    symbols: list[Any],
) -> str | None:
    caller_annotations: dict[str, str] = {}
    for symbol in symbols:
        parts = symbol.qualname.split(".")
        if parts[-1] == "__init__" and len(parts) >= 2 and parts[-2] == short:
            caller_annotations = bindings._param_annotations(symbol.signature)
            break
        if parts[-1] == short and symbol.kind != "class":
            caller_annotations = bindings._param_annotations(symbol.signature)
    if not caller_annotations:
        return None
    for accepter in accepters:
        family = {accepter}
        for symbol in symbols:
            if (
                symbol.kind == "class"
                and symbol.qualname.split(".")[-1] == accepter
                and symbol.signature
            ):
                bases = bindings.re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]*",
                    symbol.signature,
                )
                family |= set(bases)
        matches = [
            (parameter, annotation)
            for parameter, annotation in caller_annotations.items()
            if any(name in annotation for name in family if name not in {"class", short})
        ]
        if matches:
            matches.sort(
                key=lambda item: (
                    item[0].lower().rstrip("s") not in accepter.lower(),
                    len(item[1]),
                )
            )
            return f"{matches[0][0]}={accepter}({keyword_name}=...)"
    return None


def deprecation_index(bindings: ModuleType, symbols: list[Any]) -> tuple[set[str], set[str]]:
    deprecated: set[str] = set()
    live_all: set[str] = set()
    live_public: set[str] = set()
    for symbol in symbols:
        short = symbol.qualname.split(".")[-1]
        if short.startswith("_"):
            continue
        if symbol.deprecated or bindings.is_legacy_path(symbol.qualname):
            deprecated.add(short)
            continue
        live_all.add(short)
        if not any(part.startswith("_") for part in symbol.qualname.split(".")[:-1]):
            live_public.add(short)
    return deprecated - live_all, live_public


def suggest_replacements(
    deprecated_name: str,
    live_names: set[str],
    limit: int,
) -> list[str]:
    lower = deprecated_name.lower()
    candidates = [
        name
        for name in live_names
        if name.lower() != lower and (lower in name.lower() or name.lower() in lower)
    ]
    candidates.sort(
        key=lambda name: (
            name.islower() != deprecated_name.islower(),
            not (name.lower().startswith(lower) or name.lower().endswith(lower)),
            len(name),
            name,
        )
    )
    return candidates[:limit]


def is_legacy_path(bindings: ModuleType, qualname: str) -> bool:
    return any(bindings.LEGACY_SEGMENT.match(part) for part in qualname.split("."))


def known_attribute_names(symbols: list[Any]) -> set[str]:
    names: set[str] = set()
    for symbol in symbols:
        names.add(symbol.name)
        names.add(symbol.qualname.split(".")[-1])
    return {name for name in names if name and not name.startswith("_")}
