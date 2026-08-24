"""Structural API-call and keyword validation."""

from __future__ import annotations

import ast
from types import ModuleType
from typing import Any


def check_api_usage(
    bindings: ModuleType,
    code: str,
    package: str,
    symbols: list[Any],
) -> Any:
    notes: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return bindings.CodeCheckReport(
            package=package,
            parsed=False,
            checked_calls=0,
            findings=[],
            notes=[f"Generated code does not parse: {exc}"],
        )

    known = bindings._known_attribute_names(symbols)
    if not known:
        notes.append(
            f"No symbol table available for {package}; usage could not be structurally checked."
        )
        return bindings.CodeCheckReport(
            package=package,
            parsed=True,
            checked_calls=0,
            findings=[],
            notes=notes,
        )

    bound = bindings._names_bound_to_package(tree, package)
    param_index = bindings._callable_param_index(symbols)
    deprecated, live = bindings._deprecation_index(symbols)
    findings: list[Any] = []
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        short = bindings._called_short_name(node.func, bound, known)
        if short is None:
            continue
        checked += 1
        if short not in known:
            findings.append(
                bindings.ApiUsageFinding(
                    package=package,
                    attribute=short,
                    line=getattr(node, "lineno", 0),
                    reason=(
                        f"{package} defines no public symbol named {short!r}; "
                        "this call is not backed by the package's source."
                    ),
                    kind="unknown_symbol",
                )
            )
            continue
        if short in deprecated:
            replacements = bindings.suggest_replacements(short, live)
            hint = (
                f"; current replacements include: {', '.join(replacements)}"
                if replacements
                else "; use the current replacement instead of the legacy API"
            )
            findings.append(
                bindings.ApiUsageFinding(
                    package=package,
                    attribute=short,
                    line=getattr(node, "lineno", 0),
                    reason=(f"{short} is deprecated in the current release of {package}{hint}."),
                    kind="deprecated_symbol",
                )
            )
            continue
        findings.extend(bindings._check_keywords(package, short, node, param_index, symbols))
    return bindings.CodeCheckReport(
        package=package,
        parsed=True,
        checked_calls=checked,
        findings=findings,
        notes=notes,
    )


def check_keywords(
    bindings: ModuleType,
    package: str,
    short: str,
    call: ast.Call,
    param_index: dict[str, list[tuple[set[str], bool]]],
    symbols: list[Any],
) -> list[Any]:
    candidates = param_index.get(short)
    if not candidates:
        return []
    if any(has_kwargs for _, has_kwargs in candidates):
        return []
    accepted: set[str] = set()
    for params, _ in candidates:
        accepted |= params
    findings: list[Any] = []
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in accepted:
            continue
        shown = ", ".join(sorted(name for name in accepted if name != "self"))
        accepters = sorted(
            name
            for name, candidates_for_name in param_index.items()
            if name != short
            and not name.startswith("_")
            and any(keyword.arg in params for params, _ in candidates_for_name)
        )[:4]
        cross = (
            f" The keyword {keyword.arg!r} IS accepted by: {', '.join(accepters)}."
            if accepters
            else ""
        )
        if accepters:
            bridge = bindings.composition_bridge(short, keyword.arg, accepters, symbols)
            if bridge:
                cross += f" Verified composition: {short}(..., {bridge})."
        findings.append(
            bindings.ApiUsageFinding(
                package=package,
                attribute=f"{short}(..., {keyword.arg}=...)",
                line=getattr(call, "lineno", 0),
                reason=(
                    f"{short} accepts no keyword argument {keyword.arg!r} in the current release; "
                    f"accepted keywords: {shown}.{cross}"
                ),
                kind="unknown_argument",
            )
        )
    return findings
