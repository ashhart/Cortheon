"""Predictions for stale model priors tied to explicit package symbols."""

from __future__ import annotations

from typing import Any


def predict_failures(
    engine: Any,
    text: str,
    packages: list[str],
    *,
    is_legacy_path: Any,
    token_pattern: Any,
    bound_names: Any,
) -> str:
    lines: list[str] = []
    for package in packages:
        try:
            metadata, _ = engine.pypi.fetch(package)
            _, symbols, _ = engine.api_extractor.load_symbols(metadata)
        except Exception:
            continue
        legacy = {}
        for symbol in symbols:
            if symbol.deprecated or is_legacy_path(symbol.qualname):
                short = symbol.qualname.split(".")[-1]
                if not short.startswith("_"):
                    legacy[short] = symbol.qualname
        mentioned = []
        for token in token_pattern.findall(text):
            if token not in mentioned:
                mentioned.append(token)
        mentioned_legacy = [token for token in mentioned if token in legacy]
        if mentioned_legacy:
            lines.append(
                f"For {package}: these mentioned symbols your training memory knows are "
                "DEPRECATED or LEGACY in the current version: "
                f"{', '.join(mentioned_legacy[:6])}. Do not use them unless you must."
            )
        all_parts = set()
        for symbol in symbols:
            all_parts.update(symbol.qualname.split("."))
        lines.extend(
            (
                f"For {package}: your training memory may tell you to use "
                f"{token!r}, but it does NOT exist in the current source. Do not call it."
            )
            for token in bound_names(text, package)
            if token not in all_parts and len(token) > 3
        )
    return "\n".join(lines)
