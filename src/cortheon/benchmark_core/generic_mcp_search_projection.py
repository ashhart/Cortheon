"""Bounded search terms for runtime-selected document discovery."""

from __future__ import annotations

import re

_GENERIC = frozenset(
    {
        "action",
        "could",
        "cross",
        "document",
        "documents",
        "evidence",
        "focused",
        "goal",
        "lines",
        "live",
        "paths",
        "project",
        "read",
        "request",
        "search",
        "smallest",
        "source",
        "that",
        "this",
        "what",
        "with",
    }
)


def discovery_pattern(query: str) -> str | None:
    """Choose a small OR-list of literal terms already present in the request."""

    quoted = [match.group(1) for match in re.finditer(r"['\"]([^'\"]{3,160})['\"]", query)]
    pools = quoted or [query]
    selected: list[str] = []
    seen: set[str] = set()
    for pool in pools:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.:-]{2,63}", pool):
            folded = token.casefold()
            if folded in _GENERIC or folded in seen:
                continue
            seen.add(folded)
            selected.append(token)
            if len(selected) == 4:
                return "|".join(selected)
    return "|".join(selected) if selected else None
