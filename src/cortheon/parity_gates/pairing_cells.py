"""Fail-closed indexing of contender, case, and repetition cells."""

from __future__ import annotations

from typing import Any


def _cell_index(
    rows: list[Any],
    alias: str,
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    indexed: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("candidate") == alias:
            indexed.setdefault((row.get("case_id"), row.get("repetition")), []).append(row)
    return indexed


def _duplicate_count(indexed: dict[tuple[Any, Any], list[dict[str, Any]]]) -> int:
    return sum(max(0, len(rows) - 1) for rows in indexed.values())
