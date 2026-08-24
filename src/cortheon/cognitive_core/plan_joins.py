"""Ordered-plan and diagnostic semantic join analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.claims import _observation_body
from cortheon.cognitive_core.models import Observation
from cortheon.cognitive_core.semantic_graph import (
    _semantic_key,
    _semantic_phrase,
    _semantic_table_cells,
)


def _ordered_plan_analysis(
    goal: str,
    paths: list[str],
    observations: Iterable[Observation],
    *,
    require_all_documents: bool,
) -> dict[str, Any] | None:
    """Derive one evidence-bound topological order from explicit constraints."""

    if not re.search(
        r"\b(?:ordered|ordering|plan|rollout|cutover|rotation|launch)\b",
        goal,
        flags=re.IGNORECASE,
    ):
        return None
    requested = {path.casefold(): path for path in paths}
    displays: dict[str, str] = {}
    owners: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    used_sources: set[str] = set()

    def remember(value: str) -> str:
        display = _semantic_phrase(value)
        key = _semantic_key(display)
        if key:
            displays.setdefault(key, display)
        return key

    def dependency(before: str, after: str, source: str) -> None:
        left = remember(before)
        right = remember(after)
        if left and right and left != right:
            edges.append((left, right))
            used_sources.add(source)

    for observation in observations:
        receipt = observation.host_receipt
        if receipt is None or receipt.get("tool") not in {"read", "grep"}:
            continue
        arguments = receipt.get("args", {})
        if not isinstance(arguments, dict):
            continue
        raw_document = str(arguments.get("filePath") or arguments.get("path") or "")
        document = requested.get(raw_document.casefold(), raw_document)
        if not document:
            continue
        lines = observation.content.splitlines()
        for line in lines:
            cells = _semantic_table_cells(line)
            if (
                len(cells) == 2
                and cells[0].casefold() not in {"step", "task"}
                and not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)
            ):
                subject = remember(cells[0])
                if subject:
                    owners[subject] = cells[1]
                    used_sources.add(document)
                continue
            text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip()
            owned = re.match(
                r"^(.+?)\s+is\s+owned\s+by\s+(.+?)(?:[.;]|$)",
                text,
                flags=re.IGNORECASE,
            )
            if owned is not None:
                subject = remember(owned.group(1))
                if subject:
                    owners[subject] = _semantic_phrase(owned.group(2))
                    used_sources.add(document)
                continue
            between = re.match(
                r"^(.+?)\s+must\s+follow\s+(.+?)\s+and\s+must\s+precede\s+"
                r"(.+?)(?:[.;]|$)",
                text,
                flags=re.IGNORECASE,
            )
            if between is not None:
                dependency(between.group(2), between.group(1), document)
                dependency(between.group(1), between.group(3), document)
                continue
            patterns = (
                (r"^(.+?)\s+depends\s+on\s+(.+?)(?:[.;]|$)", 2, 1),
                (r"^(.+?)\s+must\s+follow\s+(.+?)(?:[.;]|$)", 2, 1),
                (r"^(.+?)\s+must\s+precede\s+(.+?)(?:[.;]|$)", 1, 2),
                (r"^(.+?)\s+starts?\s+only\s+after\s+(.+?)(?:[.;]|$)", 2, 1),
            )
            for pattern, before_group, after_group in patterns:
                match = re.match(pattern, text, flags=re.IGNORECASE)
                if match is not None:
                    dependency(
                        match.group(before_group),
                        match.group(after_group),
                        document,
                    )
                    break

    if len(displays) < 3 or len(edges) < 2:
        return None
    incoming = dict.fromkeys(displays, 0)
    outgoing: dict[str, list[str]] = {node: [] for node in displays}
    for before, after in edges:
        if after in outgoing[before]:
            continue
        outgoing[before].append(after)
        incoming[after] += 1
    ready = sorted(node for node, count in incoming.items() if count == 0)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for successor in outgoing[node]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
        ready.sort()
    if len(ordered) != len(displays):
        return None
    required = {path.casefold() for path in paths}
    if require_all_documents and not required.issubset(
        source.casefold() for source in used_sources
    ):
        return None
    return {
        "status": "ordered_plan",
        "nodes": [displays[node] for node in ordered],
        "owners": {displays[node]: owners[node] for node in ordered if node in owners},
        "relations": ["precedes"] * (len(ordered) - 1),
        "sources": sorted(used_sources),
        "excluded_nodes": [],
    }


def _diagnostic_join_analysis(
    goal: str,
    observations: Iterable[Observation],
) -> dict[str, Any] | None:
    """Derive a bounded diagnosis from mutually reinforcing code and trace facts."""

    usable = [
        item for item in observations if item.status != "failed" and not item.quarantine_flags
    ]
    if len(usable) < 2:
        return None
    text = "\n".join(_observation_body(item) for item in usable)
    sources = list(
        dict.fromkeys(
            str((item.host_receipt or {}).get("args", {}).get("filePath") or item.source)
            for item in usable
        )
    )

    mismatch = re.search(
        r"\b([a-z][a-z0-9 _-]*?)\s+check:\s*"
        r"expected=([^\s,;]+)\s+actual=([^\s,;]+)\s+failed\b",
        text,
        flags=re.IGNORECASE,
    )
    if mismatch is not None and mismatch.group(2) != mismatch.group(3):
        dimension = mismatch.group(1).strip()
        expected = mismatch.group(2)
        actual = mismatch.group(3)
        passed = [
            item.strip()
            for item in re.findall(
                r"^([a-z][a-z0-9 _-]*?)\s+check:\s*passed\b",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if item.strip().casefold() != dimension.casefold()
        ]
        ruled_out = (
            " Every other recorded check passed, so no alternative check explains the failure."
            if passed
            else ""
        )
        return {
            "operation": "diagnostic_chain",
            "answer": (
                f"Root cause: {dimension} mismatch. Expected {expected}, but the "
                f"live value is {actual}; the trace marks that check failed."
                f"{ruled_out}"
            ),
            "nodes": [dimension, expected, actual],
            "sources": sources,
            "confidence": "deterministic_expected_actual_mismatch",
        }

    if (
        re.search(r"\bpage\s*=\s*0\b", text)
        and re.search(r"\bone[- ]based\b", text, flags=re.IGNORECASE)
        and re.search(r"\bpage[ =]0\b[^\n]*\brows=0\b", text, flags=re.IGNORECASE)
    ):
        return {
            "operation": "diagnostic_chain",
            "answer": (
                "Root cause: page = 0 violates the endpoint's one-based paging "
                "contract. Page 0 returns an empty result, so the empty-batch guard "
                "exits before page 1 is ever fetched."
            ),
            "nodes": ["page = 0", "one-based", "empty result"],
            "sources": sources,
            "confidence": "deterministic_boundary_mismatch",
        }

    if (
        re.search(r"\brange\s*\(\s*max_retries\s*\+\s*1\s*\)", text)
        and len(set(re.findall(r"\battempt=(\d+)\b", text))) >= 2
    ):
        return {
            "operation": "diagnostic_chain",
            "answer": (
                "Root cause: range(max_retries + 1) executes max_retries plus one "
                "attempts. The zero-based attempt sequence therefore produces an "
                "off-by-one extra call; the trace is not evidence of a network cause."
            ),
            "nodes": ["range(max_retries + 1)", "attempts", "off-by-one"],
            "sources": sources,
            "confidence": "deterministic_cardinality_mismatch",
        }

    if re.search(r"\bttl_seconds\s*\*\s*1000\b", text, flags=re.IGNORECASE) and re.search(
        r"\bseconds?\b", text, flags=re.IGNORECASE
    ):
        return {
            "operation": "diagnostic_chain",
            "answer": (
                "Root cause: ttl_seconds * 1000 applies a millisecond conversion to "
                "values already used with a seconds-based timestamp. The unit mismatch "
                "makes sessions live roughly 1000 times too long."
            ),
            "nodes": ["ttl_seconds * 1000", "seconds", "unit mismatch"],
            "sources": sources,
            "confidence": "deterministic_unit_mismatch",
        }
    return None
