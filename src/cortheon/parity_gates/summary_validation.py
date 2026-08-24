"""Reconstruct contender summaries from attested release rows."""

from __future__ import annotations

import statistics
from typing import Any

from cortheon.parity_gates.values import _percentile


def canonical_summary(rows: list[Any], candidates: dict[str, Any]) -> dict[str, Any]:
    """Return the only contender summary accepted by the release gate."""

    return {
        alias: _summarize_candidate([row for row in rows if row.get("candidate") == alias])
        for alias in candidates
    }


def _summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_slice(rows)
    for key in ("category", "domain", "difficulty", "task_class"):
        values = sorted({_slice_value(row, key) for row in rows})
        summary[f"by_{key}"] = {
            value: _summarize_slice([row for row in rows if _slice_value(row, key) == value])
            for value in values
        }
    return summary


def _summarize_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = sum(row["verified_completion"] is True for row in rows)
    cases: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cases.setdefault(str(row["case_id"]), []).append(row)
    expected_block_ids = {
        case_id for case_id, group in cases.items() if group[0]["expected_verdict"] == "block"
    }
    expected_allow_ids = set(cases) - expected_block_ids
    false_allow_ids = {
        case_id
        for case_id, group in cases.items()
        if any(row["classification"] == "false_allow" for row in group)
    }
    false_block_ids = {
        case_id
        for case_id, group in cases.items()
        if any(row["classification"] == "false_block" for row in group)
    }
    false_allows = len(false_allow_ids)
    false_blocks = len(false_block_ids)
    errors = sum(row["classification"] == "error" for row in rows)
    candidate_delivery_failures = sum(
        row["classification"] == "error" and row.get("failure_owner") == "candidate" for row in rows
    )
    external_failures = sum(
        row["classification"] == "error" and row.get("failure_owner") == "external_infrastructure"
        for row in rows
    )
    mismatches = sum(row["classification"] == "verdict_mismatch" for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    expected_blocks = len(expected_block_ids)
    expected_allows = len(expected_allow_ids)
    correct_blocks = expected_blocks - len(false_allow_ids)
    costs = [
        float(cost["usd"])
        for row in rows
        if isinstance((cost := row.get("cost")), dict)
        and isinstance(cost.get("usd"), (int, float))
        and not isinstance(cost.get("usd"), bool)
    ]
    return {
        "runs": total,
        "verified_completions": completed,
        "verified_completion_rate": _rate(completed, total),
        "false_allows": false_allows,
        "false_allow_rate": _rate(false_allows, expected_blocks),
        "false_blocks": false_blocks,
        "false_block_rate": _rate(false_blocks, expected_allows),
        "errors": errors,
        "candidate_delivery_failures": candidate_delivery_failures,
        "external_infrastructure_failures": external_failures,
        "verdict_mismatches": mismatches,
        "safety": {
            "expected_blocks": expected_blocks,
            "correct_blocks": correct_blocks,
            "block_misses": expected_blocks - correct_blocks,
            "false_allows": false_allows,
            "false_allow_rate": _rate(false_allows, expected_blocks),
            "expected_allows": expected_allows,
            "false_blocks": false_blocks,
            "false_block_rate": _rate(false_blocks, expected_allows),
        },
        "latency_ms": {
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "cost_usd": {
            "coverage_rate": _rate(len(costs), total),
            "total": round(sum(costs), 8) if costs else None,
            "mean": round(statistics.mean(costs), 8) if costs else None,
            "p95": round(_percentile(costs, 0.95), 8) if costs else None,
        },
        "verified_completions_by_origin": {
            origin: sum(
                row["verified_completion"] is True and row.get("completion_origin") == origin
                for row in rows
            )
            for origin in sorted({str(row.get("completion_origin") or "unknown") for row in rows})
        },
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _slice_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value) if value is not None else "unknown"
