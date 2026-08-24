from __future__ import annotations

import math
import re
import statistics
import urllib.parse
from typing import Any

from cortheon.parity_benchmark_core.models import Contender


def _result_cost(
    metadata: dict[str, Any],
    contender: Contender,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    usage = metadata.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _integer_token_count(
        usage,
        "input_tokens",
        "prompt_tokens",
    )
    output_tokens = _integer_token_count(
        usage,
        "output_tokens",
        "completion_tokens",
    )
    reported_cost = metadata.get("total_cost_usd", metadata.get("cost_usd"))
    if contender.compute_cost_per_hour is not None:
        usd = max(0.0, latency_ms) * contender.compute_cost_per_hour / 3_600_000
        source = "runner_wall_clock_and_preregistered_compute_rate"
    elif (
        input_tokens is not None
        and output_tokens is not None
        and contender.input_cost_per_million is not None
        and contender.output_cost_per_million is not None
    ):
        usd = (
            input_tokens * contender.input_cost_per_million
            + output_tokens * contender.output_cost_per_million
        ) / 1_000_000
        source = "metered_from_usage_and_registered_pricing"
    elif isinstance(reported_cost, (int, float)) and math.isfinite(reported_cost):
        usd = max(0.0, float(reported_cost))
        source = "reported"
    else:
        usd = None
        source = "unavailable"
    return {
        "usd": round(usd, 8) if usd is not None else None,
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "compute_cost_per_hour": contender.compute_cost_per_hour,
    }


def _completion_origin(
    contender: Contender,
    metadata: dict[str, Any],
) -> str:
    if contender.kind != "cortheon":
        return "model_only"
    cortheon = metadata.get("cortheon")
    if not isinstance(cortheon, dict):
        return "gateway_model_only"
    agent = cortheon.get("agent")
    if isinstance(agent, dict):
        scorecard = agent.get("scorecard")
        rounds = scorecard.get("rounds_used") if isinstance(scorecard, dict) else None
        if isinstance(rounds, int):
            return "controller_only" if rounds == 0 else "substrate_plus_model"
    decision = cortheon.get("decision")
    if isinstance(decision, dict) and decision.get("verdict") == "block":
        return "controller_only"
    return "gateway_model_only"


def _contender_family(contender: Contender) -> str:
    # Explicit preregistered family always wins: a real candidate model id
    # such as "Qwen/Qwen3-32B" must never be slug-guessed into a family.
    if contender.family is not None:
        explicit = contender.family.strip()
        if explicit:
            return explicit
    # Non-parity compatibility fallback only: infer from identifiers.
    identity = " ".join([contender.name, contender.model, contender.base_url]).casefold()
    if "claude" in identity or "anthropic" in identity:
        return "anthropic"
    if "kimi" in identity or "moonshot" in identity:
        return "moonshot"
    if "openai" in identity or re.search(r"\bgpt[-_]", identity):
        return "openai"
    if "gemini" in identity or "googleapis" in identity:
        return "google"
    parsed = urllib.parse.urlparse(contender.base_url)
    if parsed.hostname and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        return parsed.hostname.casefold()
    # A loopback candidate is the local model itself, so its family is the
    # model identifier, not the shared loopback hostname.
    return re.sub(r"[^a-z0-9]+", "-", contender.model.casefold()).strip("-")


def _candidate_identity(
    contender: Contender,
    rows: list[dict[str, Any]],
    alias: str,
) -> dict[str, Any]:
    observed_models = {
        str(row.get("observed_model_id"))
        for row in rows
        if row.get("candidate") == alias
        and isinstance(row.get("observed_model_id"), str)
        and str(row.get("observed_model_id")).strip()
    }
    return {
        "name": contender.name,
        "kind": contender.kind,
        "model": (next(iter(observed_models)) if len(observed_models) == 1 else None),
        "configured_model": contender.model,
        "base_url": contender.base_url.rstrip("/"),
        "family": _contender_family(contender),
        "pricing_per_million": {
            "input": contender.input_cost_per_million,
            "output": contender.output_cost_per_million,
        },
        "compute_cost_per_hour": contender.compute_cost_per_hour,
        "runtime_sha256": contender.runtime_sha256,
    }


def _integer_token_count(usage: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_slice(rows)
    for key in ("category", "domain", "difficulty"):
        summary[f"by_{key}"] = {
            value: _summarize_slice([row for row in rows if str(row.get(key)) == value])
            for value in sorted({str(row.get(key) or "unknown") for row in rows})
        }
    return summary


def _summarize_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = sum(row["verified_completion"] is True for row in rows)
    false_allows = sum(row["classification"] == "false_allow" for row in rows)
    false_blocks = sum(row["classification"] == "false_block" for row in rows)
    errors = sum(row["classification"] == "error" for row in rows)
    candidate_delivery_failures = sum(
        row["classification"] == "error" and row.get("failure_owner") == "candidate" for row in rows
    )
    external_failures = sum(
        row["classification"] == "error" and row.get("failure_owner") == "external_infrastructure"
        for row in rows
    )
    verdict_mismatches = sum(row["classification"] == "verdict_mismatch" for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    expected_blocks = sum(row["expected_verdict"] == "block" for row in rows)
    expected_allows = sum(row["expected_verdict"] == "allow" for row in rows)
    correct_blocks = sum(
        row["expected_verdict"] == row["observed_verdict"] == "block" for row in rows
    )
    costs = [
        float(cost["usd"])
        for row in rows
        if isinstance((cost := row.get("cost")), dict) and isinstance(cost.get("usd"), (int, float))
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
        "verdict_mismatches": verdict_mismatches,
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


def _benchmark_input_sha256(metadata: dict[str, Any]) -> str | None:
    benchmark = metadata.get("_benchmark")
    value = benchmark.get("input_sha256") if isinstance(benchmark, dict) else None
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _input_symmetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, set[str]] = {}
    missing = 0
    for row in rows:
        if row.get("failure_owner") == "external_infrastructure":
            missing += 1
            continue
        digest = row.get("input_sha256")
        if not isinstance(digest, str):
            missing += 1
            continue
        by_case.setdefault(str(row.get("case_id")), set()).add(digest)
    mismatches = sorted(case_id for case_id, digests in by_case.items() if len(digests) != 1)
    return {
        "verified": bool(by_case) and not mismatches and missing == 0,
        "cases_checked": len(by_case),
        "mismatched_cases": mismatches,
        "missing_rows": missing,
    }


def _cortheon_outcome(metadata: dict[str, Any]) -> dict[str, Any] | None:
    cortheon = metadata.get("cortheon")
    outcome = cortheon.get("outcome") if isinstance(cortheon, dict) else None
    return outcome if isinstance(outcome, dict) else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
