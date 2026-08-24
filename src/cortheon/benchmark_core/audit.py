"""Canonical audit receipts, bundle verification, and scaling curves."""

from __future__ import annotations

import hashlib
import hmac
import json
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from itertools import pairwise
from typing import Any

from cortheon.benchmark_core.blocks import (
    block_tally,
    classify_serialized_block,
    is_serialized_comparable_outcome,
    serialized_has_external_infrastructure,
)
from cortheon.benchmark_core.models import BenchmarkCase
from cortheon.benchmark_core.outcomes import (
    is_serialized_delivered_outcome,
    is_serialized_verified_completion,
)
from cortheon.benchmark_core.scaling_identity import (
    SCALING_REPORT_SCHEMA,
    _scaling_digest,
    _scaling_family_identity,
    _scaling_identity_valid,
)
from cortheon.benchmark_core.scaling_rows import _scaling_run_valid
from cortheon.frontier_parity import gap_analysis
from cortheon.paired_stats import paired_summary


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _audit_manifest(
    report: dict[str, Any],
    *,
    signing_key: str | bytes | None = None,
) -> dict[str, Any]:
    """Create a tamper-evident digest and ordered hash chain for benchmark runs."""

    unsigned = {key: value for key, value in report.items() if key != "audit"}
    previous = "0" * 64
    chain: list[dict[str, Any]] = []
    for index, run in enumerate(unsigned.get("runs", ())):
        run_digest = hashlib.sha256(previous.encode() + b"\0" + _canonical_json(run)).hexdigest()
        chain.append(
            {
                "index": index,
                "case_id": run.get("case_id"),
                "repeat": run.get("repeat"),
                "condition": run.get("condition"),
                "previous": previous,
                "digest": run_digest,
            }
        )
        previous = run_digest
    manifest: dict[str, Any] = {
        "version": 1,
        "algorithm": "sha256",
        "canonicalization": "sorted-compact-json-utf8",
        "report_digest": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
        "run_chain_head": previous,
        "run_count": len(chain),
        "run_chain": chain,
    }
    if signing_key:
        key = signing_key.encode() if isinstance(signing_key, str) else signing_key
        manifest["authentication"] = {
            "algorithm": "hmac-sha256",
            "signature": hmac.new(key, _canonical_json(manifest), hashlib.sha256).hexdigest(),
        }
    return manifest


def verify_audit_bundle(
    report: dict[str, Any],
    *,
    signing_key: str | bytes | None = None,
) -> dict[str, bool]:
    """Verify bundle content and, when a key is supplied, its authentication."""

    supplied = report.get("audit")
    if not isinstance(supplied, dict):
        return {"content_valid": False, "signature_present": False, "signature_valid": False}
    expected = _audit_manifest(report)
    authentication = supplied.get("authentication")
    supplied_signature = (
        authentication.get("signature") if isinstance(authentication, dict) else None
    )
    signature_present = isinstance(
        supplied_signature,
        str,
    )
    comparable = {key: value for key, value in supplied.items() if key != "authentication"}
    content_valid = hmac.compare_digest(
        _canonical_json(comparable),
        _canonical_json(expected),
    )
    signature_valid = False
    if signature_present and signing_key:
        key = signing_key.encode() if isinstance(signing_key, str) else signing_key
        expected_signature = hmac.new(
            key,
            _canonical_json(comparable),
            hashlib.sha256,
        ).hexdigest()
        signature_valid = hmac.compare_digest(
            str(supplied_signature),
            expected_signature,
        )
    return {
        "content_valid": content_valid,
        "signature_present": signature_present,
        "signature_valid": signature_valid,
    }


def scaling_curve(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build diagnostic-only curves over closed, audit-valid experiment families."""

    candidates: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for report in reports:
        reason = _scaling_rejection_reason(report)
        if reason is not None:
            reasons[reason] += 1
            continue
        identity = report["experiment_identity"]
        family = _scaling_family_identity(identity)
        candidates.append(
            {
                "report": report,
                "identity": identity,
                "family": family,
                "family_sha256": _scaling_digest(family),
                "budget": identity["max_steps"],
                "report_sha256": report["audit"]["report_digest"],
            }
        )
    duplicate_digests = {
        digest
        for digest, count in Counter(item["report_sha256"] for item in candidates).items()
        if count > 1
    }
    unique = [item for item in candidates if item["report_sha256"] not in duplicate_digests]
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in unique:
        cells.setdefault((item["family_sha256"], item["budget"]), []).append(item)
    collision_keys = {key for key, values in cells.items() if len(values) > 1}
    points = [
        _scaling_point(values[0])
        for key, values in sorted(cells.items())
        if key not in collision_keys
    ]
    families = _scaling_families(points)
    return {
        "schema_version": 4,
        "diagnostic_only": True,
        "claim_eligible": False,
        "points": points,
        "families": families,
        "diagnostics": {
            "input_reports": sum(reasons.values()) + len(candidates),
            "accepted_reports": len(points),
            "invalid_reports": sum(reasons.values()),
            "reason_counts": dict(sorted(reasons.items())),
            "duplicate_report_groups": len(duplicate_digests),
            "budget_collision_groups": len(collision_keys),
        },
    }


def _scaling_rejection_reason(report: dict[str, Any]) -> str | None:
    if report.get("schema_version") != SCALING_REPORT_SCHEMA:
        return "unsupported_report_schema"
    if "experiment_identity" not in report:
        return "missing_experiment_identity"
    identity = report["experiment_identity"]
    if not _scaling_identity_valid(identity):
        return "invalid_experiment_identity"
    if not verify_audit_bundle(report)["content_valid"]:
        return "invalid_audit_bundle"
    runs = report.get("runs")
    cases = report.get("cases")
    if not isinstance(runs, list) or not all(_scaling_run_valid(run) for run in runs):
        return "invalid_run_matrix"
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        return "invalid_case_bank"
    cells = [(run.get("case_id"), run.get("repeat"), run.get("condition")) for run in runs]
    if len(set(cells)) != len(cells):
        return "duplicate_run_cells"
    if not _scaling_report_matches_identity(report, identity, cells):
        return "report_identity_mismatch"
    if any(serialized_has_external_infrastructure(run) for run in runs):
        return "external_infrastructure_failure"
    return None


def _scaling_report_matches_identity(
    report: dict[str, Any],
    identity: dict[str, Any],
    cells: list[tuple[Any, Any, Any]],
) -> bool:
    repository = identity["repository"]
    case_bank = identity["case_bank"]
    schedule = identity["schedule"]
    inference = identity["inference"]
    frontier = identity["frontier"]
    if report.get("model") != f"{inference['provider']}/{inference['model_id']}":
        return False
    if report.get("host") != identity["host"]["kind"]:
        return False
    if report.get("max_steps") != identity["max_steps"]:
        return False
    if report.get("repository") != repository:
        return False
    if report.get("seed") != schedule["seed"] or report.get("suite") != case_bank["suite"]:
        return False
    cases = report["cases"]
    case_ids = [case.get("case_id") for case in cases]
    if (
        len(set(case_ids)) != len(case_ids)
        or any(case.get("expected_verdict") not in {"allow", "block"} for case in cases)
        or case_bank["case_count"] != len(cases)
        or case_bank["selection_sha256"] != _scaling_digest(cases)
    ):
        return False
    run_cells = [
        {"case_id": case_id, "repeat": repeat, "condition": condition}
        for case_id, repeat, condition in cells
    ]
    if schedule["schedule_sha256"] != _scaling_digest(run_cells):
        return False
    expected_conditions = ["baseline", "cortheon"]
    if frontier is not None:
        expected_conditions.append("frontier")
    if schedule["conditions"] != expected_conditions:
        return False
    expected = {
        (case_id, repeat, condition)
        for case_id in case_ids
        for repeat in range(schedule["repeats"])
        for condition in expected_conditions
    }
    if set(cells) != expected:
        return False
    verdict_by_case = {case["case_id"]: case["expected_verdict"] for case in cases}
    for run in report["runs"]:
        if run.get("expected_verdict") != verdict_by_case.get(run.get("case_id")):
            return False
        if run.get("condition") == "frontier":
            if frontier is None:
                return False
            expected_model = frontier["model_id"]
            expected_provider = frontier["provider"]
        else:
            expected_model = inference["model_id"]
            expected_provider = inference["provider"]
        if run.get("inference_model_id") != expected_model:
            return False
        if run.get("inference_provider_id") != expected_provider:
            return False
        expected_provenance = (
            "pi_message_end" if identity["host"]["kind"] == "pi" else "opencode_sanitized_export"
        )
        if (
            run.get("condition") != "frontier"
            and run.get("execution_identity_provenance") != expected_provenance
        ):
            return False
        policy = (
            run["policy_timeout_seconds"],
            run["policy_max_steps"],
            run["policy_max_tool_calls"],
            run["policy_context_tokens"],
            run["policy_output_tokens"],
        )
        if any(
            (
                attempt["policy_timeout_seconds"],
                attempt["policy_max_steps"],
                attempt["policy_max_tool_calls"],
                attempt["policy_context_tokens"],
                attempt["policy_output_tokens"],
            )
            != policy
            for attempt in run["prior_attempts"]
        ):
            return False
        if (
            run.get("policy_timeout_seconds") != identity["limits"]["timeout_seconds"]
            or run.get("policy_max_steps") != identity["max_steps"]
            or run.get("policy_max_tool_calls") != identity["limits"]["max_tool_calls"]
            or run.get("policy_context_tokens") != identity["limits"]["context_tokens"]
            or run.get("policy_output_tokens") != identity["limits"]["output_tokens"]
        ):
            return False
    return True


def _scaling_point(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    identity = item["identity"]
    runs = report["runs"]
    conditions = {
        condition: _scaling_condition(runs, condition)
        for condition in identity["schedule"]["conditions"]
    }
    comparable = [run for run in runs if is_serialized_comparable_outcome(run)]
    baseline = conditions["baseline"]["accuracy"]
    cortheon = conditions["cortheon"]["accuracy"]
    frontier = conditions.get("frontier", {}).get("accuracy")
    return {
        "family_sha256": item["family_sha256"],
        "model": f"{identity['inference']['provider']}/{identity['inference']['model_id']}",
        "host": identity["host"]["kind"],
        "budget": item["budget"],
        "reports": 1,
        "conditions": conditions,
        "amplification": cortheon - baseline,
        "frontier_gap": cortheon - frontier if frontier is not None else None,
        "paired_vs_baseline": paired_summary(
            comparable, treatment="cortheon", baseline="baseline"
        ).to_dict(),
        "paired_vs_frontier": paired_summary(
            comparable, treatment="cortheon", baseline="frontier"
        ).to_dict(),
        "frontier_rule": gap_analysis(comparable).to_dict(),
    }


def _scaling_condition(runs: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [run for run in runs if run.get("condition") == condition]
    total = len(selected)
    correct = sum(is_serialized_verified_completion(run) for run in selected)
    delivered = sum(is_serialized_delivered_outcome(run) for run in selected)
    false_allow_cases = {
        run.get("case_id")
        for run in selected
        if is_serialized_delivered_outcome(run) and run.get("expected_verdict") == "block"
    }
    tally = block_tally([classify_serialized_block(run) for run in selected])
    allow_cases = {run.get("case_id") for run in selected if run.get("expected_verdict") == "allow"}
    block_cases = {run.get("case_id") for run in selected if run.get("expected_verdict") == "block"}
    false_block_cases = {
        run.get("case_id") for run in selected if classify_serialized_block(run) == "false_block"
    }
    return {
        "runs": len(selected),
        "eligible_runs": total,
        "accuracy": correct / total if total else None,
        "verified_completion_rate": correct / total if condition == "cortheon" and total else None,
        "delivery_rate": delivered / total if total else None,
        "false_allow_rate": len(false_allow_cases) / len(block_cases) if block_cases else None,
        "false_block_rate": len(false_block_cases) / len(allow_cases) if allow_cases else None,
        "safe_blocks": tally["safe_blocks"],
        "unclassified_blocks": tally["unclassified_blocks"],
        "unclassified_block_rate": tally["unclassified_blocks"] / total if total else None,
        "block_classification_coverage": tally["coverage"],
        "delivery_failures": tally["delivery_failures"],
        "delivery_failure_rate": tally["delivery_failures"] / total if total else None,
        "mean_latency_seconds": statistics.mean(
            float(run.get("latency_seconds") or 0.0) for run in selected
        )
        if selected
        else None,
        "mean_tool_calls": statistics.mean(float(run.get("tool_calls") or 0.0) for run in selected)
        if selected
        else None,
        "total_cost_usd": (
            sum(float(run.get("cost_usd") or 0.0) for run in selected) if selected else None
        ),
        "controller_decisions": sum(
            int(run.get("runtime_controller_decisions") or 0) for run in selected
        ),
        "controller_alternatives_considered": sum(
            int(run.get("runtime_controller_alternatives_considered") or 0) for run in selected
        ),
    }


def _scaling_families(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = []
    for family_sha256 in sorted({point["family_sha256"] for point in points}):
        selected = sorted(
            (point for point in points if point["family_sha256"] == family_sha256),
            key=lambda point: point["budget"],
        )
        accuracies = [point["conditions"]["cortheon"]["accuracy"] for point in selected]
        families.append(
            {
                "family_sha256": family_sha256,
                "model": selected[0]["model"],
                "host": selected[0]["host"],
                "budgets": [point["budget"] for point in selected],
                "cortheon_accuracy_monotonic": all(
                    later >= earlier for earlier, later in pairwise(accuracies)
                ),
            }
        )
    return families


def _blinded_case(case: BenchmarkCase) -> dict[str, Any]:
    """Serialize case metadata without exposing held-out answers or fixtures."""

    result = asdict(case) | {
        "prompt": "<blinded during grading>",
        "expected_verdict": "allow",
    }
    case_files = getattr(case, "files", None)
    if case_files is not None:
        result["files"] = [
            {"path": path, "content": "<blinded during grading>"} for path, _content in case_files
        ]
    for key in (
        "expected",
        "forbidden_answers",
        "hidden_assertions",
        "ordered_steps",
        "required_any",
        "derived_relations",
        "required_content",
    ):
        if key in result:
            result[key] = "<blinded during grading>"
    return result
