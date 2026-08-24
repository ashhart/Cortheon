"""Failure taxonomy and the content-free public projection of a run."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from cortheon.benchmark_core.outcomes import (
    is_delivered_outcome,
    is_verified_completion,
)
from cortheon.cognitive_benchmark import (
    DELIVERY_FAILURE,
    FALSE_BLOCK,
    SAFE_BLOCK,
    UNCLASSIFIED_BLOCK,
    RunResult,
    classify_block,
)


def _failure_type(result: RunResult) -> str | None:
    if result.failure_owner == "external_infrastructure":
        return "infrastructure_failure"
    if result.timed_out:
        return "timeout"
    if result.step_budget_exhausted:
        return "step_budget_exhausted"
    if result.condition != "bare" and result.substrate_telemetry_valid is not True:
        return "telemetry_failure"
    if is_delivered_outcome(result) and not result.correct and result.expected_verdict == "block":
        return "false_allow"
    # Task semantics classify authenticated restraint. Candidate and artifact
    # grades remain separate report fields and never alter this taxonomy.
    block_kind = classify_block(result)
    if block_kind == FALSE_BLOCK:
        return "false_block"
    if block_kind == SAFE_BLOCK:
        return "safe_block"
    if block_kind == UNCLASSIFIED_BLOCK:
        return "unclassified_block"
    if block_kind == DELIVERY_FAILURE:
        return "delivery_failure"
    if not is_verified_completion(result):
        return "incorrect"
    return None


def _public_run(result: RunResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "repeat": result.repeat,
        "condition": result.condition,
        "condition_registry_version": result.condition_registry_version,
        "condition_config_sha256": result.condition_config_sha256,
        "condition_implementation_sha256": result.condition_implementation_sha256,
        "condition_requires_runtime_completion": (result.condition_requires_runtime_completion),
        "condition_profile_receipt_valid": result.condition_profile_receipt_valid,
        "condition_observed_config_sha256": result.condition_observed_config_sha256,
        "condition_observed_implementation_sha256": (
            result.condition_observed_implementation_sha256
        ),
        "condition_adapter_receipt_valid": result.condition_adapter_receipt_valid,
        "condition_operator_counts": result.condition_operator_counts,
        "task_type": result.task_type,
        "delivered": result.delivered,
        "correct": result.correct,
        "verified_completion": is_verified_completion(result),
        "evaluator_outcome": asdict(result.evaluator_outcome),
        "latency_seconds": result.latency_seconds,
        "tokens": result.tokens,
        "tool_calls": result.tool_calls,
        "tool_errors": result.tool_errors,
        "timed_out": result.timed_out,
        "process_error": result.process_error is not None,
        "expected_verdict": result.expected_verdict,
        "failure_owner": result.failure_owner,
        "step_budget_exhausted": result.step_budget_exhausted,
        "cost_usd": result.cost_usd,
        "artifact_correct": result.artifact_correct,
        "candidate_correct": result.candidate_correct,
        "substrate_telemetry_valid": result.substrate_telemetry_valid,
        "runtime_sessions_started": result.runtime_sessions_started,
        "runtime_observations_accepted": result.runtime_observations_accepted,
        "runtime_sessions_completed": result.runtime_sessions_completed,
        "runtime_sessions_evidence_closed": result.runtime_sessions_evidence_closed,
        "runtime_sessions_abandoned": result.runtime_sessions_abandoned,
        "runtime_completion_withheld": result.runtime_completion_withheld,
        "failure_type": _failure_type(result),
    }
