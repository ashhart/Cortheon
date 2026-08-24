"""Closed row validation for diagnostic scaling reports."""

from __future__ import annotations

import math
from typing import Any

from cortheon.benchmark_core.blocks import serialized_failure_ownership_valid

_NONNEGATIVE_COUNTERS = (
    "tool_calls",
    "tool_errors",
    "host_tool_executions",
    "blocked_tool_calls",
    "unavailable_tool_calls",
    "observed_steps",
    "policy_max_tool_calls",
    "model_text_chars",
    "deliverable_chars",
    "runtime_sessions_started",
    "runtime_observations_accepted",
    "runtime_sessions_completed",
    "runtime_sessions_evidence_closed",
    "runtime_completion_withheld",
    "runtime_controller_decisions",
    "runtime_controller_alternatives_considered",
    "retry_count",
)
_POSITIVE_COUNTERS = (
    "policy_max_steps",
    "policy_context_tokens",
    "policy_output_tokens",
)
_ATTEMPT_KEYS = frozenset(
    {
        "attempt_index",
        "latency_seconds",
        "tokens",
        "tool_calls",
        "cost_usd",
        "timed_out",
        "process_error",
        "failure_owner",
        "terminal_status",
        "provider_id",
        "model_id",
        "identity_valid",
        "identity_provenance",
        "measurements_valid",
        "policy_timeout_seconds",
        "policy_max_steps",
        "policy_max_tool_calls",
        "policy_context_tokens",
        "policy_output_tokens",
        "budget_reason",
    }
)


def _scaling_count(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and (value > 0 if positive else value >= 0) and value <= 10**12


def _scaling_number(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _scaling_bounded_text(value: Any, *, nullable: bool = False) -> bool:
    return bool(
        (nullable and value is None) or (isinstance(value, str) and 0 < len(value) <= 2_000)
    )


def _scaling_attempt_valid(attempt: Any) -> bool:
    if not isinstance(attempt, dict) or set(attempt) != _ATTEMPT_KEYS:
        return False
    if not _scaling_count(attempt["attempt_index"], positive=True):
        return False
    if not _scaling_number(attempt["latency_seconds"]):
        return False
    if attempt["tokens"] is not None and not _scaling_count(attempt["tokens"]):
        return False
    if not _scaling_count(attempt["tool_calls"]):
        return False
    if attempt["cost_usd"] is not None and not _scaling_number(attempt["cost_usd"]):
        return False
    if type(attempt["timed_out"]) is not bool:
        return False
    if not _scaling_bounded_text(attempt["process_error"], nullable=True):
        return False
    owner = attempt["failure_owner"]
    if attempt["terminal_status"] in {"success", "withheld"}:
        if owner is not None:
            return False
    elif owner not in {"candidate", "external_infrastructure"}:
        return False
    if not _scaling_bounded_text(attempt["terminal_status"]):
        return False
    for field in ("provider_id", "model_id", "budget_reason"):
        if not _scaling_bounded_text(attempt[field], nullable=True):
            return False
    if attempt["identity_provenance"] not in {
        "pi_message_end",
        "opencode_sanitized_export",
        "unavailable",
    }:
        return False
    if (
        not _scaling_number(attempt["policy_timeout_seconds"])
        or attempt["policy_timeout_seconds"] <= 0
    ):
        return False
    if any(
        not _scaling_count(attempt[field], positive=True)
        for field in ("policy_max_steps", "policy_context_tokens", "policy_output_tokens")
    ):
        return False
    if not _scaling_count(attempt["policy_max_tool_calls"]):
        return False
    return all(type(attempt[field]) is bool for field in ("identity_valid", "measurements_valid"))


def _scaling_run_valid(run: Any) -> bool:
    if not isinstance(run, dict):
        return False
    if not isinstance(run.get("case_id"), str) or not 0 < len(run["case_id"]) <= 256:
        return False
    if not _scaling_count(run.get("repeat")):
        return False
    if run.get("condition") not in {"baseline", "cortheon", "frontier"}:
        return False
    if any(type(run.get(field)) is not bool for field in ("correct", "delivered", "timed_out")):
        return False
    if not _scaling_bounded_text(run.get("process_error"), nullable=True):
        return False
    if run.get("expected_verdict") not in {"allow", "block"}:
        return False
    if not serialized_failure_ownership_valid(run):
        return False
    for field in ("inference_provider_id", "inference_model_id"):
        if not _scaling_bounded_text(run.get(field)):
            return False
    if run.get("execution_identity_valid") is not True:
        return False
    expected_provenance = {
        "pi": "pi_message_end",
        "opencode": "opencode_sanitized_export",
    }
    if run.get("execution_identity_provenance") not in expected_provenance.values():
        return False
    if run.get("execution_measurements_valid") is not True:
        return False
    if not _scaling_number(run.get("latency_seconds")):
        return False
    if not _scaling_number(run.get("policy_timeout_seconds")) or run["policy_timeout_seconds"] <= 0:
        return False
    if not _scaling_count(run.get("tokens")) or not _scaling_number(run.get("cost_usd")):
        return False
    if any(not _scaling_count(run.get(field)) for field in _NONNEGATIVE_COUNTERS):
        return False
    if any(not _scaling_count(run.get(field), positive=True) for field in _POSITIVE_COUNTERS):
        return False
    prior = run.get("prior_attempts")
    if not isinstance(prior, list) or len(prior) != run["retry_count"]:
        return False
    if not all(_scaling_attempt_valid(attempt) for attempt in prior):
        return False
    if not _scaling_bounded_text(run.get("retry_reason"), nullable=True):
        return False
    return not (run["retry_count"] == 0 and run["retry_reason"] is not None)
