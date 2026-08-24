"""Execution provenance and metering summaries for benchmark conditions."""

from __future__ import annotations

import statistics
from typing import Any

from cortheon.benchmark_core.models import RunResult


def execution_summary(selected: list[RunResult]) -> dict[str, Any]:
    valid_identity_provenance = {"pi_message_end", "opencode_sanitized_export"}
    measurement_invalid = sum(
        item.execution_measurements_valid is not True
        or item.tokens is None
        or item.cost_usd is None
        for item in selected
    )
    policies = {
        (
            item.policy_timeout_seconds,
            item.policy_max_steps,
            item.policy_max_tool_calls,
            item.policy_context_tokens,
            item.policy_output_tokens,
        )
        for item in selected
        if item.policy_timeout_seconds > 0
        and item.policy_max_steps > 0
        and item.policy_max_tool_calls >= 0
        and item.policy_context_tokens > 0
        and item.policy_output_tokens > 0
    }
    policy_invalid = sum(
        type(item.observed_steps) is not int
        or item.observed_steps < 0
        or item.observed_steps > item.policy_max_steps
        or type(item.tool_calls) is not int
        or item.tool_calls < 0
        or item.tool_calls > item.policy_max_tool_calls
        for item in selected
    )
    prior_attempts = [attempt for item in selected for attempt in item.prior_attempts]
    return {
        "retry_count": sum(item.retry_count for item in selected),
        "retried_runs": sum(item.retry_count > 0 for item in selected),
        "prior_infrastructure_failures": sum(
            attempt.failure_owner == "external_infrastructure" for attempt in prior_attempts
        ),
        "prior_candidate_delivery_failures": sum(
            attempt.failure_owner == "candidate" for attempt in prior_attempts
        ),
        "prior_failure_ownership_invalid": sum(
            attempt.failure_owner not in {"candidate", "external_infrastructure"}
            and (attempt.process_error is not None or attempt.timed_out)
            for attempt in prior_attempts
        ),
        "execution_identity_invalid_runs": sum(
            item.execution_identity_valid is not True
            or item.execution_identity_provenance not in valid_identity_provenance
            for item in selected
        ),
        "execution_measurement_invalid_runs": measurement_invalid,
        "execution_policy": list(next(iter(policies))) if len(policies) == 1 else None,
        "execution_policy_invalid_runs": (len(selected) if len(policies) != 1 else policy_invalid),
        "mean_tokens": (
            statistics.mean(item.tokens for item in selected if item.tokens is not None)
            if selected and measurement_invalid == 0
            else None
        ),
        "total_cost_usd": (
            sum(item.cost_usd for item in selected if item.cost_usd is not None)
            if selected and measurement_invalid == 0
            else None
        ),
        "mean_cost_usd": (
            statistics.mean(item.cost_usd for item in selected if item.cost_usd is not None)
            if selected and measurement_invalid == 0
            else None
        ),
    }
