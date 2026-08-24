"""Evaluator-owned infrastructure retry provenance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from cortheon.benchmark_core._compat import facade
from cortheon.benchmark_core.models import AttemptRecord, BenchmarkCase, RunResult


def _attempt_record(result: RunResult, index: int) -> AttemptRecord:
    return AttemptRecord(
        attempt_index=index,
        latency_seconds=result.latency_seconds,
        tokens=result.tokens,
        tool_calls=result.tool_calls,
        cost_usd=result.cost_usd,
        timed_out=result.timed_out,
        process_error=result.process_error,
        failure_owner=result.failure_owner,
        terminal_status=result.evaluator_outcome.terminal_status,
        provider_id=result.inference_provider_id,
        model_id=result.inference_model_id,
        identity_valid=result.execution_identity_valid,
        identity_provenance=result.execution_identity_provenance,
        measurements_valid=result.execution_measurements_valid,
        policy_timeout_seconds=result.policy_timeout_seconds,
        policy_max_steps=result.policy_max_steps,
        policy_max_tool_calls=result.policy_max_tool_calls,
        policy_context_tokens=result.policy_context_tokens,
        policy_output_tokens=result.policy_output_tokens,
        budget_reason=result.policy_budget_reason,
    )


def _attach_retry(
    original: RunResult,
    retried: RunResult,
    *,
    reason: str,
) -> RunResult:
    if type(original) is not RunResult or type(retried) is not RunResult:
        return retried
    prior_tokens = original.tokens
    prior_cost = original.cost_usd
    retried.tokens = (
        prior_tokens + retried.tokens
        if prior_tokens is not None and retried.tokens is not None
        else None
    )
    retried.cost_usd = (
        prior_cost + retried.cost_usd
        if prior_cost is not None and retried.cost_usd is not None
        else None
    )
    retried.latency_seconds += original.latency_seconds
    retried.tool_calls += original.tool_calls
    retried.tool_errors += original.tool_errors
    retried.observed_steps += original.observed_steps
    retried.execution_identity_valid = bool(
        original.execution_identity_valid and retried.execution_identity_valid
    )
    retried.execution_measurements_valid = bool(
        original.execution_measurements_valid and retried.execution_measurements_valid
    )
    if not retried.execution_identity_valid:
        retried.execution_identity_reason = "prior_attempt_identity_invalid"
    if not retried.execution_measurements_valid:
        retried.execution_measurement_reason = "prior_attempt_measurement_invalid"
    retried.retry_count = original.retry_count + 1
    retried.retry_reason = reason
    retried.prior_attempts = (
        *original.prior_attempts,
        _attempt_record(original, original.retry_count + 1),
    )
    return retried


def _retry_after_infrastructure_death(
    job_args: argparse.Namespace,
    case: BenchmarkCase,
    repeat: int,
    condition: str,
    result: RunResult,
    *,
    probe: Any = None,
    sleep: Any = time.sleep,
    recovery_attempts: int = 12,
) -> RunResult:
    """Retry once without erasing the evaluator-owned failed attempt."""

    if condition == "frontier" or not result.process_error or result.timed_out:
        return result
    reason = "model_endpoint_down"
    check = probe or (
        lambda: facade()._model_endpoint_health(
            job_args.base_url,
            api_key=job_args.api_key,
            model_id=job_args.model_id,
            inference_timeout=15.0,
        )
    )
    try:
        check()
        return result
    except ValueError:
        # Only this evaluator-side health failure may reclassify a candidate
        # delivery failure as external infrastructure. Error text never does.
        result.failure_owner = "external_infrastructure"
    for _ in range(recovery_attempts):
        sleep(5)
        try:
            check()
            break
        except ValueError:
            continue
    else:
        return result
    retry_count = getattr(result, "retry_count", 0)
    if type(retry_count) is not int or retry_count < 0:
        retry_count = 0
    print(
        json.dumps(
            {
                "infrastructure_retry": case.case_id,
                "condition": condition,
                "reason": reason,
                "attempt": retry_count + 2,
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    retried = facade().run_job(
        job_args,
        case,
        repeat=repeat,
        treatment=condition == "cortheon",
        condition=condition,
    )
    return _attach_retry(result, retried, reason=reason)
