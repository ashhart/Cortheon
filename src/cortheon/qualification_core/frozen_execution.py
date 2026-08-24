"""Run one attested job against the frozen OpenCode comparator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

from cortheon.benchmark_core.models import BenchmarkCase, RunResult
from cortheon.benchmark_core.runner_local import run_job
from cortheon.qualification_core.conditions import (
    CONDITION_REGISTRY_VERSION,
    OLD_PLANNER,
    condition_record,
)
from cortheon.qualification_core.frozen_runtime import FrozenRuntime


def _metric_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    if set(before) != set(after):
        raise ValueError("frozen runtime metric membership changed")
    delta = {key: after[key] - before[key] for key in before}
    if any(type(value) is not int or value < 0 for value in delta.values()):
        raise ValueError("frozen runtime metrics are not monotonic integers")
    return delta


def _run_frozen_evidence(
    args: argparse.Namespace,
    case: BenchmarkCase,
    *,
    repeat: int,
    runtime: FrozenRuntime,
) -> tuple[RunResult, dict[str, int], bool, int]:
    """Execute and clean the frozen stack without asserting prior qualification."""

    if args.host != "opencode":
        raise ValueError("the frozen old planner is available only on OpenCode")
    if runtime.health().get("active_sessions") != 0:
        raise ValueError("frozen runtime was not empty before the job")
    job_args = argparse.Namespace(**vars(args))
    job_args.runtime_url = runtime.url
    job_args.evaluation_plugin_path = runtime.adapter
    before = runtime.metrics()
    result = run_job(
        job_args,
        case,
        repeat=repeat,
        treatment=True,
        condition=OLD_PLANNER,
        control_token=runtime.token,
        evaluator_control_payload=runtime.control_payload(),
    )
    cleaned = runtime.abandon_active()
    after = runtime.metrics()
    delta = _metric_delta(before, after)
    lifecycle_valid = bool(
        delta["sessions_started"] == 1
        and delta["observations_accepted"] >= 1
        and delta["sessions_completed"]
        + delta["sessions_evidence_closed"]
        + delta["sessions_abandoned"]
        == 1
        and runtime.health().get("active_sessions") == 0
        and runtime.unchanged()
    )
    return result, delta, lifecycle_valid, cleaned


def run_frozen_job(
    args: argparse.Namespace,
    case: BenchmarkCase,
    *,
    repeat: int,
    runtime: FrozenRuntime,
) -> RunResult:
    """Execute unchanged historical code with evaluator-owned cleanup."""

    result, delta, lifecycle_valid, _cleaned = _run_frozen_evidence(
        args,
        case,
        repeat=repeat,
        runtime=runtime,
    )
    record = condition_record(OLD_PLANNER, host="opencode")
    if not record["available"]:
        raise ValueError(str(record["unavailable_reason"]))
    attested = bool(
        lifecycle_valid and result.execution_identity_valid and result.execution_measurements_valid
    )
    updates: dict[str, Any] = {
        "substrate_telemetry_valid": lifecycle_valid,
        "runtime_sessions_started": delta["sessions_started"],
        "runtime_observations_accepted": delta["observations_accepted"],
        "runtime_sessions_completed": delta["sessions_completed"],
        "runtime_sessions_evidence_closed": delta["sessions_evidence_closed"],
        "runtime_sessions_abandoned": delta["sessions_abandoned"],
        "runtime_completion_withheld": delta["completion_withheld"],
        "runtime_controller_decisions": delta["controller_decisions"],
        "runtime_controller_alternatives_considered": delta["controller_alternatives_considered"],
        "condition_registry_version": CONDITION_REGISTRY_VERSION,
        "condition_config_sha256": record["config_sha256"],
        "condition_implementation_sha256": record["implementation_sha256"],
        "condition_requires_runtime_completion": True,
        "condition_profile_receipt_valid": attested,
        "condition_observed_config_sha256": record["config_sha256"] if attested else None,
        "condition_observed_implementation_sha256": (
            record["implementation_sha256"] if attested else None
        ),
        "condition_adapter_receipt_valid": attested,
        "condition_operator_counts": None,
    }
    return replace(result, **updates)
