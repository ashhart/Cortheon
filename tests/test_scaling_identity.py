"""Hostile identity and collision tests for diagnostic scaling curves."""

from __future__ import annotations

from copy import deepcopy

import pytest
from scaling_support import refresh_bindings, report, reseal

from cortheon.benchmark_core.audit import _scaling_condition
from cortheon.cognitive_benchmark import main as benchmark_main
from cortheon.cognitive_benchmark import scaling_curve


def _runs(*, treatment: tuple[bool, ...] = (True, True)) -> list[dict]:
    runs = []
    for condition, outcomes in (
        ("baseline", (False, True)),
        ("cortheon", treatment),
    ):
        runs.extend(
            {
                "case_id": f"case_{index}",
                "repeat": 0,
                "condition": condition,
                "correct": correct,
                "delivered": True,
                "timed_out": False,
                "process_error": None,
                "inference_model_id": "demo",
                "evaluator_outcome": {
                    "schema_version": 1,
                    "transport": "pi",
                    "terminal_status": "success",
                    "terminal_provenance": "pi_assistant",
                    "finish_reason": "stop",
                },
                "substrate_telemetry_valid": condition == "cortheon",
                "runtime_sessions_completed": int(condition == "cortheon"),
                "runtime_sessions_evidence_closed": 0,
                "latency_seconds": 1.0,
                "tool_calls": 0,
                "cost_usd": 0.0,
            }
            for index, correct in enumerate(outcomes)
        )
    return runs


def _set(payload: dict, path: str, value: object) -> None:
    target = payload
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _align_top_level(report: dict, path: str, value: object) -> None:
    if path == "repository.snapshot_sha256":
        report["repository"]["snapshot_sha256"] = value
    elif path == "case_bank.suite":
        report["suite"] = value
    elif path == "schedule.seed":
        report["seed"] = value
    elif path == "host.kind":
        report["host"] = value
        provenance = "opencode_sanitized_export" if value == "opencode" else "pi_message_end"
        for run in report["runs"]:
            run["execution_identity_provenance"] = provenance
    elif path == "inference.model_id":
        report["model"] = f"local/{value}"
        report["experiment_identity"]["inference"]["observed_model_id"] = value
        for run in report["runs"]:
            run["inference_model_id"] = value
    elif path == "limits.timeout_seconds":
        for run in report["runs"]:
            run["policy_timeout_seconds"] = value
    elif path == "limits.context_tokens":
        for run in report["runs"]:
            run["policy_context_tokens"] = value
    elif path == "limits.output_tokens":
        for run in report["runs"]:
            run["policy_output_tokens"] = value
    elif path == "limits.max_tool_calls":
        for run in report["runs"]:
            run["policy_max_tool_calls"] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("repository.snapshot_sha256", "9" * 64),
        ("case_bank.suite", "reasoning"),
        ("schedule.seed", 8),
        ("host.kind", "opencode"),
        ("host.configured_command", "pi-next"),
        ("host.executable_sha256", "9" * 64),
        ("host.version", "pi 2.0"),
        ("cortheon_runtime.artifact_sha256", "9" * 64),
        ("cortheon_runtime.adapter_sha256", "9" * 64),
        ("cortheon_runtime.endpoint_sha256", "9" * 64),
        ("cortheon_runtime.observed_version", "2.0"),
        ("cortheon_runtime.observed_protocol_version", "2.0.0"),
        ("cortheon_runtime.observed_source_fingerprint", "9" * 64),
        ("inference.model_id", "other"),
        ("inference.endpoint_sha256", "9" * 64),
        ("inference.registered_artifact_sha256", "9" * 64),
        ("limits.timeout_seconds", 30.0),
        ("limits.context_tokens", 4096),
        ("limits.output_tokens", 256),
        ("limits.max_tool_calls", 8),
    ],
)
def test_any_non_axis_identity_change_creates_a_separate_family(path, value):
    first = report(_runs())
    second = deepcopy(first)
    _set(second["experiment_identity"], path, value)
    _align_top_level(second, path, value)
    reseal(second)

    curve = scaling_curve([second, first])

    assert len(curve["points"]) == 2
    assert len(curve["families"]) == 2
    assert all(point["reports"] == 1 for point in curve["points"])


def test_step_budget_is_the_only_deliberate_axis_for_one_honest_family():
    low = report(_runs(treatment=(False, True)), budget=4)
    high = report(_runs(treatment=(True, True)), budget=8)

    curve = scaling_curve([high, low])

    assert [point["budget"] for point in curve["points"]] == [4, 8]
    assert len(curve["families"]) == 1
    assert curve["families"][0]["budgets"] == [4, 8]
    assert curve["diagnostic_only"] is True
    assert curve["claim_eligible"] is False


def test_missing_identity_and_old_schema_are_diagnostic_only():
    missing = report(_runs())
    missing.pop("experiment_identity")
    reseal(missing)
    stale = report(_runs())
    stale["schema_version"] = 12
    reseal(stale)

    curve = scaling_curve([stale, missing])

    assert curve["points"] == []
    assert curve["families"] == []
    assert curve["diagnostics"]["invalid_reports"] == 2
    assert set(curve["diagnostics"]["reason_counts"]) == {
        "missing_experiment_identity",
        "unsupported_report_schema",
    }


def test_malformed_registered_model_identity_never_becomes_unknown():
    malformed = report(_runs())
    malformed["experiment_identity"]["inference"]["registered_artifact_sha256"] = None
    reseal(malformed)

    curve = scaling_curve([malformed])

    assert curve["points"] == []
    assert curve["diagnostics"]["reason_counts"] == {"invalid_experiment_identity": 1}


def test_observed_model_id_must_match_the_registered_model():
    mismatched = report(_runs())
    mismatched["experiment_identity"]["inference"]["observed_model_id"] = "mutable-alias"
    reseal(mismatched)

    curve = scaling_curve([mismatched])

    assert curve["points"] == []
    assert curve["diagnostics"]["reason_counts"] == {"invalid_experiment_identity": 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", []),
        ("repeat", True),
        ("latency_seconds", float("nan")),
        ("cost_usd", float("inf")),
        ("tool_calls", -1),
        ("runtime_controller_decisions", "a billion"),
        ("runtime_controller_alternatives_considered", -1),
    ],
)
def test_malformed_run_cells_are_bounded_diagnostics(field, value):
    malformed = report(_runs())
    malformed["runs"][0][field] = value
    reseal(malformed)

    curve = scaling_curve([malformed])

    assert curve["points"] == []
    assert curve["diagnostics"]["reason_counts"] == {"invalid_run_matrix": 1}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_limits_fail_closed(value):
    malformed = report(_runs())
    malformed["experiment_identity"]["limits"]["timeout_seconds"] = value
    reseal(malformed)

    curve = scaling_curve([malformed])

    assert curve["points"] == []
    assert curve["diagnostics"]["reason_counts"] == {"invalid_experiment_identity": 1}


def test_zero_tool_call_policy_is_an_honest_bound_not_missing_identity():
    zero_tools = report(_runs())
    zero_tools["experiment_identity"]["limits"]["max_tool_calls"] = 0
    for run in zero_tools["runs"]:
        run["policy_max_tool_calls"] = 0
    reseal(zero_tools)

    curve = scaling_curve([zero_tools])

    assert len(curve["points"]) == 1
    assert curve["diagnostics"]["invalid_reports"] == 0


def test_retry_attempt_cannot_hide_a_different_execution_policy():
    retried = report(_runs())
    run = retried["runs"][0]
    run["retry_count"] = 1
    run["retry_reason"] = "model_endpoint_down"
    run["prior_attempts"] = [
        {
            "attempt_index": 1,
            "latency_seconds": 1.0,
            "tokens": 10,
            "tool_calls": 0,
            "cost_usd": 0.0,
            "timed_out": False,
            "process_error": "endpoint died",
            "failure_owner": "external_infrastructure",
            "terminal_status": "transport_error",
            "provider_id": "local",
            "model_id": "demo",
            "identity_valid": True,
            "identity_provenance": "pi_message_end",
            "measurements_valid": True,
            "policy_timeout_seconds": 30.0,
            "policy_max_steps": 4,
            "policy_max_tool_calls": 16,
            "policy_context_tokens": 8192,
            "policy_output_tokens": 512,
            "budget_reason": None,
        }
    ]
    reseal(retried)

    curve = scaling_curve([retried])

    assert curve["points"] == []
    assert curve["diagnostics"]["reason_counts"] == {"report_identity_mismatch": 1}


def test_registered_artifact_digest_must_be_canonical_lowercase_sha256():
    with pytest.raises(SystemExit, match="64 lowercase hex"):
        benchmark_main(["--inference-artifact-sha256", "A" * 64])


def test_duplicate_report_collision_is_visible_and_permutation_invariant():
    original = report(_runs())
    duplicate = deepcopy(original)

    forward = scaling_curve([original, duplicate])
    reverse = scaling_curve([duplicate, original])

    assert forward == reverse
    assert forward["points"] == []
    assert forward["diagnostics"]["duplicate_report_groups"] == 1


def test_duplicate_run_cell_is_visible_and_permutation_invariant():
    original = report(_runs())
    duplicate = deepcopy(original["runs"][0])
    original["runs"].append(duplicate)
    reseal(original)
    reversed_report = deepcopy(original)
    reversed_report["runs"].reverse()
    reseal(reversed_report)

    forward = scaling_curve([original])
    reverse = scaling_curve([reversed_report])

    assert forward["points"] == reverse["points"] == []
    assert forward["diagnostics"]["reason_counts"] == {"duplicate_run_cells": 1}
    assert reverse["diagnostics"]["reason_counts"] == {"duplicate_run_cells": 1}


def test_two_distinct_reports_cannot_share_one_family_budget_cell():
    first = report(_runs())
    second = report(_runs(treatment=(False, False)))

    curve = scaling_curve([second, first])

    assert curve["points"] == []
    assert curve["diagnostics"]["budget_collision_groups"] == 1


def test_case_bank_and_schedule_order_are_bound_into_the_family():
    first = report(_runs())
    renamed = deepcopy(first)
    renamed["cases"][0]["case_id"] = "renamed"
    for run in renamed["runs"]:
        if run["case_id"] == "case_0":
            run["case_id"] = "renamed"
    refresh_bindings(renamed)
    reordered = deepcopy(first)
    reordered["runs"].reverse()
    refresh_bindings(reordered)

    curve = scaling_curve([reordered, renamed, first])

    assert len(curve["points"]) == 3
    assert len(curve["families"]) == 3


def test_process_errors_remain_in_the_scheduled_denominator():
    broken = report(_runs())
    run = next(
        row
        for row in broken["runs"]
        if row["condition"] == "cortheon" and row["case_id"] == "case_0"
    )
    run["process_error"] = "host died"
    run["delivered"] = False
    run["evaluator_outcome"] = {
        "schema_version": 1,
        "transport": "pi",
        "terminal_status": "transport_error",
        "terminal_provenance": "pi_assistant",
        "finish_reason": "process_error",
    }
    run["failure_owner"] = "candidate"
    reseal(broken)

    condition = scaling_curve([broken])["points"][0]["conditions"]["cortheon"]

    assert condition["runs"] == condition["eligible_runs"] == 2
    assert condition["accuracy"] == 0.5
    assert condition["delivery_rate"] == 0.5
    assert condition["delivery_failures"] == 1
    assert condition["delivery_failure_rate"] == 0.5


def test_empty_scaling_condition_has_no_invented_zero_rates():
    condition = _scaling_condition(_runs(), "frontier")

    assert condition["runs"] == condition["eligible_runs"] == 0
    for field in (
        "accuracy",
        "delivery_rate",
        "false_allow_rate",
        "false_block_rate",
        "unclassified_block_rate",
        "delivery_failure_rate",
        "mean_latency_seconds",
        "mean_tool_calls",
        "total_cost_usd",
    ):
        assert condition[field] is None
