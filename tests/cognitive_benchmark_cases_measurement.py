# ruff: noqa: F401

import argparse
import json
import subprocess
from dataclasses import asdict

import pytest
from scaling_support import report as _sealed_scaling_report

from cortheon.benchmark_core.execution_provenance import ProcessCapture
from cortheon.cognitive_benchmark import (
    DiagnosticCase,
    EvaluationOutcome,
    ImportCase,
    JoinCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    RunResult,
    SemanticCase,
    _audit_manifest,
    _blinded_case,
    _condition_summary,
    _delivery_succeeded,
    _event_statistics,
    _final_text,
    _frontier_comparison,
    _grade,
    _grade_patch_workspace,
    _integer_constants,
    _model_endpoint_health,
    _north_star_coverage,
    _paired_summary,
    _pi_provider_config,
    _postflight_probe,
    _provider_config,
    _workspace_environment,
    discover_benchmark_cases,
    discover_cases,
    discover_diagnostic_cases,
    discover_join_cases,
    discover_long_horizon_cases,
    discover_patch_cases,
    discover_planning_cases,
    discover_reasoning_cases,
    discover_semantic_cases,
    isolated_repository,
    run_frontier_cli_job,
    run_job,
    scaling_curve,
    verify_audit_bundle,
)
from cortheon.cognitive_benchmark import (
    main as cognitive_benchmark_main,
)


def test_paired_metrics_count_repeats_as_one_independent_case():
    results = []
    for repeat in range(6):
        results.extend(
            [
                RunResult(
                    "case",
                    repeat,
                    "baseline",
                    True,
                    "No.",
                    True,
                    False,
                    1.0,
                    10,
                    1,
                    0,
                    False,
                    None,
                    expected_verdict="allow",
                    evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
                    substrate_telemetry_valid=None,
                ),
                RunResult(
                    "case",
                    repeat,
                    "cortheon",
                    True,
                    "Yes — import pathlib",
                    True,
                    True,
                    1.2,
                    12,
                    1,
                    0,
                    False,
                    None,
                    expected_verdict="allow",
                    evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
                    substrate_telemetry_valid=True,
                    runtime_sessions_started=1,
                    runtime_observations_accepted=1,
                    runtime_sessions_completed=1,
                ),
            ]
        )

    baseline = _condition_summary(results, "baseline")
    paired = _paired_summary(results, seed=1)

    assert baseline["false_allows"] == 0
    assert paired["pairs"] == 6
    assert paired["independent_cases"] == 1
    assert paired["cortheon_wins"] == 1
    assert paired["cortheon_losses"] == 0
    assert paired["mcnemar_exact_p"] == 1.0


def test_condition_summary_counts_only_correct_semantic_runs_as_novel_deductions():
    results = [
        RunResult(
            f"case_{index}",
            0,
            "cortheon",
            ("answer",),
            text,
            True,
            correct,
            1.0,
            10,
            0,
            0,
            False,
            None,
            evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
            task_type="semantic_cross_document",
            substrate_telemetry_valid=True,
            runtime_sessions_completed=1 if correct else 0,
        )
        for index, (text, correct) in enumerate(
            (("complete chain", True), ("plausible shortcut", False))
        )
    ]

    summary = _condition_summary(results, "cortheon")

    assert summary["novel_source_grounded_deductions"] == 1
    assert summary["novel_source_grounded_deduction_rate"] == 0.5
    assert summary["correct_without_model_tool_calls"] == 1
    assert summary["verified_without_model_tool_calls"] == 1


def test_evidence_closed_answer_is_counted_as_verified_completion():
    result = RunResult(
        "case",
        0,
        "cortheon",
        ("answer",),
        "model-owned synthesis",
        True,
        True,
        1.0,
        10,
        1,
        0,
        False,
        None,
        evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
        task_type="novel_abductive_synthesis",
        substrate_telemetry_valid=True,
        runtime_sessions_started=1,
        runtime_observations_accepted=3,
        runtime_sessions_evidence_closed=1,
    )

    summary = _condition_summary([result], "cortheon")

    assert summary["substrate_telemetry_valid"] is True
    assert summary["runtime_sessions_evidence_closed"] == 1
    assert summary["verified_completions"] == 1


def test_frontier_control_does_not_pollute_small_model_pairing():
    results = [
        RunResult(
            "case",
            0,
            condition,
            True,
            "Yes — import pathlib",
            True,
            True,
            1.0,
            10,
            1,
            0,
            False,
            None,
            evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
            substrate_telemetry_valid=(condition == "cortheon"),
            runtime_sessions_completed=1 if condition == "cortheon" else 0,
        )
        for condition in ("baseline", "cortheon", "frontier")
    ]

    paired = _paired_summary(results, seed=1)

    assert paired["pairs"] == 1
    assert paired["invalid_pairs"] == 0


def test_frontier_comparison_requires_balanced_clean_outcomes():
    cortheon = {
        "eligible_runs": 8,
        "infrastructure_failures": 0,
        "eligible_accuracy": 0.875,
        "false_allow_rate": 0.0,
        "unclassified_blocks": 0,
        "delivery_failures": 0,
        "latency_seconds": {"mean": 2.0},
    }
    frontier = {
        "eligible_runs": 8,
        "infrastructure_failures": 0,
        "eligible_accuracy": 0.875,
        "false_allow_rate": 0.0,
        "unclassified_blocks": 0,
        "delivery_failures": 0,
        "latency_seconds": {"mean": 1.0},
    }

    comparison = _frontier_comparison(cortheon, frontier)

    assert comparison["complete_balanced_runs"]
    assert comparison["deliveries_accounted"]
    assert comparison["scoped_frontier_parity_observed"]
    assert comparison["eligible_accuracy_gap"] == 0.0
    assert comparison["mean_latency_ratio"] == 2.0


def test_infrastructure_failures_do_not_become_safety_errors_or_valid_pairs():
    results = [
        RunResult(
            "case",
            0,
            "baseline",
            True,
            "Yes — import pathlib",
            False,
            False,
            10.0,
            100,
            1,
            0,
            False,
            "opencode exited 1",
            expected_verdict="allow",
            failure_owner="external_infrastructure",
        ),
        RunResult(
            "case",
            0,
            "cortheon",
            True,
            "Yes — import pathlib",
            True,
            True,
            1.0,
            100,
            1,
            0,
            False,
            None,
            expected_verdict="allow",
            substrate_telemetry_valid=True,
            runtime_sessions_started=1,
            runtime_observations_accepted=1,
            runtime_sessions_completed=1,
        ),
    ]

    baseline = _condition_summary(results, "baseline")
    paired = _paired_summary(results, seed=1)

    assert baseline["eligible_runs"] == 0
    assert baseline["infrastructure_failures"] == 1
    assert baseline["false_allows"] == 0
    assert baseline["false_blocks"] == 0
    assert paired["pairs"] == 0
    assert paired["invalid_pairs"] == 1


def test_tool_workload_counters_exclude_infrastructure_failed_runs():
    def _run(process_error, *, errors=0, executed=0, blocked=0, unavailable=0):
        return RunResult(
            "case",
            0,
            "baseline",
            True,
            "Yes — import pathlib",
            process_error is None,
            process_error is None,
            1.0,
            10,
            1,
            errors,
            False,
            process_error,
            expected_verdict="allow",
            failure_owner=("external_infrastructure" if process_error else None),
            host_tool_executions=executed,
            blocked_tool_calls=blocked,
            unavailable_tool_calls=unavailable,
        )

    results = [
        _run(None, errors=1, executed=2, blocked=3, unavailable=4),
        _run("pi exited 137", errors=5, executed=5, blocked=5, unavailable=5),
    ]

    summary = _condition_summary(results, "baseline")

    # Like model_tool_calls, the workload counters aggregate over eligible
    # runs only; the process-error run contributes nothing.
    assert summary["model_tool_calls"] == 1
    assert summary["tool_errors"] == 1
    assert summary["host_tool_executions"] == 2
    assert summary["blocked_tool_calls"] == 3
    assert summary["unavailable_tool_calls"] == 4
    # The infrastructure failure itself stays separately visible.
    assert summary["runs"] == 2
    assert summary["eligible_runs"] == 1
    assert summary["infrastructure_failures"] == 1
    assert summary["process_errors"] == 1


def test_substrate_telemetry_counts_released_uncertified_terminal() -> None:
    """An abandoned session with accepted observations proves execution.

    Regression: released-uncertified runs end abandoned rather than
    completed, and were marked telemetry-invalid, failing the
    substrate_execution_observed gate on otherwise clean runs.
    """

    from cortheon.cognitive_benchmark import _substrate_telemetry_valid

    engaged_released = {
        "sessions_started": 1,
        "observations_accepted": 4,
        "sessions_completed": 0,
        "sessions_evidence_closed": 0,
        "sessions_abandoned": 1,
        "completion_withheld": 2,
    }
    assert _substrate_telemetry_valid(engaged_released) is True

    certified = dict(engaged_released, sessions_completed=1, sessions_abandoned=0)
    assert _substrate_telemetry_valid(certified) is True

    never_engaged = dict(engaged_released, observations_accepted=0)
    assert _substrate_telemetry_valid(never_engaged) is False

    double_terminal = dict(engaged_released, sessions_completed=1)
    assert _substrate_telemetry_valid(double_terminal) is False

    assert _substrate_telemetry_valid(None) is False


def test_infrastructure_retry_only_fires_when_endpoint_was_dead() -> None:
    """Retry is bounded to probed endpoint death, never real outcomes.

    Regression: baseline-arm jobs died with empty answers whenever the
    serving endpoint degraded mid-benchmark, invalidating pairs; a failure
    while the endpoint still answered must never be retried.
    """

    import argparse as _argparse
    from unittest import mock

    from cortheon.cognitive_benchmark import (
        RunResult,
        _retry_after_infrastructure_death,
    )

    args = _argparse.Namespace(base_url="", api_key="", model_id="m")
    case = mock.Mock(case_id="case-x")
    failed = mock.Mock(spec=RunResult, process_error="opencode exited 137", timed_out=False)
    healthy_probe = mock.Mock()

    # Endpoint answers: real outcome, no retry.
    out = _retry_after_infrastructure_death(
        args, case, 0, "baseline", failed, probe=healthy_probe, sleep=lambda _: None
    )
    assert out is failed

    # Endpoint dead then recovered: one retry via run_job.
    dead_then_alive = mock.Mock(side_effect=[ValueError("down"), ValueError("down"), None])
    retried = mock.Mock(spec=RunResult)
    with mock.patch("cortheon.cognitive_benchmark.run_job", return_value=retried) as runner:
        out = _retry_after_infrastructure_death(
            args,
            case,
            0,
            "baseline",
            failed,
            probe=dead_then_alive,
            sleep=lambda _: None,
        )
    assert out is retried
    assert runner.call_count == 1

    # Endpoint never recovers: keep the original failure.
    always_dead = mock.Mock(side_effect=ValueError("down"))
    with mock.patch("cortheon.cognitive_benchmark.run_job") as runner:
        out = _retry_after_infrastructure_death(
            args,
            case,
            0,
            "baseline",
            failed,
            probe=always_dead,
            sleep=lambda _: None,
            recovery_attempts=3,
        )
    assert out is failed
    assert runner.call_count == 0

    # Successful results and frontier jobs pass through untouched.
    ok = mock.Mock(spec=RunResult, process_error=None, timed_out=False)
    assert _retry_after_infrastructure_death(args, case, 0, "baseline", ok) is ok
    assert _retry_after_infrastructure_death(args, case, 0, "frontier", failed) is failed


def test_infrastructure_retry_covers_missing_assistant_answer() -> None:
    """An events-without-text run retries once; step exhaustion never can.

    Regression: JSON-format runs that ended with no assistant message were
    classified as infrastructure but only retried when the model endpoint
    probe failed, so a healthy-endpoint harness anomaly consumed the pair.
    """

    import argparse as _argparse
    from unittest import mock

    from cortheon.cognitive_benchmark import (
        RunResult,
        _retry_after_infrastructure_death,
    )

    args = _argparse.Namespace(base_url="", api_key="", model_id="m")
    case = mock.Mock(case_id="case-y")
    failed = mock.Mock(
        spec=RunResult,
        process_error="opencode returned no assistant answer",
        timed_out=False,
    )
    retried = mock.Mock(spec=RunResult)
    with mock.patch("cortheon.cognitive_benchmark.run_job", return_value=retried) as runner:
        probe = mock.Mock(side_effect=[ValueError("down"), None])
        out = _retry_after_infrastructure_death(
            args,
            case,
            0,
            "cortheon",
            failed,
            probe=probe,
            sleep=lambda _seconds: None,
        )
    assert out is retried
    assert runner.call_count == 1
    assert probe.call_count == 2
