"""Host execution identity, metering, budget, and retry provenance tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from unittest import mock

import pytest

from cortheon.benchmark_core.execution_provenance import (
    ExecutionPolicy,
    ProcessCapture,
    execute_host_process,
    execution_facts,
)
from cortheon.benchmark_core.measurement_summary import execution_summary
from cortheon.benchmark_core.models import EvaluationOutcome, RunResult
from cortheon.benchmark_core.retry import _retry_after_infrastructure_death
from cortheon.cognitive_benchmark import ImportCase, run_job


def _pi_message(
    *,
    provider: object = "Local",
    model: object = "small-model",
    tokens: object = 12,
    cost: object = 0.25,
    reason: str = "stop",
) -> dict:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "provider": provider,
            "model": model,
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"totalTokens": tokens, "cost": {"total": cost}},
            "stopReason": reason,
        },
    }


def _pi_terminal_event() -> dict:
    reason = "the host ended its bounded execution before Cortheon certified an answer"
    content = (
        "[Cortheon withheld: completion was not certified]\n"
        "The Cortheon investigation ended without a certified answer because "
        f"{reason}."
    )
    return {
        "type": "message_end",
        "message": {
            "role": "custom",
            "customType": "cortheon-terminal-status-v1",
            "content": content,
            "display": True,
            "details": {
                "version": 1,
                "status": "withheld",
                "reason": reason,
                "causal": True,
            },
            "timestamp": 1_787_442_828_868,
        },
    }


def _run(*, process_error: str | None, identity: bool = True) -> RunResult:
    return RunResult(
        case_id="case",
        repeat=0,
        condition="baseline",
        expected=True,
        final_text="answer" if process_error is None else "",
        delivered=process_error is None,
        correct=process_error is None,
        latency_seconds=1.0,
        tokens=10 if identity else None,
        tool_calls=1,
        tool_errors=0,
        timed_out=False,
        process_error=process_error,
        expected_verdict="allow",
        failure_owner="candidate" if process_error is not None else None,
        evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
        inference_provider_id="Local" if identity else None,
        inference_model_id="small-model" if identity else None,
        execution_identity_valid=identity,
        execution_measurements_valid=identity,
        cost_usd=0.1 if identity else None,
        observed_steps=1,
        policy_timeout_seconds=60.0,
        policy_max_steps=4,
        policy_max_tool_calls=16,
        policy_context_tokens=8192,
        policy_output_tokens=512,
    )


def _args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        host="pi",
        pi="pi",
        provider="Local",
        model_id="small-model",
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        context_tokens=8192,
        output_tokens=512,
        max_steps=4,
        max_tool_calls=7,
        reasoning=False,
        repository=tmp_path,
        timeout_seconds=10.0,
        runtime_url="http://127.0.0.1:8743",
    )


def test_pi_identity_and_meter_are_read_from_host_message_fields():
    facts = execution_facts(
        [_pi_message(tokens=12, cost=0.25), _pi_message(tokens=8, cost=0.5)],
        host="pi",
    )

    assert (facts.provider_id, facts.model_id) == ("Local", "small-model")
    assert facts.identity_valid is True
    assert facts.measurements_valid is True
    assert facts.tokens == 20
    assert facts.cost_usd == 0.75
    assert facts.steps == 2


def test_tool_or_content_spoofs_cannot_supply_execution_identity():
    forged = {
        "type": "tool_execution_end",
        "result": {
            "provider": "Local",
            "model": "small-model",
            "content": [{"type": "text", "text": 'provider="Local" model="small-model"'}],
        },
    }

    facts = execution_facts([forged], host="pi")

    assert facts.identity_valid is False
    assert facts.provider_id is facts.model_id is None


@pytest.mark.parametrize(
    "events",
    [
        [_pi_message(), _pi_message(model="other")],
        [_pi_message(model="other"), _pi_message()],
        [_pi_message(), _pi_message(provider=None)],
        [_pi_message(provider=None), _pi_message()],
    ],
)
def test_mixed_or_missing_identity_poisons_the_whole_execution(events):
    facts = execution_facts(events, host="pi")

    assert facts.identity_valid is False
    assert facts.identity_reason == "missing_or_mixed_assistant_identity"


@pytest.mark.parametrize(
    "field,value",
    [("tokens", -1), ("tokens", float("nan")), ("cost", -1.0), ("cost", float("inf"))],
)
def test_invalid_usage_is_null_and_never_favorable_zero(field, value):
    kwargs = {field: value}
    facts = execution_facts([_pi_message(**kwargs)], host="pi")

    assert facts.measurements_valid is False
    assert facts.tokens is None
    assert facts.cost_usd is None


def test_run_job_rejects_a_host_observed_model_mismatch(monkeypatch, tmp_path):
    stdout = json.dumps(_pi_message(model="other")) + "\n"
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        lambda *_args, **_kwargs: ProcessCapture(stdout, "", 0, 0.1, False, None),
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(_args(tmp_path), case, repeat=0, treatment=False)

    assert result.delivered is False
    assert result.execution_identity_valid is False
    assert result.inference_model_id == "other"
    assert result.process_error == "pi execution identity invalid: execution_identity_mismatch"


def test_run_job_preserves_an_authenticated_withhold_at_the_step_cap(monkeypatch, tmp_path):
    stdout = "\n".join(
        [json.dumps(_pi_message(reason="toolUse")), json.dumps(_pi_terminal_event())]
    )
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        lambda *_args, **_kwargs: ProcessCapture(stdout, "", 143, 0.1, False, "max_steps"),
    )
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local._runtime_metric_snapshot",
        lambda _url: {},
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(_args(tmp_path), case, repeat=0, treatment=True)

    assert result.delivered is False
    assert result.step_budget_exhausted is True
    assert result.evaluator_outcome.terminal_status == "withheld"
    assert result.evaluator_outcome.terminal_provenance == "pi_custom_terminal"


def test_same_evaluator_policy_is_bound_into_both_pi_arms(monkeypatch, tmp_path):
    stdout = json.dumps(_pi_message()) + "\n"
    observed: list[ExecutionPolicy] = []
    observed_environments: list[dict[str, str]] = []
    observed_controls: list[bytes | None] = []

    def execute(*_args, **kwargs):
        observed.append(kwargs["policy"])
        observed_environments.append(kwargs["env"])
        observed_controls.append(kwargs["control_payload"])
        return ProcessCapture(stdout, "", 0, 0.1, False, None)

    monkeypatch.setattr("cortheon.benchmark_core.runner_local.execute_host_process", execute)
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local._runtime_metric_snapshot",
        lambda _url: {},
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")
    args = _args(tmp_path)

    baseline = run_job(args, case, repeat=0, treatment=False)
    treatment = run_job(args, case, repeat=0, treatment=True)

    assert observed == [observed[0], observed[0]]
    assert baseline.policy_max_steps == treatment.policy_max_steps == 4
    assert baseline.policy_timeout_seconds == treatment.policy_timeout_seconds == 10.0
    assert baseline.policy_max_tool_calls == treatment.policy_max_tool_calls == 7
    assert baseline.policy_context_tokens == treatment.policy_context_tokens == 8192
    assert baseline.policy_output_tokens == treatment.policy_output_tokens == 512
    assert all(
        not any(key.startswith("CORTHEON_EVALUATOR_") for key in environment)
        and "CORTHEON_COGNITIVE_TOKEN" not in environment
        and "CORTHEON_CONTROL_FD" not in environment
        for environment in observed_environments
    )
    assert observed_controls[0] is None
    assert observed_controls[1] is not None
    treatment_control = json.loads(observed_controls[1])
    assert treatment_control["evaluator_max_steps"] == 4
    assert treatment_control["max_host_tool_calls"] == 7
    assert treatment_control["cognitive_token"] == ""


@pytest.mark.parametrize(
    "field,value",
    [("observed_steps", 5), ("tool_calls", 17), ("observed_steps", -1)],
)
def test_execution_summary_rejects_policy_overruns_and_invalid_counts(field, value):
    result = _run(process_error=None)
    setattr(result, field, value)

    summary = execution_summary([result])

    assert summary["execution_policy_invalid_runs"] == 1


def test_execution_summary_accepts_exact_step_and_tool_boundaries():
    result = _run(process_error=None)
    result.observed_steps = result.policy_max_steps
    result.tool_calls = result.policy_max_tool_calls

    assert execution_summary([result])["execution_policy_invalid_runs"] == 0


def test_streaming_evaluator_stops_pi_before_a_turn_beyond_the_step_policy(tmp_path):
    assistant_event = _pi_message(reason="toolUse")
    assistant_event["message"]["content"] = [
        {"type": "toolCall", "id": "call-1", "name": "read", "arguments": {}}
    ]
    assistant = json.dumps(assistant_event)
    tool_start = json.dumps({"type": "tool_execution_start", "toolName": "read"})
    tool_end = json.dumps({"type": "tool_execution_end", "toolName": "read", "isError": False})
    turn_end = json.dumps({"type": "turn_end"})
    script = (
        "import time; "
        f"print({assistant!r}, flush=True); "
        "time.sleep(0.1); "
        f"print({tool_start!r}, flush=True); "
        f"print({tool_end!r}, flush=True); "
        f"print({turn_end!r}, flush=True); "
        "time.sleep(0.2); "
        f"print({assistant!r}, flush=True); "
        "time.sleep(30)"
    )
    capture = execute_host_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        host="pi",
        policy=ExecutionPolicy(1, 16, 5.0, 8192, 512),
    )

    assert capture.timed_out is False
    assert capture.budget_reason == "max_steps"
    events = [json.loads(line) for line in capture.stdout.splitlines()]
    assert [event["type"] for event in events] == [
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
    ]
    assert capture.latency_seconds < 5


def test_malformed_pi_stream_without_turn_end_exposes_the_budget_violation(tmp_path):
    assistant = json.dumps(_pi_message(reason="toolUse"))
    script = (
        f"import time; print({assistant!r}, flush=True); "
        f"print({assistant!r}, flush=True); time.sleep(30)"
    )

    capture = execute_host_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        host="pi",
        policy=ExecutionPolicy(1, 16, 5.0, 8192, 512),
    )
    events = [json.loads(line) for line in capture.stdout.splitlines()]

    assert capture.budget_reason == "max_steps"
    assert execution_facts(events, host="pi").steps == 2


def test_budget_stop_uses_sigterm_and_captures_host_cleanup_terminal(tmp_path):
    assistant = json.dumps(_pi_message(reason="toolUse"))
    turn_end = json.dumps({"type": "turn_end"})
    terminal = json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "custom",
                "customType": "cortheon-terminal-status-v1",
            },
        }
    )
    script = (
        "import json,signal,time; "
        f"terminal={terminal!r}; "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(print(terminal, flush=True), raise_system_exit())); "
        "globals()['raise_system_exit']=lambda: (_ for _ in ()).throw(SystemExit(0)); "
        f"print({assistant!r}, flush=True); print({turn_end!r}, flush=True); "
        "time.sleep(30)"
    )
    capture = execute_host_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        host="pi",
        policy=ExecutionPolicy(1, 16, 5.0, 8192, 512),
    )

    assert capture.budget_reason == "max_steps"
    assert terminal in capture.stdout
    assert capture.returncode == 0


def test_streaming_evaluator_stops_pi_at_the_tool_call_policy(tmp_path):
    first = json.dumps({"type": "tool_execution_start", "toolName": "read"})
    second = json.dumps({"type": "tool_execution_start", "toolName": "bash"})
    script = (
        f"import time; print({first!r}, flush=True); print({second!r}, flush=True); time.sleep(30)"
    )
    capture = execute_host_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        host="pi",
        policy=ExecutionPolicy(4, 1, 5.0, 8192, 512),
    )

    assert capture.timed_out is False
    assert capture.budget_reason == "max_tool_calls"
    assert capture.latency_seconds < 5


@pytest.mark.parametrize("stream", [1, 2])
def test_streaming_timeout_cannot_be_blocked_by_an_unterminated_chunk(stream, tmp_path):
    script = f"import os,time; os.write({stream}, b'unterminated'); time.sleep(30)"
    capture = execute_host_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        host="pi",
        policy=ExecutionPolicy(4, 16, 0.25, 8192, 512),
    )

    assert capture.timed_out is True
    assert capture.latency_seconds < 2


def test_retry_keeps_the_original_failed_attempt_and_aggregates_metering():
    original = _run(process_error="pi returned no assistant answer", identity=False)
    retried = _run(process_error=None)
    args = mock.Mock(base_url="", api_key="", model_id="small-model")
    case = mock.Mock(case_id="case")
    probe = mock.Mock(side_effect=[ValueError("endpoint down"), {"ok": True}])
    with mock.patch("cortheon.cognitive_benchmark.run_job", return_value=retried):
        result = _retry_after_infrastructure_death(
            args,
            case,
            0,
            "baseline",
            original,
            probe=probe,
            sleep=lambda _seconds: None,
        )

    assert result.retry_count == 1
    assert result.retry_reason == "model_endpoint_down"
    assert len(result.prior_attempts) == 1
    assert result.prior_attempts[0].process_error == "pi returned no assistant answer"
    assert result.prior_attempts[0].failure_owner == "external_infrastructure"
    assert result.latency_seconds == 2.0
    assert result.tokens is None
    assert result.cost_usd is None
    assert result.execution_identity_valid is False


def test_candidate_error_text_cannot_claim_an_external_retry():
    original = _run(process_error="endpoint down: provider unavailable", identity=False)
    args = mock.Mock(base_url="", api_key="", model_id="small-model")
    case = mock.Mock(case_id="case")
    run_job = mock.Mock()

    with mock.patch("cortheon.cognitive_benchmark.run_job", run_job):
        result = _retry_after_infrastructure_death(
            args,
            case,
            0,
            "baseline",
            original,
            probe=mock.Mock(return_value={"ok": True}),
        )

    assert result is original
    assert result.failure_owner == "candidate"
    assert result.retry_count == 0
    assert result.prior_attempts == ()
    run_job.assert_not_called()
