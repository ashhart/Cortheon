# ruff: noqa: F401, I001

import argparse
import json
import subprocess
from dataclasses import asdict

import pytest
from scaling_support import report as _sealed_scaling_report

from cognitive_benchmark_cases_execution import _process_capture

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


def test_frontier_run_results_keep_zero_defaults_for_pi_classifications():
    result = RunResult(
        case_id="case",
        repeat=0,
        condition="frontier",
        expected=True,
        final_text="answer",
        delivered=True,
        correct=True,
        latency_seconds=0.1,
        tokens=10,
        tool_calls=2,
        tool_errors=0,
        timed_out=False,
        process_error=None,
        evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
    )
    assert result.host_tool_executions == 0
    assert result.blocked_tool_calls == 0
    assert result.unavailable_tool_calls == 0


def test_delivery_fails_closed_on_timeout_or_process_error():
    success = EvaluationOutcome("pi", "success", "pi_assistant", "stop")
    assert _delivery_succeeded(
        "Yes — import pathlib",
        timed_out=False,
        process_error=None,
        evaluator_outcome=success,
    )
    assert not _delivery_succeeded(
        "Yes — import pathlib",
        timed_out=True,
        process_error=None,
        evaluator_outcome=success,
    )
    assert not _delivery_succeeded(
        "Yes — import pathlib",
        timed_out=False,
        process_error="opencode exited 1",
        evaluator_outcome=success,
    )


def test_pi_run_without_an_assistant_message_is_infrastructure_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        lambda *_args, **_kwargs: _process_capture(""),
    )
    args = argparse.Namespace(
        host="pi",
        pi="pi",
        provider="Local",
        model_id="small-model",
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        context_tokens=8_192,
        output_tokens=512,
        reasoning=False,
        repository=tmp_path,
        timeout_seconds=10,
        runtime_url="http://127.0.0.1:8743",
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(args, case, repeat=0, treatment=False)

    assert result.process_error == "pi produced no output"
    assert not result.delivered
    summary = _condition_summary([result], "baseline")
    assert result.failure_owner == "candidate"
    assert summary["eligible_runs"] == 1
    assert summary["candidate_delivery_failures"] == 1
    assert summary["infrastructure_failures"] == 0


def test_timed_out_run_keeps_diagnostics_but_cannot_deliver(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    commands = []
    environments = []

    def time_out(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        return _process_capture(
            '{"type":"text","part":{"text":"Yes — import pathlib"}}\n',
            timed_out=True,
        )

    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        time_out,
    )
    args = argparse.Namespace(
        host="opencode",
        opencode="opencode",
        provider="Local",
        model_id="small-model",
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        context_tokens=8_192,
        output_tokens=512,
        repository=tmp_path,
        timeout_seconds=10,
        runtime_url="http://127.0.0.1:8743",
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(args, case, repeat=0, treatment=False)

    assert "--pure" in commands[0]
    assert environments[0]["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert environments[0]["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "1"
    assert environments[0]["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
    assert environments[0]["OPENCODE_CONFIG_DIR"]
    assert environments[0]["OPENCODE_TEST_HOME"]
    assert environments[0]["XDG_CONFIG_HOME"]
    assert environments[0]["XDG_DATA_HOME"]
    assert result.timed_out
    assert result.process_error is None
    assert result.final_text == "Yes — import pathlib"
    assert not result.delivered
    assert not result.correct
    summary = _condition_summary([result], "baseline")
    assert summary["eligible_runs"] == 1
    assert summary["infrastructure_failures"] == 0
    assert summary["timeouts"] == 1


def test_opencode_terminal_step_budget_is_an_eligible_task_failure(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "src").mkdir()

    def exhaust_budget(command, **_kwargs):
        return _process_capture(
            (
                '{"type":"text","part":{"text":'
                '"**CRITICAL - MAXIMUM STEPS REACHED**\\n\\n'
                'The model did not finish."}}\n'
            ),
            timed_out=True,
        )

    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        exhaust_budget,
    )
    args = argparse.Namespace(
        host="opencode",
        opencode="opencode",
        provider="Local",
        model_id="small-model",
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        context_tokens=8_192,
        output_tokens=512,
        max_steps=4,
        repository=tmp_path,
        timeout_seconds=10,
        runtime_url="http://127.0.0.1:8743",
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(args, case, repeat=0, treatment=False)

    assert result.step_budget_exhausted
    assert not result.timed_out
    assert result.process_error is None
    assert not result.delivered
    assert not result.correct


def test_pi_provider_catalog_is_isolated_and_declares_model_limits():
    args = argparse.Namespace(
        provider="Local",
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        model_id="small-model",
        reasoning=True,
        context_tokens=16_384,
        output_tokens=1_024,
    )

    catalog = _pi_provider_config(args)
    provider = catalog["providers"]["Local"]
    model = provider["models"][0]

    assert provider["authHeader"] is False
    assert model["id"] == "small-model"
    assert model["reasoning"] is True
    assert model["contextWindow"] == 16_384
    assert model["maxTokens"] == 1_024


def test_opencode_provider_config_caps_model_steps_equally():
    args = argparse.Namespace(
        provider="Local",
        base_url="http://127.0.0.1:9000/v1",
        api_key="secret",
        model_id="small-model",
        context_tokens=8_192,
        output_tokens=512,
        max_steps=4,
    )

    baseline = json.loads(_provider_config(args, treatment=False))
    cortheon = json.loads(_provider_config(args, treatment=True))

    assert baseline["agent"]["build"]["steps"] == 4
    assert cortheon["agent"]["build"]["steps"] == 4
    baseline_model = baseline["provider"]["Local"]["models"]["small-model"]
    cortheon_model = cortheon["provider"]["Local"]["models"]["small-model"]
    assert baseline_model["limit"] == {"context": 8_192, "output": 512}
    assert cortheon_model["limit"] == baseline_model["limit"]
    assert baseline["plugin"] == []
    assert len(cortheon["plugin"]) == 1


def test_model_endpoint_preflight_requires_the_requested_model(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return self.body

    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/models"):
            return Response(b'{"data":[{"id":"small-model"}]}')
        return Response(b'{"choices":[{"message":{"content":"OK"}}]}')

    monkeypatch.setattr(
        "cortheon.cognitive_benchmark.urllib.request.urlopen",
        open_request,
    )

    result = _model_endpoint_health(
        "http://127.0.0.1:9000/v1",
        api_key="secret",
        model_id="small-model",
        inference_timeout=7,
    )

    assert result["ok"] is True
    assert requests[0][0].full_url.endswith("/v1/models")
    assert requests[0][0].get_header("Authorization") == "Bearer secret"
    assert requests[0][1] == 5
    assert requests[1][0].full_url.endswith("/v1/chat/completions")
    assert requests[1][0].get_method() == "POST"
    assert requests[1][1] == 7
    assert result["inference_probe_ok"] is True


def test_model_endpoint_preflight_rejects_listed_but_unresponsive_model(monkeypatch):
    class ModelsResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"data":[{"id":"sleeping-model"}]}'

    calls = 0

    def open_request(_request, timeout):
        nonlocal calls
        assert timeout in {1, 5}
        calls += 1
        if calls == 1:
            return ModelsResponse()
        raise TimeoutError("inference timed out")

    monkeypatch.setattr(
        "cortheon.cognitive_benchmark.urllib.request.urlopen",
        open_request,
    )

    with pytest.raises(ValueError, match="failed a live inference probe"):
        _model_endpoint_health(
            "http://127.0.0.1:9000/v1",
            api_key="",
            model_id="sleeping-model",
            inference_timeout=1,
        )


def test_postflight_probe_records_failure_without_discarding_results():
    def fail():
        raise ValueError("model probe timed out")

    assert _postflight_probe(fail) == {
        "ok": False,
        "error": "model probe timed out",
    }


def test_main_writes_invalid_audit_report_when_postflight_fails(
    monkeypatch,
    tmp_path,
):
    runtime_calls = 0

    def runtime_health(_url):
        nonlocal runtime_calls
        runtime_calls += 1
        if runtime_calls == 1:
            return {"ok": True, "storage": "memory_only"}
        raise ValueError("runtime disappeared after the balanced run")

    cases = [
        ImportCase("case_a", "src/a.py", "pathlib", True, "prompt a"),
        ImportCase("case_b", "src/b.py", "json", False, "prompt b"),
    ]

    def run_balanced(_args, case, *, repeat, treatment, condition):
        return RunResult(
            case_id=case.case_id,
            repeat=repeat,
            condition=condition,
            expected=case.expected,
            final_text="verified",
            delivered=True,
            correct=True,
            latency_seconds=0.1,
            tokens=10,
            tool_calls=1,
            tool_errors=0,
            timed_out=False,
            process_error=None,
            inference_model_id="small-model",
            substrate_telemetry_valid=True if treatment else None,
            runtime_sessions_started=1 if treatment else 0,
            runtime_observations_accepted=1 if treatment else 0,
            runtime_sessions_completed=1 if treatment else 0,
        )

    monkeypatch.setattr(
        "cortheon.cognitive_benchmark._runtime_health",
        runtime_health,
    )
    monkeypatch.setattr(
        "cortheon.cognitive_benchmark._model_endpoint_health",
        lambda *_args, **_kwargs: {
            "ok": True,
            "model_id": "small-model",
            "inference_probe_ok": True,
        },
    )
    monkeypatch.setattr(
        "cortheon.cognitive_benchmark.discover_benchmark_cases",
        lambda *_args, **_kwargs: cases,
    )
    monkeypatch.setattr(
        "cortheon.cognitive_benchmark.run_job",
        run_balanced,
    )
    output = tmp_path / "postflight-failure.json"

    exit_code = cognitive_benchmark_main(
        [
            "--repository",
            str(tmp_path),
            "--cases",
            "2",
            "--repeats",
            "1",
            "--suite",
            "imports",
            "--model-id",
            "small-model",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert exit_code == 0
    assert report["qualification_valid"] is False
    assert report["proof_gates"]["postflight_healthy"] is False
    assert report["runtime"]["postflight_ok"] is False
    assert report["runtime"]["postflight"]["error"] == (
        "runtime disappeared after the balanced run"
    )
    assert len(report["runs"]) == 4
    assert report["audit"]["run_count"] == 4
