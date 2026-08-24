"""OpenCode evaluator receipt identity and metering tests."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import pytest

from cortheon.benchmark_core.execution_provenance import ProcessCapture, execution_facts
from cortheon.benchmark_core.opencode_receipt import (
    OpenCodeReceipt,
    capture_opencode_receipt,
)
from cortheon.cognitive_benchmark import ImportCase, run_job


def _events(session_id: str = "ses_real") -> list[dict]:
    return [
        {"type": "step_start", "sessionID": session_id, "part": {}},
        {"type": "text", "sessionID": session_id, "part": {"text": "answer"}},
        {
            "type": "step_finish",
            "sessionID": session_id,
            "part": {
                "reason": "stop",
                "cost": 0.2,
                "tokens": {
                    "input": 10,
                    "output": 5,
                    "reasoning": 2,
                    "cache": {"read": 3, "write": 1},
                },
            },
        },
    ]


def _receipt(session_id: str = "ses_real") -> OpenCodeReceipt:
    tokens = {
        "input": 10,
        "output": 5,
        "reasoning": 2,
        "cache": {"read": 3, "write": 1},
    }
    return OpenCodeReceipt(
        {
            "info": {
                "id": session_id,
                "version": "1.18.18",
                "model": {"providerID": "Local", "id": "small-model"},
                "tokens": tokens,
                "cost": 0.2,
            },
            "messages": [
                {"info": {"role": "user", "sessionID": session_id}, "parts": []},
                {
                    "info": {
                        "role": "assistant",
                        "sessionID": session_id,
                        "providerID": "Local",
                        "modelID": "small-model",
                        "tokens": tokens,
                        "cost": 0.2,
                        "finish": "stop",
                    },
                    "parts": [],
                },
            ],
        },
        session_id,
        "1.18.18",
        None,
    )


def _args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        host="opencode",
        opencode="opencode",
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


def test_identity_and_meter_reconcile_with_sanitized_export():
    facts = execution_facts(_events(), host="opencode", opencode_receipt=_receipt())

    assert facts.tokens == 21
    assert facts.cost_usd == 0.2
    assert facts.measurements_valid is True
    assert facts.identity_valid is True
    assert facts.identity_provenance == "opencode_sanitized_export"
    assert (facts.provider_id, facts.model_id) == ("Local", "small-model")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["info"].update(id="ses_other"),
        lambda value: value["info"].update(version="1.18.17"),
        lambda value: value["info"]["model"].update(id="other"),
        lambda value: value["messages"][1]["info"].update(providerID="forged"),
        lambda value: value["messages"][1]["info"].update(cost=0.3),
        lambda value: value["messages"][1]["info"].update(finish="tool-calls"),
        lambda value: value["info"].update(cost=0.3),
        lambda value: value["info"]["tokens"].update(output=999),
    ],
)
def test_receipt_mutations_fail_identity_and_measurement_closed(mutation):
    receipt = _receipt()
    data = copy.deepcopy(receipt.data)
    assert data is not None
    mutation(data)

    facts = execution_facts(
        _events(),
        host="opencode",
        opencode_receipt=OpenCodeReceipt(data, "ses_real", "1.18.18", None),
    )

    assert facts.identity_valid is False
    assert facts.measurements_valid is False
    assert facts.tokens is None
    assert facts.cost_usd is None


def test_session_id_rejects_nested_spoof_and_top_level_mix(tmp_path):
    nested = _events()
    nested[1]["part"]["sessionID"] = "ses_forged"
    no_binary = str(tmp_path / "must-not-run")
    nested_receipt = capture_opencode_receipt(
        no_binary, nested, cwd=tmp_path, env=os.environ.copy()
    )
    assert nested_receipt.error == "nested_session_id_mismatch"

    mixed = _events()
    mixed[1]["sessionID"] = "ses_other"
    receipt = capture_opencode_receipt(no_binary, mixed, cwd=tmp_path, env=os.environ.copy())
    assert receipt.error == "mixed_top_level_session_ids"


@pytest.mark.parametrize(
    "events,error",
    [
        (
            [{"type": "step_finish", "sessionID": "ses_real", "part": {}}],
            "invalid_session_event_order",
        ),
        (
            [{"type": "step_start", "sessionID": "ses_real", "part": {}}],
            "incomplete_session_event_order",
        ),
        (
            [{"type": "step_start", "part": {}}, {"type": "step_finish", "part": {}}],
            "missing_top_level_session_id",
        ),
    ],
)
def test_receipt_requires_closed_ordered_top_level_session(events, error, tmp_path):
    receipt = capture_opencode_receipt(
        str(tmp_path / "must-not-run"), events, cwd=tmp_path, env=os.environ.copy()
    )
    assert receipt.error == error


def test_receipt_export_is_sanitized_pure_and_memory_only(monkeypatch, tmp_path):
    expected = _receipt()
    commands: list[list[str]] = []

    def command(args, **_kwargs):
        commands.append(args)
        if args == ["opencode", "--version"]:
            return 0, b"1.18.18\n", None
        assert expected.data is not None
        return 0, json.dumps(expected.data).encode(), None

    monkeypatch.setattr("cortheon.benchmark_core.opencode_receipt._bounded_command", command)

    receipt = capture_opencode_receipt("opencode", _events(), cwd=tmp_path, env=os.environ.copy())

    assert receipt.error is None
    assert commands == [
        ["opencode", "--version"],
        ["opencode", "export", "ses_real", "--sanitize", "--pure"],
    ]
    assert list(tmp_path.iterdir()) == []


def _patch_run(monkeypatch, receipt: OpenCodeReceipt) -> None:
    stdout = "\n".join(json.dumps(event) for event in _events())
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        lambda *_args, **_kwargs: ProcessCapture(stdout, "", 0, 0.1, False, None),
    )
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.capture_opencode_receipt",
        lambda *_args, **_kwargs: receipt,
    )


def test_run_job_captures_receipt_before_isolated_config_closes(monkeypatch, tmp_path):
    stdout = "\n".join(json.dumps(event) for event in _events())
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        lambda *_args, **_kwargs: ProcessCapture(stdout, "", 0, 0.1, False, None),
    )
    observed: list[tuple[str, str]] = []

    def capture(_executable, _events, *, cwd, env):
        observed.append((str(cwd), env["OPENCODE_CONFIG_DIR"]))
        assert Path(env["OPENCODE_CONFIG_DIR"]).is_dir()
        return _receipt()

    monkeypatch.setattr("cortheon.benchmark_core.runner_local.capture_opencode_receipt", capture)
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(_args(tmp_path), case, repeat=0, treatment=False)

    assert len(observed) == 1
    assert result.process_error is None
    assert result.execution_identity_valid is True
    assert result.execution_identity_provenance == "opencode_sanitized_export"
    assert result.execution_measurements_valid is True
    assert (result.inference_provider_id, result.inference_model_id) == (
        "Local",
        "small-model",
    )
    assert (result.tokens, result.cost_usd) == (21, 0.2)


def test_run_job_fails_closed_when_sanitized_export_is_missing(monkeypatch, tmp_path):
    _patch_run(
        monkeypatch,
        OpenCodeReceipt(None, "ses_real", "1.18.18", "sanitized_export_failed"),
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(_args(tmp_path), case, repeat=0, treatment=False)

    assert result.delivered is False
    assert result.execution_identity_valid is False
    assert result.execution_measurements_valid is False
    assert result.process_error == ("opencode execution identity invalid: sanitized_export_failed")


def test_run_job_rejects_a_consistent_but_unrequested_export_model(monkeypatch, tmp_path):
    receipt = _receipt()
    data = copy.deepcopy(receipt.data)
    assert data is not None
    data["info"]["model"]["id"] = "other"
    data["messages"][1]["info"]["modelID"] = "other"
    _patch_run(monkeypatch, OpenCodeReceipt(data, "ses_real", "1.18.18", None))
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(_args(tmp_path), case, repeat=0, treatment=False)

    assert result.inference_model_id == "other"
    assert result.execution_identity_valid is False
    assert result.process_error == (
        "opencode execution identity invalid: execution_identity_mismatch"
    )
