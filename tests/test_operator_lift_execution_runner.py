from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.execution_models import ExecutionConfig, ScheduledCell
from cortheon.operator_lift.execution_runner import (
    _action_trace,
    _materialize_public_workspace,
    run_cell,
)
from cortheon.operator_lift.execution_schedule import execution_manifest


class _Endpoint(BaseHTTPRequestHandler):
    answer: ClassVar[dict[str, Any]] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        messages = request["messages"]
        tool_messages = [message for message in messages if message["role"] == "tool"]
        forced = request["tool_choice"]["function"]["name"]
        if forced == "host_read":
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "read-projection",
                        "type": "function",
                        "function": {
                            "name": "host_read",
                            "arguments": json.dumps({"path": "public-projection.json"}),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        elif forced == "host_reason":
            leading = self.answer["leading"]
            rival = self.answer["rival"]
            falsification = self.answer["falsification"]
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "reason",
                        "type": "function",
                        "function": {
                            "name": "host_reason",
                            "arguments": json.dumps(
                                {
                                    "hypotheses": [
                                        {
                                            "statement": (
                                                f"{leading['cause']} causes "
                                                f"{leading['outcome']} in {leading['scope']}"
                                            ),
                                            "falsification_test": (
                                                f"Apply {falsification['intervention']} and "
                                                f"observe {falsification['result']}"
                                            ),
                                        },
                                        {
                                            "statement": (
                                                f"{rival['cause']} causes {rival['outcome']} "
                                                f"in {rival['scope']}"
                                            ),
                                            "falsification_test": (
                                                "Hold the rival mechanism fixed and compare"
                                            ),
                                        },
                                    ]
                                }
                            ),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        else:
            observation = json.loads(tool_messages[-1]["content"])
            evidence_ids = observation.get("accepted_evidence_ids") or ["ev1"]
            leading = self.answer["leading"]
            rival = self.answer["rival"]
            falsification = self.answer["falsification"]
            arguments = {
                "answer": self.answer,
                "claims": [
                    {
                        "claim": (
                            f"The leading hypothesis is that {leading['cause']} caused "
                            f"{leading['outcome']} in the {leading['scope']} scope, supported "
                            "by the directly read task record showing a 500-request broker "
                            "and bursts of 900."
                        ),
                        "evidence_ids": evidence_ids,
                    },
                    {
                        "claim": (
                            f"The distinct rival is that {rival['cause']} caused "
                            f"{rival['outcome']} in the {rival['scope']} scope."
                        ),
                        "evidence_ids": evidence_ids,
                    },
                    {
                        "claim": (
                            f"The falsifying intervention is {falsification['intervention']}; "
                            f"if {falsification['result']}, that refutes "
                            f"{falsification['refutes']}."
                        ),
                        "evidence_ids": evidence_ids,
                    },
                ],
                "hypotheses": [
                    {
                        "statement": (
                            f"{leading['cause']} caused {leading['outcome']} in "
                            f"{leading['scope']} accounts."
                        ),
                        "falsification_test": (
                            f"Apply {falsification['intervention']} and check whether "
                            f"{falsification['result']}."
                        ),
                        "status": "supported",
                        "evidence_ids": evidence_ids,
                    },
                    {
                        "statement": (
                            f"{rival['cause']} caused {rival['outcome']} in "
                            f"{rival['scope']} accounts."
                        ),
                        "falsification_test": "Hold cohort selection fixed and compare outcomes.",
                        "status": "uncertain",
                        "evidence_ids": evidence_ids,
                    },
                ],
                "completion_evidence_ids": evidence_ids,
            }
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "complete",
                        "type": "function",
                        "function": {
                            "name": "host_complete",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        body = json.dumps(
            {
                "model": "small",
                "choices": [{"message": message, "finish_reason": reason}],
                "usage": {"total_tokens": 5, "cost": 0.0},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _valid_hypothesis(case) -> dict[str, Any]:
    leading = case.oracle["leading"]
    rival = case.oracle["rivals"][0]
    falsification = case.oracle["falsification"]
    return {
        "leading": dict(zip(("cause", "outcome", "scope"), leading, strict=True)),
        "rival": dict(zip(("cause", "outcome", "scope"), rival, strict=True)),
        "falsification": dict(
            zip(("intervention", "result", "refutes"), falsification, strict=True)
        ),
    }


def test_public_workspace_contains_projection_and_reveals_but_no_oracle_module(tmp_path) -> None:
    case = next(case for case in development_cases() if case.operator == "adaptive_stopping")
    _materialize_public_workspace(tmp_path, case)
    paths = {
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert "public-projection.json" in paths
    assert all(
        path.startswith(("evidence/", "actions/")) or path == "public-projection.json"
        for path in paths
    )
    assert not any("oracle" in path for path in paths)


def test_action_trace_comes_only_from_successful_evaluator_tool_results() -> None:
    case = next(case for case in development_cases() if case.operator == "adaptive_stopping")
    action_id = case.action_catalog[0][0]
    events = [
        {
            "type": "tool_request",
            "origin": "host",
            "name": "host_read",
            "call_id": "call-1",
            "arguments": {"path": f"actions/{action_id}.txt"},
        },
        {
            "type": "tool_result",
            "origin": "host",
            "status": "result",
            "call_id": "call-1",
        },
    ]
    trace = _action_trace(case, events)
    assert [item["action_id"] for item in trace] == [action_id]
    events[-1]["status"] = "error"
    assert _action_trace(case, events) == []


def test_config_is_loopback_only_and_api_key_is_not_public() -> None:
    config = ExecutionConfig("http://127.0.0.1:9000/v1", "local", "small", "secret")
    config.validate()
    assert "secret" not in repr(config.public_identity())
    bad = ExecutionConfig("https://example.com/v1", "local", "small", "secret")
    try:
        bad.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("non-loopback endpoint was accepted")


def test_schedule_cell_identity_is_stable() -> None:
    manifest = execution_manifest()
    case = development_cases()[0]
    cell = ScheduledCell(1, case.case_id, case.operator, "full", 0)
    assert cell.cell_id == f"{case.case_id}--full--r0"
    assert manifest.full_condition.condition_id == "full"


def test_real_generic_subprocess_binds_identity_and_returns_submission() -> None:
    case = development_cases()[0]
    manifest = execution_manifest()
    _Endpoint.answer = _valid_hypothesis(case)
    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    config = ExecutionConfig(
        f"http://127.0.0.1:{endpoint.server_address[1]}/v1",
        "local",
        "small",
        "not-exported",
        timeout_seconds=10,
        output_tokens=512,
        max_steps=3,
        max_tool_calls=4,
    )
    cell = ScheduledCell(1, case.case_id, case.operator, "full", 0)
    try:
        result = run_cell(config, manifest, case, cell)
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        thread.join(timeout=2)
    assert result.summary["transcript_valid"] is True
    assert result.summary["identity_valid"] is True
    assert result.summary["identity_provenance"] == "generic_mcp_evaluator_transcript"
    assert result.submission["safe"] is True
    assert result.submission["delivered"] is True
    assert result.submission["response"] == _Endpoint.answer
    assert "not-exported" not in repr(result)
