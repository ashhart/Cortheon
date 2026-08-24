"""End-to-end runner attestation for the evaluator-wrapped generic host."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from cortheon.benchmark_core.models import ImportCase
from cortheon.benchmark_core.runner_local import run_job


class _ModelEndpoint(BaseHTTPRequestHandler):
    prompts: ClassVar[list[str]] = []
    tool_names: ClassVar[list[set[str]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        assert request["model"] == "small"
        system = str(request["messages"][0]["content"])
        self.prompts.append(system)
        names = {tool["function"]["name"] for tool in request["tools"]}
        self.tool_names.append(names)
        if "Cortheon is already active" not in system:
            message = {"role": "assistant", "content": "No."}
            reason = "stop"
        elif not any(
            message["role"] == "tool"
            or (
                message["role"] == "system"
                and "cortheon_runtime_tool" in str(message.get("content"))
            )
            for message in request["messages"]
        ):
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "host-1",
                        "type": "function",
                        "function": {
                            "name": "host_search",
                            "arguments": json.dumps({"pattern": "pathlib", "path": "missing.py"}),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        else:
            arguments = {
                "answer": "No.",
                "claims": [
                    {"claim": "missing.py does not import pathlib.", "evidence_ids": ["ev1"]}
                ],
                "hypotheses": [
                    {
                        "statement": "The pathlib import is absent.",
                        "falsification_test": "Search missing.py for pathlib.",
                        "status": "supported",
                        "evidence_ids": ["ev1"],
                    }
                ],
                "completion_evidence_ids": ["ev1"],
            }
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "complete-1",
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
                "choices": [
                    {
                        "message": message,
                        "finish_reason": reason,
                    }
                ],
                "usage": {"total_tokens": 3, "cost": 0.0},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def test_runner_marks_generic_transcript_evaluator_wrapped_and_claim_valid(
    tmp_path: Path,
) -> None:
    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), _ModelEndpoint)
    _ModelEndpoint.prompts.clear()
    _ModelEndpoint.tool_names.clear()
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    args = argparse.Namespace(
        host="generic_mcp",
        repository=tmp_path,
        runtime_url="http://127.0.0.1:1",
        base_url=f"http://127.0.0.1:{endpoint.server_address[1]}/v1",
        api_key="",
        provider="local",
        model_id="small",
        timeout_seconds=5.0,
        context_tokens=4_096,
        output_tokens=128,
        max_steps=2,
        max_tool_calls=4,
        generic_web_command=None,
    )
    case = ImportCase(
        case_id="generic-runner",
        path="missing.py",
        module="pathlib",
        expected=False,
        prompt="Does missing.py import pathlib? Answer yes or no.",
    )
    (tmp_path / "missing.py").write_text("import json\n", encoding="utf-8")
    try:
        bare = run_job(args, case, repeat=0, treatment=False)
        full = run_job(args, case, repeat=0, treatment=True)
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        thread.join(timeout=2)

    for result in (bare, full):
        assert result.delivered and result.correct
        assert result.final_text == "No."
        assert result.host_assurance == "evaluator_wrapped"
        assert result.host_transcript_valid is True
        assert result.host_transcript_sha256 is not None
        assert result.host_identity_sha256 is not None
        assert result.execution_identity_valid
        assert result.execution_identity_provenance == "generic_mcp_evaluator_transcript"
        assert result.evaluator_outcome.transport == "generic_mcp"
        assert result.condition_profile_receipt_valid is True
    assert bare.runtime_sessions_started == 0
    assert bare.condition_observed_config_sha256 is None
    assert bare.condition_observed_implementation_sha256 is None
    assert all("Cortheon" not in prompt for prompt in _ModelEndpoint.prompts[:1])
    assert not _ModelEndpoint.tool_names[0] & {
        "cortheon_complete",
        "cortheon_retract",
        "cortheon_abandon",
    }
    assert full.runtime_sessions_started == 1
    assert full.runtime_sessions_completed == 1
    assert "Cortheon is already active" in _ModelEndpoint.prompts[1]
    assert _ModelEndpoint.tool_names[1:] == [{"host_complete"}]
