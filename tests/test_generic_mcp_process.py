"""Real child/control-FD proof for the evaluator-owned generic MCP host."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.condition_execution import CONTROL_FD_ENV
from cortheon.benchmark_core.generic_mcp_source import (
    EXPECTED_DIGEST_ENV,
    generic_source_sha256,
)
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.qualification_core.conditions import execution_profile

_REQUEST_ENTERED = threading.Event()
_RELEASE_RESPONSE = threading.Event()
_LAUNCHER = Path(__file__).parents[1] / "src/cortheon/benchmark_core/generic_mcp_launcher.py"


class _Endpoint(BaseHTTPRequestHandler):
    block = False

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        if self.block:
            _REQUEST_ENTERED.set()
            _RELEASE_RESPONSE.wait(timeout=10)
        body = json.dumps(
            {
                "model": "small",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Bare child answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 9},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def test_child_reads_control_fd_and_owns_the_only_terminal(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    marker = "isolated-child"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    profile = execution_profile("bare", "a" * 64)
    profile["nonce"] = "2" * 32
    control = {
        "schema_version": 1,
        "evaluation_profile": profile,
        "cognitive_token": "",
        "evaluator_max_steps": None,
        "auto_enable": False,
        "benchmark_capture_candidate": False,
        "max_host_tool_calls": 4,
    }
    payload = {
        "schema_version": 1,
        "task_id": "child-task",
        "goal": "Does example.py import pathlib?",
        "task_kind": "code",
        "require_web": False,
        "workspace": str(tmp_path),
        "workspace_nonce": marker,
        "resource_paths": [],
        "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
        "api_key": "child-secret-key",
        "provider_id": "local",
        "model_id": "small",
        "timeout_seconds": 2,
        "output_tokens": 128,
        "max_steps": 2,
        "max_tool_calls": 4,
        "tests": {},
        "web_command": None,
    }
    read_fd, write_fd = os.pipe()
    try:
        environment = {key: item for key, item in os.environ.items() if key != "PYTHONPATH"}
        environment.update(
            {
                CONTROL_FD_ENV: str(read_fd),
                EXPECTED_DIGEST_ENV: generic_source_sha256(),
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-I", str(_LAUNCHER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
            env=environment,
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        read_fd = -1
        os.write(write_fd, json.dumps(control, separators=(",", ":")).encode())
        os.close(write_fd)
        write_fd = -1
        stdout, stderr = process.communicate(json.dumps(payload), timeout=10)
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert process.returncode == 0, stderr
    assert "child-secret-key" not in stdout + stderr
    events = [json.loads(line) for line in stdout.splitlines()]
    assert validate_transcript(events)
    assert events[-1]["text"] == "Bare child answer"
    assert sum(event["type"] == "terminal" for event in events) == 1


def test_sigterm_during_blocked_model_call_closes_runtime_and_fails_claim(
    tmp_path: Path,
) -> None:
    _Endpoint.block = True
    _REQUEST_ENTERED.clear()
    _RELEASE_RESPONSE.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    marker = "signal-child"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "4" * 32
    control = {
        "schema_version": 1,
        "evaluation_profile": profile,
        "cognitive_token": "",
        "evaluator_max_steps": None,
        "auto_enable": False,
        "benchmark_capture_candidate": False,
        "max_host_tool_calls": 4,
    }
    payload = {
        "schema_version": 1,
        "task_id": "signal-task",
        "goal": "Does example.py import pathlib?",
        "task_kind": "code",
        "require_web": False,
        "workspace": str(tmp_path),
        "workspace_nonce": marker,
        "resource_paths": [],
        "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
        "api_key": "",
        "provider_id": "local",
        "model_id": "small",
        "timeout_seconds": 30,
        "output_tokens": 128,
        "max_steps": 2,
        "max_tool_calls": 4,
        "tests": {},
        "web_command": None,
    }
    read_fd, write_fd = os.pipe()
    environment = {key: item for key, item in os.environ.items() if key != "PYTHONPATH"}
    environment.update(
        {
            CONTROL_FD_ENV: str(read_fd),
            EXPECTED_DIGEST_ENV: generic_source_sha256(),
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-I", str(_LAUNCHER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env=environment,
        pass_fds=(read_fd,),
    )
    os.close(read_fd)
    os.write(write_fd, json.dumps(control, separators=(",", ":")).encode())
    os.close(write_fd)
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload))
    process.stdin.close()
    try:
        assert _REQUEST_ENTERED.wait(timeout=5)
        process.terminate()
        process.wait(timeout=5)
        stdout = process.stdout.read() if process.stdout is not None else ""
        stderr = process.stderr.read() if process.stderr is not None else ""
    finally:
        _RELEASE_RESPONSE.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        _Endpoint.block = False

    assert process.returncode == 2, stderr
    events = [json.loads(line) for line in stdout.splitlines()]
    terminal = events[-1]
    assert terminal["type"] == "terminal"
    assert terminal["runtime_closed"] is True
    assert terminal["active_sessions"] == 0
    assert any(
        event["type"] == "runtime_transition" and event["transition"] == "abandon"
        for event in events
    )
    assert not validate_transcript(events)


def test_isolated_launcher_rejects_a_wrong_evaluator_source_digest(tmp_path: Path) -> None:
    environment = {
        key: item for key, item in os.environ.items() if key not in {"PYTHONPATH", CONTROL_FD_ENV}
    }
    environment[EXPECTED_DIGEST_ENV] = "0" * 64
    completed = subprocess.run(
        [sys.executable, "-I", str(_LAUNCHER)],
        input="",
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        timeout=5,
        check=False,
    )
    assert completed.returncode != 0
    assert "source digest changed before launch" in completed.stderr
