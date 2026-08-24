"""Shared mock servers and Pi runner for Pi adapter behavior tests.

Not a pytest module: imported by the recovery behavior tests and the
source-versus-installed-wheel comparison tests.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

RuntimeScript = Callable[[str, dict[str, Any]], Any]
# A script returns (status, object) for a normal JSON response, the string
# "invalid-json" for unparseable bytes, or "non-object" for valid JSON that
# is not an object.

# Tool-result texts the Cortheon adapter substitutes when it blocks a call.
BLOCKED_RESULT_MARKERS = (
    "Cortheon has all the evidence",
    "Cortheon already certified",
    "Cortheon reached its host tool budget",
    "Cortheon has accepted sufficient independent evidence",
    "Cortheon's bounded completion continuation budget is exhausted",
)
# Pi's result for a model tool call naming a tool that does not exist. The
# exact wording is validated against real Pi output by the doom-loop tests.
UNAVAILABLE_RESULT_MARKERS = ("not found",)


def _is_unavailable_result(text: str) -> bool:
    return any(marker in text for marker in UNAVAILABLE_RESULT_MARKERS)


def model_tool_attempts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every tool the model attempted, including blocked and unavailable ones.

    A tool_execution_start is evidence of a model tool attempt only: Pi emits
    it before the adapter blocks the call and for tools that do not exist.
    """
    return [event for event in events if event.get("type") == "tool_execution_start"]


def host_executions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool executions that really ran on the host: tool_execution_end events
    whose result is neither one of the Cortheon adapter's block reasons nor
    Pi's "tool not found" answer for an unavailable tool. Pi emits
    tool_execution_start/end even for blocked and nonexistent tools, so the
    end-result text is the only reliable discriminator."""
    return [
        event
        for event in events
        if event.get("type") == "tool_execution_end"
        and not _result_text(event).startswith(BLOCKED_RESULT_MARKERS)
        and not _is_unavailable_result(_result_text(event))
    ]


def blocked_executions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("type") == "tool_execution_end"
        and _result_text(event).startswith(BLOCKED_RESULT_MARKERS)
    ]


def unavailable_tool_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """tool_execution_end events for calls to tools Pi cannot find."""
    return [
        event
        for event in events
        if event.get("type") == "tool_execution_end"
        and _is_unavailable_result(_result_text(event))
        and not _result_text(event).startswith(BLOCKED_RESULT_MARKERS)
    ]


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    ).encode()


def _model_chunks(turn: dict[str, Any], index: int) -> bytes:
    delta: dict[str, Any] = {"role": "assistant"}
    if turn.get("text"):
        delta["content"] = turn["text"]
    if turn.get("tool_calls"):
        delta["tool_calls"] = [
            {
                "index": position,
                "id": f"call-{index}-{position}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
            for position, (name, arguments) in enumerate(turn["tool_calls"])
        ]
    chunks = [
        {
            "id": f"mock-{index}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mock-small",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": None},
            ],
        },
        {
            "id": f"mock-{index}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mock-small",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
            ],
            "usage": {
                "prompt_tokens": 10 * index,
                "completion_tokens": 5 * index,
                "total_tokens": 15 * index,
            },
        },
    ]
    return _sse(chunks)


class StateServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer whose handler reads shared state from .state."""

    state: dict[str, Any]


class ModelHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        server: StateServer = self.server  # type: ignore[assignment]
        state: dict[str, Any] = server.state
        delay = float(state.get("delay") or 0)
        if delay > 0:
            time.sleep(delay)
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        state["requests"].append(request)
        script = state.get("model_script")
        if script is not None and script(request) == "connection-reset":
            self.close_connection = True
            return
        turns = state["turns"]
        index = len(state["requests"])
        turn = turns[min(index - 1, len(turns) - 1)]
        body = _model_chunks(turn, index)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RuntimeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/healthz":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"_unparseable": raw.decode("utf-8", "replace")}
        server: StateServer = self.server  # type: ignore[assignment]
        state: dict[str, Any] = server.state
        state["records"].append((self.path, body))
        result = state["script"](self.path, body)
        if result == "connection-reset":
            # Drop the connection with no response: a transport-level reset.
            self.close_connection = True
            return
        if result == "invalid-json":
            payload = b"{{not json at all"
            status = 200
        elif result == "non-object":
            payload = b"[1, 2, 3]"
            status = 200
        else:
            status, obj = result
            payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Servers:
    """One model server and one runtime server with a shared shutdown."""

    def __init__(self, model_state: dict[str, Any], runtime_state: dict[str, Any]) -> None:
        self.model = _start(ModelHandler, model_state)
        self.runtime = _start(RuntimeHandler, runtime_state)

    def __enter__(self) -> Servers:
        return self

    def __exit__(self, *_exc: object) -> None:
        for server in (self.model, self.runtime):
            server.shutdown()
            server.server_close()


def _start(handler: type[BaseHTTPRequestHandler], state: dict[str, Any]) -> StateServer:
    server = StateServer(("127.0.0.1", 0), handler)
    server.state = state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_pi(
    extension: Path,
    prompt: str | list[str],
    *,
    model_port: int,
    runtime_port: int,
    workspace: Path,
    tmp_path: Path,
    timeout: int = 90,
    extra_env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "providers": {
            "mock": {
                "baseUrl": f"http://127.0.0.1:{model_port}/v1",
                "api": "openai-completions",
                "apiKey": "test-only",
                "authHeader": False,
                "models": [
                    {
                        "id": "mock-small",
                        "name": "mock-small",
                        "reasoning": False,
                        "input": ["text"],
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                        "contextWindow": 8_192,
                        "maxTokens": 700,
                    }
                ],
            }
        }
    }
    (agent_dir / "models.json").write_text(json.dumps(model), encoding="utf-8")
    # File-backed capture keeps intentionally looping runs from stalling the
    # test harness with unbounded in-memory output.
    with contextlib.ExitStack() as stack:
        stdout = stack.enter_context(open(stdout_path, "w")) if stdout_path else None
        stderr = stack.enter_context(open(stderr_path, "w")) if stderr_path else None
        completed = subprocess.run(
            [
                "pi",
                "--provider",
                "mock",
                "--model",
                "mock-small",
                "--mode",
                "json",
                "--print",
                "--no-session",
                "--no-extensions",
                "--no-skills",
                "--no-context-files",
                "--approve",
                "--extension",
                str(extension),
                *(prompt if isinstance(prompt, list) else [prompt]),
            ],
            cwd=workspace,
            env=_pi_env(agent_dir, runtime_port, extra_env),
            stdout=stdout if stdout_path else subprocess.PIPE,
            stderr=stderr if stderr_path else subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    if stdout_path:
        completed.stdout = stdout_path.read_text(encoding="utf-8")
    if stderr_path:
        completed.stderr = stderr_path.read_text(encoding="utf-8")
    return completed


def _pi_env(agent_dir: Path, runtime_port: int, extra_env: dict[str, str] | None) -> dict[str, str]:
    return {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "PI_TELEMETRY": "0",
        "CORTHEON_AUTO_ENABLE": "1",
        "CORTHEON_RUNTIME_URL": f"http://127.0.0.1:{runtime_port}",
        **(extra_env or {}),
    }


def assistant_answers(completed: subprocess.CompletedProcess[str]) -> list[str]:
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    return [
        block["text"]
        for event in events
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
        for block in event["message"]["content"]
        if block.get("type") == "text"
    ]


def parse_events(stdout_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout_text.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _result_text(event: dict[str, Any]) -> str:
    for block in event.get("result", {}).get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def executed_tool_calls(completed: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
    """One entry per host tool call that actually executed (not blocked)."""
    return host_executions(parse_events(completed.stdout))


def _message_texts(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    return []


def continuation_requests(model_state: dict[str, Any]) -> int:
    """Model requests triggered by Cortheon's automatic follow-up prompts."""
    count = 0
    for request in model_state["requests"]:
        messages = request.get("messages", [])
        if any(
            isinstance(message, dict)
            and any(text.startswith("[CORTHEON_CONTINUE]") for text in _message_texts(message))
            for message in messages
        ):
            count += 1
    return count


def require_pi() -> bool:
    return shutil.which("pi") is not None
