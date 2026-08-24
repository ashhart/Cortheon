"""A persistent Pi RPC process for multi-prompt adapter tests.

Not a pytest module: imported by tests that need one Pi process (and one
extension instance with its module-global state) to survive across prompts.
"""

from __future__ import annotations

import json
import select
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pi_recovery_helpers import _pi_env


class PiRpcSession:
    """One persistent pi process in RPC mode: LF-delimited JSON commands on
    stdin, streamed JSON events plus per-command responses on stdout.

    Keeps the extension instance (and its module-global state) alive across
    multiple user prompts, which --print runs cannot do.
    """

    def __init__(
        self,
        extension: Path,
        *,
        model_port: int,
        runtime_port: int,
        workspace: Path,
        tmp_path: Path,
        extra_env: dict[str, str] | None = None,
    ) -> None:
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
        self.events: list[dict[str, Any]] = []
        self.proc = subprocess.Popen(
            [
                "pi",
                "--provider",
                "mock",
                "--model",
                "mock-small",
                "--mode",
                "rpc",
                "--no-session",
                "--no-extensions",
                "--no-skills",
                "--no-context-files",
                "--approve",
                "--extension",
                str(extension),
            ],
            cwd=workspace,
            env=_pi_env(agent_dir, runtime_port, extra_env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def prompt(self, message: str, *, timeout: float = 30.0) -> list[dict[str, Any]]:
        """Send one prompt and return the events streamed for that turn.

        Waits for the prompt's response AND an authoritative agent_settled
        event (emitted after the turn and every queued follow-up settle) so
        callers never observe a turn that is still running. The read itself
        is deadline-safe: select() with the remaining budget, never a fixed
        sleep, and a hard error if the stream closes mid-turn."""
        command_id = f"prompt-{len(self.events)}"
        start = len(self.events)
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps({"id": command_id, "type": "prompt", "message": message}) + "\n"
        )
        self.proc.stdin.flush()
        state = {"response": False, "settled": False}

        def settled_turn(event: dict[str, Any]) -> bool:
            if (
                event.get("type") == "response"
                and event.get("command") == "prompt"
                and event.get("id") == command_id
            ):
                state["response"] = True
            if event.get("type") == "agent_settled":
                state["settled"] = True
            return state["response"] and state["settled"]

        self._read_until(settled_turn, timeout=timeout)
        return self.events[start:]

    def _read_until(
        self,
        stop: Callable[[dict[str, Any]], bool],
        *,
        timeout: float,
        allow_timeout: bool = False,
    ) -> None:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _frames, _errors = select.select([fd], [], [], remaining)
            if not ready:
                break
            line = self.proc.stdout.readline()
            if not line:
                if allow_timeout:
                    return
                raise AssertionError("pi RPC stream closed unexpectedly")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.events.append(event)
            if stop(event):
                return
        if not allow_timeout:
            raise AssertionError(f"pi RPC timed out after {timeout}s")

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)
