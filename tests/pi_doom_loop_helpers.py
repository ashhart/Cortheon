"""Shared scripts, turns, and workspaces for the Pi doom-loop tests.

Not a pytest module: imported by the doom-loop, mutation, and RPC stale-state
test modules so the mock runtime behavior stays defined in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CERTIFIED = "CORTHEON CERTIFIED: the total count is 3."
PROMPT = "Report the total count of keys listed in facts/a.txt."
PLAIN_PROMPT = "Say hello."
CEILING = "8"
# Only tools that always exist in Pi: a "Tool not found" error result is not
# a blocked result and would prevent batch termination (the original hang).
TOOL_TURN: dict[str, Any] = {
    "tool_calls": [
        ("read", {"path": "facts/a.txt"}),
        ("bash", {"command": "echo hi"}),
    ]
}
ANSWER_TURN: dict[str, Any] = {"text": "The total count of keys is three."}
# A hostile mixed batch: one valid host tool plus one tool Pi cannot find.
# The unavailable tool's result carries no terminate flag, so per-tool
# terminate:true alone cannot end this batch — this is the exact production
# failure the adapter must stop with a whole-operation abort.
MIXED_TURN: dict[str, Any] = {
    "tool_calls": [
        ("read", {"path": "facts/a.txt"}),
        ("totally_unavailable_probe", {"query": "anything"}),
    ]
}
# The same hostile batch with the unavailable tool FIRST: the abort fires
# before any valid tool's tool_call handler could run, so the state
# transition before the abort is the only thing that can mark answer-only.
MIXED_TURN_UNAVAILABLE_FIRST: dict[str, Any] = {
    "tool_calls": [
        ("totally_unavailable_probe", {"query": "anything"}),
        ("read", {"path": "facts/a.txt"}),
    ]
}


def finish_script(completes: bool):
    """A runtime that reaches finish after one observation; /v1/complete
    certifies only when ``completes`` is true."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "fin-1",
                    "status": "observing",
                    "session": {"deliverable": "answer"},
                    "next_action": {
                        "type": "harness_tool",
                        "instruction": "Read the fact file.",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Count the keys.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "fin-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1"],
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/complete" and completes:
            return (
                200,
                {"session_id": "fin-1", "status": "complete", "answer": CERTIFIED},
            )
        return 200, {"status": "ok"}

    return script


def never_finished_script(state: dict[str, Any], deliverable: str = "answer"):
    """A runtime that never finishes: every observation demands another."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "loop-1",
                    "status": "observing",
                    "session": {"deliverable": deliverable},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Count the keys.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            state["observes"] = state.get("observes", 0) + 1
            return (
                200,
                {
                    "session_id": "loop-1",
                    "status": "observing",
                    "accepted_evidence_ids": [f"ev-{state['observes']}"],
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": f"req-{state['observes']}",
                            "capability": "reason",
                            "query": "Count the keys.",
                        },
                    },
                },
            )
        if path == "/v1/complete" and state.get("recover"):
            return (
                200,
                {"session_id": "loop-1", "status": "complete", "answer": CERTIFIED},
            )
        return 200, {"status": "ok"}

    return script


def workspace(tmp_path) -> Path:
    """An isolated workspace with one three-key fact file."""
    root = Path(tmp_path) / "workspace"
    (root / "facts").mkdir(parents=True)
    (root / "facts" / "a.txt").write_text(
        "alpha key amber\nbeta key bronze\ngamma key copper\n", encoding="utf-8"
    )
    return root
