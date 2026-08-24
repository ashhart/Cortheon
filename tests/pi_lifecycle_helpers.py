"""Shared fixtures for the terminal-lifecycle (withheld answer) Pi tests.

The runtime script and run wrapper reproduce the precise round-23 live
lifecycle: a causal novel-synthesis treatment that accepts every host
observation and always demands one more, so it can only end at the host
tool budget — never at finish or completion. Constants cover the
withheld marker, the ordinary wrong-shaped answer the false allow
delivered, and the turn shapes used to probe the bounded answer-only
window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pi_causal_helpers import CAUSAL_PROMPT
from pi_doom_loop_helpers import workspace
from pi_recovery_helpers import Servers, run_pi

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"
WITHHELD_MARKER = "[Cortheon withheld:"
CAP = "8"
# Ordinary prose: a plausible-sounding answer with no Cause/Rival/Test
# structure — exactly the wrong shape the live run delivered unvalidated.
ORDINARY_ANSWER = (
    "The two ledgers clash because they share shard key copper, so entries "
    "collide during nightly rotation."
)
STAGE_ENTRY_TYPE = "cortheon-benchmark-causal-stage-v1"
CANDIDATE_ENTRY_TYPE = "cortheon-benchmark-candidate-v1"
# A tool-call-shaped turn in the answer-only window: one existing host tool.
SINGLE_TOOL_TURN: dict[str, Any] = {"tool_calls": [("read", {"path": "facts/a.txt"})]}
# A mixed turn: assistant text emitted alongside a tool call, so the message
# is not answerable and must leave the disposition intact for the final text.
MIXED_TEXT_TOOL_TURN: dict[str, Any] = {
    "text": ORDINARY_ANSWER,
    "tool_calls": [("read", {"path": "facts/a.txt"})],
}
PLAIN_PROMPT = "Summarize the two fact files in one sentence."
PLAIN_ANSWER = "Both ledgers reuse the shard key copper."


def never_finishing_causal_script(
    state: dict[str, Any],
    *,
    abandon_fails: bool = False,
):
    """A causal document-synthesis runtime that accepts every observation
    and always demands one more: the treatment can only end at the host
    tool budget, never at finish or completion. With ``abandon_fails`` the
    /v1/abandon transport itself fails after the controlled block is set."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "causal-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal")},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            state["observes"] = state.get("observes", 0) + 1
            return (
                200,
                {
                    "session_id": "causal-1",
                    "status": "observing",
                    "accepted_evidence_ids": [f"ev-{state['observes']}"],
                    "context": {"goal": CAUSAL_PROMPT},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": f"req-{state['observes']}",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/abandon" and abandon_fails:
            return "invalid-json"
        return 200, {"status": "ok"}

    return script


def run_lifecycle(
    extension: Path,
    tmp_path: Path,
    servers: Servers,
) -> Any:
    """Run the causal prompt through Pi with the bounded host-tool budget."""
    return run_pi(
        extension,
        CAUSAL_PROMPT,
        model_port=servers.model.server_port,
        runtime_port=servers.runtime.server_port,
        workspace=workspace(tmp_path),
        tmp_path=tmp_path,
        timeout=45,
        extra_env={
            "CORTHEON_MAX_HOST_TOOL_CALLS": CAP,
            "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1",
        },
    )


def custom_entry_data(completed: Any, custom_type: str) -> list[dict[str, Any]]:
    """Data payloads of every benchmark custom entry of one type."""
    from pi_recovery_helpers import parse_events

    return [
        entry.get("data", {})
        for event in parse_events(completed.stdout)
        if event.get("type") == "entry_appended"
        for entry in [event.get("entry", {})]
        if entry.get("type") == "custom" and entry.get("customType") == custom_type
    ]
