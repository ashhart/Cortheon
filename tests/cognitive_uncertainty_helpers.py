"""Shared fixtures and drivers for the uncertainty-visibility tests.

One neutral investigation, one certified answer skeleton, and the rival
hypotheses both uncertainty modules assert against, so the visibility
contract and the identifying rule exercise the same session rather than
two drifting copies of it.
"""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_core.runtime import CognitiveRuntime

GOAL = (
    "Read facts/a.txt and facts/b.txt. Diagnose the causal explanation "
    "for the collision, disprove the rival, and give a discriminating test."
)
READ_A = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/a.txt"}}\n'
    "Northstar path A uses collision key amber."
)
READ_B_NEUTRAL = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/b.txt"}}\n'
    "Path B reuses key amber. Compaction is scheduled nightly."
)
CAUSE_AND_TEST = (
    "Cause: The collision occurs because both paths reuse the Northstar key "
    "amber.\n"
    "Test: Assign distinct keys; this distinguishing test would falsify the "
    "wrong mechanism: Cause predicts the collision disappears whereas Rival "
    "predicts the collision remains."
)
CLAIM = {
    "claim": (
        "The causal explanation for the collision is that both paths reuse "
        "the Northstar key amber; the rival alternative remains uncertain "
        "pending the falsification test."
    ),
    "evidence_ids": ["ev1", "ev2"],
}
CAUSE = {
    "statement": "The collision occurs because both paths reuse the Northstar key amber.",
    "falsification_test": "Assign distinct keys.",
    "status": "supported",
    "evidence_ids": ["ev1", "ev2"],
}
COMPACTION_RIVAL = {
    "statement": "Instead, cache compaction is the competing alternative.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}
ATLAS_RIVAL = {
    "statement": "Atlas scheduling contributes to the collision.",
    "falsification_test": "Query the Atlas schedule log.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}


def start_session() -> tuple[CognitiveRuntime, str]:
    """A runtime holding both neutral reads, ready to complete."""

    runtime = CognitiveRuntime()
    started = runtime.start(GOAL)
    session_id = started["session"]["session_id"]
    runtime.observe(
        session_id,
        [
            {
                "kind": "documentation",
                "content": READ_A,
                "status": "verified",
                "source": "pi:read:facts/a.txt",
            },
            {
                "kind": "documentation",
                "content": READ_B_NEUTRAL,
                "status": "verified",
                "source": "pi:read:facts/b.txt",
            },
        ],
        request_id="req1",
    )
    return runtime, session_id


def complete_answer(
    runtime: CognitiveRuntime,
    session_id: str,
    answer: str,
    hypotheses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return runtime.complete(
        session_id,
        answer=answer,
        claims=[CLAIM],
        hypotheses=hypotheses or [CAUSE, COMPACTION_RIVAL],
        completion_evidence_ids=["ev1", "ev2"],
    )


def rival_answer(line: str) -> str:
    """The certified answer skeleton carrying one rival line."""

    return CAUSE_AND_TEST.replace("Test:", line + "\nTest:")


def failed_checks(result: dict[str, Any]) -> set[str]:
    return {item["name"] for item in result["verification"]["checks"] if not item["passed"]}


def visibility(result: dict[str, Any]) -> dict[str, Any]:
    return next(
        item
        for item in result["verification"]["checks"]
        if item["name"] == "uncertainty_visibility"
    )
