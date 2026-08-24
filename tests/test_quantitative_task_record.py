"""Quantitative claims require an exact evaluator-bound task record."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cortheon.cognitive_runtime import CognitiveRuntime


def _profile(
    *,
    kind: str,
    source: str | None,
    tool: str,
    outcome: str,
    evidence_number: int = 500,
) -> dict[str, Any]:
    runtime = CognitiveRuntime(require_host_receipts=False)
    started = runtime.start("Identify the broker threshold.", task_kind="general")
    session_id = started["session"]["session_id"]
    receipt = json.dumps(
        {
            "tool": tool,
            "outcome": outcome,
            "args": {"path": "public-projection.json"},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    observation: dict[str, object] = {
        "kind": kind,
        "content": (
            f"[CORTHEON_HOST_EVIDENCE] {receipt}\n"
            f"The broker threshold is {evidence_number} requests."
        ),
    }
    if source is not None:
        observation["source"] = source
    runtime.observe(
        session_id,
        [observation],
        request_id=started["next_action"]["request"]["request_id"],
    )
    completed = runtime.complete(
        session_id,
        answer="The broker threshold is 500 requests.",
        claims=[
            {
                "claim": "The broker threshold is 500 requests.",
                "evidence_ids": ["ev1"],
            }
        ],
        hypotheses=[
            {
                "statement": "The broker threshold is 500 requests.",
                "falsification_test": "Read the task record.",
                "status": "supported",
                "evidence_ids": ["ev1"],
            }
        ],
        completion_evidence_ids=["ev1"],
    )
    return completed["verification"]["claim_verification"][0]


@pytest.mark.parametrize(
    ("kind", "source", "tool", "outcome"),
    [
        ("documentation", "README.md", "read", "result"),
        ("artifact", None, "read", "result"),
        ("artifact", "public-projection.json", "grep", "match"),
        ("artifact", "public-projection.json", "read", "error"),
    ],
)
def test_unbound_or_unverified_records_do_not_establish_quantities(
    kind: str,
    source: str | None,
    tool: str,
    outcome: str,
) -> None:
    profile = _profile(kind=kind, source=source, tool=tool, outcome=outcome)
    assert not profile["passed"]
    assert any("directly read task record" in gap for gap in profile["gaps"])


def test_task_record_must_contain_every_claimed_number() -> None:
    profile = _profile(
        kind="artifact",
        source="public-projection.json",
        tool="read",
        outcome="result",
        evidence_number=400,
    )
    assert not profile["passed"]
    assert any("material number" in gap for gap in profile["gaps"])
