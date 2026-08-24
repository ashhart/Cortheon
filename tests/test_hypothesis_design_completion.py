"""Hypothesis-design tasks certify proposed tests without inventing test results."""

from __future__ import annotations

import json

from cortheon.cognitive_runtime import CognitiveRuntime


def test_exact_structured_hypothesis_design_can_complete() -> None:
    runtime = CognitiveRuntime(require_host_receipts=False)
    goal = (
        "Frame the strongest causal hypothesis and one genuinely distinct rival. "
        "Return their cause, outcome and scope IDs, then a falsifying intervention "
        "with the result that would refute the leader. IDs appear in the evidence."
    )
    started = runtime.start(goal, task_kind="general")
    session_id = started["session"]["session_id"]
    record = {
        "evidence": [
            [
                "source_a",
                "[cause=legacy_broker_overload] Weekend accounts use a 500-request "
                "broker; bursts are 900.",
            ],
            [
                "source_b",
                "[outcome=activation_drop scope=weekend] Only weekend migration "
                "activation fell; measurement sampling is unchanged.",
            ],
        ],
        "response_schema": {
            "type": "closed_json_object",
            "additional_fields": False,
            "fields": {
                "leading": ["cause", "outcome", "scope"],
                "rival": ["cause", "outcome", "scope"],
                "falsification": ["intervention", "result", "refutes"],
            },
            "vocabulary": [
                "activation_drop",
                "cohort_selection_bias",
                "drop_persists",
                "legacy_broker_overload",
                "route_new_broker",
                "weekend",
            ],
        },
    }
    receipt = json.dumps(
        {"tool": "read", "outcome": "result", "args": {"path": "public-projection.json"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    runtime.observe(
        session_id,
        [
            {
                "kind": "artifact",
                "source": "public-projection.json",
                "content": (
                    f"[CORTHEON_HOST_EVIDENCE] {receipt}\n{json.dumps(record, sort_keys=True)}"
                ),
            }
        ],
        request_id=started["next_action"]["request"]["request_id"],
    )
    answer = json.dumps(
        {
            "leading": {
                "cause": "legacy_broker_overload",
                "outcome": "activation_drop",
                "scope": "weekend",
            },
            "rival": {
                "cause": "cohort_selection_bias",
                "outcome": "activation_drop",
                "scope": "weekend",
            },
            "falsification": {
                "intervention": "route_new_broker",
                "result": "drop_persists",
                "refutes": "legacy_broker_overload",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    result = runtime.complete(
        session_id,
        answer=answer,
        claims=[
            {
                "claim": (
                    "The leading hypothesis is that legacy_broker_overload caused "
                    "activation_drop in the weekend scope, supported by the task record "
                    "showing a 500-request broker and bursts of 900."
                ),
                "evidence_ids": ["ev1"],
            },
            {
                "claim": (
                    "The distinct rival is that cohort_selection_bias caused "
                    "activation_drop in the weekend scope."
                ),
                "evidence_ids": ["ev1"],
            },
            {
                "claim": (
                    "The falsifying intervention is route_new_broker; if drop_persists, "
                    "that refutes legacy_broker_overload."
                ),
                "evidence_ids": ["ev1"],
            },
        ],
        hypotheses=[
            {
                "statement": ("legacy_broker_overload caused activation_drop in weekend accounts."),
                "falsification_test": "route_new_broker and check whether drop_persists.",
                "status": "supported",
                "evidence_ids": ["ev1"],
            },
            {
                "statement": ("cohort_selection_bias caused activation_drop in weekend accounts."),
                "falsification_test": ("Hold cohort selection fixed and compare activation_drop."),
                "status": "uncertain",
                "evidence_ids": ["ev1"],
            },
        ],
        completion_evidence_ids=["ev1"],
    )

    assert result["status"] == "complete"
    assert result["answer"] == answer
