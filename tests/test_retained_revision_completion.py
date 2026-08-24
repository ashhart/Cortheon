"""Retained hypotheses use the same bound revision path as changed hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

from test_contradiction_revision_completion import _executor, _RevisionModel

from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal
from cortheon.qualification_core.conditions import execution_profile


def test_retained_no_change_revision_is_bound_and_delivered(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_07")
    projection = public_case(case)
    assert projection["response_schema"]["hypothesis_vocabulary"] == ["h_bug_fixed"]
    change_map = projection["response_schema"]["effect_changes_hypothesis"]
    retained_effect = next(effect for effect, changes in change_map.items() if changes is False)
    (tmp_path / "public-projection.json").write_text(
        json.dumps(projection, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    model = _RevisionModel(
        revision_record={
            "prior": "h_bug_fixed",
            "original_source": "source_a",
            "decisive_source": "source_b",
            "decisive_effect": retained_effect,
            "revised": "h_bug_fixed",
        },
        answer={
            "prior": "h_bug_fixed",
            "prior_status": case.oracle["expected"][1],
            "revised": "h_bug_fixed",
            "decisive_source": "source_b",
        },
        claims=[
            {
                "claim": "The signed production test in source_b confirms h_bug_fixed.",
                "evidence_ids": ["ev1"],
            }
        ],
        hypotheses=[
            {
                "statement": "source_b confirms h_bug_fixed.",
                "falsification_test": "Check for a newer signed production test.",
                "status": "supported",
                "evidence_ids": ["ev1"],
            }
        ],
    )

    result = GenericMcpHost(
        task_id="retained-revision",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
        resource_paths=("public-projection.json",),
    ).run(_goal(case), task_kind="general")

    assert result.delivered
    assert validate_transcript(list(result.events))
    assert model.offers == [
        (["host_read"], "host_read"),
        (["host_reason"], "host_reason"),
        (["host_complete"], "host_complete"),
    ]
    assert model.tool_results[1]["reasoning_binding"]
