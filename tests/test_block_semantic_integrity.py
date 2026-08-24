"""Hostile checks that task semantics, not candidate grades, classify blocks."""

from block_taxonomy_support import scaling_report
from test_block_taxonomy import _block

from cortheon.benchmark_core.audit import scaling_curve
from cortheon.benchmark_core.stats import (
    FALSE_BLOCK,
    SAFE_BLOCK,
    _condition_summary,
    classify_block,
)
from cortheon.qualification_factory import _failure_type


def test_contradictory_grades_do_not_override_expected_allow():
    result = _block(artifact_correct=False, candidate_correct=True)

    assert classify_block(result) == FALSE_BLOCK
    assert _failure_type(result) == "false_block"
    summary = _condition_summary([result], "cortheon")
    assert summary["false_blocks"] == 1
    assert summary["safe_blocks"] == 0
    assert summary["unclassified_blocks"] == 0


def test_contradictory_grades_do_not_override_expected_block():
    result = _block(
        artifact_correct=True,
        candidate_correct=False,
        expected_verdict="block",
    )

    assert classify_block(result) == SAFE_BLOCK


def test_serialized_missing_verdict_is_rejected_from_scaling_curve():
    report = scaling_report(
        [
            {
                "condition": "cortheon",
                "delivered": False,
                "artifact_correct": False,
                "candidate_correct": True,
                "expected_verdict": None,
            },
            {
                "condition": "baseline",
                "delivered": False,
                "artifact_correct": True,
                "candidate_correct": False,
                "expected_verdict": None,
            },
        ]
    )

    curve = scaling_curve([report])
    assert curve["points"] == []
    assert curve["diagnostics"]["reason_counts"] == {"invalid_run_matrix": 1}
