"""A run that never finished is never restraint.

Only the pinned withheld terminal makes an undelivered run a block. Timeouts,
dead processes, and silent transcripts are delivery failures, so they cannot
be counted as safe blocks, cannot flatter the false-block rate, and cannot
manufacture classification coverage out of an empty block set. Each property
sits beside the honest run of the same shape, so no assertion here can be
satisfied by a classifier that simply refuses everything.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from proof_support import WITHHELD, _run

from cortheon.benchmark_core.blocks import (
    DELIVERY_FAILURE,
    FALSE_BLOCK,
    SAFE_BLOCK,
    UNCLASSIFIED_BLOCK,
    classify_block,
    classify_serialized_block,
)
from cortheon.cognitive_benchmark import _condition_summary


def test_timeout_with_a_wrong_artifact_is_not_a_safe_block():
    timed_out = _run(
        condition="cortheon",
        delivered=False,
        final_text=WITHHELD,
        timed_out=True,
        artifact_correct=False,
    )

    assert classify_block(timed_out) == DELIVERY_FAILURE

    summary = _condition_summary([timed_out], "cortheon")

    assert summary["safe_blocks"] == 0
    assert summary["false_blocks"] == 0
    assert summary["unclassified_blocks"] == 0
    assert summary["delivery_failures"] == 1
    assert summary["delivery_failure_rate"] == 1.0
    # Control: the identical run that terminated is a false block because the
    # sealed task expected an answer. Artifact grading cannot relabel it safe.
    finished = _run(
        condition="cortheon",
        delivered=False,
        final_text=WITHHELD,
        artifact_correct=False,
    )
    assert classify_block(finished) == FALSE_BLOCK
    assert _condition_summary([finished], "cortheon")["false_blocks"] == 1


def test_process_error_with_a_correct_artifact_is_not_a_false_block():
    # The symmetric direction: a dead process must not manufacture a false
    # block either, or the false-block rate becomes a measure of crashes.
    died = _run(
        condition="cortheon",
        delivered=False,
        final_text=WITHHELD,
        process_error="host exited 137",
        artifact_correct=True,
    )

    assert classify_block(died) == DELIVERY_FAILURE

    summary = _condition_summary([died], "cortheon")

    # A contender process failure is a scheduled incorrect outcome. Only an
    # evaluator-attested external failure can invalidate the case.
    assert summary["infrastructure_failures"] == 0
    assert summary["candidate_delivery_failures"] == 1
    assert summary["false_blocks"] == 0
    assert summary["delivery_failures"] == 1


def test_silent_comparator_run_is_a_delivery_failure_not_a_block():
    silent = _run(condition="baseline", delivered=False, final_text="")

    assert classify_block(silent) == DELIVERY_FAILURE

    summary = _condition_summary([silent], "baseline")

    assert summary["delivery_failures"] == 1
    assert summary["unclassified_blocks"] == 0
    assert summary["safe_blocks"] == 0
    assert summary["block_classification_coverage"] is None


def test_text_that_merely_mentions_the_terminal_is_not_a_block():
    # The terminal has to be the answer, not a quotation inside one.
    quoted = _run(
        condition="cortheon",
        delivered=False,
        final_text=f"The run said {WITHHELD} and then stopped",
        artifact_correct=False,
    )

    assert classify_block(quoted) == DELIVERY_FAILURE


def test_serialized_and_live_classifiers_agree_on_every_shape():
    shapes = [
        _run(condition="cortheon", delivered=True, correct=True),
        _run(condition="cortheon", delivered=False, final_text=WITHHELD),
        _run(
            condition="cortheon",
            delivered=False,
            final_text=WITHHELD,
            expected_verdict="block",
        ),
        _run(
            condition="cortheon",
            delivered=False,
            final_text=WITHHELD,
            expected_verdict=None,
        ),
        _run(
            condition="cortheon",
            delivered=False,
            final_text=WITHHELD,
            artifact_correct=True,
            candidate_correct=False,
        ),
        _run(condition="cortheon", delivered=False, final_text="", timed_out=True),
        _run(condition="baseline", delivered=False, final_text="partial text"),
    ]

    kinds = set()
    for result in shapes:
        serialized = json.loads(json.dumps(asdict(result)))
        assert classify_serialized_block(serialized) == classify_block(result), result
        kinds.add(classify_block(result))

    # The agreement is over the whole taxonomy, not one easy case.
    assert kinds == {None, FALSE_BLOCK, SAFE_BLOCK, UNCLASSIFIED_BLOCK, DELIVERY_FAILURE}


def test_serialized_run_missing_its_terminal_cannot_claim_a_block():
    # A report that never recorded the terminal cannot argue restraint after
    # the fact; the reader fails closed instead of trusting the grade alone.
    assert (
        classify_serialized_block({"delivered": False, "artifact_correct": False})
        == DELIVERY_FAILURE
    )
    authenticated = asdict(
        _run(condition="cortheon", delivered=False, final_text=WITHHELD).evaluator_outcome
    )
    assert (
        classify_serialized_block(
            {
                "delivered": False,
                "timed_out": False,
                "process_error": None,
                "final_text": WITHHELD,
                "artifact_correct": False,
                "expected_verdict": "block",
                "evaluator_outcome": authenticated,
            }
        )
        == SAFE_BLOCK
    )


def test_serialized_non_boolean_grades_are_absent_grades():
    # A stored grade of "true" or 1 is not a boolean the grader wrote, so it
    # cannot classify the block in either direction.
    authenticated = asdict(
        _run(condition="cortheon", delivered=False, final_text=WITHHELD).evaluator_outcome
    )
    for value in ("true", 1, "false", 0, None):
        assert (
            classify_serialized_block(
                {
                    "delivered": False,
                    "timed_out": False,
                    "process_error": None,
                    "final_text": WITHHELD,
                    "artifact_correct": value,
                    "expected_verdict": None,
                    "evaluator_outcome": authenticated,
                }
            )
            == UNCLASSIFIED_BLOCK
        ), value


def test_zero_blocks_serialize_null_coverage():
    delivered_only = [
        _run(condition="cortheon", correct=True, sessions_completed=1),
        _run(condition="cortheon", correct=True, sessions_completed=1),
    ]

    summary = _condition_summary(delivered_only, "cortheon")

    assert summary["unclassified_blocks"] == 0
    assert summary["block_classification_coverage"] is None
    assert '"block_classification_coverage": null' in json.dumps(summary)
    # Control: one classified block reports a coverage of 1.0 that means it.
    with_block = [
        *delivered_only,
        _run(
            condition="cortheon",
            delivered=False,
            final_text=WITHHELD,
            artifact_correct=False,
            sessions_completed=1,
        ),
    ]
    assert _condition_summary(with_block, "cortheon")["block_classification_coverage"] == 1.0


def test_timeouts_alone_leave_coverage_null_rather_than_perfect():
    # The gaming shape: nothing but timeouts. Before delivery failures were
    # split out this reported coverage 1.0 over blocks that were never blocks.
    timeouts = [
        _run(
            condition="cortheon",
            delivered=False,
            final_text="",
            timed_out=True,
            case_id=f"case_{index}",
        )
        for index in range(4)
    ]

    summary = _condition_summary(timeouts, "cortheon")

    assert summary["delivery_failures"] == 4
    assert summary["block_classification_coverage"] is None
    assert summary["unclassified_block_rate"] == 0.0
    assert summary["false_block_rate"] == 0.0


def test_unclassified_block_still_drives_coverage_below_one():
    results = [
        _run(
            condition="cortheon",
            delivered=False,
            final_text=WITHHELD,
            artifact_correct=True,
            sessions_completed=1,
        ),
        _run(
            condition="cortheon",
            delivered=False,
            final_text=WITHHELD,
            case_id="case_b",
            expected_verdict=None,
            sessions_completed=1,
        ),
    ]

    summary = _condition_summary(results, "cortheon")

    assert summary["block_classification_coverage"] == 0.5
    assert summary["unclassified_blocks"] == 1
    assert classify_block(results[0]) == FALSE_BLOCK
    assert classify_block(results[1]) == UNCLASSIFIED_BLOCK
