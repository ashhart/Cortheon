"""Hostile C11 tests for semantic blocks and evaluator-owned failure attribution."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

from parity_gates_support import CANDIDATE_ALIAS, build_report

from cortheon.benchmark import Contender, ModelResult, run_benchmark
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.parity import evaluate_frontier_parity
from cortheon.parity_benchmark_core.oracle_taxonomy import ORACLE_SPECS
from cortheon.parity_gates.paired_validation import canonical_paired_comparisons
from cortheon.parity_gates.report_metrics import validate_release_report
from cortheon.parity_gates.summary_validation import canonical_summary


def _recompute(report: dict) -> None:
    report["summary"] = canonical_summary(report["rows"], report["candidates"])
    report["paired_comparisons"] = canonical_paired_comparisons(
        report["rows"], report["candidates"], report["methodology"]["seed"]
    )


def _candidate_row(report: dict, *, expected: str = "allow") -> dict:
    return next(
        row
        for row in report["rows"]
        if row["candidate"] == CANDIDATE_ALIAS and row["expected_verdict"] == expected
    )


def _make_failure(row: dict, owner: str) -> None:
    row.update(
        {
            "passed": False,
            "verified_completion": False,
            "verification_assurance": "not_graded",
            "proof_eligible": False,
            "grade_failures": ["delivery_failure"],
            "observed_verdict": "error",
            "classification": "error",
            "completion_origin": "error",
            "evaluator_outcome": {
                "schema_version": 1,
                "transport": "cli",
                "terminal_status": "incomplete",
                "terminal_provenance": "process_exit",
                "finish_reason": "empty_output",
            },
            "failure_owner": owner,
            "error": "delivery_failure",
        }
    )
    row.pop("answer", None)
    row.pop("verdict_source", None)


def _make_withhold(row: dict) -> None:
    expected_block = row["expected_verdict"] == "block"
    row.update(
        {
            "passed": expected_block,
            "verified_completion": expected_block,
            "proof_eligible": True,
            "verification_assurance": ORACLE_SPECS[row["task_class"]].assurance,
            "grade_failures": [] if expected_block else ["withheld_expected_allow"],
            "observed_verdict": "block",
            "classification": "correct" if expected_block else "false_block",
            "completion_origin": "controller_only",
            "evaluator_outcome": {
                "schema_version": 1,
                "transport": "pi",
                "terminal_status": "withheld",
                "terminal_provenance": "pi_custom_terminal",
                "finish_reason": "withheld",
            },
            "failure_owner": None,
        }
    )
    row.pop("answer", None)


def test_authenticated_withhold_follows_sealed_task_semantics() -> None:
    report, contract, _digest = build_report()
    allow = _candidate_row(report, expected="allow")
    block = _candidate_row(report, expected="block")
    _make_withhold(allow)
    _make_withhold(block)
    _recompute(report)

    validate_release_report(report, contract)
    summary = report["summary"][CANDIDATE_ALIAS]
    assert summary["false_blocks"] == 1
    assert block["verified_completion"] is True


def test_candidate_failure_stays_paired_and_blocks_promotion() -> None:
    report, contract, digest = build_report()
    row = _candidate_row(report)
    _make_failure(row, "candidate")
    _recompute(report)

    validate_release_report(report, contract)
    candidate = report["summary"][CANDIDATE_ALIAS]
    assert candidate["candidate_delivery_failures"] == 1
    comparison = next(
        item for item in report["paired_comparisons"] if item["left"] == CANDIDATE_ALIAS
    )
    assert comparison["paired_runs"] > 0
    assert comparison["invalid_cases"] == 0
    decision = evaluate_frontier_parity(report, contract, contract_sha256=digest)
    assert "zero_candidate_delivery_failures" in decision["failure_reasons"]


def test_external_failure_invalidates_whole_repeat_cluster() -> None:
    report, contract, digest = build_report()
    row = _candidate_row(report)
    case_id = row["case_id"]
    _make_failure(row, "external_infrastructure")
    _recompute(report)

    validate_release_report(report, contract)
    comparison = next(
        item for item in report["paired_comparisons"] if item["left"] == CANDIDATE_ALIAS
    )
    assert comparison["invalid_cases"] == 1
    assert comparison["paired_runs"] == len(report["rows"]) // 3 - 5
    assert all(
        entry["case_id"] != case_id
        for entry in report["rows"]
        if entry["candidate"] == CANDIDATE_ALIAS
        and entry["failure_owner"] == "external_infrastructure"
        and entry is not row
    )
    decision = evaluate_frontier_parity(report, contract, contract_sha256=digest)
    assert "external_infrastructure_failure_ceiling" in decision["failure_reasons"]


def test_owner_flip_cannot_be_hidden_by_recomputed_caches() -> None:
    report, contract, _digest = build_report()
    row = _candidate_row(report)
    row["failure_owner"] = "candidate"
    _recompute(report)

    try:
        validate_release_report(report, contract)
    except ValueError as error:
        assert "failure owner" in str(error)
    else:  # pragma: no cover - hostile mutation must fail
        raise AssertionError("terminal row accepted a fabricated failure owner")


def test_live_runner_classifies_withhold_without_candidate_text() -> None:
    contender = Contender("cortheon", "cortheon", "http://local", "small", "")
    case = {
        "id": "allow_case",
        "category": "research",
        "domain": "research",
        "difficulty": "hard",
        "expected_verdict": "allow",
        "grader": {"type": "document_relations"},
    }
    result = ModelResult(
        answer="ignored",
        latency_ms=1.0,
        metadata={},
        evaluator_outcome=EvaluationOutcome("pi", "withheld", "pi_custom_terminal", "withheld"),
    )
    with patch("cortheon.benchmark.call_contender", return_value=result):
        report = run_benchmark(
            [contender],
            [case],
            repetitions=1,
            seed=1,
            timeout=1,
            max_tokens=8,
            include_answers=False,
        )

    assert report["schema_version"] == 6
    assert report["rows"][0]["classification"] == "false_block"
    assert report["rows"][0]["failure_owner"] is None


def _small_pair(
    treatment: list[bool],
    bare: list[bool],
) -> list[dict]:
    rows = []
    for repetition, (treatment_ok, bare_ok) in enumerate(zip(treatment, bare, strict=True), 1):
        for alias, correct in (
            ("candidate_1", treatment_ok),
            ("candidate_2", bare_ok),
        ):
            rows.append(
                {
                    "case_id": "case",
                    "repetition": repetition,
                    "candidate": alias,
                    "verified_completion": correct,
                    "failure_owner": None if correct else "candidate",
                }
            )
    return rows


def _small_comparison(rows: list[dict]) -> dict:
    return canonical_paired_comparisons(
        rows,
        {"candidate_1": {}, "candidate_2": {}},
        seed=7,
    )[0]


def test_candidate_failures_score_win_loss_and_tie_without_disappearing() -> None:
    win = _small_comparison(_small_pair([True], [False]))
    loss = _small_comparison(_small_pair([False], [True]))
    tie = _small_comparison(_small_pair([False], [False]))

    assert (win["left_wins"], win["right_wins"], win["ties"]) == (1, 0, 0)
    assert (loss["left_wins"], loss["right_wins"], loss["ties"]) == (0, 1, 0)
    assert (tie["left_wins"], tie["right_wins"], tie["ties"]) == (0, 0, 1)


def test_repeats_cluster_and_permutation_do_not_inflate_independence() -> None:
    rows = _small_pair([True] * 6, [False] * 6)

    forward = _small_comparison(rows)
    reverse = _small_comparison(list(reversed(rows)))

    assert forward == reverse
    assert forward["paired_runs"] == 6
    assert forward["paired_cases"] == 1
    assert forward["left_wins"] == 1


def test_duplicate_or_missing_cells_invalidate_the_whole_case() -> None:
    complete = _small_pair([True, True], [False, False])
    duplicate = [*complete, deepcopy(complete[0])]
    missing = complete[1:]

    duplicated = _small_comparison(duplicate)
    incomplete = _small_comparison(missing)

    assert duplicated["duplicate_cells"] == 1
    assert duplicated["invalid_cases"] == 1
    assert duplicated["paired_runs"] == 0
    assert incomplete["invalid_cases"] == 1
    assert incomplete["paired_runs"] == 0
