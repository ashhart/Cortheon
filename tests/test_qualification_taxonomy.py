"""Failure taxonomy over the block classification."""

from qualification_support import WITHHELD_TERMINAL, _result

from cortheon.qualification_factory import _failure_type


def _blocked(case_id, **overrides):
    return _result(
        case_id,
        0,
        "cortheon",
        False,
        delivered=False,
        final_text=WITHHELD_TERMINAL,
        telemetry=True,
        **overrides,
    )


def test_failure_taxonomy_distinguishes_false_allow_and_false_block():
    false_allow = _result("a", 0, "cortheon", False, telemetry=True, expected_verdict="block")
    false_block = _blocked("b", artifact_correct=True)
    safe_block = _blocked("c", artifact_correct=False, expected_verdict="block")
    unclassified_block = _blocked("d", expected_verdict=None)

    assert _failure_type(false_allow) == "false_allow"
    assert _failure_type(false_block) == "false_block"
    assert _failure_type(safe_block) == "safe_block"
    assert _failure_type(unclassified_block) == "unclassified_block"


def test_undelivered_run_without_a_withheld_terminal_is_a_delivery_failure():
    # No answer and no block notice. Reporting this as an artifact failure
    # would credit the run with restraint it never exercised.
    silent = _result(
        "e",
        0,
        "cortheon",
        False,
        delivered=False,
        final_text="",
        artifact_correct=False,
        telemetry=True,
    )

    assert _failure_type(silent) == "delivery_failure"


def test_timeout_outranks_the_block_taxonomy_even_on_a_withheld_terminal():
    # The timeout is reported as itself. It is the earlier, more specific
    # fact, and it is never an artifact failure.
    timed_out = _result(
        "f",
        0,
        "cortheon",
        False,
        delivered=False,
        final_text=WITHHELD_TERMINAL,
        artifact_correct=False,
        telemetry=True,
        timed_out=True,
    )

    assert _failure_type(timed_out) == "timeout"
