"""Independent-case pairing: repeats measure stability, never independence."""

from qualification_support import WITHHELD_TERMINAL, _cell, _result

from cortheon.cognitive_benchmark import ImportCase
from cortheon.qualification_factory import (
    CellRun,
    _aggregate_pairing,
    _independent_pairing,
    _sealed_task_digest,
)


def test_repeats_measure_stability_without_inflating_independence_or_ci():
    once = []
    repeated = []
    for case_id, cortheon_correct, baseline_correct in (
        ("a", True, False),
        ("b", True, True),
    ):
        once.extend(
            [
                _result(
                    case_id,
                    0,
                    "full",
                    cortheon_correct,
                    telemetry=True,
                ),
                _result(case_id, 0, "bare", baseline_correct),
            ]
        )
        for repeat in range(5):
            repeated.extend(
                [
                    _result(
                        case_id,
                        repeat,
                        "full",
                        cortheon_correct,
                        telemetry=True,
                    ),
                    _result(case_id, repeat, "bare", baseline_correct),
                ]
            )

    one_summary, _, _ = _independent_pairing(
        once,
        treatment="full",
        comparison="bare",
        repeats=(0,),
        seed=11,
    )
    repeated_summary, _, _ = _independent_pairing(
        repeated,
        treatment="full",
        comparison="bare",
        repeats=tuple(range(5)),
        seed=11,
    )

    assert repeated_summary["independent_cases"] == 2
    assert repeated_summary["repeat_pairs"] == 10
    assert repeated_summary["accuracy_delta"] == one_summary["accuracy_delta"]
    assert repeated_summary["accuracy_delta_95_ci"] == one_summary["accuracy_delta_95_ci"]


def test_six_winning_repeats_remain_one_underpowered_independent_case():
    results = [
        _result(
            "one",
            repeat,
            condition,
            condition == "full",
            telemetry=condition == "full",
        )
        for repeat in range(6)
        for condition in ("full", "bare")
    ]

    summary, deltas, invalid = _independent_pairing(
        results,
        treatment="full",
        comparison="bare",
        repeats=tuple(range(6)),
        seed=11,
    )

    assert summary["repeat_pairs"] == 6
    assert summary["independent_cases"] == 1
    assert summary["treatment_wins"] == 1
    assert summary["paired_sign_test_exact_p"] == 1.0
    assert deltas == {"one": 1.0}
    assert invalid == set()


def test_invalid_repeat_invalidates_the_independent_case():
    results = [
        _result("a", 0, "full", True, telemetry=True),
        _result("a", 0, "bare", False),
        _result("a", 1, "full", True, telemetry=True),
        _result(
            "a",
            1,
            "bare",
            False,
            process_error="private path and error",
            failure_owner="external_infrastructure",
        ),
    ]

    summary, deltas, invalid = _independent_pairing(
        results,
        treatment="full",
        comparison="bare",
        repeats=(0, 1),
        seed=3,
    )

    assert summary["qualified_independent_cases"] == 0
    assert summary["invalid_pairs"] == 1
    assert deltas == {}
    assert invalid == {"a"}


def test_duplicate_cell_is_invalid_and_cannot_win_by_input_order():
    rows = [
        _result("a", 0, "full", True, telemetry=True),
        _result("a", 0, "full", False, telemetry=True),
        _result("a", 0, "bare", False),
    ]

    forward = _independent_pairing(
        rows,
        treatment="full",
        comparison="bare",
        repeats=(0,),
        seed=3,
    )
    reverse = _independent_pairing(
        list(reversed(rows)),
        treatment="full",
        comparison="bare",
        repeats=(0,),
        seed=3,
    )

    assert forward == reverse
    summary, deltas, invalid = forward
    assert summary["duplicate_cells"] == 1
    assert summary["qualified_independent_cases"] == 0
    assert deltas == {}
    assert invalid == {"a"}


def test_a_timed_out_comparator_invalidates_the_pair_it_could_have_lost():
    # The same rule the benchmark proof applies: a comparator that ran out of
    # wall clock observed no outcome, so it cannot hand the treatment a delta.
    results = [
        _result("a", 0, "full", True, telemetry=True),
        _result(
            "a",
            0,
            "bare",
            False,
            delivered=False,
            final_text="",
            timed_out=True,
            failure_owner="external_infrastructure",
        ),
    ]

    summary, deltas, invalid = _independent_pairing(
        results,
        treatment="full",
        comparison="bare",
        repeats=(0,),
        seed=3,
    )

    assert summary["valid_repeat_pairs"] == 0
    assert summary["invalid_pairs"] == 1
    assert summary["treatment_wins"] == 0
    assert deltas == {}
    assert invalid == {"a"}


def test_a_withheld_block_stays_a_scorable_comparator_outcome():
    # The control. The treatment declined with a candidate in hand, which is
    # an outcome, so the pair is valid and the comparator wins it.
    results = [
        _result(
            "a",
            0,
            "full",
            False,
            delivered=False,
            final_text=WITHHELD_TERMINAL,
            artifact_correct=False,
            telemetry=True,
        ),
        _result("a", 0, "bare", True),
    ]

    summary, deltas, invalid = _independent_pairing(
        results,
        treatment="full",
        comparison="bare",
        repeats=(0,),
        seed=3,
    )

    assert summary["valid_repeat_pairs"] == 1
    assert summary["invalid_pairs"] == 0
    assert summary["comparison_wins"] == 1
    assert deltas == {"a": -1.0}
    assert invalid == set()


def test_cross_cell_reuse_does_not_create_fake_independent_cases():
    first = CellRun(
        cell=_cell(cell_id="first"),
        case_ids=("seed-1-case",),
        task_digests={"seed-1-case": "same-sealed-task"},
        results=[],
        pairing={
            "repeat_pairs": 1,
            "valid_repeat_pairs": 1,
            "invalid_pairs": 0,
            "unstable_cases": 0,
        },
        case_deltas={"seed-1-case": 1.0},
        invalid_case_ids=set(),
        repository_unchanged=True,
        environment_stable=True,
        runtime={"storage": "memory_only"},
        inference={"ok": True},
        host_version="1",
        contrasts={
            "full_vs_bare": {
                "repeat_pairs": 1,
                "valid_repeat_pairs": 1,
                "invalid_pairs": 0,
                "unstable_cases": 0,
            }
        },
        contrast_case_deltas={"full_vs_bare": {"seed-1-case": 1.0}},
        contrast_invalid_case_ids={"full_vs_bare": set()},
    )
    second = CellRun(
        cell=_cell(cell_id="second"),
        case_ids=("seed-2-case",),
        task_digests={"seed-2-case": "same-sealed-task"},
        results=[],
        pairing={
            "repeat_pairs": 1,
            "valid_repeat_pairs": 1,
            "invalid_pairs": 0,
            "unstable_cases": 0,
        },
        case_deltas={"seed-2-case": 0.0},
        invalid_case_ids=set(),
        repository_unchanged=True,
        environment_stable=True,
        runtime={"storage": "memory_only"},
        inference={"ok": True},
        host_version="1",
        contrasts={
            "full_vs_bare": {
                "repeat_pairs": 1,
                "valid_repeat_pairs": 1,
                "invalid_pairs": 0,
                "unstable_cases": 0,
            }
        },
        contrast_case_deltas={"full_vs_bare": {"seed-2-case": 0.0}},
        contrast_invalid_case_ids={"full_vs_bare": set()},
    )

    aggregate = _aggregate_pairing([first, second], contrast="full_vs_bare", seed=5)

    assert aggregate["independent_cases"] == 1
    assert aggregate["cell_case_exposures"] == 2
    assert aggregate["accuracy_delta"] == 0.5


def test_sealed_task_identity_ignores_seed_derived_case_ids():
    first = ImportCase(
        case_id="case_seed_one",
        path="src/example.py",
        module="pathlib",
        expected=True,
        prompt="Inspect the file.",
    )
    second = ImportCase(
        case_id="case_seed_two",
        path="src/example.py",
        module="pathlib",
        expected=True,
        prompt="Inspect the same file without changing it.",
    )

    assert _sealed_task_digest(first) == _sealed_task_digest(second)
