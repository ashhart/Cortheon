"""Held-out case selection and the digests that pin it.

The benchmark module is imported lazily, inside the call: issuing a pack must
not drag contender execution into the process that owns the evaluator's
secrets.
"""

from __future__ import annotations

from typing import Any

from cortheon.parity_benchmark_core.oracle_taxonomy import TASK_CLASSES, proof_binding


def normalize_and_select(
    raw_cases: list[Any],
    *,
    seed: int,
    holdout_fraction: float,
    rotation_index: int,
    rotation_size: int,
    min_cases_per_task_class: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize the submitted cases and draw the held-out selection.

    Returns the full normalized bank and the selected subset. Every grader
    must already carry a frozen external oracle, because a pack whose answers
    can move after sealing is not a held-out pack.
    """

    # Imported lazily to keep the pack issuer separate from contender execution.
    from cortheon.benchmark import _case_has_frozen_oracle, _normalize_cases, select_case_bank

    normalized_cases = _normalize_cases(
        {"cases": raw_cases},
        built_in=False,
        allow_external_patch_tests=True,
    )
    if not all(_case_has_frozen_oracle(case) for case in normalized_cases):
        raise ValueError(
            "every grader must declare oracle_provenance=frozen_external_pack "
            "and include any required answer_key"
        )
    selection = select_case_bank(
        normalized_cases,
        split="heldout",
        seed=seed,
        holdout_fraction=holdout_fraction,
        rotation_index=rotation_index,
        rotation_size=rotation_size,
    )
    if not selection:
        raise ValueError("the requested held-out selection is empty")
    selection = _rebalance_task_class_coverage(
        normalized_cases, selection, minimum=min_cases_per_task_class
    )
    return normalized_cases, selection


def selection_sha256(selection: list[dict[str, Any]]) -> str:
    """The case-bank digest the report must reproduce to prove the selection."""

    from cortheon.benchmark import _case_bank_hash

    return _case_bank_hash(selection)


def validate_task_class_coverage(selection: list[dict[str, Any]], minimum: int) -> None:
    counts = {
        task_class: sum(
            case.get("task_class") == task_class and proof_binding(case) is not None
            for case in selection
        )
        for task_class in TASK_CLASSES
    }
    if any(count < minimum for count in counts.values()):
        missing = sorted(key for key, count in counts.items() if count < minimum)
        raise ValueError(
            "sealed selection lacks proof cases for task classes: " + ", ".join(missing)
        )


def _rebalance_task_class_coverage(
    bank: list[dict[str, Any]], selection: list[dict[str, Any]], *, minimum: int
) -> list[dict[str, Any]]:
    result = list(selection)
    if len(result) < minimum * len(TASK_CLASSES):
        raise ValueError("held-out selection is too small for the registered task-class floor")
    selected_ids = {case["id"] for case in result}
    counts = {
        task_class: sum(case.get("task_class") == task_class for case in result)
        for task_class in TASK_CLASSES
    }
    for task_class in sorted(TASK_CLASSES):
        while counts[task_class] < minimum:
            replacement = next(
                (
                    case
                    for case in bank
                    if case["id"] not in selected_ids
                    and case.get("task_class") == task_class
                    and proof_binding(case) is not None
                ),
                None,
            )
            donor_index = next(
                (
                    index
                    for index in range(len(result) - 1, -1, -1)
                    if result[index].get("task_class") not in TASK_CLASSES
                    or counts[str(result[index].get("task_class"))] > minimum
                ),
                None,
            )
            if replacement is None or donor_index is None:
                raise ValueError(f"case bank cannot satisfy task-class floor for {task_class}")
            removed = result[donor_index]
            removed_class = removed.get("task_class")
            if removed_class in TASK_CLASSES:
                counts[str(removed_class)] -= 1
            selected_ids.remove(removed["id"])
            result[donor_index] = replacement
            selected_ids.add(replacement["id"])
            counts[task_class] += 1
    return result
