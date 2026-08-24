"""Hostile semantic tests for six closed structured oracle classes."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
from north_star_oracle_support import cases, encoded

from cortheon.parity_benchmark_core.casepack import _normalize_cases
from cortheon.parity_benchmark_core.grading import grade_answer


@pytest.mark.parametrize(
    "task_class",
    [
        "ambiguity_resolution",
        "constraint_bound_planning",
        "cross_file_numeric_join",
        "evidence_bound_debugging",
        "long_horizon_execution",
        "semantic_cross_document_reasoning",
    ],
)
def test_valid_structured_answers_accept_explanatory_paraphrases(task_class: str) -> None:
    case, answer = cases()[task_class]
    _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)

    terse = grade_answer(case, encoded(answer, "Short explanation."))
    paraphrase = grade_answer(case, encoded(answer, "A wholly different explanation in prose."))

    assert terse["passed"] is True and terse["proof_eligible"] is True
    assert paraphrase["passed"] is True
    assert terse["method"] == case["grader"]["type"]


def test_ambiguity_rejects_copied_prompt_and_wrong_discriminator() -> None:
    case, answer = cases()["ambiguity_resolution"]
    wrong = deepcopy(answer)
    wrong["discriminators"][0]["value"] = "USD"

    assert grade_answer(case, case["prompt"])["passed"] is False
    assert "wrong_discriminators" in grade_answer(case, encoded(wrong))["failures"]


def test_planning_rejects_omission_wrong_quantity_and_forbidden_order() -> None:
    case, answer = cases()["constraint_bound_planning"]
    omitted = deepcopy(answer)
    omitted["steps"].pop(1)
    quantity = deepcopy(answer)
    quantity["constraints"][0]["value"] = "2h"
    reversed_steps = deepcopy(answer)
    reversed_steps["steps"].reverse()

    assert grade_answer(case, encoded(omitted))["passed"] is False
    assert "wrong_constraints" in grade_answer(case, encoded(quantity))["failures"]
    assert "forbidden_step_order" in grade_answer(case, encoded(reversed_steps))["failures"]


def test_numeric_requires_exact_typed_facts_and_every_source_in_result() -> None:
    case, answer = cases()["cross_file_numeric_join"]
    wrong = deepcopy(answer)
    wrong["facts"][0]["value"] = 120
    wrong_unit = deepcopy(answer)
    wrong_unit["facts"][0]["unit"] = "widgets"
    shortcut = deepcopy(answer)
    shortcut["derivation"] = [
        {"id": "total", "op": "multiply", "args": ["rate", "days"], "unit": "widgets"}
    ]
    wrong_dimension = deepcopy(answer)
    wrong_dimension["derivation"][0]["unit"] = "USD"

    assert "wrong_source_facts" in grade_answer(case, encoded(wrong))["failures"]
    assert "wrong_source_facts" in grade_answer(case, encoded(wrong_unit))["failures"]
    assert "wrong_derivation_result" in grade_answer(case, encoded(shortcut))["failures"]
    assert "wrong_derivation_result" in grade_answer(case, encoded(wrong_dimension))["failures"]


def test_numeric_issuance_rejects_substring_and_unbound_units() -> None:
    case, _answer = cases()["cross_file_numeric_join"]
    case["documents"][0]["text"] = "Daily output is 120 widgets/day."
    import hashlib

    case["grader"]["oracle"]["source_bindings"][0]["sha256"] = hashlib.sha256(
        case["documents"][0]["text"].encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="fact and unit"):
        _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)


def test_debugging_rejects_plausible_wrong_cause_and_swapped_evidence() -> None:
    case, answer = cases()["evidence_bound_debugging"]
    wrong = deepcopy(answer)
    wrong["cause"] = "NETWORK_CONGESTION"
    swapped = deepcopy(answer)
    swapped["evidence"][0]["source_ids"], swapped["evidence"][1]["source_ids"] = (
        swapped["evidence"][1]["source_ids"],
        swapped["evidence"][0]["source_ids"],
    )

    assert "wrong_cause" in grade_answer(case, encoded(wrong))["failures"]
    assert "wrong_evidence_chain" in grade_answer(case, encoded(swapped))["failures"]


def test_horizon_rejects_reverse_order_missing_gate_and_incomplete_path() -> None:
    case, answer = cases()["long_horizon_execution"]
    reverse = deepcopy(answer)
    reverse["steps"].reverse()
    no_gate = deepcopy(answer)
    no_gate["gates"] = []
    incomplete = deepcopy(answer)
    incomplete["steps"].pop(1)

    assert "incomplete_horizon" in grade_answer(case, encoded(reverse))["failures"]
    assert grade_answer(case, encoded(no_gate))["passed"] is False
    assert grade_answer(case, encoded(incomplete))["passed"] is False


def test_semantic_documents_reject_direction_polarity_quantity_and_source_swaps() -> None:
    case, answer = cases()["semantic_cross_document_reasoning"]
    mutations = []
    for key, value in (
        ("subject", "pods"),
        ("polarity", "negative"),
        ("quantity", 84),
    ):
        changed = deepcopy(answer)
        changed["hops"][0][key] = value
        mutations.append(changed)
    swapped = deepcopy(answer)
    swapped["hops"][0]["source_ids"] = ["benchmark://semantic/b"]
    mutations.append(swapped)

    assert all(grade_answer(case, encoded(value))["passed"] is False for value in mutations)


@pytest.mark.parametrize(
    ("task_class", "document_index", "old", "new"),
    [
        ("constraint_bound_planning", 0, "Add schema", "Inspect schema"),
        ("evidence_bound_debugging", 1, "POOL_UNDERSIZED", "NETWORK_CONGESTION"),
        ("long_horizon_execution", 2, "OWNER_ELENA", "OWNER_MAYA"),
        ("semantic_cross_document_reasoning", 1, "active", "paused"),
    ],
)
def test_issuance_rejects_source_premise_mutations_even_with_updated_document_hash(
    task_class: str, document_index: int, old: str, new: str
) -> None:
    case, _answer = cases()[task_class]
    document = case["documents"][document_index]
    document["text"] = document["text"].replace(old, new)
    source_id = document["uri"]
    binding = next(
        item for item in case["grader"]["oracle"]["source_bindings"] if item["id"] == source_id
    )
    binding["sha256"] = hashlib.sha256(document["text"].encode()).hexdigest()

    with pytest.raises(ValueError, match=r"premise|present"):
        _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)
