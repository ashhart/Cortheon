from __future__ import annotations

from copy import deepcopy

import pytest

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.models import LiftCase
from cortheon.operator_lift.oracles import grade_case


def _valid(case: LiftCase) -> dict:
    if case.operator == "hypothesis_framing":
        leading = tuple(case.oracle["leading"])
        rival = next(iter(case.oracle["rivals"]))
        falsification = tuple(case.oracle["falsification"])
        return {
            "leading": dict(zip(("cause", "outcome", "scope"), leading, strict=True)),
            "rival": dict(zip(("cause", "outcome", "scope"), rival, strict=True)),
            "falsification": dict(
                zip(("intervention", "result", "refutes"), falsification, strict=True)
            ),
        }
    if case.operator == "discriminating_evidence":
        return dict(
            zip(
                ("probe_id", "positive_supports", "negative_supports"),
                case.oracle["expected"],
                strict=True,
            )
        )
    if case.operator == "contradiction_revision":
        return dict(
            zip(
                ("prior", "prior_status", "revised", "decisive_source"),
                case.oracle["expected"],
                strict=True,
            )
        )
    if case.operator == "cross_source_derivation":
        subject, relation, object_id = case.oracle["conclusion"]
        return {
            "subject": subject,
            "relation": relation,
            "object": object_id,
            "premises": [
                dict(
                    zip(
                        ("source_id", "subject", "relation", "object"),
                        premise,
                        strict=True,
                    )
                )
                for premise in case.oracle["premises"]
            ],
        }
    actions = tuple(case.oracle["expected_actions"])
    costs = {action_id: cost for action_id, _description, cost in case.action_catalog}
    return {
        "actions": list(actions),
        "decision": case.oracle["decision"],
        "total_cost": sum(costs[action] for action in actions),
        "stop_reason": "sufficient",
    }


def _case(operator: str) -> LiftCase:
    return next(case for case in development_cases() if case.operator == operator)


def test_all_sixty_private_oracles_accept_their_structured_solution() -> None:
    for case in development_cases():
        result = grade_case(case, _valid(case))
        assert result.correct and result.proof_eligible, (case.case_id, result)


@pytest.mark.parametrize("payload", [{}, {"answer": "because therefore"}, ["all", "keywords"]])
def test_keyword_salads_and_copied_shapes_fail(payload) -> None:
    for case in development_cases():
        assert not grade_case(case, payload).correct


def test_hypothesis_oracle_requires_distinct_rival_and_bound_falsifier() -> None:
    case = _case("hypothesis_framing")
    valid = _valid(case)
    same = deepcopy(valid)
    same["rival"] = same["leading"]
    assert not grade_case(case, same).correct
    wrong_target = deepcopy(valid)
    wrong_target["falsification"]["refutes"] = "cohort_selection_bias"
    assert not grade_case(case, wrong_target).correct
    swapped = deepcopy(valid)
    swapped["leading"], swapped["rival"] = swapped["rival"], swapped["leading"]
    assert not grade_case(case, swapped).correct


def test_discrimination_oracle_rejects_reversed_and_nonseparating_outcomes() -> None:
    case = _case("discriminating_evidence")
    valid = _valid(case)
    reversed_outcomes = deepcopy(valid)
    reversed_outcomes["positive_supports"], reversed_outcomes["negative_supports"] = (
        reversed_outcomes["negative_supports"],
        reversed_outcomes["positive_supports"],
    )
    assert not grade_case(case, reversed_outcomes).correct
    same = deepcopy(valid)
    same["negative_supports"] = same["positive_supports"]
    assert not grade_case(case, same).correct
    copied_all = {**valid, "other_probe": case.action_catalog[1][0]}
    assert not grade_case(case, copied_all).correct


def test_revision_oracle_requires_polarity_change_and_decisive_source() -> None:
    case = _case("contradiction_revision")
    valid = _valid(case)
    unchanged = deepcopy(valid)
    unchanged["revised"] = unchanged["prior"]
    assert not grade_case(case, unchanged).correct
    unsupported = deepcopy(valid)
    unsupported["decisive_source"] = "source_a"
    assert not grade_case(case, unsupported).correct
    positive = deepcopy(valid)
    positive["prior_status"] = "supported"
    assert not grade_case(case, positive).correct


def test_derivation_oracle_requires_direction_source_roles_and_every_edge() -> None:
    case = _case("cross_source_derivation")
    valid = _valid(case)
    reversed_relation = deepcopy(valid)
    reversed_relation["subject"], reversed_relation["object"] = (
        reversed_relation["object"],
        reversed_relation["subject"],
    )
    assert not grade_case(case, reversed_relation).correct
    missing = deepcopy(valid)
    missing["premises"].pop(1)
    assert not grade_case(case, missing).correct
    misattributed = deepcopy(valid)
    misattributed["premises"][0]["source_id"] = "source_b"
    assert not grade_case(case, misattributed).correct
    prompt_copy = {
        "subject": case.prompt,
        "relation": "steward",
        "object": case.prompt,
        "premises": [],
    }
    assert not grade_case(case, prompt_copy).correct


def test_stopping_oracle_rejects_premature_redundant_and_post_sufficiency_work() -> None:
    case = next(
        case
        for case in development_cases()
        if case.operator == "adaptive_stopping" and len(case.oracle["expected_actions"]) == 2
    )
    valid = _valid(case)
    premature = deepcopy(valid)
    premature["actions"] = premature["actions"][:1]
    premature["total_cost"] -= 1
    assert not grade_case(case, premature).correct
    extra = deepcopy(valid)
    extra["actions"].append(case.action_catalog[-1][0])
    extra["total_cost"] += case.action_catalog[-1][2]
    assert "continued_after_sufficiency" in grade_case(case, extra).reasons
    false_cost = deepcopy(valid)
    false_cost["total_cost"] += 1
    assert not grade_case(case, false_cost).correct
