"""Task-specific structured and executable operator-lift oracles."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from cortheon.operator_lift.models import LiftCase, OracleResult


def _closed(value: Any, fields: set[str]) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) and set(value) == fields else None


def _triple(value: Any, fields: tuple[str, str, str]) -> tuple[str, str, str] | None:
    item = _closed(value, set(fields))
    if item is None or not all(isinstance(item[field], str) for field in fields):
        return None
    return tuple(item[field] for field in fields)  # type: ignore[return-value]


def _result(correct: bool, *reasons: str) -> OracleResult:
    return OracleResult(
        correct=correct,
        proof_eligible=True,
        reasons=() if correct else tuple(reasons) or ("oracle_mismatch",),
    )


def _grade_hypothesis(case: LiftCase, response: Mapping[str, Any]) -> OracleResult:
    item = _closed(response, {"leading", "rival", "falsification"})
    if item is None:
        return _result(False, "response_shape")
    leading = _triple(item["leading"], ("cause", "outcome", "scope"))
    rival = _triple(item["rival"], ("cause", "outcome", "scope"))
    falsification = _triple(item["falsification"], ("intervention", "result", "refutes"))
    if leading is None or leading != tuple(case.oracle["leading"]):
        return _result(False, "leading_relation")
    allowed_rivals = {tuple(value) for value in case.oracle["rivals"]}
    if rival is None or rival not in allowed_rivals or rival == leading:
        return _result(False, "distinct_rival")
    if falsification is None or falsification != tuple(case.oracle["falsification"]):
        return _result(False, "falsification_binding")
    if falsification[2] != leading[0]:
        return _result(False, "falsification_target")
    return _result(True)


def _grade_discrimination(case: LiftCase, response: Mapping[str, Any]) -> OracleResult:
    item = _closed(response, {"probe_id", "positive_supports", "negative_supports"})
    if item is None or not all(isinstance(value, str) for value in item.values()):
        return _result(False, "response_shape")
    actual = (item["probe_id"], item["positive_supports"], item["negative_supports"])
    if actual != tuple(case.oracle["expected"]):
        return _result(False, "probe_or_outcome_direction")
    hypotheses = tuple(case.oracle["hypotheses"])
    if actual[1] == actual[2] or set(actual[1:]) != set(hypotheses):
        return _result(False, "probe_does_not_separate")
    if actual[0] not in {action[0] for action in case.action_catalog}:
        return _result(False, "unknown_probe")
    return _result(True)


def _grade_revision(case: LiftCase, response: Mapping[str, Any]) -> OracleResult:
    fields = ("prior", "prior_status", "revised", "decisive_source")
    item = _closed(response, set(fields))
    if item is None or not all(isinstance(item[field], str) for field in fields):
        return _result(False, "response_shape")
    actual = tuple(item[field] for field in fields)
    if actual != tuple(case.oracle["expected"]):
        return _result(False, "revision_relation_or_source")
    status_map = case.response_schema.get("effect_status_map")
    change_map = case.response_schema.get("effect_changes_hypothesis")
    if not isinstance(status_map, Mapping) or not isinstance(change_map, Mapping):
        return _result(False, "revision_contract")
    declared = {
        change_map.get(effect)
        for effect, status in status_map.items()
        if status == actual[1] and type(change_map.get(effect)) is bool
    }
    if len(declared) != 1:
        return _result(False, "ambiguous_revision_status")
    changed = actual[0] != actual[2]
    if changed is not declared.pop():
        return _result(False, "revision_change_semantics")
    if actual[3] not in {source_id for source_id, _ in case.evidence}:
        return _result(False, "unknown_decisive_source")
    return _result(True)


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))


def _premise_stated(case: LiftCase, premise: tuple[str, str, str, str]) -> bool:
    source_id, subject, relation, object_id = premise
    content = dict(case.evidence).get(source_id)
    if content is None:
        return False
    expected = _normalized(f"{subject} {relation} {object_id}")
    return expected in _normalized(content)


def _grade_derivation(case: LiftCase, response: Mapping[str, Any]) -> OracleResult:
    item = _closed(response, {"subject", "relation", "object", "premises"})
    if item is None:
        return _result(False, "response_shape")
    conclusion = (item["subject"], item["relation"], item["object"])
    if not all(isinstance(value, str) for value in conclusion):
        return _result(False, "conclusion_shape")
    if conclusion != tuple(case.oracle["conclusion"]):
        return _result(False, "conclusion_direction_or_polarity")
    raw_premises = item["premises"]
    if not isinstance(raw_premises, Sequence) or isinstance(raw_premises, (str, bytes)):
        return _result(False, "premise_shape")
    premises: list[tuple[str, str, str, str]] = []
    fields = ("source_id", "subject", "relation", "object")
    for raw in raw_premises:
        entry = _closed(raw, set(fields))
        if entry is None or not all(isinstance(entry[field], str) for field in fields):
            return _result(False, "premise_shape")
        premises.append(tuple(entry[field] for field in fields))  # type: ignore[arg-type]
    expected = [tuple(value) for value in case.oracle["premises"]]
    if premises != expected:
        return _result(False, "premise_order_or_source_binding")
    if len({premise[0] for premise in premises}) != len(premises):
        return _result(False, "source_not_independently_necessary")
    if not all(_premise_stated(case, premise) for premise in premises):
        return _result(False, "unstated_premise")
    if premises[0][1] != conclusion[0] or premises[-1][3] != conclusion[2]:
        return _result(False, "path_endpoint")
    if any(left[3] != right[1] for left, right in pairwise(premises)):
        return _result(False, "path_disconnected")
    # This explicit full-path check is also the leave-one-source-out oracle:
    # deleting any independently sourced edge disconnects an endpoint or join.
    for index in range(len(premises)):
        reduced = premises[:index] + premises[index + 1 :]
        if reduced and (
            reduced[0][1] == conclusion[0]
            and reduced[-1][3] == conclusion[2]
            and all(left[3] == right[1] for left, right in pairwise(reduced))
        ):
            return _result(False, "source_not_necessary")
    return _result(True)


def _grade_stopping(case: LiftCase, response: Mapping[str, Any]) -> OracleResult:
    item = _closed(response, {"actions", "decision", "total_cost", "stop_reason"})
    if item is None:
        return _result(False, "response_shape")
    actions = item["actions"]
    if (
        not isinstance(actions, Sequence)
        or isinstance(actions, (str, bytes))
        or not all(isinstance(action, str) for action in actions)
    ):
        return _result(False, "action_shape")
    expected = tuple(case.oracle["expected_actions"])
    actual = tuple(actions)
    if actual != expected:
        if actual[: len(expected)] == expected:
            return _result(False, "continued_after_sufficiency")
        return _result(False, "premature_or_wrong_probe")
    costs = {action_id: cost for action_id, _description, cost in case.action_catalog}
    if len(set(actual)) != len(actual) or any(action not in costs for action in actual):
        return _result(False, "repeated_or_unknown_probe")
    expected_cost = sum(costs[action] for action in actual)
    if type(item["total_cost"]) is not int or item["total_cost"] != expected_cost:
        return _result(False, "cost_binding")
    if item["decision"] != case.oracle["decision"]:
        return _result(False, "decision")
    if item["stop_reason"] != "sufficient":
        return _result(False, "stop_justification")
    observations = tuple(tuple(value) for value in case.oracle["observations"])
    observation_ids = {action for action, _value in observations}
    if any(action not in observation_ids for action in expected):
        return _result(False, "oracle_observation_path")
    return _result(True)


_GRADERS = {
    "hypothesis_framing": _grade_hypothesis,
    "discriminating_evidence": _grade_discrimination,
    "contradiction_revision": _grade_revision,
    "cross_source_derivation": _grade_derivation,
    "adaptive_stopping": _grade_stopping,
}


def grade_case(case: LiftCase, response: Mapping[str, Any]) -> OracleResult:
    """Grade a structured answer with the case's operator-specific oracle."""

    if not isinstance(response, Mapping):
        return _result(False, "response_not_object")
    return _GRADERS[case.operator](case, response)
