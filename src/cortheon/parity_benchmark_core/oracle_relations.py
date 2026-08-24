"""Typed numeric, semantic-document, and abductive oracles."""

from __future__ import annotations

from decimal import Decimal, DivisionByZero
from typing import Any

from cortheon.parity_benchmark_core.oracle_common import (
    closed_object,
    decimal_value,
    record_map,
    string_set,
)


def grade_numeric(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    if not closed_object(answer, {"facts", "derivation", "result"}, {"explanation"}):
        return ["invalid_answer_schema"]
    expected_facts = record_map(oracle.get("facts"), ("id", "value", "unit", "source_id"))
    observed_facts = record_map(answer.get("facts"), ("id", "value", "unit", "source_id"))
    failures: list[str] = []
    if expected_facts is None or observed_facts != expected_facts:
        failures.append("wrong_source_facts")
    allowed_operations = string_set(oracle.get("allowed_operations"), minimum=1, maximum=4)
    computed = _execute_derivation(
        answer.get("derivation"), observed_facts or {}, allowed_operations or set()
    )
    result = answer.get("result")
    expected_result = oracle.get("result")
    result_ref = result.get("ref") if isinstance(result, dict) else None
    if (
        computed is None
        or not isinstance(result, dict)
        or set(result) != {"ref", "value", "unit"}
        or not isinstance(expected_result, dict)
        or not isinstance(result_ref, str)
        or result_ref not in computed[0]
        or decimal_value(result.get("value")) != computed[0].get(result_ref)
        or decimal_value(result.get("value")) != decimal_value(expected_result.get("value"))
        or result.get("unit") != expected_result.get("unit")
    ):
        failures.append("wrong_derivation_result")
    elif allowed_operations is None or computed[1].get(result_ref) != string_set(
        oracle.get("necessary_fact_ids"), minimum=2
    ):
        failures.append("unused_or_unbound_source_fact")
    return failures


def grade_semantic_documents(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    if not closed_object(answer, {"hops", "conclusion", "necessary_source_ids"}, {"explanation"}):
        return ["invalid_answer_schema"]
    fields = ("id", "subject", "relation", "object", "polarity", "quantity", "unit", "source_ids")
    expected = record_map(oracle.get("hops"), fields)
    observed = record_map(answer.get("hops"), fields)
    failures: list[str] = []
    if (
        expected is None
        or observed is None
        or set(expected) != set(observed)
        or any(not _semantic_hop_equal(expected[key], observed[key]) for key in expected)
    ):
        failures.append("wrong_semantic_hops")
    if answer.get("conclusion") != oracle.get("conclusion"):
        failures.append("wrong_conclusion")
    if string_set(answer.get("necessary_source_ids"), minimum=2) != string_set(
        oracle.get("necessary_source_ids"), minimum=2
    ):
        failures.append("missing_necessary_source")
    return failures


def grade_abduction(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    required = {"hypotheses", "selected_hypothesis", "premises", "discriminator", "conclusion"}
    if not closed_object(answer, required, {"explanation"}):
        return ["invalid_answer_schema"]
    failures: list[str] = []
    selected = _proposition(oracle.get("selected_proposition"))
    rivals = {
        proposition
        for item in oracle.get("accepted_rivals") or []
        if (proposition := _proposition(item)) is not None
    }
    hypotheses = _hypotheses(answer.get("hypotheses"))
    if (
        selected is None
        or len(rivals) < 2
        or hypotheses is None
        or hypotheses.get(selected) != "selected"
        or not any(hypotheses.get(rival) == "ruled_out" for rival in rivals)
        or any(item not in rivals | {selected} for item in hypotheses)
    ):
        failures.append("wrong_hypothesis_set")
    if _proposition(answer.get("selected_hypothesis")) != selected:
        failures.append("wrong_selected_hypothesis")
    premise_fields = ("id", "proposition_id", "fact", "source_id")
    expected_premises = record_map(oracle.get("premises"), premise_fields)
    observed_premises = record_map(answer.get("premises"), premise_fields)
    if expected_premises is None or observed_premises != expected_premises:
        failures.append("wrong_source_bound_premises")
    discriminator_fields = {"observation", "supports", "rules_out", "source_id"}
    discriminator = answer.get("discriminator")
    expected_discriminator = oracle.get("discriminator")
    if (
        not isinstance(discriminator, dict)
        or set(discriminator) != discriminator_fields
        or not isinstance(expected_discriminator, dict)
        or discriminator.get("observation") != expected_discriminator.get("observation")
        or discriminator.get("source_id") != expected_discriminator.get("source_id")
        or _proposition(discriminator.get("supports")) != selected
        or _proposition(discriminator.get("rules_out")) not in rivals
    ):
        failures.append("wrong_discriminating_observation")
    if answer.get("conclusion") != oracle.get("conclusion"):
        failures.append("wrong_novel_conclusion")
    required_sources = string_set(oracle.get("necessary_source_ids"), minimum=2)
    dependencies = string_set(oracle.get("conclusion_dependencies"), minimum=2)
    observed_records = list((observed_premises or {}).values())
    observed_sources = {item.get("source_id") for item in observed_records}
    observed_propositions = {item.get("proposition_id") for item in observed_records}
    if (
        required_sources is None
        or dependencies is None
        or required_sources != observed_sources
        or dependencies != observed_propositions
        or any(
            dependencies
            <= {
                item.get("proposition_id")
                for item in observed_records
                if item.get("source_id") != omitted
            }
            for omitted in required_sources
        )
    ):
        failures.append("leave_one_source_out_failure")
    return failures


def _proposition(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, dict) or set(value) != {"subject", "relation", "object"}:
        return None
    subject = value.get("subject")
    relation = value.get("relation")
    object_value = value.get("object")
    if (
        not isinstance(subject, str)
        or not subject
        or not isinstance(relation, str)
        or not relation
        or not isinstance(object_value, str)
        or not object_value
    ):
        return None
    return subject, relation, object_value


def _hypotheses(value: Any) -> dict[tuple[str, str, str], str] | None:
    if not isinstance(value, list) or not 2 <= len(value) <= 8:
        return None
    result: dict[tuple[str, str, str], str] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"proposition", "status"}:
            return None
        proposition = _proposition(item.get("proposition"))
        status = item.get("status")
        if proposition is None or status not in {"selected", "ruled_out"} or proposition in result:
            return None
        result[proposition] = status
    return result


def _semantic_hop_equal(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    scalar = ("id", "subject", "relation", "object", "polarity", "unit")
    return (
        all(observed.get(key) == expected.get(key) for key in scalar)
        and decimal_value(observed.get("quantity")) == decimal_value(expected.get("quantity"))
        and string_set(observed.get("source_ids"), minimum=1)
        == string_set(expected.get("source_ids"), minimum=1)
    )


def _execute_derivation(
    value: Any,
    facts: dict[str, dict[str, Any]],
    allowed_operations: set[str],
) -> tuple[dict[str, Decimal], dict[str, set[str]], dict[str, dict[str, int]]] | None:
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        return None
    values: dict[str, Decimal] = {}
    units: dict[str, dict[str, int]] = {}
    for key, item in facts.items():
        number = decimal_value(item.get("value"))
        dimension = _unit_dimension(item.get("unit"))
        if number is None or dimension is None:
            return None
        values[key] = number
        units[key] = dimension
    provenance = {key: {key} for key in facts}
    for operation in value:
        if not isinstance(operation, dict) or set(operation) != {"id", "op", "args", "unit"}:
            return None
        identifier = operation.get("id")
        args = operation.get("args")
        if (
            not isinstance(identifier, str)
            or identifier in values
            or not isinstance(args, list)
            or len(args) != 2
            or any(arg not in values for arg in args)
        ):
            return None
        left, right = values[args[0]], values[args[1]]
        op = str(operation.get("op"))
        if op not in allowed_operations:
            return None
        declared_dimension = _unit_dimension(operation.get("unit"))
        expected_dimension = _operation_dimension(op, units[args[0]], units[args[1]])
        if declared_dimension is None or expected_dimension != declared_dimension:
            return None
        try:
            result = _calculate(op, left, right)
        except DivisionByZero:
            return None
        if result is None or not result.is_finite():
            return None
        values[identifier] = result
        units[identifier] = declared_dimension
        provenance[identifier] = provenance[args[0]] | provenance[args[1]]
    return values, provenance, units


def _unit_dimension(value: Any) -> dict[str, int] | None:
    if not isinstance(value, str) or not value or len(value) > 80:
        return None
    if value == "scalar":
        return {}
    parts = value.split("/")
    if len(parts) > 2:
        return None
    dimensions: dict[str, int] = {}
    for index, group in enumerate(parts):
        atoms = group.split("*")
        if any(not atom or not atom.replace("_", "").isalnum() for atom in atoms):
            return None
        sign = 1 if index == 0 else -1
        for atom in atoms:
            dimensions[atom] = dimensions.get(atom, 0) + sign
            if dimensions[atom] == 0:
                del dimensions[atom]
    return dimensions


def _operation_dimension(
    op: str, left: dict[str, int], right: dict[str, int]
) -> dict[str, int] | None:
    if op in {"add", "subtract"}:
        return dict(left) if left == right else None
    if op not in {"multiply", "divide"}:
        return None
    result = dict(left)
    sign = 1 if op == "multiply" else -1
    for atom, exponent in right.items():
        result[atom] = result.get(atom, 0) + sign * exponent
        if result[atom] == 0:
            del result[atom]
    return result


def _calculate(op: str, left: Decimal, right: Decimal) -> Decimal | None:
    if op == "add":
        return left + right
    if op == "subtract":
        return left - right
    if op == "multiply":
        return left * right
    if op == "divide":
        return left / right
    return None
