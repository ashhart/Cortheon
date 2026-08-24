"""Structured intent, planning, debugging, and horizon oracles."""

from __future__ import annotations

from typing import Any

from cortheon.parity_benchmark_core.oracle_common import (
    closed_object,
    record_map,
    string_set,
)


def grade_ambiguity(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not closed_object(
        answer,
        {"resolved_intent", "decision", "discriminators"},
        {"clarification_id", "explanation"},
    ):
        return ["invalid_answer_schema"]
    if answer.get("resolved_intent") != oracle.get("resolved_intent"):
        failures.append("wrong_resolved_intent")
    if answer.get("decision") != oracle.get("decision"):
        failures.append("wrong_clarification_decision")
    expected = record_map(oracle.get("discriminators"), ("id", "value", "source_id"))
    observed = record_map(answer.get("discriminators"), ("id", "value", "source_id"))
    if expected is None or observed != expected:
        failures.append("wrong_discriminators")
    if oracle.get("decision") == "clarify":
        accepted = string_set(oracle.get("accepted_clarification_ids"), minimum=1)
        if accepted is None or answer.get("clarification_id") not in accepted:
            failures.append("wrong_clarification")
    elif "clarification_id" in answer:
        failures.append("unnecessary_clarification")
    return failures


def grade_constraint_plan(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    if not closed_object(answer, {"steps", "dependencies", "constraints"}, {"explanation"}):
        return ["invalid_answer_schema"]
    expected_steps = record_map(oracle.get("steps"), ("id", "action", "source_id"))
    observed_steps = record_map(answer.get("steps"), ("id", "action", "source_id"))
    failures: list[str] = []
    if expected_steps is None or observed_steps != expected_steps:
        failures.append("incomplete_or_wrong_steps")
    expected_edges = _edge_set(oracle.get("dependencies"))
    observed_edges = _edge_set(answer.get("dependencies"))
    if expected_edges is None or observed_edges != expected_edges:
        failures.append("wrong_dependencies")
    expected_constraints = record_map(
        oracle.get("constraints"), ("id", "step_id", "operator", "value", "unit", "source_id")
    )
    observed_constraints = record_map(
        answer.get("constraints"), ("id", "step_id", "operator", "value", "unit", "source_id")
    )
    if expected_constraints is None or observed_constraints != expected_constraints:
        failures.append("wrong_constraints")
    order = [item.get("id") for item in answer.get("steps", []) if isinstance(item, dict)]
    positions = {step_id: index for index, step_id in enumerate(order)}
    step_ids = set(observed_steps or {})
    if (
        observed_edges is None
        or not _valid_dag(step_ids, observed_edges)
        or any(
            left not in positions or right not in positions or positions[left] >= positions[right]
            for left, right in observed_edges
        )
    ):
        failures.append("forbidden_step_order")
    forbidden = _edge_set(oracle.get("forbidden_dependencies"))
    if forbidden is None or any(
        left in positions and right in positions and positions[left] < positions[right]
        for left, right in forbidden
    ):
        failures.append("forbidden_step_order")
    return failures


def grade_debugging(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    required = {"symptom", "cause", "fix", "verification", "evidence"}
    if not closed_object(answer, required, {"explanation"}):
        return ["invalid_answer_schema"]
    failures = [
        f"wrong_{field}"
        for field in ("symptom", "cause", "fix", "verification")
        if answer.get(field) != oracle.get(field)
    ]
    expected = record_map(oracle.get("evidence"), ("stage", "source_ids"), key="stage")
    observed = record_map(answer.get("evidence"), ("stage", "source_ids"), key="stage")
    if (
        expected is None
        or observed is None
        or set(expected) != set(observed)
        or any(
            string_set(observed[key]["source_ids"], minimum=1)
            != string_set(expected[key]["source_ids"], minimum=1)
            for key in expected
        )
    ):
        failures.append("wrong_evidence_chain")
    return failures


def grade_horizon(oracle: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    if not closed_object(
        answer,
        {"steps", "dependencies", "gates", "terminal_step_id", "final_owner"},
        {"summary", "explanation"},
    ):
        return ["invalid_answer_schema"]
    failures: list[str] = []
    for field, keys in (
        ("steps", ("id", "action", "source_id")),
        ("gates", ("id", "after_step", "condition", "source_id")),
    ):
        if record_map(answer.get(field), keys) != record_map(oracle.get(field), keys):
            failures.append(f"incomplete_or_wrong_{field}")
    expected_edges = _edge_set(oracle.get("dependencies"))
    observed_edges = _edge_set(answer.get("dependencies"))
    if expected_edges is None or observed_edges != expected_edges:
        failures.append("wrong_dependencies")
    step_ids = set(record_map(answer.get("steps"), ("id", "action", "source_id")) or {})
    terminal = answer.get("terminal_step_id")
    if terminal != oracle.get("terminal_step_id") or not isinstance(terminal, str):
        failures.append("wrong_terminal_step")
    elif observed_edges is None or not _valid_dag(step_ids, observed_edges):
        failures.append("invalid_horizon_graph")
    else:
        ancestors = _ancestors(terminal, observed_edges)
        gates = record_map(answer.get("gates"), ("id", "after_step", "condition", "source_id"))
        order = [item.get("id") for item in answer.get("steps", []) if isinstance(item, dict)]
        positions = {step_id: index for index, step_id in enumerate(order)}
        if (
            gates is None
            or step_ids - {terminal} - ancestors
            or any(
                positions.get(left, -1) >= positions.get(right, -1)
                for left, right in observed_edges
            )
            or any(
                gate.get("after_step") not in ancestors
                or positions.get(str(gate.get("after_step")), -1) >= positions.get(terminal, -1)
                for gate in gates.values()
            )
        ):
            failures.append("incomplete_horizon")
    if answer.get("final_owner") != oracle.get("final_owner"):
        failures.append("wrong_final_owner")
    return failures


def _edge_set(value: Any) -> set[tuple[str, str]] | None:
    if not isinstance(value, list) or len(value) > 128:
        return None
    edges: set[tuple[str, str]] = set()
    for edge in value:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(not isinstance(item, str) or not item for item in edge)
        ):
            return None
        edges.add((edge[0], edge[1]))
    return edges if len(edges) == len(value) else None


def _valid_dag(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
    if any(left not in nodes or right not in nodes or left == right for left, right in edges):
        return False
    incoming = dict.fromkeys(nodes, 0)
    outgoing = {node: set() for node in nodes}
    for left, right in edges:
        incoming[right] += 1
        outgoing[left].add(right)
    ready = [node for node, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for successor in outgoing[node]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
    return visited == len(nodes)


def _ancestors(node: str, edges: set[tuple[str, str]]) -> set[str]:
    result: set[str] = set()
    pending = [left for left, right in edges if right == node]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(left for left, right in edges if right == current)
    return result
