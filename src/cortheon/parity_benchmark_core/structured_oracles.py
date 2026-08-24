"""Dispatcher for proof-grade structured North Star oracles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cortheon.parity_benchmark_core.oracle_common import answer_object, evidence_digest
from cortheon.parity_benchmark_core.oracle_graphs import (
    grade_ambiguity,
    grade_constraint_plan,
    grade_debugging,
    grade_horizon,
)
from cortheon.parity_benchmark_core.oracle_relations import (
    grade_abduction,
    grade_numeric,
    grade_semantic_documents,
)
from cortheon.parity_benchmark_core.oracle_taxonomy import proof_binding
from cortheon.parity_benchmark_core.oracle_web import grade_current_web, validate_public_web_prompt

_GRADERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], list[str]]] = {
    "ambiguity_resolution": grade_ambiguity,
    "constraint_bound_planning": grade_constraint_plan,
    "cross_file_numeric_join": grade_numeric,
    "current_web_research": grade_current_web,
    "evidence_bound_debugging": grade_debugging,
    "long_horizon_execution": grade_horizon,
    "novel_abductive_synthesis": grade_abduction,
    "semantic_cross_document_reasoning": grade_semantic_documents,
}


def grade_structured_oracle(case: dict[str, Any], answer: str) -> tuple[list[str], str | None]:
    binding = proof_binding(case)
    if binding is None:
        return ["invalid_oracle_binding"], None
    task_class, _spec = binding
    if task_class == "repository_patching":
        return [], None
    payload = answer_object(answer)
    if payload is None:
        return ["invalid_structured_answer"], None
    grader = _GRADERS.get(task_class)
    if grader is None:
        return ["unsupported_structured_oracle"], None
    failures = grader(case["grader"]["oracle"], payload)
    return failures, evidence_digest(task_class, payload)


def validate_private_oracle(case: dict[str, Any]) -> None:
    """Reject malformed or internally inconsistent private oracle payloads."""

    binding = proof_binding(case)
    if binding is None:
        raise ValueError(f"case {case.get('id')} has an invalid proof binding")
    task_class, _spec = binding
    if task_class == "repository_patching":
        return
    oracle = case["grader"]["oracle"]
    private_fields = {
        "ambiguity_resolution": {
            "resolved_intent",
            "decision",
            "discriminators",
            "accepted_clarification_ids",
            "source_bindings",
        },
        "constraint_bound_planning": {
            "steps",
            "dependencies",
            "constraints",
            "forbidden_dependencies",
            "source_bindings",
        },
        "cross_file_numeric_join": {
            "facts",
            "allowed_operations",
            "necessary_fact_ids",
            "derivation",
            "result",
            "source_bindings",
        },
        "current_web_research": {
            "as_of",
            "revalidated_at",
            "valid_until",
            "truth_digest",
            "revalidated_truth_digest",
            "sources",
            "origin_equivalence",
            "claims",
            "contradictions",
            "acquisition_attestation",
        },
        "evidence_bound_debugging": {
            "symptom",
            "cause",
            "fix",
            "verification",
            "evidence",
            "evidence_facts",
            "source_bindings",
        },
        "long_horizon_execution": {
            "steps",
            "dependencies",
            "gates",
            "terminal_step_id",
            "final_owner",
            "owner_source_id",
            "source_bindings",
        },
        "novel_abductive_synthesis": {
            "selected_proposition",
            "accepted_rivals",
            "premises",
            "discriminator",
            "conclusion",
            "necessary_source_ids",
            "conclusion_dependencies",
            "source_bindings",
        },
        "semantic_cross_document_reasoning": {
            "hops",
            "conclusion",
            "necessary_source_ids",
            "source_premises",
            "source_bindings",
        },
    }[task_class]
    if set(oracle) != private_fields:
        raise ValueError(f"case {case.get('id')} oracle payload fields are not closed")
    public_fields = {
        "ambiguity_resolution": {"resolved_intent", "decision", "discriminators"},
        "constraint_bound_planning": {"steps", "dependencies", "constraints"},
        "cross_file_numeric_join": {"facts", "derivation", "result"},
        "current_web_research": {"as_of", "sources", "claims", "contradictions"},
        "evidence_bound_debugging": {"symptom", "cause", "fix", "verification", "evidence"},
        "long_horizon_execution": {
            "steps",
            "dependencies",
            "gates",
            "terminal_step_id",
            "final_owner",
        },
        "novel_abductive_synthesis": {
            "hypotheses",
            "selected_hypothesis",
            "premises",
            "discriminator",
            "conclusion",
        },
        "semantic_cross_document_reasoning": {
            "hops",
            "conclusion",
            "necessary_source_ids",
        },
    }[task_class]
    candidate = {key: oracle.get(key) for key in public_fields}
    if task_class == "ambiguity_resolution" and oracle.get("decision") == "clarify":
        choices = oracle.get("accepted_clarification_ids")
        candidate["clarification_id"] = (
            choices[0] if isinstance(choices, list) and choices else None
        )
    if task_class == "novel_abductive_synthesis":
        selected = oracle.get("selected_proposition")
        rivals = oracle.get("accepted_rivals")
        rival = rivals[0] if isinstance(rivals, list) and rivals else None
        candidate["hypotheses"] = [
            {"proposition": selected, "status": "selected"},
            {"proposition": rival, "status": "ruled_out"},
        ]
        candidate["selected_hypothesis"] = selected
    if task_class == "current_web_research":
        candidate["sources"] = [
            {"canonical_url": item.get("canonical_url")}
            for item in oracle.get("sources") or []
            if isinstance(item, dict)
        ]
    failures = _GRADERS[task_class](oracle, candidate)
    if failures:
        raise ValueError(f"case {case.get('id')} has invalid oracle payload: {failures[0]}")
    if task_class == "current_web_research":
        validate_public_web_prompt(case)
