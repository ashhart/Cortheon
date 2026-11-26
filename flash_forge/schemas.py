"""Causal JSON schemas enforced by TensorFold's libtfgrammar + grammar_budget.

Key principle for local MoE models: field order is causal.
Generating observed_facts -> invariants -> edge_cases BEFORE plan/verdict
forces the KV cache to condition the decision on failure modes it would otherwise skip.
"""

from __future__ import annotations

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "inputs_and_types": {"type": "array", "items": {"type": "string"}},
        "required_signatures": {"type": "array", "items": {"type": "string"}},
        "hard_invariants": {"type": "array", "items": {"type": "string"}},
        "tricky_edge_cases": {"type": "array", "items": {"type": "string"}},
        "complexity_bound": {"type": "string"},
        "common_traps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "goal",
        "inputs_and_types",
        "required_signatures",
        "hard_invariants",
        "tricky_edge_cases",
        "complexity_bound",
        "common_traps",
    ],
    "additionalProperties": False,
}

ANSWER_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "answer_type": {
            "type": "string",
            "enum": ["integer", "fraction", "float", "boolean", "string", "expression", "list"],
        },
        "confidence_notes": {"type": "string"},
    },
    "required": ["final_answer", "answer_type", "confidence_notes"],
    "additionalProperties": False,
}

REDTEAM_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "checked_invariants": {"type": "array", "items": {"type": "string"}},
        "bug_found": {"type": "boolean"},
        "bug_summary": {"type": "string"},
        "counterexample_description": {"type": "string"},
    },
    "required": [
        "checked_invariants",
        "bug_found",
        "bug_summary",
        "counterexample_description",
    ],
    "additionalProperties": False,
}

LESSON_DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "symptom": {"type": "string"},
        "dead_end": {"type": "string"},
        "verified_rule": {"type": "string"},
    },
    "required": ["domain", "symptom", "dead_end", "verified_rule"],
    "additionalProperties": False,
}
