"""Compile a task into a bounded public cognitive program."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cortheon.cognitive_graph import content_id
from cortheon.cognitive_protocol import evaluation_operator

_PURPOSES = {
    "orient": "Fix the deliverable, constraints, and strongest failure mode.",
    "decompose": (
        "Split the goal into ordered sub-goals, each with its own completion "
        "obligation, so partial work cannot be reported as a whole answer."
    ),
    "inspect_surface": "Acquire the smallest live task surface from the host.",
    "frame_hypotheses": "Create distinct falsifiable explanations.",
    "discriminate": "Choose evidence that best separates the explanations.",
    "connect_sources": "Join separately supported propositions into a candidate inference.",
    "resolve_contradiction": "Seek a bounded authority or scope distinction for conflicts.",
    "localize_cause": "Trace evidence to the smallest causal implementation boundary.",
    "verify_change": "Require a final diff followed by a host-verified behavioral test.",
    "establish_freshness": "Establish when time-sensitive evidence was published and retrieved.",
    "corroborate": "Test the claim against an independent source lineage.",
    "synthesize": "Separate observations, derivations, and residual uncertainty.",
    "challenge": "Attack the strongest conclusion with the best counterexample.",
    "verify": "Bind claims and completion obligations to accepted evidence.",
    "stop_inconclusive": "Stop without an answer when budget or evidence is exhausted.",
}

_SEQUENCES = {
    "code_change": (
        "orient",
        "inspect_surface",
        "frame_hypotheses",
        "discriminate",
        "localize_cause",
        "connect_sources",
        "verify_change",
        "challenge",
        "synthesize",
        "verify",
    ),
    "code_understanding": (
        "orient",
        "inspect_surface",
        "frame_hypotheses",
        "discriminate",
        "connect_sources",
        "challenge",
        "synthesize",
        "verify",
    ),
    "research_answer": (
        "orient",
        "frame_hypotheses",
        "discriminate",
        "establish_freshness",
        "corroborate",
        "connect_sources",
        "resolve_contradiction",
        "challenge",
        "synthesize",
        "verify",
    ),
    "document_synthesis": (
        "orient",
        "inspect_surface",
        "frame_hypotheses",
        "discriminate",
        "connect_sources",
        "resolve_contradiction",
        "challenge",
        "synthesize",
        "verify",
    ),
    "decision": (
        "orient",
        "inspect_surface",
        "frame_hypotheses",
        "discriminate",
        "connect_sources",
        "challenge",
        "synthesize",
        "verify",
    ),
    "answer": (
        "orient",
        "inspect_surface",
        "frame_hypotheses",
        "discriminate",
        "connect_sources",
        "challenge",
        "synthesize",
        "verify",
    ),
}


def _apply_effort(sequence: tuple[str, ...], effort: str, *, split: bool) -> tuple[str, ...]:
    """Shape the operator sequence by effort tier.

    The tier is the host's prompt-derived signal for how much structure a
    task warrants; without it a trivial lookup and a deep multi-source task
    would compile to byte-identical programs.
    """

    tier = effort.strip().lower()
    operators = list(sequence)

    if split and "decompose" not in operators:
        # Immediately after orient: the split governs everything downstream.
        operators.insert(1 if operators and operators[0] == "orient" else 0, "decompose")

    if tier == "quick":
        # Drop the operators that only pay for themselves on hard tasks. Verify
        # and synthesize always survive: cheap tasks may still be answered wrong.
        droppable = {"challenge", "resolve_contradiction", "corroborate", "decompose"}
        reduced = [item for item in operators if item not in droppable]
        return tuple(reduced or operators)

    if tier == "deep" and "corroborate" not in operators:
        # A deep task earns an independent source lineage before synthesis.
        index = operators.index("synthesize") if "synthesize" in operators else len(operators)
        operators.insert(index, "corroborate")
    return tuple(operators)


def compile_program(
    *,
    goal: str,
    task_kind: str,
    deliverable: str,
    effort: str,
    requirements: Iterable[tuple[str, str]],
    max_turns: int,
    max_observations: int,
    evaluation_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requirement_list = [(str(requirement_id), str(proof)) for requirement_id, proof in requirements]
    # Decomposition is implemented in cortheon.decomposition but not yet shipped:
    # the 150 KB runtime budget has no room for it. See tests/test_decomposition.py.
    sequence = _apply_effort(
        _SEQUENCES.get(deliverable, _SEQUENCES["answer"]),
        effort,
        split=False,
    )
    disabled_groups = {
        "retrieval": {
            "inspect_surface",
            "establish_freshness",
            "corroborate",
            "localize_cause",
        },
        "verification": {"verify", "verify_change"},
        "hypothesis_framing": {"frame_hypotheses"},
        "discriminating_evidence": {"discriminate"},
        "contradiction_revision": {"resolve_contradiction", "challenge"},
        "cross_source_derivation": {"connect_sources"},
    }
    disabled_operators = {
        operator
        for group, grouped_operators in disabled_groups.items()
        if not evaluation_operator(evaluation_profile, group)
        for operator in grouped_operators
    }
    sequence = tuple(
        operator
        for operator in sequence
        if all(
            evaluation_operator(evaluation_profile, group) or operator not in grouped_operators
            for group, grouped_operators in disabled_groups.items()
        )
        and not (
            evaluation_profile is not None
            and evaluation_profile["config"]["intercepts_final"] is False
            and operator == "synthesize"
        )
    )
    obligations = [
        {"requirement_id": requirement_id, "proof": proof}
        for requirement_id, proof in requirement_list
    ]
    identity = {
        "goal_digest": content_id("goal", goal),
        "task_kind": task_kind,
        "deliverable": deliverable,
        "effort": effort,
        "operators": sequence,
        "proof_obligations": obligations,
    }
    return {
        "program_id": content_id("cp", identity),
        "task_kind": task_kind,
        "deliverable": deliverable,
        "operators": [
            {
                "operator_id": operator_id,
                "purpose": _PURPOSES[operator_id],
                "ordinal": index,
            }
            for index, operator_id in enumerate(sequence, 1)
        ],
        "proof_obligations": obligations,
        "budgets": {
            "turns": max_turns,
            "observations": max_observations,
        },
        "stop_conditions": [
            "all material requirements and claims are evidence-covered",
            "the conclusion survives an adversarial challenge",
            "no evidence request remains pending",
            "budget exhaustion yields an inconclusive result",
            *(
                ["zero expected utility stops optional evidence gathering"]
                if evaluation_operator(evaluation_profile, "adaptive_stopping")
                else []
            ),
        ],
        "disabled_operators": sorted(disabled_operators),
    }


def select_operator(
    program: Mapping[str, Any],
    next_action: Mapping[str, Any],
    *,
    has_derivation: bool,
    has_conflict: bool,
) -> dict[str, Any]:
    action_type = str(next_action.get("type") or "")
    request = next_action.get("request")
    request = request if isinstance(request, Mapping) else {}
    parameters = request.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    capability = str(request.get("capability") or "")
    purpose = str(parameters.get("purpose") or "")
    required = next_action.get("required_fields")
    required = set(required) if isinstance(required, (list, tuple)) else set()
    if has_conflict:
        operator_id = "resolve_contradiction"
    elif has_derivation:
        operator_id = "connect_sources"
    elif action_type == "harness_tool" and request.get("hypothesis_id"):
        operator_id = "discriminate"
    elif purpose == "freshness_check":
        operator_id = "establish_freshness"
    elif purpose in {"corroboration", "primary_fetch", "contradiction_check"}:
        operator_id = "corroborate"
    elif capability in {"diff", "test"} and program.get("deliverable") == "code_change":
        operator_id = "verify_change"
    elif action_type == "harness_tool":
        operator_id = "inspect_surface"
    elif "hypotheses" in required:
        operator_id = "frame_hypotheses"
    elif action_type == "challenge":
        operator_id = "challenge"
    elif action_type in {"verify", "complete"}:
        operator_id = "verify"
    elif action_type == "finish":
        operator_id = "stop_inconclusive"
    else:
        operator_id = "synthesize"
    operators = {
        str(item.get("operator_id")): item
        for item in program.get("operators", ())
        if isinstance(item, Mapping)
    }
    selected = operators.get(operator_id)
    if operator_id in set(program.get("disabled_operators", ())):
        return {
            "operator_id": "none",
            "purpose": "No cognitive intervention is enabled for this move.",
            "ordinal": None,
        }
    if selected is None:
        selected = {
            "operator_id": operator_id,
            "purpose": _PURPOSES[operator_id],
            "ordinal": None,
        }
    return dict(selected)
