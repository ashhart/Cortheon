"""Sixty independent development cases for causal operator attribution.

Each case is one independent cluster. The cases share response protocols, not
stories, entities, causal mechanisms, evidence graphs, or optimal probes.
"""

from __future__ import annotations

from cortheon.operator_lift.models import LiftCase

_STRUCTURED = {"type": "closed_json_object", "additional_fields": False}


def _e(first: str, second: str, third: str | None = None) -> tuple[tuple[str, str], ...]:
    items = (("source_a", first), ("source_b", second))
    return items if third is None else (*items, ("source_c", third))


def _hyp(
    number: int,
    family: str,
    evidence: tuple[tuple[str, str], ...],
    leading: tuple[str, str, str],
    rival: tuple[str, str, str],
    falsification: tuple[str, str, str],
) -> LiftCase:
    case_id = f"hypothesis_{number:02d}"
    return LiftCase(
        case_id=case_id,
        cluster_id=f"cluster_{case_id}",
        operator="hypothesis_framing",
        causal_family=family,
        prompt=(
            "Frame the strongest causal hypothesis and one genuinely distinct rival. Return their cause, outcome and scope IDs, then a falsifying intervention with the result that would refute the leader. IDs appear in the evidence."
        ),
        evidence=evidence,
        response_schema={
            **_STRUCTURED,
            "fields": {
                "leading": ["cause", "outcome", "scope"],
                "rival": ["cause", "outcome", "scope"],
                "falsification": ["intervention", "result", "refutes"],
            },
            "vocabulary": sorted({*leading, *rival, *falsification}),
            "field_vocabulary": {
                "leading": {
                    "cause": sorted({leading[0], rival[0]}),
                    "outcome": [leading[1]],
                    "scope": [leading[2]],
                },
                "rival": {
                    "cause": sorted({leading[0], rival[0]}),
                    "outcome": [rival[1]],
                    "scope": [rival[2]],
                },
                "falsification": {
                    "intervention": [falsification[0]],
                    "result": [falsification[1]],
                    "refutes": sorted({leading[0], rival[0]}),
                },
            },
        },
        oracle={"leading": leading, "rivals": (rival,), "falsification": falsification},
    )


def _probe(
    number: int,
    family: str,
    evidence: tuple[tuple[str, str], ...],
    hypotheses: tuple[str, str],
    actions: tuple[tuple[str, str, int], ...],
    expected: tuple[str, str, str],
) -> LiftCase:
    case_id = f"discriminate_{number:02d}"
    return LiftCase(
        case_id=case_id,
        cluster_id=f"cluster_{case_id}",
        operator="discriminating_evidence",
        causal_family=family,
        prompt=(
            f"The live hypotheses are {hypotheses[0]} and {hypotheses[1]}. Choose exactly one available probe whose opposite outcomes separate them. Return probe_id, positive_supports, and negative_supports."
        ),
        evidence=evidence,
        response_schema={
            **_STRUCTURED,
            "fields": ["probe_id", "positive_supports", "negative_supports"],
            "field_vocabulary": {
                "probe_id": [action_id for action_id, _description, _cost in actions],
                "positive_supports": list(hypotheses),
                "negative_supports": list(hypotheses),
            },
        },
        oracle={"hypotheses": hypotheses, "expected": expected},
        action_catalog=actions,
    )


def _revision(
    number: int,
    family: str,
    evidence: tuple[tuple[str, str], ...],
    expected: tuple[str, str, str, str],
    *,
    effect_contract: dict[str, tuple[str, bool]],
) -> LiftCase:
    case_id = f"revision_{number:02d}"
    if not 2 <= len(effect_contract) <= 8:
        raise ValueError("revision effect contract is invalid")
    normalized: dict[str, tuple[str, bool]] = {}
    for effect, semantics in effect_contract.items():
        if (
            not isinstance(effect, str)
            or not effect
            or not isinstance(semantics, tuple)
            or len(semantics) != 2
            or not isinstance(semantics[0], str)
            or not semantics[0]
            or type(semantics[1]) is not bool
        ):
            raise ValueError("revision effect contract is invalid")
        normalized[effect] = semantics
    by_status: dict[str, bool] = {}
    for status, changes in normalized.values():
        if status in by_status and by_status[status] is not changes:
            raise ValueError("revision status has conflicting change semantics")
        by_status[status] = changes
    if set(by_status.values()) != {False, True}:
        raise ValueError("revision effect contract needs change and retention outcomes")
    status_map = {effect: status for effect, (status, _changes) in normalized.items()}
    change_map = {effect: changes for effect, (_status, changes) in normalized.items()}
    expected_change = expected[0] != expected[2]
    matching_effects = [
        effect
        for effect, status in status_map.items()
        if status == expected[1] and change_map[effect] is expected_change
    ]
    if len(matching_effects) != 1:
        raise ValueError("revision answer is ambiguous under its effect contract")
    fields = ["prior", "prior_status", "revised", "decisive_source"]
    shift = (number - 1) % len(fields)
    fields = fields[shift:] + fields[:shift]
    return LiftCase(
        case_id=case_id,
        cluster_id=f"cluster_{case_id}",
        operator="contradiction_revision",
        causal_family=family,
        prompt=(
            "Evidence arrives in source order. Bind the original hypothesis, its new "
            "status, the revised or retained hypothesis, and the decisive source that "
            "determined whether a change was required."
        ),
        evidence=evidence,
        response_schema={
            **_STRUCTURED,
            "fields": fields,
            "hypothesis_vocabulary": list(dict.fromkeys((expected[0], expected[2]))),
            "status_vocabulary": sorted(set(status_map.values())),
            "effect_status_map": status_map,
            "effect_changes_hypothesis": change_map,
        },
        oracle={"expected": expected},
    )


def _join(
    number: int,
    family: str,
    evidence: tuple[tuple[str, str], ...],
    conclusion: tuple[str, str, str],
    premises: tuple[tuple[str, str, str, str], ...],
) -> LiftCase:
    case_id = f"derivation_{number:02d}"
    return LiftCase(
        case_id=case_id,
        cluster_id=f"cluster_{case_id}",
        operator="cross_source_derivation",
        causal_family=family,
        prompt=(
            "Derive the requested relation from separately sourced premises. Return subject, relation, object, and the exact ordered source-bound premise path."
        ),
        evidence=evidence,
        response_schema={
            **_STRUCTURED,
            "fields": ["subject", "relation", "object", "premises"],
            "premise_fields": ["source_id", "subject", "relation", "object"],
            "relation_vocabulary": [conclusion[1]],
            "token_vocabulary": sorted(
                {value for premise in premises for value in premise} | set(conclusion)
            ),
        },
        oracle={"conclusion": conclusion, "premises": premises},
    )


def _stop(
    number: int,
    family: str,
    evidence: tuple[tuple[str, str], ...],
    actions: tuple[tuple[str, str, int], ...],
    expected_actions: tuple[str, ...],
    decision: str,
    observations: tuple[tuple[str, str], ...],
) -> LiftCase:
    case_id = f"stopping_{number:02d}"
    revealed = dict(observations)
    for action_id, _description, _cost in actions:
        revealed.setdefault(action_id, f"uninformative_{action_id}")
    return LiftCase(
        case_id=case_id,
        cluster_id=f"cluster_{case_id}",
        operator="adaptive_stopping",
        causal_family=family,
        prompt=(
            "Choose probes sequentially, then stop as soon as the decision is identified. Return the ordered action IDs, decision ID, total_cost, and stop_reason=sufficient."
        ),
        evidence=evidence,
        response_schema={
            **_STRUCTURED,
            "fields": ["actions", "decision", "total_cost", "stop_reason"],
            "decision_vocabulary": [decision, "undetermined"],
        },
        oracle={
            "expected_actions": expected_actions,
            "decision": decision,
            "observations": tuple(
                (action_id, revealed[action_id]) for action_id, _description, _cost in actions
            ),
        },
        action_catalog=actions,
    )
