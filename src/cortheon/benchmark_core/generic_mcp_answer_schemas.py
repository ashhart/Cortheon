"""Closed answer schemas derived only from evaluator-bound public case data."""

from __future__ import annotations

from typing import Any


def derivation_schema(
    payload: dict[str, Any],
    response: dict[str, Any],
    fields: list[Any],
) -> dict[str, Any] | None:
    premise_fields = response.get("premise_fields")
    relations = response.get("relation_vocabulary")
    tokens = response.get("token_vocabulary")
    evidence = payload.get("evidence")
    if (
        fields != ["subject", "relation", "object", "premises"]
        or premise_fields != ["source_id", "subject", "relation", "object"]
        or not _vocabulary(relations)
        or not _vocabulary(tokens)
        or not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 16
    ):
        return None
    source_ids = [item.get("source_id") for item in evidence if isinstance(item, dict)]
    if len(source_ids) != len(evidence) or not _vocabulary(source_ids):
        return None
    token_schema = {"type": "string", "enum": tokens}
    premise = {
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "enum": source_ids},
            "subject": token_schema,
            "relation": token_schema,
            "object": token_schema,
        },
        "required": premise_fields,
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "subject": token_schema,
            "relation": {"type": "string", "enum": relations},
            "object": token_schema,
            "premises": {
                "type": "array",
                "items": premise,
                "minItems": len(evidence),
                "maxItems": len(evidence),
            },
        },
        "required": fields,
        "additionalProperties": False,
    }


def stopping_schema(
    payload: dict[str, Any],
    response: dict[str, Any],
    fields: list[Any],
) -> dict[str, Any] | None:
    decisions = response.get("decision_vocabulary")
    actions = payload.get("actions")
    if (
        fields != ["actions", "decision", "total_cost", "stop_reason"]
        or not _vocabulary(decisions)
        or not isinstance(actions, list)
        or not 1 <= len(actions) <= 16
    ):
        return None
    action_ids = [item.get("action_id") for item in actions if isinstance(item, dict)]
    if len(action_ids) != len(actions) or not _vocabulary(action_ids):
        return None
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "string", "enum": action_ids},
                "minItems": 1,
                "maxItems": len(action_ids),
            },
            "decision": {"type": "string", "enum": decisions},
            "total_cost": {"type": "integer"},
            "stop_reason": {"type": "string", "enum": ["sufficient"]},
        },
        "required": fields,
        "additionalProperties": False,
    }


def _vocabulary(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and len(value) <= 128
        and all(isinstance(item, str) and 0 < len(item) <= 256 for item in value)
        and len(set(value)) == len(value)
    )
