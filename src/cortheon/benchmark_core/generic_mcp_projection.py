"""Exact model-facing tool projection for the generic evaluator host."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.generic_mcp_answer_schemas import (
    derivation_schema,
    stopping_schema,
)
from cortheon.cognitive_core.runtime_completion import _parse_public_revision_contract
from cortheon.cognitive_mcp import tool_definitions


def lifecycle_tools() -> list[dict[str, Any]]:
    selected = {"cortheon_complete", "cortheon_retract", "cortheon_abandon"}
    return [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["inputSchema"],
            },
        }
        for item in tool_definitions()
        if item["name"] in selected
    ]


def completion_answer_schema(
    root: Path,
    resource_paths: tuple[str, ...],
) -> dict[str, Any] | None:
    """Build a closed answer schema from one evaluator-bound public resource."""

    if len(resource_paths) != 1:
        return None
    try:
        candidate = (root / resource_paths[0]).resolve(strict=True)
        candidate.relative_to(root.resolve(strict=True))
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    response = payload.get("response_schema") if isinstance(payload, dict) else None
    fields = response.get("fields") if isinstance(response, dict) else None
    vocabulary = response.get("field_vocabulary") if isinstance(response, dict) else None
    if isinstance(fields, list) and isinstance(response, dict):
        revision = _flat_revision_schema(payload, response, fields)
        derivation = derivation_schema(payload, response, fields)
        stopping = stopping_schema(payload, response, fields)
        return revision or derivation or stopping or _flat_vocabulary_schema(response, fields)
    if (
        not isinstance(fields, dict)
        or not isinstance(vocabulary, dict)
        or set(fields) != set(vocabulary)
    ):
        return None
    properties: dict[str, Any] = {}
    for object_name, raw_fields in fields.items():
        allowed_fields = vocabulary.get(object_name)
        if (
            not isinstance(object_name, str)
            or not isinstance(raw_fields, list)
            or not raw_fields
            or not all(isinstance(item, str) for item in raw_fields)
            or not isinstance(allowed_fields, dict)
            or set(raw_fields) != set(allowed_fields)
        ):
            return None
        nested: dict[str, Any] = {}
        for field in raw_fields:
            allowed = allowed_fields[field]
            if (
                not isinstance(allowed, list)
                or not 1 <= len(allowed) <= 32
                or not all(isinstance(item, str) and 0 < len(item) <= 256 for item in allowed)
                or len(set(allowed)) != len(allowed)
            ):
                return None
            nested[field] = {"type": "string", "enum": allowed}
        properties[object_name] = {
            "type": "object",
            "properties": nested,
            "required": raw_fields,
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }


def _flat_vocabulary_schema(
    response: dict[str, Any],
    fields: list[Any],
) -> dict[str, Any] | None:
    vocabulary = response.get("field_vocabulary")
    if (
        not fields
        or not all(isinstance(field, str) for field in fields)
        or not isinstance(vocabulary, dict)
        or set(fields) != set(vocabulary)
    ):
        return None
    properties: dict[str, Any] = {}
    for field in fields:
        allowed = vocabulary[field]
        if (
            not isinstance(allowed, list)
            or not allowed
            or not all(isinstance(item, str) and item for item in allowed)
            or len(set(allowed)) != len(allowed)
        ):
            return None
        properties[field] = {"type": "string", "enum": allowed}
    return {
        "type": "object",
        "properties": properties,
        "required": fields,
        "additionalProperties": False,
    }


def _flat_revision_schema(
    payload: dict[str, Any],
    response: dict[str, Any],
    fields: list[Any],
) -> dict[str, Any] | None:
    """Project the public contradiction-revision vocabulary into closed JSON Schema."""

    contract = _parse_public_revision_contract(payload)
    if contract is None:
        return None
    hypotheses = contract["hypotheses"]
    sources = contract["sources"]
    statuses = sorted(set(contract["status_map"].values()))
    effect_status_map = contract["status_map"]
    effect_changes = contract["change_map"]
    return {
        "type": "object",
        "properties": {
            "prior": {"type": "string", "enum": hypotheses},
            "prior_status": {"type": "string", "enum": statuses},
            "revised": {"type": "string", "enum": hypotheses},
            "decisive_source": {"type": "string", "enum": sources},
        },
        "required": fields,
        "additionalProperties": False,
        "x-cortheon-effect-status-map": effect_status_map,
        "x-cortheon-effect-changes-hypothesis": effect_changes,
    }


def host_complete_tool(
    answer_schema: dict[str, Any] | None = None,
    reasoning_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    source = next(item for item in tool_definitions() if item["name"] == "cortheon_complete")
    parameters = json.loads(json.dumps(source["inputSchema"]))
    parameters["properties"].pop("session_id")
    parameters["required"].remove("session_id")
    parameters["properties"]["answer"] = (
        json.loads(json.dumps(answer_schema))
        if answer_schema is not None
        else {"oneOf": [{"type": "string"}, {"type": "object"}]}
    )
    parameters["properties"]["answer"]["description"] = (
        "The solved answer. When the evidence includes response_schema, pass a JSON "
        "object matching its exact fields. Do not encode that object as a string. If a "
        "hypothesis is uncertain, restate that hypothesis in a clause that explicitly says "
        "it remains uncertain."
    )
    hypothesis = parameters["properties"]["hypotheses"]["items"]["properties"]
    hypothesis["statement"]["description"] = (
        "One concrete competing interpretation or causal explanation grounded in the "
        "accepted evidence. Name the actual component or mechanism. Do not use generic "
        "phrases such as 'the leading condition' or 'an unstated boundary'."
    )
    hypothesis["falsification_test"]["description"] = (
        "The smallest observable check or clarification that would separate this exact "
        "hypothesis from its named rival."
    )
    if reasoning_binding is not None:
        parameters["properties"]["reasoning_binding"] = {
            "const": json.loads(json.dumps(reasoning_binding))
        }
        parameters["required"].append("reasoning_binding")
    return {
        "type": "function",
        "function": {
            "name": "host_complete",
            "description": (
                "Submit the solved model-authored answer, material claims, requested "
                "hypotheses, and exact evidence links to Cortheon. Never copy the response "
                "schema as the answer. The host injects only the active ephemeral session id."
            ),
            "parameters": parameters,
        },
    }


def reasoning_binding(result: dict[str, Any]) -> dict[str, str]:
    """Validate the content-free binding returned by the evaluator runtime."""

    value = result.get("reasoning_binding")
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "reasoning_binding_sha256"}
        or value.get("schema_version") != "1"
        or not all(
            isinstance(value.get(key), str)
            and len(value[key]) == 64
            and all(character in "0123456789abcdef" for character in value[key])
            for key in ("reasoning_binding_sha256",)
        )
    ):
        raise RuntimeError("runtime returned an invalid reasoning binding")
    return dict(value)


def host_reason_tool() -> dict[str, Any]:
    """Expose one bounded public hypothesis step before final completion."""

    hypothesis = {
        "type": "object",
        "properties": {
            "statement": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "falsification_test": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2_000,
            },
        },
        "required": ["statement", "falsification_test"],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": "host_reason",
            "description": (
                "Submit compact public competing hypotheses and an observable "
                "falsification test for each. The host records this reasoning step in "
                "Cortheon before it offers final completion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": hypothesis,
                        "minItems": 2,
                        "maxItems": 7,
                    }
                },
                "required": ["hypotheses"],
                "additionalProperties": False,
            },
        },
    }


def host_discrimination_tool(answer_schema: dict[str, Any]) -> dict[str, Any]:
    """Expose a closed public test-design step before final completion."""

    return {
        "type": "function",
        "function": {
            "name": "host_reason",
            "description": (
                "Choose the probe whose opposite outcomes distinguish the named hypotheses. "
                "Record the probe and the hypothesis supported by each outcome."
            ),
            "parameters": json.loads(json.dumps(answer_schema)),
        },
    }


def host_derivation_tool(answer_schema: dict[str, Any]) -> dict[str, Any]:
    """Expose a closed public source-chain derivation step."""

    return {
        "type": "function",
        "function": {
            "name": "host_reason",
            "description": (
                "Join every ordered source-bound premise. Return the first premise's "
                "subject, the requested terminal relation, the final premise's object, "
                "and the exact normalized premise path."
            ),
            "parameters": json.loads(json.dumps(answer_schema)),
        },
    }


def host_repair_tool() -> dict[str, Any]:
    """Expose one public draft-repair step after verification finds a gap."""

    return {
        "type": "function",
        "function": {
            "name": "host_reason",
            "description": (
                "Revise the previous draft to resolve Cortheon's stated verification gap. "
                "Return the complete replacement draft, not commentary about the gap."
            ),
            "parameters": {
                "type": "object",
                "properties": {"draft": {"type": "string", "minLength": 1, "maxLength": 20_000}},
                "required": ["draft"],
                "additionalProperties": False,
            },
        },
    }


def host_completion_repair_tool(
    previous: dict[str, Any],
    answer_schema: dict[str, Any] | None,
    reasoning_record: dict[str, str] | None,
) -> dict[str, Any]:
    """Offer one answer revision while freezing prior model-authored support fields."""

    tool = host_complete_tool(answer_schema, reasoning_record)
    parameters = tool["function"]["parameters"]
    for field in ("claims", "hypotheses", "completion_evidence_ids"):
        if field not in previous:
            raise ValueError(f"repair completion lacks prior {field}")
        parameters["properties"][field] = {"const": previous[field]}
    if reasoning_record is not None:
        parameters["properties"]["reasoning_binding"] = {"const": reasoning_record}
    tool["function"]["description"] = (
        "Revise only the solved answer to resolve Cortheon's stated wording gap. The "
        "host freezes the model's prior claims, hypotheses, statuses, and evidence links."
    )
    return tool


def host_revision_tool(answer_schema: dict[str, Any]) -> dict[str, Any]:
    """Expose a public source-comparison step before final completion."""

    properties = answer_schema.get("properties") if isinstance(answer_schema, dict) else None
    effect_status_map = (
        answer_schema.get("x-cortheon-effect-status-map")
        if isinstance(answer_schema, dict)
        else None
    )
    prior = properties.get("prior") if isinstance(properties, dict) else None
    revised = properties.get("revised") if isinstance(properties, dict) else None
    source = properties.get("decisive_source") if isinstance(properties, dict) else None
    if (
        not all(isinstance(item, dict) for item in (prior, revised, source))
        or not isinstance(effect_status_map, dict)
        or not effect_status_map
    ):
        raise RuntimeError("revision reasoning requires a closed public answer schema")

    return {
        "type": "function",
        "function": {
            "name": "host_reason",
            "description": (
                "Compare the source that established the prior with the decisive later "
                "source. State whether the decisive source supports or refutes the prior "
                "before constructing the final revision answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prior": json.loads(json.dumps(prior)),
                    "original_source": json.loads(json.dumps(source)),
                    "decisive_source": json.loads(json.dumps(source)),
                    "decisive_effect": {
                        "type": "string",
                        "enum": list(effect_status_map),
                    },
                    "revised": json.loads(json.dumps(revised)),
                },
                "required": [
                    "prior",
                    "original_source",
                    "decisive_source",
                    "decisive_effect",
                    "revised",
                ],
                "additionalProperties": False,
            },
        },
    }


def bind_tool_arguments(
    tool: dict[str, Any],
    arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    if arguments is None:
        return tool
    bound = json.loads(json.dumps(tool))
    properties = bound["function"]["parameters"]["properties"]
    if not set(arguments) <= set(properties):
        raise RuntimeError("runtime argument projection exceeded the tool schema")
    bound["function"]["parameters"]["properties"] = {
        key: {"const": value} for key, value in arguments.items()
    }
    bound["function"]["parameters"]["required"] = list(arguments)
    return bound
