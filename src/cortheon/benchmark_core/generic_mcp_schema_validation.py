"""Closed model-tool catalogue validation for generic MCP transcripts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import bounded_identifier, canonical_json


def valid_tool_arguments(arguments: Any, tool: Any) -> bool:
    """Validate model arguments against the exact offered JSON Schema subset."""

    function = tool.get("function") if isinstance(tool, dict) else None
    schema = function.get("parameters") if isinstance(function, dict) else None
    return isinstance(arguments, dict) and _matches(arguments, schema, depth=0)


def _matches(value: Any, schema: Any, *, depth: int) -> bool:
    if not isinstance(schema, dict) or depth > 12:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if set(schema) == {"const"}:
        return True
    if "enum" in schema and (not isinstance(schema["enum"], list) or value not in schema["enum"]):
        return False
    alternatives = schema.get("oneOf")
    if alternatives is not None:
        return bool(
            isinstance(alternatives, list)
            and sum(_matches(value, item, depth=depth + 1) for item in alternatives) == 1
        )
    kind = schema.get("type")
    if kind == "string":
        return bool(
            isinstance(value, str)
            and len(value) >= schema.get("minLength", 0)
            and len(value) <= schema.get("maxLength", 200_000)
        )
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return type(value) is bool
    if kind == "array":
        return bool(
            isinstance(value, list)
            and len(value) >= schema.get("minItems", 0)
            and len(value) <= schema.get("maxItems", 128)
            and all(_matches(item, schema.get("items"), depth=depth + 1) for item in value)
        )
    if kind != "object" or not isinstance(value, dict):
        return False
    properties = schema.get("properties")
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False
    if not all(isinstance(item, str) for item in required) or not set(required) <= set(value):
        return False
    extras = set(value) - set(properties)
    additional = schema.get("additionalProperties", True)
    if extras and additional is False:
        return False
    if (
        extras
        and isinstance(additional, dict)
        and not all(_matches(value[key], additional, depth=depth + 1) for key in extras)
    ):
        return False
    return all(
        key not in value or _matches(value[key], child, depth=depth + 1)
        for key, child in properties.items()
    )


def valid_tool_catalogue(
    value: Any,
    names: list[Any],
    bounded_text: Callable[..., bool],
) -> bool:
    if not isinstance(value, list) or len(value) != len(names):
        return False
    observed: list[str] = []
    for tool in value:
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(tool, dict)
            or set(tool) != {"type", "function"}
            or tool.get("type") != "function"
            or not isinstance(function, dict)
            or set(function) != {"name", "description", "parameters"}
            or not bounded_identifier(function.get("name"))
            or not bounded_text(function.get("description"), 2_000)
            or not isinstance(function.get("parameters"), dict)
            or len(canonical_json(function["parameters"])) > 50_000
        ):
            return False
        observed.append(function["name"])
    return observed == names
