"""Bounded validation and coercion of tools/call arguments."""

from __future__ import annotations

import json
from typing import Any

from cortheon.cognitive_mcp_core.protocol import (
    HOST_EVIDENCE_PREFIX,
    HOST_RECEIPT_OUTCOMES,
)


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string_list(arguments: dict[str, Any], key: str) -> list[str]:
    value = _coerce_json_array(arguments.get(key), key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return value


def _required_string_list(arguments: dict[str, Any], key: str) -> list[str]:
    value = _coerce_json_array(arguments.get(key), key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a non-empty array of strings")
    return value


def _required_object_list(arguments: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = _coerce_json_array(arguments.get(key), key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty array of objects")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must contain only objects")
    return value


def _optional_object_list(arguments: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = _coerce_json_array(arguments.get(key), key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be an array of objects")
    return value


def _coerce_json_array(value: Any, key: str) -> Any:
    """Repair the common small-model error of JSON-encoding an array twice."""

    if not isinstance(value, str):
        return value
    if len(value) > 100_000:
        raise ValueError(f"{key} JSON string exceeds the 100000-character limit")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded


def _observations_with_host_receipts(
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for raw in _required_object_list(arguments, "observations"):
        observation = dict(raw)
        receipt = observation.pop("host_receipt", None)
        if receipt is None:
            observations.append(observation)
            continue
        if not isinstance(receipt, dict):
            raise ValueError("observation.host_receipt must be an object")
        tool = receipt.get("tool")
        executor = receipt.get("executor")
        outcome = receipt.get("outcome")
        receipt_arguments = receipt.get("args")
        if (
            not isinstance(tool, str)
            or not tool
            or len(tool) > 64
            or any(not (character.isalnum() or character in "_.:-") for character in tool)
        ):
            raise ValueError("observation.host_receipt.tool must be a bounded host tool name")
        if outcome not in HOST_RECEIPT_OUTCOMES:
            raise ValueError(
                "observation.host_receipt.outcome must be one of: "
                + ", ".join(sorted(HOST_RECEIPT_OUTCOMES))
            )
        if not isinstance(receipt_arguments, dict):
            raise ValueError("observation.host_receipt.args must be an object")
        if executor is not None and (
            not isinstance(executor, str)
            or not executor
            or len(executor) > 64
            or any(not (character.isalnum() or character in "_.:-") for character in executor)
        ):
            raise ValueError("observation.host_receipt.executor must be a bounded host tool name")
        try:
            encoded_arguments = json.dumps(
                receipt_arguments,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("observation.host_receipt.args must be JSON serializable") from exc
        if len(encoded_arguments) > 4_000:
            raise ValueError("observation.host_receipt.args exceeds the 4000-character limit")
        content = observation.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("observation.content must be a non-empty string")
        encoded_receipt = json.dumps(
            {
                "tool": tool,
                **({"executor": executor} if executor is not None else {}),
                "outcome": outcome,
                "args": receipt_arguments,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        if not content.startswith(HOST_EVIDENCE_PREFIX):
            observation["content"] = f"{HOST_EVIDENCE_PREFIX}{encoded_receipt}\n{content}"
        observation.setdefault("source", f"mcp-host:{tool}")
        observations.append(observation)
    return observations
