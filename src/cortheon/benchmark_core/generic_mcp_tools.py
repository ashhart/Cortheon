"""Closed tool catalogue and at-most-once receipt binding for generic MCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import bounded_identifier, payload_sha256

HOST_TOOL_NAMES = frozenset(
    {
        "host_search",
        "host_read",
        "host_read_many",
        "host_diff",
        "host_test",
        "host_web_search",
        "host_web_fetch",
    }
)
BRIDGE_TOOL_NAMES = frozenset({"host_complete", "host_reason"})


class ToolBudgetExhausted(RuntimeError):
    """The evaluator admitted every configured host-tool call."""


def registrable_tool_name(name: Any) -> bool:
    return bool(
        bounded_identifier(name)
        and (
            name in HOST_TOOL_NAMES
            or name in BRIDGE_TOOL_NAMES
            or str(name).startswith("cortheon_")
        )
    )


@dataclass(frozen=True, slots=True)
class ToolRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]
    request_sha256: str

    @classmethod
    def create(cls, call_id: str, name: str, arguments: dict[str, Any]) -> ToolRequest:
        if not isinstance(call_id, str) or not 0 < len(call_id) <= 128:
            raise ValueError("tool call id must be bounded")
        if not registrable_tool_name(name):
            raise ValueError(f"tool is outside the closed catalogue: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        return cls(call_id, name, arguments, payload_sha256([name, arguments]))


@dataclass(frozen=True, slots=True)
class ToolExecution:
    request: ToolRequest
    status: str
    content: str
    receipt: dict[str, Any]

    @property
    def result_sha256(self) -> str:
        return payload_sha256([self.status, self.content, self.receipt])


class ToolLedger:
    """Bind results to wrapper-created calls; never ingest model receipts."""

    def __init__(self, *, maximum_calls: int) -> None:
        if type(maximum_calls) is not int or not 1 <= maximum_calls <= 128:
            raise ValueError("maximum_calls must be from 1 to 128")
        self.maximum_calls = maximum_calls
        self._requests: dict[str, ToolRequest] = {}
        self._results: dict[str, ToolExecution] = {}

    def request(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolRequest:
        proposed = ToolRequest.create(call_id, name, arguments)
        existing = self._requests.get(call_id)
        if existing is not None:
            if existing != proposed:
                raise ValueError("tool call id was reused with different arguments")
            return existing
        if len(self._requests) >= self.maximum_calls:
            raise ToolBudgetExhausted("generic MCP host tool budget exhausted")
        self._requests[call_id] = proposed
        return proposed

    def record(
        self,
        request: ToolRequest,
        *,
        status: str,
        content: str,
        receipt: dict[str, Any],
    ) -> ToolExecution:
        if self._requests.get(request.call_id) != request:
            raise ValueError("tool result has no matching wrapper request")
        if request.call_id in self._results:
            raise ValueError("tool result was submitted twice")
        if status not in {"result", "match", "no_match", "changed", "passed", "failed", "error"}:
            raise ValueError("tool result status is invalid")
        if not isinstance(content, str) or len(content) > 40_000:
            raise ValueError("tool result content must be bounded")
        expected = {
            "tool": request.name.removeprefix("host_"),
            "executor": "generic_mcp_wrapper",
            "outcome": status,
            "args": request.arguments,
        }
        if receipt != expected:
            raise ValueError("tool receipt was not created from the wrapper request")
        result = ToolExecution(request, status, content, dict(receipt))
        self._results[request.call_id] = result
        return result

    def cached(self, call_id: str) -> ToolExecution | None:
        return self._results.get(call_id)


def host_tool_definitions(*, web_enabled: bool) -> list[dict[str, Any]]:
    string = {"type": "string"}
    tools = [
        ("host_search", {"pattern": string, "path": string}),
        ("host_read", {"path": string, "start_line": {"type": "integer"}}),
        ("host_read_many", {"paths": {"type": "array", "items": string}}),
        ("host_diff", {"paths": {"type": "array", "items": string}}),
        ("host_test", {"test_id": string}),
    ]
    if web_enabled:
        tools.extend(
            [
                ("host_web_search", {"query": string}),
                ("host_web_fetch", {"url": string}),
            ]
        )
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Evaluator-owned bounded host capability.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        }
        for name, properties in tools
    ]
