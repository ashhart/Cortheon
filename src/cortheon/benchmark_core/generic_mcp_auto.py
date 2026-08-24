"""Evaluator-owned execution of exact runtime-selected host actions."""

from __future__ import annotations

import json
from typing import Any


def execute_projected_action(host: Any, messages: list[dict[str, Any]]) -> bool:
    if getattr(host.model, "evaluator_executes_exact_tools", False) is not True:
        return False
    runtime = host.runtime
    if runtime is None or host.profile["config"]["intercepts_final"] is False:
        return False
    name = runtime.projected_host_tool()
    arguments = runtime.projected_arguments(name) if name is not None else None
    if name is None or arguments is None:
        return False
    call_id = f"runtime-{len(host.transcript.events)}"
    request = host.executor.ledger.request(call_id, name, arguments)
    host.transcript.record(
        "runtime_tool_decision",
        {
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "request_sha256": request.request_sha256,
        },
    )
    result, terminal = host._tool_call(call_id, name, arguments)
    if terminal:
        raise RuntimeError("runtime-selected host evidence became terminal")
    messages.append(
        {
            "role": "system",
            "content": json.dumps(
                {"cortheon_runtime_tool": name, "result": result}, separators=(",", ":")
            ),
        }
    )
    return True
