"""The facade patch seam the pre-split module had, proved across the split.

Monkeypatching ``cortheon.cognitive_mcp`` used to change what the running
code saw, because every lookup the module made resolved through its own
globals. Each test here patches only the facade, proves the implementation
module that actually performs the lookup used the replacement, and proves
that assigning the original back leaves every binding and the observable
behavior exactly as they were before the patch.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from cognitive_mcp_helpers import request

import cortheon.cognitive_mcp as mcp
from cortheon.cognitive_mcp_core import arguments, protocol, server, stdio, tools

LIST_TOOLS = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def _call(name: str, call_id: int, **payload: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": payload},
    }


def _started(server_object: mcp.CortheonMcpServer, call_id: int) -> dict[str, Any]:
    response = request(server_object, _call("cortheon_start", call_id, goal="Does it import json?"))
    return response["result"]["structuredContent"]


def test_tool_definitions_patch_reaches_server_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    original = mcp.tool_definitions
    sentinel = [{"name": "sentinel"}]
    before = request(mcp.CortheonMcpServer(), LIST_TOOLS)["result"]["tools"]

    monkeypatch.setattr(mcp, "tool_definitions", lambda **_: sentinel)
    assert server.tool_definitions is not original, "server.handle must see the patch"
    # stdio resolves no such name, so the bridge must not invent one there;
    # getattr states that dynamically without asserting a static attribute.
    assert getattr(stdio, "tool_definitions", None) is None
    assert request(mcp.CortheonMcpServer(), LIST_TOOLS)["result"]["tools"] is sentinel

    monkeypatch.undo()
    assert mcp.tool_definitions is original
    assert server.tool_definitions is original
    assert tools.tool_definitions is original
    assert request(mcp.CortheonMcpServer(), LIST_TOOLS)["result"]["tools"] == before


def test_observation_certifier_patch_reaches_the_tools_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = mcp._observations_with_host_receipts
    seen: list[dict[str, Any]] = []

    def recording(raw: dict[str, Any]) -> list[dict[str, Any]]:
        seen.append(raw)
        return [{"kind": "code", "content": "[patched] import json\n"}]

    server_object = mcp.CortheonMcpServer(mcp.CognitiveRuntime())
    started = _started(server_object, 10)
    monkeypatch.setattr(mcp, "_observations_with_host_receipts", recording)
    assert server._observations_with_host_receipts is recording

    observed = request(
        server_object,
        _call(
            "cortheon_observe",
            11,
            session_id=started["session"]["session_id"],
            request_id=started["next_action"]["request"]["request_id"],
            observations=[{"kind": "code", "content": "original"}],
        ),
    )["result"]["structuredContent"]
    assert seen and seen[0]["observations"] == [{"kind": "code", "content": "original"}]
    assert observed["accepted_evidence_ids"] == ["ev1"]

    monkeypatch.undo()
    assert mcp._observations_with_host_receipts is original
    assert server._observations_with_host_receipts is original
    assert arguments._observations_with_host_receipts is original
    # The real certifier is back on the live path: it prefixes host receipts.
    certified = original(
        {
            "observations": [
                {
                    "kind": "code",
                    "content": "import json\n",
                    "host_receipt": {"tool": "grep", "outcome": "match", "args": {"p": "json"}},
                }
            ]
        }
    )
    assert certified[0]["content"].startswith(mcp.HOST_EVIDENCE_PREFIX)


def test_write_patch_reaches_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    original = mcp._write
    written: list[dict[str, Any]] = []
    monkeypatch.setattr(mcp, "_write", lambda _stdout, message: written.append(message))
    assert stdio._write is not original

    incoming = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}}) + "\n"
    mcp.serve(mcp.CortheonMcpServer(mcp.CognitiveRuntime()), io.StringIO(incoming), io.StringIO())
    assert [message["id"] for message in written] == [7]

    monkeypatch.undo()
    assert mcp._write is original and stdio._write is original
    outgoing = io.StringIO()
    mcp.serve(mcp.CortheonMcpServer(mcp.CognitiveRuntime()), io.StringIO(incoming), outgoing)
    assert json.loads(outgoing.getvalue())["id"] == 7


def test_serve_and_runtime_patches_reach_main(monkeypatch: pytest.MonkeyPatch) -> None:
    original_serve, original_runtime = mcp.serve, mcp.CognitiveRuntime
    served: list[mcp.CortheonMcpServer] = []
    built: list[dict[str, Any]] = []

    def fake_runtime(**kwargs: Any) -> Any:
        built.append(kwargs)
        return original_runtime(**kwargs)

    monkeypatch.setattr(mcp, "serve", served.append)
    monkeypatch.setattr(mcp, "CognitiveRuntime", fake_runtime)
    assert stdio.serve is not original_serve and stdio.CognitiveRuntime is fake_runtime

    mcp.main(["--max-sessions", "7", "--advanced"])
    assert built == [{"max_sessions": 7, "ttl_seconds": 1_800.0}]
    assert len(served) == 1 and served[0].advanced is True

    monkeypatch.undo()
    assert mcp.serve is original_serve and stdio.serve is original_serve
    assert mcp.CognitiveRuntime is original_runtime
    assert stdio.CognitiveRuntime is original_runtime
    assert server.CognitiveRuntime is original_runtime


def test_protocol_helper_patch_reaches_handle_and_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    original = mcp._error
    unknown = {"jsonrpc": "2.0", "id": 4, "method": "nope", "params": {}}
    before = request(mcp.CortheonMcpServer(), unknown)

    monkeypatch.setattr(
        mcp, "_error", lambda request_id, code, message: {"patched": [request_id, code, message]}
    )
    assert server._error is not original and stdio._error is not original
    assert request(mcp.CortheonMcpServer(), unknown)["patched"][:2] == [
        4,
        mcp.JSONRPC_METHOD_NOT_FOUND,
    ]

    monkeypatch.undo()
    assert mcp._error is original
    assert server._error is original and stdio._error is original
    assert protocol._error is original
    assert request(mcp.CortheonMcpServer(), unknown) == before


def test_a_module_that_rebound_the_name_itself_keeps_its_own_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity, not the name alone, decides where a facade patch lands."""

    def independent(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"independent": [request_id, code]}

    monkeypatch.setattr(stdio, "_error", independent)
    monkeypatch.setattr(mcp, "_error", lambda request_id, code, message: {"facade": request_id})

    assert stdio._error is independent, "an independent binding must not be overwritten"
    assert server._error is mcp._error, "a binding still holding the old object must follow"

    monkeypatch.undo()
    assert stdio._error is protocol._error and server._error is protocol._error


def test_a_name_the_facade_never_had_is_added_without_touching_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp, "brand_new_name", object(), raising=False)

    assert not hasattr(server, "brand_new_name")
    assert not hasattr(stdio, "brand_new_name")
