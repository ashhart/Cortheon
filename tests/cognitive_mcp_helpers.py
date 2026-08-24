"""Typed request helper shared by the split cognitive MCP tests.

``CortheonMcpServer.handle`` answers with ``None`` for a notification - a
JSON-RPC message carrying no ``id``. Every message a test sends here carries
one, so ``None`` is a real failure rather than something to index into
blindly, and saying so once keeps the response narrowed for the type checker
at every call site.
"""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_mcp import CortheonMcpServer


def request(server: CortheonMcpServer, message: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON-RPC request and assert the server answered it."""

    response = server.handle(message)
    assert response is not None, (
        f"handle() returned no response for {message.get('method')!r} id={message.get('id')!r}"
    )
    return response
