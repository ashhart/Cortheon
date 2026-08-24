"""JSON-RPC constants and the shared response envelopes."""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = {"2025-06-18", "2025-03-26", "2024-11-05"}

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603
MAX_MESSAGE_CHARS = 1_000_000
HOST_EVIDENCE_PREFIX = "[CORTHEON_HOST_EVIDENCE] "
HOST_RECEIPT_OUTCOMES = {
    "result",
    "match",
    "no_match",
    "changed",
    "passed",
    "failed",
    "error",
}


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, separators=(",", ":"), sort_keys=True),
            }
        ],
        "structuredContent": payload,
    }
