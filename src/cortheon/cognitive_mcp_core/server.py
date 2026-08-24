"""MCP request dispatch and the cognitive lifecycle handlers."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from cortheon import __version__
from cortheon.cognitive_mcp_core.arguments import (
    _observations_with_host_receipts,
    _optional_object_list,
    _optional_string,
    _optional_string_list,
    _required_object_list,
    _required_string,
    _required_string_list,
)
from cortheon.cognitive_mcp_core.protocol import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOLS,
    _error,
    _result,
    _tool_result,
)
from cortheon.cognitive_mcp_core.tools import tool_definitions
from cortheon.cognitive_protocol import protocol_capabilities
from cortheon.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeError


class CortheonMcpServer:
    """MCP adapter that exposes cognition but no filesystem or execution tools."""

    def __init__(
        self,
        runtime: CognitiveRuntime | None = None,
        *,
        advanced: bool = False,
        evaluation_profile: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime or CognitiveRuntime(require_host_receipts=True)
        self.advanced = advanced
        self.evaluation_profile = evaluation_profile

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return _error(
                None,
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC message must be an object.",
            )
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if params is None:
            params = {}
        if not isinstance(method, str):
            return _error(request_id, JSONRPC_INVALID_REQUEST, "Missing JSON-RPC method.")
        if "id" not in message:
            return None
        if not isinstance(params, dict):
            return _error(
                request_id,
                JSONRPC_INVALID_PARAMS,
                "JSON-RPC params must be an object.",
            )
        try:
            if method == "initialize":
                return _result(request_id, self._initialize(params))
            if method == "ping":
                return _result(request_id, {})
            if method == "tools/list":
                return _result(
                    request_id,
                    {"tools": tool_definitions(advanced=self.advanced)},
                )
            if method == "tools/call":
                return _result(request_id, self._call_tool(params))
        except (ValueError, CognitiveRuntimeError) as exc:
            return _error(request_id, JSONRPC_INVALID_PARAMS, str(exc))
        except Exception as exc:  # pragma: no cover - defensive stdio boundary.
            return _error(
                request_id,
                JSONRPC_INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}",
            )
        if request_id is None:
            return None
        return _error(
            request_id,
            JSONRPC_METHOD_NOT_FOUND,
            f"Unsupported method: {method}",
        )

    @staticmethod
    def _initialize(params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        protocol = requested if requested in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
        return {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "cortheon",
                "title": "Cortheon Cognitive Runtime",
                "version": __version__,
            },
            "cortheonProtocol": protocol_capabilities(),
            "instructions": (
                "Cortheon is an ephemeral test-time reasoning substrate. Start a bounded "
                "investigation with cortheon_start, let Cortheon request evidence, use the "
                "host's own tools to obtain it, and return focused results with "
                "cortheon_observe. Investigate incrementally: satisfy each request within "
                "its tool_call_budget and observe before reading more; never batch a "
                "whole exploration before the first observation. For most tasks, then "
                "call cortheon_complete once with "
                "public hypotheses, evidence-linked claims, and the answer. The default "
                "surface intentionally exposes only start, observe, complete, and abandon; "
                "operators can restart with --advanced for the low-level protocol. Never "
                "fabricate an observation or send whole files. Cortheon stores no project "
                "files, does not execute tools, and discards all task state on successful "
                "completion, abandonment, or expiry. For web evidence, provide the exact "
                "URL, a timezone-aware retrieved_at timestamp, published_at when known, "
                "and the purpose returned in the evidence request. For code, document, "
                "command, diff, and test evidence, include host_receipt with the exact host "
                "operation, its actual arguments and outcome, plus the optional executor "
                "name when the harness uses a generic shell tool. If an accepted "
                "observation proves wrong or mis-linked, call cortheon_retract with its "
                "ev* ids instead of abandoning the investigation. If conversation "
                "context was compacted or lost, call cortheon_resume to recover active "
                "investigations and continue instead of asking the user to restate "
                "the goal. Cite only returned ev* evidence ids; request ids are "
                "never evidence ids."
            ),
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            raise ValueError("tools/call requires a string tool name")
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")

        if name == "cortheon_start":
            payload = self.runtime.start(
                _required_string(arguments, "goal"),
                constraints=_optional_string_list(arguments, "constraints"),
                effort=_optional_string(arguments, "effort") or "quick",
                task_kind=_optional_string(arguments, "task_kind") or "auto",
                strictness=_optional_string(arguments, "strictness") or "standard",
                evaluation_profile=self.evaluation_profile,
            )
        elif name == "cortheon_step":
            payload = self.runtime.step(
                _required_string(arguments, "session_id"),
                hypotheses=_optional_object_list(arguments, "hypotheses"),
                hypothesis_updates=_optional_object_list(
                    arguments,
                    "hypothesis_updates",
                ),
                open_questions=_optional_string_list(arguments, "open_questions"),
                draft=_optional_string(arguments, "draft"),
            )
        elif name == "cortheon_observe":
            session_id = _required_string(arguments, "session_id")
            try:
                request_id = _required_string(arguments, "request_id")
                observations = _observations_with_host_receipts(arguments)
            except ValueError as exc:
                raise self._malformed_observe(session_id, arguments, exc) from exc
            payload = self.runtime.observe(
                session_id,
                observations,
                request_id=request_id,
            )
        elif name == "cortheon_resume":
            limit = arguments.get("limit", 3)
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            payload = self.runtime.describe_sessions(limit=limit)
        elif name == "cortheon_retract":
            payload = self.runtime.retract(
                _required_string(arguments, "session_id"),
                _required_string_list(arguments, "evidence_ids"),
                reason=_optional_string(arguments, "reason") or "",
            )
        elif name == "cortheon_challenge":
            payload = self.runtime.challenge(
                _required_string(arguments, "session_id"),
                draft=_required_string(arguments, "draft"),
                claims=_required_object_list(arguments, "claims"),
            )
        elif name == "cortheon_verify":
            payload = self.runtime.verify(
                _required_string(arguments, "session_id"),
                answer=_required_string(arguments, "answer"),
                claims=_required_object_list(arguments, "claims"),
                completion_evidence_ids=_optional_string_list(
                    arguments,
                    "completion_evidence_ids",
                ),
            )
        elif name == "cortheon_complete":
            payload = self.runtime.complete(
                _required_string(arguments, "session_id"),
                answer=_required_string(arguments, "answer"),
                claims=_required_object_list(arguments, "claims"),
                hypotheses=_required_object_list(arguments, "hypotheses"),
                completion_evidence_ids=_required_string_list(
                    arguments,
                    "completion_evidence_ids",
                ),
            )
        elif name == "cortheon_abandon":
            payload = self.runtime.finish(
                _required_string(arguments, "session_id"),
                mode="abandon",
            )
        elif name == "cortheon_finish":
            payload = self.runtime.finish(
                _required_string(arguments, "session_id"),
                mode=_optional_string(arguments, "mode") or "complete",
                answer=_optional_string(arguments, "answer"),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        if not self.advanced:
            payload = _compact_payload(payload)
        return _tool_result(payload)

    def _malformed_observe(
        self,
        session_id: str,
        arguments: dict[str, Any],
        error: ValueError,
    ) -> ValueError:
        """Turn an observe shape error into a copyable example that still heals.

        Counted here against the pending request's attempt ladder, or the
        model can loop at the tool boundary forever.
        """

        healed: dict[str, Any] = {}
        raw_request_id = arguments.get("request_id")
        with contextlib.suppress(ValueError, CognitiveRuntimeError):
            healed = self.runtime.note_failed_submission(
                session_id,
                request_id=raw_request_id if isinstance(raw_request_id, str) else None,
            )
        example = json.dumps(
            {
                "session_id": session_id,
                "request_id": healed.get("request_id") or "<request_id from next_action>",
                "observations": [
                    {
                        "kind": "code",
                        "content": "<focused excerpt from the tool you actually ran>",
                        "host_receipt": {
                            "tool": "grep",
                            "outcome": "match",
                            "args": {"pattern": "<pattern>", "path": "src/module.py"},
                        },
                    }
                ],
            },
            separators=(",", ":"),
        )
        message = f"{error}. Correct example call: {example}"
        if healed.get("waived"):
            message += (
                " That evidence request has now been waived after repeated failed "
                "submissions; call cortheon_complete with the narrowest answer the "
                "accepted evidence supports."
            )
        return ValueError(message)


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Never direct the compact surface toward a hidden low-level tool."""

    action = payload.get("next_action")
    if not isinstance(action, dict):
        return payload
    submit_via = action.get("submit_via")
    if submit_via in {
        "cortheon_observe",
        "cortheon_complete",
        "cortheon_retract",
        "cortheon_abandon",
    }:
        return payload

    original_instruction = action.get("instruction")
    suffix = (
        f" Address this gate: {original_instruction}"
        if isinstance(original_instruction, str) and original_instruction.strip()
        else ""
    )
    payload["next_action"] = {
        "type": "complete",
        "instruction": (
            "Call cortheon_complete now with the narrowest answer supported by the live "
            "evidence. Use only returned ev* ids for every hypothesis, claim, and "
            "completion_evidence_ids entry." + suffix
        ),
        "required_fields": [
            "hypotheses",
            "claims",
            "answer",
            "completion_evidence_ids",
        ],
        "submit_via": "cortheon_complete",
    }
    payload["guidance"] = (
        "The compact surface performs reasoning, challenge, verification, completion, "
        "and erasure transactionally through cortheon_complete. Do not call hidden "
        "low-level protocol tools."
    )
    return payload
