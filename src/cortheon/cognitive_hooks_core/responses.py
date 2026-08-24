"""Response shaping, request scheduling, denial, and uncertified release."""

from __future__ import annotations

import copy
from typing import Any

from cortheon.cognitive_hooks_core.state import (
    MAX_TOOL_DENIALS_PER_TURN,
    UNCERTIFIED_RELEASE_CAVEAT,
    HookTurn,
    _bounded_cognition,
)
from cortheon.cognitive_hooks_core.tracker_base import TrackerBase
from cortheon.cognitive_repair import is_test_path


class ResponseMixin(TrackerBase):
    """Build every hook reply and own the requests the hook schedules itself."""

    @staticmethod
    def _protected_diff_paths(
        state: HookTurn,
        changed_paths: set[str],
    ) -> set[str]:
        return {
            path
            for path in changed_paths
            if path in state.protected_test_paths
            or (state.protects_all_tests and is_test_path(path))
        }

    @staticmethod
    def _clear_pending_request(state: HookTurn) -> None:
        state.pending_request = None
        state.pending_origin = None

    @staticmethod
    def _schedule_adapter_request(
        state: HookTurn,
        *,
        capability: str,
        query: str,
        success_condition: str,
        parameters: dict[str, Any],
    ) -> None:
        state.pending_request = {
            "request_id": f"hook_{capability}",
            "capability": capability,
            "query": query,
            "reason": "Close the next observable host-side patch-loop gate.",
            "success_condition": success_condition,
            "parameters": copy.deepcopy(parameters),
            "status": "pending",
        }
        state.pending_origin = "adapter"
        state.last_success_condition = success_condition[:1_500]
        state.last_next_action = {
            "type": "harness_tool",
            "instruction": (
                "Run this exact operation with the host. The Cortheon hook will "
                "capture and verify the result automatically."
            ),
            "request": copy.deepcopy(state.pending_request),
        }

    def _release_uncertified(self, key: str, state: HookTurn) -> dict[str, Any]:
        """End a stuck turn by releasing the answer with an explicit caveat.

        Repeated failures trip a per-session circuit breaker that degrades
        Cortheon to passthrough."""

        self._discard_runtime_session(state)
        del self._turns[key]
        self._record_turn_failure(state.host_session_hash)
        self._metrics["hook_uncertified_releases"] += 1
        return {
            "tracked": True,
            "allow": True,
            "terminal": True,
            "certified": False,
            "uncertified": True,
            "caveat": UNCERTIFIED_RELEASE_CAVEAT,
            "systemMessage": UNCERTIFIED_RELEASE_CAVEAT,
        }

    def _deny_requested_capability(
        self,
        state: HookTurn,
        capability: str,
    ) -> dict[str, Any]:
        if state.tool_denials >= MAX_TOOL_DENIALS_PER_TURN:
            return {
                "tracked": True,
                "allow": True,
                "degraded": True,
                **self._response(state),
            }
        state.tool_denials += 1
        self._metrics["hook_tools_denied"] += 1
        request = state.pending_request or {}
        query = str(request.get("query") or "")[:500]
        return {
            "tracked": True,
            "allow": False,
            "reason": (
                f"Cortheon is already active. Use a host {capability or 'evidence'} "
                f"tool for this request: {query}"
            ),
            **self._response(state),
        }

    def _capture_next_action(
        self,
        state: HookTurn,
        payload: dict[str, Any],
    ) -> None:
        cognition = payload.get("cognition")
        state.cognition = (
            _bounded_cognition(cognition) if isinstance(cognition, dict) else state.cognition
        )
        next_action = payload.get("next_action")
        state.last_next_action = (
            copy.deepcopy(next_action) if isinstance(next_action, dict) else None
        )
        request = (
            next_action.get("request")
            if isinstance(next_action, dict) and next_action.get("type") == "harness_tool"
            else None
        )
        state.pending_request = copy.deepcopy(request) if isinstance(request, dict) else None
        state.pending_origin = "runtime" if isinstance(request, dict) else None
        if isinstance(request, dict):
            state.last_success_condition = str(request.get("success_condition") or "")[:1_500]

    def _response(self, state: HookTurn) -> dict[str, Any]:
        response: dict[str, Any] = {
            **self._public_state(state),
            "automatic": state.automatic,
        }
        if state.cognition is not None:
            response["cognition"] = copy.deepcopy(state.cognition)
        if state.pending_request is not None:
            response["next_action"] = {
                "type": "harness_tool",
                "instruction": (
                    "Run this evidence request with the host's real tools. "
                    "The Codex hook will observe the result automatically."
                ),
                "request": copy.deepcopy(state.pending_request),
            }
        elif state.last_next_action is not None and not str(
            state.last_next_action.get("submit_via") or ""
        ).startswith("cortheon_"):
            response["next_action"] = copy.deepcopy(state.last_next_action)
        return response

    @staticmethod
    def _public_state(state: HookTurn) -> dict[str, bool]:
        return {
            "started": state.started,
            "observed": state.observed,
            "certified": state.certified,
        }
