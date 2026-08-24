"""Host lifecycle entry points and the manual, model-driven enforcement path."""

from __future__ import annotations

import time
from typing import Any

from cortheon.cognitive_hooks_core.responses import ResponseMixin
from cortheon.cognitive_hooks_core.state import (
    MAX_STOP_CONTINUATIONS_PER_TURN,
    HookTurn,
    _bounded,
    cortheon_tool_phase,
)


class LifecycleMixin(ResponseMixin):
    """Route each host hook event to the automatic or manual enforcement path."""

    def pre_tool(
        self,
        host: str,
        host_session_id: str,
        turn_id: str,
        tool_name: str,
        *,
        tool_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key, _host_hash = self._identifiers(host, host_session_id, turn_id)
        normalized_tool = _bounded(tool_name, "tool_name")
        phase = cortheon_tool_phase(normalized_tool)
        with self._lock:
            self._purge_expired()
            state = self._turns.get(key)
            if state is None:
                return {"tracked": False, "allow": True}
            state.updated_at = time.monotonic()
            if state.certified or phase is not None:
                return {"tracked": True, "allow": True, **self._response(state)}
            if state.automatic:
                return self._automatic_pre_tool(
                    state,
                    normalized_tool,
                    tool_input=tool_input or {},
                )
            if not state.started:
                return self._nudge_start(state)
            return {"tracked": True, "allow": True, **self._response(state)}

    def post_tool(
        self,
        host: str,
        host_session_id: str,
        turn_id: str,
        tool_name: str,
        *,
        succeeded: bool,
        certified: bool = False,
        tool_output: str = "",
        tool_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key, _host_hash = self._identifiers(host, host_session_id, turn_id)
        normalized_tool = _bounded(tool_name, "tool_name")
        if not isinstance(tool_output, str):
            raise ValueError("tool_output must be a string")
        phase = cortheon_tool_phase(normalized_tool)
        with self._lock:
            self._purge_expired()
            state = self._turns.get(key)
            if state is None:
                return {"tracked": False}
            state.updated_at = time.monotonic()
            if state.automatic and phase is None:
                return self._automatic_post_tool(
                    state,
                    normalized_tool,
                    succeeded=succeeded,
                    tool_output=tool_output,
                    tool_metadata=tool_metadata or {},
                )
            if succeeded and phase == "start":
                state.started = True
            elif succeeded and phase == "observe" and state.started:
                state.observed = True
            elif (
                succeeded and certified and phase == "complete" and state.started and state.observed
            ):
                self._mark_certified(state)
            return {"tracked": True, **self._response(state)}

    def stop(
        self,
        host: str,
        host_session_id: str,
        turn_id: str,
        *,
        answer: str | None = None,
    ) -> dict[str, Any]:
        key, _host_hash = self._identifiers(host, host_session_id, turn_id)
        with self._lock:
            self._purge_expired()
            state = self._turns.get(key)
            if state is None:
                return {"tracked": False, "allow": True}
            if state.automatic:
                return self._automatic_stop(key, state, answer)
            if state.certified:
                del self._turns[key]
                return {"tracked": True, "allow": True, "certified": True}
            return self._manual_stop(key, state)

    def _manual_stop(self, key: str, state: HookTurn) -> dict[str, Any]:
        state.updated_at = time.monotonic()
        state.stop_continuations += 1
        self._metrics["hook_continuations"] += 1
        if state.stop_continuations > MAX_STOP_CONTINUATIONS_PER_TURN:
            return self._release_uncertified(key, state)
        if not state.started:
            reason = (
                "Completion is withheld: call `cortheon_start` with the user's "
                "exact goal and follow its `next_action`."
            )
        elif not state.observed:
            reason = (
                "Completion is withheld: obtain the requested host evidence and "
                "call `cortheon_observe` with its focused result."
            )
        else:
            reason = (
                "Completion is withheld: call `cortheon_complete`; if rejected, "
                "follow the returned `next_action`."
            )
        return {
            "tracked": True,
            "allow": False,
            "reason": reason,
            **self._response(state),
        }

    def _nudge_start(self, state: HookTurn) -> dict[str, Any]:
        """Let investigation tools through pre-start; enforcement waits at stop.

        Weak models doom-loop on pre-start denials (they retry the denied tool
        instead of reading the denial), so Cortheon steps back during
        investigation and stands in at completion time.
        """

        return {
            "tracked": True,
            "allow": True,
            "guidance": (
                "Cortheon is active for this turn. Investigate freely with host "
                "tools, then call `cortheon_start` with the user's exact goal "
                "before answering; completion is withheld until certified."
            ),
            **self._response(state),
        }
