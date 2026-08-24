"""Turn registration, automatic start-up, and whole-session teardown."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from cortheon.cognitive_hooks_core.responses import ResponseMixin
from cortheon.cognitive_hooks_core.state import (
    MAX_TURN_FAILURES_PER_HOST_SESSION,
    HookTurn,
    _bounded,
)
from cortheon.cognitive_repair import (
    protected_test_paths,
    protects_tests,
    requested_check_invocation,
    requested_test_invocation,
)


class RegistrationMixin(ResponseMixin):
    """Admit a host turn, optionally starting the runtime for it."""

    def register(
        self,
        host: str,
        host_session_id: str,
        turn_id: str,
        *,
        goal: str | None = None,
        effort: str = "quick",
        strictness: str = "standard",
        task_kind: str = "auto",
    ) -> dict[str, Any]:
        key, host_hash = self._identifiers(host, host_session_id, turn_id)
        normalized_goal = _bounded(goal, "goal", maximum=8_000) if goal is not None else None
        goal_hash = (
            hashlib.sha256(normalized_goal.casefold().encode()).hexdigest()
            if normalized_goal is not None
            else ""
        )
        with self._lock:
            self._purge_expired()
            if self._session_failures.get(host_hash, 0) >= MAX_TURN_FAILURES_PER_HOST_SESSION:
                self._metrics["hook_degraded_registrations"] += 1
                return {
                    "tracked": False,
                    "degraded": True,
                    "started": False,
                    "observed": False,
                    "certified": False,
                    "automatic": False,
                    "reason": (
                        "Cortheon degraded to passthrough for this host session after "
                        "repeated uncertified turns."
                    ),
                }
            state = self._turns.get(key)
            if state is None:
                prior = next(
                    (
                        (prior_key, prior_state)
                        for prior_key, prior_state in self._turns.items()
                        if prior_state.host_session_hash == host_hash
                        and prior_state.automatic
                        and not prior_state.certified
                    ),
                    None,
                )
                if prior is not None and (
                    goal_hash == "" or not prior[1].goal_hash or prior[1].goal_hash == goal_hash
                ):
                    # Same in-flight investigation continuing on a fresh turn:
                    # keep its evidence but grant a fresh continuation budget.
                    prior_key, state = prior
                    del self._turns[prior_key]
                    self._turns[key] = state
                    state.stop_continuations = 0
                    state.tool_denials = 0
                elif prior is not None:
                    # A different goal arrived; the stale investigation must not
                    # poison it, so discard instead of migrating.
                    prior_key, prior_state = prior
                    del self._turns[prior_key]
                    self._discard_runtime_session(prior_state)
                if state is None:
                    if len(self._turns) >= self.max_turns:
                        self._evict_oldest()
                    state = HookTurn(
                        host_session_hash=host_hash,
                        goal_hash=goal_hash,
                        updated_at=time.monotonic(),
                    )
                    self._turns[key] = state
                    self._metrics["hook_turns_registered"] += 1
            state.updated_at = time.monotonic()
            if (
                normalized_goal is not None
                and self.runtime is not None
                and state.cortheon_session_id is None
            ):
                started = self.runtime.start(
                    normalized_goal,
                    effort=effort,
                    task_kind=task_kind,
                    strictness=strictness,
                )
                state.started = True
                state.automatic = True
                state.goal_hash = goal_hash
                state.cortheon_session_id = str(started["session"]["session_id"])
                state.deliverable = str(started.get("session", {}).get("deliverable") or "")
                state.test_invocation = requested_test_invocation(normalized_goal)
                state.check_invocation = requested_check_invocation(normalized_goal)
                state.protected_test_paths = protected_test_paths(normalized_goal)
                state.protects_all_tests = protects_tests(normalized_goal)
                self._capture_next_action(state, started)
                self._metrics["hook_auto_started"] += 1
            return self._response(state)

    def end_session(self, host: str, host_session_id: str) -> dict[str, Any]:
        host_hash = self._host_hash(host, host_session_id)
        with self._lock:
            matching = [
                (key, state)
                for key, state in self._turns.items()
                if state.host_session_hash == host_hash
            ]
            for key, state in matching:
                self._discard_runtime_session(state)
                del self._turns[key]
            self._session_failures.pop(host_hash, None)
            return {"ok": True, "removed_turns": len(matching)}
