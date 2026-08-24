"""Shared tracker state: construction, metrics, identity, and expiry."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from typing import Any

from cortheon.cognitive_hooks_core.state import HookTurn, _bounded


class TrackerBase:
    """State every hook mixin shares, plus the cross-module method contract."""

    def __init__(
        self,
        *,
        runtime: Any | None = None,
        max_turns: int = 512,
        ttl_seconds: float = 1_800.0,
    ) -> None:
        if not 1 <= max_turns <= 100_000:
            raise ValueError("max_turns must be between 1 and 100000")
        if not 1.0 <= ttl_seconds <= 86_400.0:
            raise ValueError("ttl_seconds must be between 1 and 86400")
        self.runtime = runtime
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._turns: dict[str, HookTurn] = {}
        self._session_failures: dict[str, int] = {}
        self._lock = threading.RLock()
        self._metrics = {
            "hook_turns_registered": 0,
            "hook_turns_certified": 0,
            "hook_tools_denied": 0,
            "hook_continuations": 0,
            "hook_turns_expired": 0,
            "hook_auto_started": 0,
            "hook_auto_observations": 0,
            "hook_auto_completed": 0,
            "hook_auto_withheld": 0,
            "hook_auto_repairs_derived": 0,
            "hook_auto_repair_candidates_advanced": 0,
            "hook_auto_patches_applied": 0,
            "hook_auto_edit_reconciliations": 0,
            "hook_auto_tests_passed": 0,
            "hook_auto_checks_scheduled": 0,
            "hook_auto_checks_passed": 0,
            "hook_protected_mutations_denied": 0,
            "hook_uncertified_releases": 0,
            "hook_degraded_registrations": 0,
        }

    @property
    def active_turns(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._turns)

    @property
    def metrics(self) -> dict[str, int]:
        with self._lock:
            self._purge_expired()
            return {**self._metrics, "active_hook_turns": len(self._turns)}

    def _record_turn_failure(self, host_hash: str) -> None:
        if len(self._session_failures) >= 1_024:
            self._session_failures.clear()
        self._session_failures[host_hash] = self._session_failures.get(host_hash, 0) + 1

    def _mark_certified(self, state: HookTurn) -> None:
        if not state.certified:
            state.certified = True
            self._metrics["hook_turns_certified"] += 1

    def _discard_runtime_session(self, state: HookTurn) -> None:
        if self.runtime is None or state.cortheon_session_id is None:
            return
        with contextlib.suppress(KeyError, RuntimeError, ValueError):
            self.runtime.finish(state.cortheon_session_id, mode="abandon")
        state.cortheon_session_id = None

    @classmethod
    def _identifiers(
        cls,
        host: str,
        host_session_id: str,
        turn_id: str,
    ) -> tuple[str, str]:
        normalized_host = _bounded(host, "host", maximum=64).lower()
        normalized_session = _bounded(
            host_session_id,
            "host_session_id",
            maximum=512,
        )
        normalized_turn = _bounded(turn_id, "turn_id", maximum=512)
        host_hash = cls._host_hash(normalized_host, normalized_session)
        key = hashlib.sha256(
            f"{normalized_host}\0{normalized_session}\0{normalized_turn}".encode()
        ).hexdigest()
        return key, host_hash

    @staticmethod
    def _host_hash(host: str, host_session_id: str) -> str:
        normalized_host = _bounded(host, "host", maximum=64).lower()
        normalized_session = _bounded(
            host_session_id,
            "host_session_id",
            maximum=512,
        )
        return hashlib.sha256(f"{normalized_host}\0{normalized_session}".encode()).hexdigest()

    def _purge_expired(self) -> None:
        threshold = time.monotonic() - self.ttl_seconds
        expired = [key for key, state in self._turns.items() if state.updated_at < threshold]
        for key in expired:
            self._discard_runtime_session(self._turns[key])
            del self._turns[key]
        self._metrics["hook_turns_expired"] += len(expired)

    def _evict_oldest(self) -> None:
        if not self._turns:
            return
        oldest = min(self._turns, key=lambda key: self._turns[key].updated_at)
        self._discard_runtime_session(self._turns[oldest])
        del self._turns[oldest]
        self._metrics["hook_turns_expired"] += 1

    # Contract stubs: the lifecycle mixin dispatches into the automatic and
    # patch-loop mixins, which are composed onto the concrete tracker.

    def _automatic_pre_tool(
        self,
        state: HookTurn,
        tool_name: str,
        *,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _automatic_post_tool(
        self,
        state: HookTurn,
        tool_name: str,
        *,
        succeeded: bool,
        tool_output: str,
        tool_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _automatic_stop(
        self,
        key: str,
        state: HookTurn,
        answer: str | None,
    ) -> dict[str, Any]:
        raise NotImplementedError
