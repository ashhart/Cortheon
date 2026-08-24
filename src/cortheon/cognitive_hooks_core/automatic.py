"""The automatic path: rewrite host calls before they run, and gate the stop."""

from __future__ import annotations

import copy
import time
from typing import Any

from cortheon.cognitive_hooks_core.host_tools import (
    _attempts_protected_mutation,
    _is_apply_patch_tool,
    _is_shell_tool,
    _safe_command,
)
from cortheon.cognitive_hooks_core.responses import ResponseMixin
from cortheon.cognitive_hooks_core.state import (
    MAX_PATCH_STOP_CONTINUATIONS_PER_TURN,
    MAX_STOP_CONTINUATIONS_PER_TURN,
    HookTurn,
    _continuation_reason,
)


class AutomaticMixin(ResponseMixin):
    """Drive an automatic turn: admit, redirect, or deny each host action."""

    def _automatic_pre_tool(
        self,
        state: HookTurn,
        tool_name: str,
        *,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        if _attempts_protected_mutation(
            tool_name,
            tool_input,
            protected_paths=state.protected_test_paths,
            protects_all_tests=state.protects_all_tests,
        ):
            state.tool_denials += 1
            self._metrics["hook_tools_denied"] += 1
            self._metrics["hook_protected_mutations_denied"] += 1
            return {
                "tracked": True,
                "allow": False,
                "reason": (
                    "Cortheon blocked a mutation to a user-protected test file. "
                    "Change only the implementation and preserve the test."
                ),
                **self._response(state),
            }
        if (
            state.awaiting_host_result
            and state.pending_origin == "adapter"
            and str((state.pending_request or {}).get("capability") or "") == "edit"
            and state.repair_plan is not None
        ):
            state.awaiting_host_result = False
            self._metrics["hook_auto_edit_reconciliations"] += 1
            self._schedule_adapter_request(
                state,
                capability="diff",
                query=(
                    "The host edit path did not emit a usable result event. "
                    f"Inspect the live Git diff for {state.repair_plan.path} "
                    "instead of assuming the edit succeeded."
                ),
                success_condition=(
                    "Return a non-empty Git diff proving the implementation "
                    "change and showing that protected tests were untouched."
                ),
                parameters={"path": state.repair_plan.path},
            )
        request = state.pending_request
        if request is None:
            state.awaiting_host_result = False
            state.pending_host_command = ""
            state.pending_host_input = {}
            return {"tracked": True, "allow": True, **self._response(state)}
        capability = str(request.get("capability") or "")
        if capability == "edit":
            return self._automatic_edit_pre_tool(state, tool_name)
        # The strict phase is the adapter-driven patch loop (diff/test/edit).
        # Runtime-issued investigation requests stay lenient: the model may run
        # its own commands and the hook harvests the results as evidence.
        strict = state.pending_origin == "adapter"
        if _is_shell_tool(tool_name):
            if strict:
                command = _safe_command(request)
                if command is not None:
                    state.awaiting_host_result = True
                    state.pending_host_command = ""
                    state.pending_host_input = {}
                    response = {"tracked": True, "allow": True, **self._response(state)}
                    response["updated_input"] = {"command": command}
                    return response
                state.awaiting_host_result = False
                state.pending_host_input = {}
                return self._deny_requested_capability(state, capability)
            state.awaiting_host_result = True
            state.pending_host_command = str(
                tool_input.get("command") or tool_input.get("cmd") or ""
            )[:500]
            state.pending_host_input = copy.deepcopy(tool_input)
            return {"tracked": True, "allow": True, **self._response(state)}
        if strict:
            state.pending_host_input = {}
            return self._deny_requested_capability(state, capability)
        # Purpose-built read/search tools are model-owned too. Preserve their
        # structured input so the receipt records what actually ran.
        state.awaiting_host_result = True
        state.pending_host_command = ""
        state.pending_host_input = copy.deepcopy(tool_input)
        return {"tracked": True, "allow": True, **self._response(state)}

    def _automatic_edit_pre_tool(
        self,
        state: HookTurn,
        tool_name: str,
    ) -> dict[str, Any]:
        plan = state.repair_plan
        if plan is None:
            return self._deny_requested_capability(state, "edit")
        patch = plan.patch()
        state.awaiting_host_result = True
        state.pending_host_input = {}
        response = {"tracked": True, "allow": True, **self._response(state)}
        if _is_apply_patch_tool(tool_name):
            response["updated_input"] = {"patch": patch}
            return response
        if _is_shell_tool(tool_name):
            response["updated_input"] = {
                "command": (f"apply_patch <<'CORTHEON_PATCH'\n{patch}CORTHEON_PATCH")
            }
            return response
        state.awaiting_host_result = False
        return self._deny_requested_capability(state, "edit")

    def _automatic_stop(
        self,
        key: str,
        state: HookTurn,
        answer: str | None,
    ) -> dict[str, Any]:
        state.updated_at = time.monotonic()
        if (
            state.awaiting_host_result
            and state.pending_origin == "adapter"
            and str((state.pending_request or {}).get("capability") or "") == "edit"
            and state.repair_plan is not None
        ):
            state.awaiting_host_result = False
            self._metrics["hook_auto_edit_reconciliations"] += 1
            self._schedule_adapter_request(
                state,
                capability="diff",
                query=(
                    "The host edit path did not emit a usable result event. "
                    f"Inspect the live Git diff for {state.repair_plan.path}."
                ),
                success_condition=(
                    "Return a non-empty Git diff proving the implementation "
                    "change and showing that protected tests were untouched."
                ),
                parameters={"path": state.repair_plan.path},
            )
        normalized_answer = (answer or "").strip()
        runtime = self.runtime
        if (
            runtime is not None
            and state.cortheon_session_id is not None
            and state.evidence_ids
            and normalized_answer
            and state.pending_request is None
        ):
            completion_answer = normalized_answer[:4_000]
            hypothesis = completion_answer[:1_500]
            falsification = (
                state.last_success_condition or "Re-run the requested host evidence operation."
            )[:1_500]
            try:
                completed = runtime.complete(
                    state.cortheon_session_id,
                    answer=completion_answer,
                    claims=[
                        {
                            "claim": completion_answer,
                            "evidence_ids": list(state.evidence_ids),
                        }
                    ],
                    hypotheses=[
                        {
                            "statement": hypothesis,
                            "falsification_test": falsification,
                            "status": "supported",
                            "evidence_ids": list(state.evidence_ids),
                        }
                    ],
                    completion_evidence_ids=list(state.evidence_ids),
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                completed = {
                    "status": "needs_evidence",
                    "next_action": state.pending_request,
                    "error": str(exc)[:500],
                }
            if completed.get("status") == "complete":
                self._mark_certified(state)
                self._metrics["hook_auto_completed"] += 1
                del self._turns[key]
                return {
                    "tracked": True,
                    "allow": True,
                    "certified": True,
                    "automatic": True,
                }
            self._capture_next_action(state, completed)
            self._metrics["hook_auto_withheld"] += 1

        state.stop_continuations += 1
        self._metrics["hook_continuations"] += 1
        continuation_limit = (
            MAX_PATCH_STOP_CONTINUATIONS_PER_TURN
            if state.deliverable == "code_change"
            else MAX_STOP_CONTINUATIONS_PER_TURN
        )
        if state.stop_continuations > continuation_limit:
            return self._release_uncertified(key, state)
        return {
            "tracked": True,
            "allow": False,
            "reason": _continuation_reason(state),
            **self._response(state),
        }
