"""Harvest each host result and advance the verified edit/diff/test loop."""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_hooks_core.observations import _host_observations, _read_snapshots
from cortheon.cognitive_hooks_core.responses import ResponseMixin
from cortheon.cognitive_hooks_core.state import HookTurn
from cortheon.cognitive_repair import (
    RepairPlan,
    changed_paths_from_diff,
    derive_repair_candidates,
)


class PatchLoopMixin(ResponseMixin):
    """Observe the completed host action and schedule the next bounded step."""

    def _automatic_post_tool(
        self,
        state: HookTurn,
        tool_name: str,
        *,
        succeeded: bool,
        tool_output: str,
        tool_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        request = state.pending_request
        if not state.awaiting_host_result or request is None or state.cortheon_session_id is None:
            return {"tracked": True, **self._response(state)}
        state.awaiting_host_result = False
        host_command = state.pending_host_command
        state.pending_host_command = ""
        host_input = state.pending_host_input
        state.pending_host_input = {}
        capability = str(request.get("capability") or "")
        origin = state.pending_origin
        if capability == "edit":
            if succeeded and state.repair_plan is not None:
                state.patch_applied = True
                self._metrics["hook_auto_patches_applied"] += 1
                self._schedule_adapter_request(
                    state,
                    capability="diff",
                    query=(f"Capture the focused live diff for {state.repair_plan.path}."),
                    success_condition=(
                        "Return a non-empty Git diff proving the implementation "
                        "change and showing that protected tests were untouched."
                    ),
                    parameters={"path": state.repair_plan.path},
                )
            else:
                self._clear_pending_request(state)
                state.last_next_action = {
                    "type": "reason",
                    "instruction": (
                        "The bounded host edit failed. Inspect the failure, revise "
                        "the implementation repair, and do not claim completion."
                    ),
                }
            return {"tracked": True, **self._response(state)}
        if not succeeded and capability != "grep" and origin == "adapter":
            if (
                capability == "test"
                and not state.check_pending
                and state.repair_candidates
                and state.repair_candidate_index + 1 < len(state.repair_candidates)
            ):
                # Best-of-N: several candidates satisfied the observed
                # examples; the real test suite is the judge. Swap the
                # failed candidate for the next-ranked one and rerun the
                # verified edit -> diff -> test loop.
                current = state.repair_candidates[state.repair_candidate_index]
                state.repair_candidate_index += 1
                successor = state.repair_candidates[state.repair_candidate_index]
                state.repair_plan = RepairPlan(
                    path=successor.path,
                    old_text=current.new_text,
                    new_text=successor.new_text,
                    function_name=successor.function_name,
                    examples=successor.examples,
                )
                self._metrics["hook_auto_repair_candidates_advanced"] += 1
                self._schedule_adapter_request(
                    state,
                    capability="edit",
                    query=(
                        "The previous candidate repair failed the real test "
                        f"suite; apply the next verified candidate to {successor.path}."
                    ),
                    success_condition=(
                        f"Change only {successor.path}; preserve every protected test."
                    ),
                    parameters={
                        "path": successor.path,
                        "function": successor.function_name,
                        "examples": successor.examples,
                    },
                )
                return {"tracked": True, **self._response(state)}
            state.check_pending = False
            self._clear_pending_request(state)
            state.last_next_action = {
                "type": "reason",
                "instruction": (
                    f"The host {capability or 'evidence'} operation failed. "
                    "Correct the implementation or command, then gather fresh "
                    "diff and test evidence."
                ),
            }
            return {"tracked": True, **self._response(state)}
        observations = _host_observations(
            request,
            tool_name,
            tool_output,
            succeeded=succeeded,
            host_command=host_command,
            host_input=host_input,
            tool_metadata=tool_metadata,
        )
        if not observations:
            return {"tracked": True, **self._response(state)}
        runtime = self.runtime
        if runtime is None:
            return {"tracked": True, **self._response(state)}
        try:
            observed = runtime.observe(
                state.cortheon_session_id,
                observations,
                request_id=(
                    str(request["request_id"])
                    if origin == "runtime" and request.get("request_id")
                    else None
                ),
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            return {
                "tracked": True,
                "observation_error": str(exc)[:500],
                **self._response(state),
            }
        accepted = [str(item) for item in observed.get("accepted_evidence_ids", ())]
        if all(item.get("status") != "failed" for item in observations):
            state.evidence_ids.extend(item for item in accepted if item not in state.evidence_ids)
        state.observed = bool(state.evidence_ids)
        self._capture_next_action(state, observed)
        self._metrics["hook_auto_observations"] += len(accepted)
        if capability == "read_many":
            for path, content in _read_snapshots(observations):
                state.read_snapshots[path] = content
        response_request = (observed.get("next_action") or {}).get("request")
        read_still_pending = isinstance(response_request, dict) and (
            response_request.get("request_id") == request.get("request_id")
        )
        if (
            capability == "read_many"
            and state.deliverable == "code_change"
            and not read_still_pending
        ):
            plans = derive_repair_candidates(list(state.read_snapshots.items()))
            plan = plans[0] if plans else None
            if plan is not None and state.test_invocation is not None:
                state.repair_plan = plan
                state.repair_candidates = plans
                state.repair_candidate_index = 0
                self._metrics["hook_auto_repairs_derived"] += 1
                self._schedule_adapter_request(
                    state,
                    capability="edit",
                    query=(
                        f"Apply the bounded one-line repair to {plan.path}; "
                        "the host hook will provide the exact patch."
                    ),
                    success_condition=(f"Change only {plan.path}; preserve every protected test."),
                    parameters={
                        "path": plan.path,
                        "function": plan.function_name,
                        "examples": plan.examples,
                    },
                )
        elif capability == "diff" and origin == "adapter":
            changed_paths = changed_paths_from_diff(tool_output)
            if (
                changed_paths
                and not self._protected_diff_paths(state, changed_paths)
                and state.test_invocation is not None
            ):
                if not state.patch_applied:
                    state.patch_applied = True
                    self._metrics["hook_auto_patches_applied"] += 1
                self._schedule_adapter_request(
                    state,
                    capability="test",
                    query=(
                        "Run the exact test command requested by the user after "
                        "the captured final diff."
                    ),
                    success_condition=("Return exit status zero and the focused passing summary."),
                    parameters={
                        "command": [
                            state.test_invocation.executable,
                            *state.test_invocation.arguments,
                        ]
                    },
                )
            else:
                self._clear_pending_request(state)
                state.last_next_action = {
                    "type": "reason",
                    "instruction": (
                        "The diff was empty or touched a protected test. Restore "
                        "the tests and produce an implementation-only change."
                    ),
                }
        elif capability == "test" and origin == "adapter":
            self._clear_pending_request(state)
            if succeeded:
                was_check_step = state.check_pending
                state.check_pending = False
                if was_check_step:
                    self._metrics["hook_auto_checks_passed"] += 1
                else:
                    self._metrics["hook_auto_tests_passed"] += 1
                if not was_check_step and state.check_invocation is not None:
                    # Chain the user's requested quality check as one more
                    # deterministic adapter step after the tests pass.
                    state.check_pending = True
                    self._metrics["hook_auto_checks_scheduled"] += 1
                    self._schedule_adapter_request(
                        state,
                        capability="test",
                        query=(
                            "Run the exact quality check requested by the user "
                            "after the passing test."
                        ),
                        success_condition=(
                            "Return exit status zero from the requested lint or type check."
                        ),
                        parameters={
                            "command": [
                                state.check_invocation.executable,
                                *state.check_invocation.arguments,
                            ]
                        },
                    )
                else:
                    state.last_next_action = {
                        "type": "finish",
                        "instruction": (
                            "The host captured the final diff and a post-change passing "
                            "test. Report the concrete change and verified result."
                        ),
                    }
        return {"tracked": True, **self._response(state)}
