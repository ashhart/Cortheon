"""Evaluator-owned model/MCP host with sticky terminal interception."""

from __future__ import annotations

import json
from typing import Any

from cortheon.benchmark_core import generic_mcp_protocol as protocol
from cortheon.benchmark_core import generic_mcp_turns as turns
from cortheon.benchmark_core.generic_mcp_auto import execute_projected_action
from cortheon.benchmark_core.generic_mcp_brief import (
    evidence_brief,
)
from cortheon.benchmark_core.generic_mcp_events import (
    host_tool_result_event,
    mcp_tool_result_event,
    runtime_transition_event,
)
from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_lifecycle import execute_lifecycle_call
from cortheon.benchmark_core.generic_mcp_model import OpenAiModelClient
from cortheon.benchmark_core.generic_mcp_projection import (
    bind_tool_arguments,
    completion_answer_schema,
    host_complete_tool,
    host_completion_repair_tool,
    host_derivation_tool,
    host_discrimination_tool,
    host_reason_tool,
    host_repair_tool,
    host_revision_tool,
    lifecycle_tools,
)
from cortheon.benchmark_core.generic_mcp_result import GenericHostResult
from cortheon.benchmark_core.generic_mcp_runtime import EvaluatorMcpRuntime
from cortheon.benchmark_core.generic_mcp_source import generic_source_sha256, resource_records
from cortheon.benchmark_core.generic_mcp_terminal import StickyTerminal
from cortheon.benchmark_core.generic_mcp_tools import ToolBudgetExhausted, host_tool_definitions


class GenericMcpHost:
    def __init__(
        self,
        *,
        task_id: str,
        evaluation_profile: dict[str, Any],
        model: OpenAiModelClient,
        executor: IsolatedExecutor,
        max_steps: int,
        require_web: bool = False,
        resource_paths: tuple[str, ...] = (),
        wrapper_source_sha256: str | None = None,
    ) -> None:
        if not 1 <= max_steps <= 32:
            raise ValueError("max_steps must be from 1 to 32")
        nonce = evaluation_profile.get("nonce")
        if not isinstance(nonce, str):
            raise ValueError("evaluator profile requires a nonce")
        self.profile = json.loads(protocol.canonical_json(evaluation_profile))
        self.model = model
        self.executor = executor
        self.max_steps = max_steps
        self.require_web = require_web
        self.wrapper_source_sha256 = wrapper_source_sha256 or generic_source_sha256()
        operators = self.profile.get("config", {}).get("operators", {})
        no_operators = isinstance(operators, dict) and not any(operators.values())
        self.is_placebo = bool(no_operators and self.profile["config"]["intercepts_final"] is True)
        self.is_bare = bool(no_operators and not self.is_placebo)
        self.resource_paths = tuple(resource_paths)
        self.resource_records = resource_records(self.executor.root, self.resource_paths)
        self.completion_answer_schema = completion_answer_schema(
            self.executor.root, self.resource_paths
        )
        self.runtime = (
            None
            if no_operators
            else EvaluatorMcpRuntime(self.profile, resource_paths=self.resource_paths)
        )
        self.transcript = protocol.GenericMcpTranscript(task_id, nonce)
        self.terminal = StickyTerminal(max_continuations=1)
        self._last_rejected_completion_sha256: str | None = None
        self._rejected_completion_count = 0
        self._repair_completion_arguments: dict[str, Any] | None = None
        self._reasoning_binding: dict[str, str] | None = None
        self._placebo_review_requested = False
        self._stale_host_call_retry_phases: set[str] = set()

    def _model_tools(self) -> tuple[list[dict[str, Any]], str]:
        host_tools = host_tool_definitions(web_enabled=self.executor.web_provider is not None)
        if self.runtime is None or self.profile["config"]["intercepts_final"] is False:
            return host_tools + (lifecycle_tools() if self.runtime is not None else []), "auto"
        projected = self.runtime.projected_host_tool()
        if projected is not None:
            selected = [tool for tool in host_tools if tool["function"]["name"] == projected]
            if len(selected) != 1:
                raise RuntimeError("runtime projected an unavailable host tool")
            return [
                bind_tool_arguments(selected[0], self.runtime.projected_arguments(projected))
            ], projected
        if self.runtime.projects_hypothesis_reasoning():
            return [host_reason_tool()], "host_reason"
        if self.runtime.projects_revision_reasoning():
            if self.completion_answer_schema is None:
                raise RuntimeError("revision reasoning requires a public answer schema")
            return [host_revision_tool(self.completion_answer_schema)], "host_reason"
        if self.runtime.projects_discrimination_reasoning():
            if self.completion_answer_schema is None:
                raise RuntimeError("discrimination reasoning requires a public answer schema")
            return [host_discrimination_tool(self.completion_answer_schema)], "host_reason"
        if self.runtime.projects_derivation_reasoning():
            if self.completion_answer_schema is None:
                raise RuntimeError("derivation reasoning requires a public answer schema")
            return [host_derivation_tool(self.completion_answer_schema)], "host_reason"
        if self.runtime.projects_repair_reasoning():
            if (
                self._last_rejected_completion_sha256
                and self._repair_completion_arguments is not None
                and self.runtime.projects_answer_repair()
            ):
                return [
                    host_completion_repair_tool(
                        self._repair_completion_arguments,
                        self.completion_answer_schema,
                        self._reasoning_binding,
                    )
                ], "host_complete"
            return [host_repair_tool()], "host_reason"
        return [
            host_complete_tool(self.completion_answer_schema, self._reasoning_binding)
        ], "host_complete"

    def _system_prompt(self) -> str:
        if self.is_bare or self.is_placebo:
            return protocol.BARE_SYSTEM_PROMPT
        if self.profile["config"]["intercepts_final"] is True:
            return protocol.WRAPPED_SYSTEM_PROMPT
        return protocol.TREATMENT_SYSTEM_PROMPT

    def run(self, goal: str, *, task_kind: str = "auto") -> GenericHostResult:
        capabilities = {
            "isolated_workspace": True,
            "closed_tool_catalogue": True,
            "intercepts_final": self.profile["config"]["intercepts_final"],
            "sticky_terminal": True,
            "current_web": self.executor.web_provider is not None,
        }
        self.transcript.record(
            "task_start",
            {
                "assurance": protocol.GENERIC_MCP_ASSURANCE,
                "condition_sha256": self.profile["config_sha256"],
                "evaluation_profile": self.profile,
                "model_requested": self.model.model_id,
                "provider_requested": self.model.provider_id,
                "endpoint_sha256": self.model.endpoint_sha256,
                "wrapper_source_sha256": self.wrapper_source_sha256,
                "intervention_prompt_sha256": protocol.payload_sha256(self._system_prompt()),
                "identity_provenance": "evaluator_requested_endpoint_response_model",
                "capabilities": capabilities,
                "runtime_used": self.runtime is not None,
                "condition_intercepts_final": self.profile["config"]["intercepts_final"],
                "web_provider": self.executor.web_identity,
                "resource_paths": list(self.resource_paths),
                "resource_records": list(self.resource_records),
                "task_kind": task_kind,
            },
        )
        if self.require_web and not capabilities["current_web"]:
            return self._handshake_failure("required generic web capability is absent")
        tokens = steps = calls = 0
        cost_usd: float | None = 0.0
        candidate = ""
        try:
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {"role": "user", "content": goal},
            ]
            if self.runtime is not None:
                started = self.runtime.start(goal, task_kind=task_kind)
                self._runtime_event("start", started)
                messages.append(
                    {"role": "system", "content": json.dumps(started, separators=(",", ":"))}
                )
            while steps < self.max_steps and not self.transcript.terminal:
                if execute_projected_action(self, messages):
                    calls += 1
                    continue
                tools, tool_choice = self._model_tools()
                turn = self.model.complete(messages, tools, tool_choice=tool_choice)
                turn, forced_binding = turns.bind_forced_turn(turn, tools, tool_choice)
                steps += 1
                tokens += turn.tokens
                cost_usd = (
                    cost_usd + turn.cost_usd
                    if cost_usd is not None and turn.cost_usd is not None
                    else None
                )
                candidate = turn.content or candidate
                self.transcript.record(
                    "message",
                    {
                        "role": "assistant",
                        "message_id": f"assistant-{steps}",
                        "content": turn.content,
                        "tool_call_ids": [item.call_id for item in turn.tool_calls],
                        "finish_reason": turn.finish_reason,
                        "tokens": turn.tokens,
                        "provider_requested": turn.provider_id,
                        "model_observed": turn.model_id,
                        "identity_provenance": turn.identity_provenance,
                        "cost_usd": turn.cost_usd,
                        "available_tools": [tool["function"]["name"] for tool in tools],
                        "tool_choice": tool_choice,
                        "tool_catalogue": tools,
                        "tool_catalogue_sha256": protocol.payload_sha256(tools),
                        "forced_binding": forced_binding,
                    },
                )
                messages.append(turns.assistant_message(turn))
                if turn.tool_calls:
                    if turns.reject_invalid_call_ids(self, turn.tool_calls):
                        calls += len(turn.tool_calls)
                        break
                    if turns.reject_invalid_tool_names(self, turn.tool_calls):
                        calls += len(turn.tool_calls)
                        break
                    if turns.reject_duplicate_call_ids(self, turn.tool_calls):
                        calls += len(turn.tool_calls)
                        break
                    stale_results = turns.retry_stale_host_calls(
                        self,
                        turn.tool_calls,
                        tools,
                        tool_choice,
                    )
                    if stale_results is not None:
                        calls += len(turn.tool_calls)
                        messages.extend(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": json.dumps(result, separators=(",", ":")),
                            }
                            for call_id, result in stale_results
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    f"The previous host tool is no longer active. Call "
                                    f"{tool_choice} now using its offered schema."
                                ),
                            }
                        )
                        continue
                    if turns.reject_unoffered_calls(self, turn.tool_calls, tools, tool_choice):
                        calls += len(turn.tool_calls)
                        break
                    if turns.reject_duplicate_forced_calls(self, turn.tool_calls, tool_choice):
                        calls += len(turn.tool_calls)
                        break
                    if turns.reject_invalid_calls(self, turn.tool_calls, tools):
                        calls += len(turn.tool_calls)
                        break
                    for call in turn.tool_calls:
                        calls += 1
                        result, terminal = self._tool_call(call.call_id, call.name, call.arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": json.dumps(result, separators=(",", ":")),
                            }
                        )
                        if terminal:
                            break
                    continue
                if not turn.content.strip():
                    messages.append(
                        {"role": "system", "content": "Return a bounded answer or MCP action."}
                    )
                    continue
                if self.is_placebo and not self._placebo_review_requested:
                    self._placebo_review_requested = True
                    messages.append(
                        {"role": "system", "content": protocol.EQUAL_BUDGET_REVIEW_PROMPT}
                    )
                    continue
                if (
                    self.is_bare
                    or self.is_placebo
                    or self.profile["config"]["intercepts_final"] is False
                ):
                    closed = self._abandon_runtime()
                    self._emit_receipt()
                    self._terminal(self.terminal.released(turn.content, runtime_closed=closed))
                    break
                if self.terminal.premature_final() == "continue":
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your answer was not certified. Call host_complete now "
                                "with evidence-linked claims and hypotheses, or abandon."
                            ),
                        }
                    )
                    continue
                assert self.runtime is not None
                closed = self._abandon_runtime()
                self._emit_receipt()
                self._terminal(
                    self.terminal.withheld(
                        "completion was not certified within the continuation budget",
                        runtime_closed=closed,
                    )
                )
            if not self.transcript.terminal:
                closed = self._abandon_runtime()
                self._emit_receipt()
                self._terminal(
                    self.terminal.withheld("model step budget exhausted", runtime_closed=closed)
                )
        except ToolBudgetExhausted:
            closed = self._abandon_runtime()
            self._emit_receipt()
            self._terminal(
                self.terminal.withheld("host tool budget exhausted", runtime_closed=closed)
            )
            return self._result(tokens, cost_usd, steps, calls, process_error=None)
        except Exception as exc:
            closed = False
            cleanup_error: Exception | None = None
            try:
                closed = self._abandon_runtime()
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
            if candidate:
                self._terminal(self.terminal.fail_open(candidate, type(exc).__name__.lower()))
            else:
                self._terminal(
                    self.terminal.withheld(
                        f"generic MCP host failed: {type(exc).__name__}",
                        runtime_closed=closed,
                    )
                )
            detail = str(exc)
            if cleanup_error is not None:
                detail = f"{detail}; cleanup failed: {type(cleanup_error).__name__}"
            return self._result(tokens, cost_usd, steps, calls, process_error=detail[:500])
        return self._result(tokens, cost_usd, steps, calls, process_error=None)

    def _tool_call(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        origin = (
            "mcp"
            if name in {"host_complete", "host_reason"}
            else ("host" if name.startswith("host_") else "mcp")
        )
        request = self.executor.ledger.request(call_id, name, arguments)
        self.transcript.record(
            "tool_request",
            {
                "call_id": call_id,
                "origin": origin,
                "name": name,
                "arguments": arguments,
                "request_sha256": request.request_sha256,
            },
        )
        if origin == "host":
            if self.runtime is not None:
                test_command = self.executor.tests.get(str(arguments.get("test_id")))
                allowed = self.runtime.validate_host_arguments(
                    name,
                    arguments,
                    test_command=test_command,
                )
                if not allowed:
                    receipt = {
                        "tool": name.removeprefix("host_"),
                        "executor": "generic_mcp_wrapper",
                        "outcome": "error",
                        "args": arguments,
                    }
                    rejected = self.executor.ledger.record(
                        request,
                        status="error",
                        content="Host arguments did not match the active runtime request.",
                        receipt=receipt,
                    )
                    observed = {"status": rejected.status, "content": rejected.content}
                    self._tool_result(rejected, observed)
                    return observed, False
            execution = self.executor.execute(call_id, name, arguments)
            if self.runtime is None:
                observed = {"status": execution.status, "content": execution.content}
                self._tool_result(execution, observed)
                return observed, False
            observed = self.runtime.observe(execution)
            self._tool_result(execution, observed)
            self._runtime_event("observe", observed)
            return (
                evidence_brief(execution, observed) if len(self.resource_paths) == 1 else observed
            ), False
        if self.runtime is None:
            raise RuntimeError("bare condition cannot call Cortheon lifecycle tools")
        return execute_lifecycle_call(
            self,
            call_id,
            name,
            arguments,
            request.request_sha256,
        )

    def _tool_result(self, execution: Any, observed: dict[str, Any]) -> None:
        self.transcript.record(
            "tool_result",
            host_tool_result_event(
                execution,
                observed,
                self.runtime.session_id if self.runtime is not None else None,
            ),
        )

    def _mcp_tool_result(
        self,
        call_id: str,
        request_sha256: str,
        result: dict[str, Any],
        *,
        transition: str | None = None,
    ) -> None:
        self.transcript.record(
            "tool_result",
            mcp_tool_result_event(
                call_id,
                request_sha256,
                result,
                self.runtime.session_id if self.runtime is not None else None,
                transition,
            ),
        )

    def _runtime_event(self, transition: str, payload: dict[str, Any]) -> None:
        assert self.runtime is not None
        assert self.runtime.session_id is not None
        self.transcript.record(
            "runtime_transition",
            runtime_transition_event(transition, self.runtime.session_id, payload),
        )

    def _emit_receipt(self) -> None:
        receipt = self.runtime.evaluation_receipt() if self.runtime is not None else None
        if receipt is not None:
            self.transcript.record("evaluation_receipt", {"receipt": receipt})

    def _abandon_runtime(self) -> bool:
        if self.runtime is None:
            return True
        was_closed = self.runtime.closed
        closed = self.runtime.abandon()
        if closed and not was_closed:
            self._runtime_event("abandon", {"status": "abandoned", "next_action": None})
        return closed

    def _terminal(self, payload: dict[str, object]) -> None:
        self.transcript.record("terminal", dict(payload))

    def _handshake_failure(self, message: str) -> GenericHostResult:
        self._terminal(self.terminal.withheld(message, runtime_closed=True))
        return self._result(0, None, 0, 0, process_error=message)

    def _result(
        self,
        tokens: int,
        cost_usd: float | None,
        steps: int,
        calls: int,
        *,
        process_error: str | None,
    ) -> GenericHostResult:
        terminal = self.transcript.events[-1]
        return GenericHostResult(
            tuple(self.transcript.events),
            str(terminal.get("text", "")),
            terminal.get("disposition") in {"release", "fail_open"},
            process_error,
            tokens,
            cost_usd,
            steps,
            calls,
        )
