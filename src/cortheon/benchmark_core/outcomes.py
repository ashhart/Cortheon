"""Evaluator-owned terminal provenance and verified-completion rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

OUTCOME_SCHEMA_VERSION = 1

Transport = Literal[
    "pi",
    "opencode",
    "frontier_cli",
    "openai_chat",
    "openai_responses",
    "cli",
    "generic_mcp",
]
TerminalStatus = Literal[
    "success",
    "withheld",
    "incomplete",
    "tool_only",
    "transport_error",
    "missing",
]
TerminalProvenance = Literal[
    "pi_assistant",
    "pi_custom_terminal",
    "pi_candidate_assistant",
    "opencode_step_finish",
    "frontier_result",
    "chat_finish_reason",
    "responses_status",
    "process_exit",
    "none",
    "generic_mcp_terminal",
]

_OUTCOME_KEYS = frozenset(
    {
        "schema_version",
        "transport",
        "terminal_status",
        "terminal_provenance",
        "finish_reason",
    }
)
_TRANSPORTS = frozenset(
    {
        "pi",
        "opencode",
        "frontier_cli",
        "openai_chat",
        "openai_responses",
        "cli",
        "generic_mcp",
    }
)
_STATUSES = frozenset(
    {"success", "withheld", "incomplete", "tool_only", "transport_error", "missing"}
)
_PROVENANCES = frozenset(
    {
        "pi_assistant",
        "pi_custom_terminal",
        "pi_candidate_assistant",
        "opencode_step_finish",
        "frontier_result",
        "chat_finish_reason",
        "responses_status",
        "process_exit",
        "none",
        "generic_mcp_terminal",
    }
)
_SUCCESS_TERMINALS = frozenset(
    {
        ("pi", "pi_assistant", "stop"),
        ("opencode", "opencode_step_finish", "stop"),
        ("frontier_cli", "frontier_result", "success"),
        ("openai_chat", "chat_finish_reason", "stop"),
        ("openai_responses", "responses_status", "completed"),
        ("cli", "process_exit", "exit_0"),
        ("generic_mcp", "generic_mcp_terminal", "certified"),
        ("generic_mcp", "generic_mcp_terminal", "stop"),
    }
)
_WITHHELD_TERMINALS = frozenset(
    {
        ("pi", "pi_custom_terminal", "withheld"),
        ("pi", "pi_candidate_assistant", "stop"),
        ("generic_mcp", "generic_mcp_terminal", "bounded_incomplete"),
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """A terminal fact observed by the evaluator, never inferred from answer text."""

    transport: Transport
    terminal_status: TerminalStatus
    terminal_provenance: TerminalProvenance
    finish_reason: str | None
    schema_version: int = OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.transport not in _TRANSPORTS:
            raise ValueError(f"unknown evaluation transport: {self.transport!r}")
        if self.terminal_status not in _STATUSES:
            raise ValueError(f"unknown terminal status: {self.terminal_status!r}")
        if self.terminal_provenance not in _PROVENANCES:
            raise ValueError(f"unknown terminal provenance: {self.terminal_provenance!r}")
        if self.finish_reason is not None and (
            not isinstance(self.finish_reason, str) or len(self.finish_reason) > 128
        ):
            raise ValueError("finish_reason must be a bounded string or null")


def missing_outcome(transport: Transport) -> EvaluationOutcome:
    return EvaluationOutcome(
        transport=transport,
        terminal_status="missing",
        terminal_provenance="none",
        finish_reason=None,
    )


class _RunLike(Protocol):
    condition: str
    correct: bool
    delivered: bool
    timed_out: bool
    process_error: str | None
    evaluator_outcome: EvaluationOutcome
    runtime_sessions_started: int
    runtime_observations_accepted: int
    runtime_sessions_completed: int
    runtime_sessions_evidence_closed: int
    runtime_sessions_abandoned: int
    substrate_telemetry_valid: bool | None


def _mapping_outcome(value: Any) -> EvaluationOutcome | None:
    if not isinstance(value, Mapping) or set(value) != _OUTCOME_KEYS:
        return None
    if type(value.get("schema_version")) is not int:
        return None
    try:
        outcome = EvaluationOutcome(
            schema_version=value["schema_version"],
            transport=value["transport"],
            terminal_status=value["terminal_status"],
            terminal_provenance=value["terminal_provenance"],
            finish_reason=value["finish_reason"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    return outcome if outcome.schema_version == OUTCOME_SCHEMA_VERSION else None


def is_exact_terminal_success(outcome: EvaluationOutcome | Mapping[str, Any]) -> bool:
    parsed = outcome if isinstance(outcome, EvaluationOutcome) else _mapping_outcome(outcome)
    return bool(
        parsed
        and parsed.schema_version == OUTCOME_SCHEMA_VERSION
        and parsed.terminal_status == "success"
        and (parsed.transport, parsed.terminal_provenance, parsed.finish_reason)
        in _SUCCESS_TERMINALS
    )


def is_authenticated_withhold(outcome: EvaluationOutcome | Mapping[str, Any]) -> bool:
    parsed = outcome if isinstance(outcome, EvaluationOutcome) else _mapping_outcome(outcome)
    return bool(
        parsed
        and parsed.schema_version == OUTCOME_SCHEMA_VERSION
        and parsed.terminal_status == "withheld"
        and (parsed.transport, parsed.terminal_provenance, parsed.finish_reason)
        in _WITHHELD_TERMINALS
    )


def is_task_terminal_success(
    outcome: EvaluationOutcome | Mapping[str, Any],
    expected_verdict: Any,
) -> bool:
    """A completed answer, or evaluator-authenticated restraint when required."""

    return is_exact_terminal_success(outcome) or bool(
        expected_verdict == "block" and is_authenticated_withhold(outcome)
    )


def is_delivered_outcome(run: _RunLike) -> bool:
    return bool(
        run.delivered
        and not run.timed_out
        and run.process_error is None
        and is_exact_terminal_success(run.evaluator_outcome)
    )


def is_verified_completion(run: _RunLike) -> bool:
    if not (run.correct and is_delivered_outcome(run)):
        return False
    if run.condition == "cortheon":
        requires_runtime = True
    elif getattr(run, "condition_registry_version", None) is None:
        return run.condition in {"baseline", "frontier"}
    else:
        if run.condition not in {
            "bare",
            "retrieval_only",
            "verification_only",
            "old_planner",
            "full",
            "without_hypothesis_framing",
            "without_discriminating_evidence",
            "without_contradiction_revision",
            "without_cross_source_derivation",
            "without_adaptive_stopping",
        }:
            return False
        requires_runtime = getattr(run, "condition_requires_runtime_completion", None)
        if (
            getattr(run, "condition_profile_receipt_valid", None) is not True
            or type(requires_runtime) is not bool
        ):
            return False
        if run.condition == "bare":
            return bool(
                requires_runtime is False
                and getattr(run, "condition_adapter_receipt_valid", None) is None
                and run.runtime_sessions_started == 0
                and run.runtime_observations_accepted == 0
                and run.runtime_sessions_completed == 0
                and run.runtime_sessions_evidence_closed == 0
                and run.runtime_sessions_abandoned == 0
            )
        if run.condition == "old_planner":
            return bool(
                getattr(run, "condition_adapter_receipt_valid", None) is True
                and run.substrate_telemetry_valid is True
                and run.runtime_sessions_started == 1
                and run.runtime_observations_accepted >= 1
                and run.runtime_sessions_completed
                + run.runtime_sessions_evidence_closed
                + run.runtime_sessions_abandoned
                == 1
            )
        if getattr(run, "condition_adapter_receipt_valid", None) is not True:
            return False
    if requires_runtime is False:
        return True
    closed = run.runtime_sessions_completed + run.runtime_sessions_evidence_closed
    return run.substrate_telemetry_valid is True and closed == 1


def is_serialized_delivered_outcome(run: Mapping[str, Any]) -> bool:
    return bool(
        run.get("delivered") is True
        and run.get("timed_out") is False
        and run.get("process_error") is None
        and is_exact_terminal_success(run.get("evaluator_outcome", {}))
    )


def is_serialized_verified_completion(run: Mapping[str, Any]) -> bool:
    if run.get("correct") is not True or not is_serialized_delivered_outcome(run):
        return False
    if run.get("condition") == "cortheon":
        requires_runtime = True
    elif run.get("condition_registry_version") is None:
        return run.get("condition") in {"baseline", "frontier"}
    else:
        if run.get("condition") not in {
            "bare",
            "retrieval_only",
            "verification_only",
            "old_planner",
            "full",
            "without_hypothesis_framing",
            "without_discriminating_evidence",
            "without_contradiction_revision",
            "without_cross_source_derivation",
            "without_adaptive_stopping",
        }:
            return False
        requires_runtime = run.get("condition_requires_runtime_completion")
        if (
            run.get("condition_profile_receipt_valid") is not True
            or type(requires_runtime) is not bool
        ):
            return False
        if run.get("condition") == "bare":
            return bool(
                requires_runtime is False
                and run.get("condition_adapter_receipt_valid") is None
                and all(
                    run.get(field) == 0
                    for field in (
                        "runtime_sessions_started",
                        "runtime_observations_accepted",
                        "runtime_sessions_completed",
                        "runtime_sessions_evidence_closed",
                        "runtime_sessions_abandoned",
                    )
                )
            )
        if run.get("condition") == "old_planner":
            return bool(
                run.get("condition_adapter_receipt_valid") is True
                and run.get("substrate_telemetry_valid") is True
                and run.get("runtime_sessions_started") == 1
                and type(run.get("runtime_observations_accepted")) is int
                and run["runtime_observations_accepted"] >= 1
                and sum(
                    run.get(field, -1)
                    for field in (
                        "runtime_sessions_completed",
                        "runtime_sessions_evidence_closed",
                        "runtime_sessions_abandoned",
                    )
                )
                == 1
            )
        if run.get("condition_adapter_receipt_valid") is not True:
            return False
    if requires_runtime is False:
        return True
    completed = run.get("runtime_sessions_completed")
    evidence_closed = run.get("runtime_sessions_evidence_closed")
    return bool(
        type(completed) is int
        and type(evidence_closed) is int
        and completed >= 0
        and evidence_closed >= 0
        and completed + evidence_closed == 1
        and run.get("substrate_telemetry_valid") is True
    )
