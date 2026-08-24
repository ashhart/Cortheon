"""Validate evaluator-owned terminal facts on sealed report rows."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.outcomes import (
    is_authenticated_withhold,
    is_exact_terminal_success,
)
from cortheon.parity_gates.errors import ParityContractError

_EVALUATOR_OUTCOME_KEYS = frozenset(
    {"schema_version", "transport", "terminal_status", "terminal_provenance", "finish_reason"}
)
_EVALUATOR_TRANSPORTS = frozenset(
    {"pi", "opencode", "frontier_cli", "openai_chat", "openai_responses", "cli"}
)
_TERMINAL_STATUSES = frozenset(
    {"success", "withheld", "incomplete", "tool_only", "transport_error", "missing"}
)
_TERMINAL_PROVENANCES = frozenset(
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
    }
)


def validate_evaluator_outcome(row: dict[str, Any], index: int) -> None:
    """Require a closed outcome and align it with the row's grading state."""

    path = f"rows[{index}].evaluator_outcome"
    outcome = row.get("evaluator_outcome")
    if not isinstance(outcome, dict):
        raise ParityContractError(f"{path} must be an object")
    observed = set(outcome)
    if observed != _EVALUATOR_OUTCOME_KEYS:
        missing = sorted(_EVALUATOR_OUTCOME_KEYS - observed)
        extra = sorted(observed - _EVALUATOR_OUTCOME_KEYS)
        raise ParityContractError(f"{path} fields are not closed: missing={missing}, extra={extra}")
    if type(outcome.get("schema_version")) is not int or outcome["schema_version"] != 1:
        raise ParityContractError(f"{path}.schema_version must be 1")
    if outcome.get("transport") not in _EVALUATOR_TRANSPORTS:
        raise ParityContractError(f"{path}.transport is unknown")
    if outcome.get("terminal_status") not in _TERMINAL_STATUSES:
        raise ParityContractError(f"{path}.terminal_status is unknown")
    if outcome.get("terminal_provenance") not in _TERMINAL_PROVENANCES:
        raise ParityContractError(f"{path}.terminal_provenance is unknown")
    finish_reason = outcome.get("finish_reason")
    if finish_reason is not None and (
        not isinstance(finish_reason, str) or len(finish_reason) > 128
    ):
        raise ParityContractError(f"{path}.finish_reason must be a bounded string or null")
    terminal_success = is_exact_terminal_success(outcome)
    authenticated_withhold = is_authenticated_withhold(outcome)
    owner = row.get("failure_owner")
    if row.get("classification") == "error":
        if terminal_success or authenticated_withhold:
            raise ParityContractError(f"{path} cannot claim success for an error row")
        if owner not in {"candidate", "external_infrastructure"}:
            raise ParityContractError(f"{path} error has no closed failure owner")
    elif not (terminal_success or authenticated_withhold):
        raise ParityContractError(f"{path} is not an exact terminal success")
    elif owner is not None:
        raise ParityContractError(f"{path} terminal outcome cannot have a failure owner")
