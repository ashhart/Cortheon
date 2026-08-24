"""Parse host event streams into evaluator-owned terminal outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from cortheon.benchmark_core.outcomes import EvaluationOutcome, Transport
from cortheon.benchmark_core.pi_terminal import (
    _pi_terminal_text,
    _pi_withheld_reason,
)

CANDIDATE_ENTRY_TYPE = "cortheon-benchmark-candidate-v1"
CANDIDATE_ENTRY_VERSION = 1
CANDIDATE_STAGES = frozenset({"completion", "causal_synthesis"})
CANDIDATE_MAX_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class ParsedTransportOutcome:
    final_text: str
    outcome: EvaluationOutcome
    candidate: str | None = None


def _candidate_from_entry(event: Mapping[str, Any]) -> str | None:
    if event.get("type") != "entry_appended":
        return None
    entry = event.get("entry")
    if not isinstance(entry, Mapping) or entry.get("type") != "custom":
        return None
    if entry.get("customType") != CANDIDATE_ENTRY_TYPE:
        return None
    if not isinstance(entry.get("id"), str) or not isinstance(entry.get("timestamp"), str):
        return None
    data = entry.get("data")
    if not isinstance(data, Mapping) or set(data) != {"version", "stage", "candidate"}:
        return None
    if type(data["version"]) is not int or data["version"] != CANDIDATE_ENTRY_VERSION:
        return None
    if data["stage"] not in CANDIDATE_STAGES:
        return None
    candidate = data["candidate"]
    if not isinstance(candidate, str) or not candidate or len(candidate) > CANDIDATE_MAX_CHARS:
        return None
    return candidate


def captured_candidate(events: Iterable[dict[str, Any]]) -> str | None:
    """Return the last exact candidate entry; malformed latest entries poison."""

    observed = False
    candidate: str | None = None
    for event in events:
        entry = event.get("entry")
        if (
            event.get("type") == "entry_appended"
            and isinstance(entry, Mapping)
            and entry.get("customType") == CANDIDATE_ENTRY_TYPE
        ):
            observed = True
            candidate = _candidate_from_entry(event)
    return candidate if observed else None


def _assistant_content(message: Mapping[str, Any]) -> tuple[str, bool]:
    content = message.get("content")
    if not isinstance(content, list):
        return "", False
    text = "".join(
        str(item.get("text", ""))
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ).strip()
    has_tool = any(
        isinstance(item, Mapping) and item.get("type") in {"toolCall", "tool_call", "tool_use"}
        for item in content
    )
    return text, has_tool


def _pi_outcome(events: list[dict[str, Any]]) -> ParsedTransportOutcome:
    parsed = ParsedTransportOutcome(
        "",
        EvaluationOutcome("pi", "missing", "none", None),
    )
    last_output_index = -1
    candidate_index = -1
    candidate: str | None = None
    for index, event in enumerate(events):
        entry = event.get("entry")
        if (
            event.get("type") == "entry_appended"
            and isinstance(entry, Mapping)
            and entry.get("customType") == CANDIDATE_ENTRY_TYPE
        ):
            candidate_index = index
            candidate = _candidate_from_entry(event)
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping):
            continue
        custom = _pi_terminal_text(dict(message))
        if custom is not None:
            paired = candidate if candidate_index > last_output_index else None
            parsed = ParsedTransportOutcome(
                custom,
                EvaluationOutcome("pi", "withheld", "pi_custom_terminal", "withheld"),
                paired,
            )
            last_output_index = index
            continue
        if message.get("role") != "assistant":
            continue
        text, has_tool = _assistant_content(message)
        reason = message.get("stopReason")
        finish_reason = reason if isinstance(reason, str) else None
        if (
            parsed.outcome.terminal_provenance == "pi_custom_terminal"
            and text == parsed.final_text
            and not has_tool
            and finish_reason == "stop"
        ):
            # A message_end hook can emit the authenticated custom terminal
            # before Pi serializes the assistant replacement carrying the
            # exact same bounded text. It is one terminal, not a later output.
            continue
        if (
            not text
            and not has_tool
            and finish_reason not in {"toolUse", "tool_calls"}
            and parsed.outcome.terminal_provenance == "pi_custom_terminal"
        ):
            # Pi emits an empty aborted assistant envelope after an extension
            # aborts a bounded tool turn. It is lifecycle noise, not a later
            # answer or terminal. It can only be ignored after the exact
            # custom receipt that caused the abort; otherwise a later empty
            # envelope remains authoritative and poisons an earlier output.
            continue
        paired = candidate if candidate_index > last_output_index else None
        if text and _pi_withheld_reason(text) is not None and paired is not None:
            outcome = EvaluationOutcome("pi", "withheld", "pi_candidate_assistant", finish_reason)
            parsed = ParsedTransportOutcome(text, outcome, paired)
        elif text.startswith("[Cortheon withheld"):
            parsed = ParsedTransportOutcome(
                text,
                EvaluationOutcome("pi", "incomplete", "pi_assistant", finish_reason),
            )
        elif finish_reason == "stop" and text and not has_tool:
            parsed = ParsedTransportOutcome(
                text,
                EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
            )
        elif has_tool or finish_reason in {"toolUse", "tool_calls"}:
            parsed = ParsedTransportOutcome(
                text,
                EvaluationOutcome("pi", "tool_only", "pi_assistant", finish_reason),
            )
        else:
            parsed = ParsedTransportOutcome(
                text,
                EvaluationOutcome("pi", "incomplete", "pi_assistant", finish_reason),
            )
        last_output_index = index
    return parsed


def _opencode_outcome(events: list[dict[str, Any]]) -> ParsedTransportOutcome:
    parsed = ParsedTransportOutcome("", EvaluationOutcome("opencode", "missing", "none", None))
    pending_text = ""
    for event in events:
        part = event.get("part")
        if not isinstance(part, Mapping):
            continue
        if event.get("type") == "text":
            pending_text = str(part.get("text", "")).strip()
            continue
        if event.get("type") != "step_finish":
            continue
        raw_reason = part.get("reason")
        reason = raw_reason if isinstance(raw_reason, str) else None
        if reason == "stop" and pending_text:
            status = "success"
        elif reason in {"tool-calls", "tool_calls", "toolUse"}:
            status = "tool_only"
        else:
            status = "incomplete"
        parsed = ParsedTransportOutcome(
            pending_text,
            EvaluationOutcome("opencode", status, "opencode_step_finish", reason),
        )
        pending_text = ""
    if pending_text:
        return ParsedTransportOutcome(
            pending_text,
            EvaluationOutcome("opencode", "incomplete", "none", None),
        )
    return parsed


def _generic_mcp_outcome(events: list[dict[str, Any]]) -> ParsedTransportOutcome:
    from cortheon.benchmark_core.generic_mcp_validation import validate_transcript

    terminal = events[-1] if events and events[-1].get("type") == "terminal" else None
    text = str(terminal.get("text", "")) if isinstance(terminal, dict) else ""
    reason = terminal.get("finish_reason") if isinstance(terminal, dict) else None
    if not validate_transcript(events):
        return ParsedTransportOutcome(
            text,
            EvaluationOutcome(
                "generic_mcp", "transport_error", "process_exit", "invalid_transcript"
            ),
        )
    assert isinstance(terminal, dict) and isinstance(reason, str)
    status = "withheld" if terminal["disposition"] == "withhold" else "success"
    return ParsedTransportOutcome(
        text,
        EvaluationOutcome("generic_mcp", status, "generic_mcp_terminal", reason),
    )


def parse_transport_outcome(
    events: Iterable[dict[str, Any]],
    *,
    host: str,
) -> ParsedTransportOutcome:
    selected = list(events)
    if host == "pi":
        return _pi_outcome(selected)
    if host == "opencode":
        return _opencode_outcome(selected)
    if host == "generic_mcp":
        return _generic_mcp_outcome(selected)
    raise ValueError(f"unsupported event transport: {host!r}")


def failed_transport_outcome(
    transport: Transport,
    *,
    status: Literal["incomplete", "tool_only", "transport_error", "missing"],
    finish_reason: str,
) -> EvaluationOutcome:
    if transport == "pi":
        return EvaluationOutcome(transport, status, "pi_assistant", finish_reason)
    if transport == "opencode":
        return EvaluationOutcome(transport, status, "opencode_step_finish", finish_reason)
    if transport == "frontier_cli":
        return EvaluationOutcome(transport, status, "frontier_result", finish_reason)
    if transport == "openai_chat":
        return EvaluationOutcome(transport, status, "chat_finish_reason", finish_reason)
    if transport == "openai_responses":
        return EvaluationOutcome(transport, status, "responses_status", finish_reason)
    return EvaluationOutcome(transport, status, "process_exit", finish_reason)


def frontier_result_outcome(payload: Any, final: str) -> EvaluationOutcome:
    if not isinstance(payload, Mapping):
        return EvaluationOutcome("frontier_cli", "missing", "none", None)
    subtype = payload.get("subtype")
    stop_reason = payload.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason in {"length", "max_tokens"}:
        return EvaluationOutcome("frontier_cli", "incomplete", "frontier_result", stop_reason)
    if (
        payload.get("type") == "result"
        and subtype == "success"
        and payload.get("is_error") is False
    ):
        status = "success" if final else "tool_only"
        return EvaluationOutcome("frontier_cli", status, "frontier_result", "success")
    reason = subtype if isinstance(subtype, str) else "invalid_result_envelope"
    return EvaluationOutcome("frontier_cli", "incomplete", "frontier_result", reason[:128])
