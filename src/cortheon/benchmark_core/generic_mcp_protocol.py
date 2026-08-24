"""Evaluator-owned transcript primitives for claim-grade generic MCP runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

GENERIC_MCP_TRANSCRIPT_VERSION = 1
GENERIC_MCP_ASSURANCE = "evaluator_wrapped"
BARE_SYSTEM_PROMPT = (
    "You are inside an evaluator-owned isolated host. Use only the listed host tools, "
    "then answer the user directly. No external reasoning substrate is active."
)
EQUAL_BUDGET_REVIEW_PROMPT = (
    "Review your previous candidate once using the same evidence and tools. Correct any error "
    "you notice, then return the final answer. No new method or task hint is provided."
)
TREATMENT_SYSTEM_PROMPT = (
    "You are inside an evaluator-owned generic MCP host. Use only the listed tools. "
    "Cortheon is already active. Never invent observations or receipts; the host submits "
    "tool evidence automatically. Complete with cortheon_complete, or abandon."
)
WRAPPED_SYSTEM_PROMPT = (
    "You are inside an evaluator-owned generic MCP host. Use only the one listed tool. "
    "Cortheon is already active and the host submits tool evidence automatically. Never "
    "invent observations or receipts. When offered host_complete, submit your own solved "
    "answer, material claims, requested hypotheses, and exact accepted evidence IDs."
)
MAX_EVENT_CHARS = 200_000
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
SHA256 = re.compile(r"[0-9a-f]{64}")
EVENT_TYPES = frozenset(
    {
        "task_start",
        "message",
        "runtime_tool_decision",
        "tool_request",
        "tool_result",
        "runtime_transition",
        "evaluation_receipt",
        "terminal",
    }
)
COMMON_KEYS = {"schema_version", "sequence", "task_id", "nonce", "type"}
EVENT_KEYS = {
    "task_start": {
        "assurance",
        "condition_sha256",
        "evaluation_profile",
        "model_requested",
        "provider_requested",
        "endpoint_sha256",
        "wrapper_source_sha256",
        "intervention_prompt_sha256",
        "identity_provenance",
        "capabilities",
        "runtime_used",
        "condition_intercepts_final",
        "web_provider",
        "resource_paths",
        "resource_records",
        "task_kind",
    },
    "message": {
        "role",
        "message_id",
        "content",
        "tool_call_ids",
        "finish_reason",
        "tokens",
        "provider_requested",
        "model_observed",
        "identity_provenance",
        "cost_usd",
        "available_tools",
        "tool_choice",
        "tool_catalogue",
        "tool_catalogue_sha256",
        "forced_binding",
    },
    "runtime_tool_decision": {"call_id", "name", "arguments", "request_sha256"},
    "tool_request": {"call_id", "origin", "name", "arguments", "request_sha256"},
    "tool_result": {
        "call_id",
        "origin",
        "status",
        "content",
        "request_sha256",
        "result_sha256",
        "receipt",
        "accepted_evidence_ids",
        "runtime_transition_sha256",
    },
    "runtime_transition": {
        "transition",
        "session_id",
        "status",
        "next_action",
        "transition_sha256",
    },
    "evaluation_receipt": {"receipt"},
    "terminal": {
        "disposition",
        "text",
        "provenance",
        "finish_reason",
        "runtime_closed",
        "candidate_sha256",
        "active_sessions",
    },
}
OPERATOR_KEYS = {
    "retrieval",
    "verification",
    "hypothesis_framing",
    "discriminating_evidence",
    "contradiction_revision",
    "cross_source_derivation",
    "adaptive_stopping",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def encoded_payload_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def bounded_identifier(value: Any) -> bool:
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


@dataclass(slots=True)
class GenericMcpTranscript:
    task_id: str
    nonce: str
    events: list[dict[str, Any]] = field(default_factory=list)
    terminal: bool = False

    def __post_init__(self) -> None:
        if not bounded_identifier(self.task_id) or not bounded_identifier(self.nonce):
            raise ValueError("transcript identity must be bounded")

    def record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.terminal:
            raise RuntimeError("generic MCP transcript is terminal")
        if event_type not in EVENT_TYPES or not isinstance(payload, dict):
            raise ValueError("generic MCP event is invalid")
        event = {
            "schema_version": GENERIC_MCP_TRANSCRIPT_VERSION,
            "sequence": len(self.events),
            "task_id": self.task_id,
            "nonce": self.nonce,
            "type": event_type,
            **payload,
        }
        if len(canonical_json(event)) > MAX_EVENT_CHARS:
            raise ValueError("generic MCP event exceeded its bound")
        self.events.append(event)
        self.terminal = event_type == "terminal"
        return event


TerminalDisposition = Literal["release", "withhold", "fail_open"]


def terminal_payload(
    disposition: TerminalDisposition,
    text: str,
    *,
    provenance: str,
    finish_reason: str,
    runtime_closed: bool,
) -> dict[str, object]:
    if not isinstance(text, str) or len(text) > 20_000:
        raise ValueError("terminal text must be bounded")
    return {
        "disposition": disposition,
        "text": text,
        "provenance": provenance,
        "finish_reason": finish_reason,
        "runtime_closed": runtime_closed,
        "candidate_sha256": encoded_payload_sha256(text),
        "active_sessions": 0 if runtime_closed else -1,
    }
