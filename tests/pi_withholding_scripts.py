"""Ambiguity-resolution withholding scripts for the terminal tests.

Not a pytest module: these runtimes accept one observation and withhold
every completion — with a fixed action, a varying action, or a request
followed by a finish — so the candidate is withheld and the model wanders
into unrequested tools.
"""

from __future__ import annotations

from typing import Any


def withholding_ambiguity_script():
    """An ambiguity-resolution runtime: accepts the first observation,
    holds no further request, and withholds every completion with a
    bounded next action — the candidate is withheld twice and the model
    then wanders into tools nobody requested."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "amb-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal")},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Clarify the phrase.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "amb-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1"],
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete":
            return (
                200,
                {
                    "session_id": "amb-1",
                    "status": "needs_evidence",
                    "next_action": {
                        "type": "verify",
                        "submit_via": "cortheon_challenge",
                    },
                },
            )
        return 200, {"status": "ok"}

    return script


def varying_action_withholding_script():
    """An ambiguity-resolution runtime that withholds every completion with a
    different bounded next action each time (verify, then challenge, then
    verify, ...): every withhold is genuine request/action progress, so only
    the continuation cap — never the repeated-fingerprint gate — can bound
    the chain."""

    counter = {"withholds": 0}

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "amb-vary-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal")},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Clarify the phrase.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "amb-vary-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1"],
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete":
            counter["withholds"] += 1
            action = "verify" if counter["withholds"] % 2 else "challenge"
            return (
                200,
                {
                    "session_id": "amb-vary-1",
                    "status": "needs_evidence",
                    "next_action": {
                        "type": action,
                        "submit_via": "cortheon_challenge",
                    },
                },
            )
        return 200, {"status": "ok"}

    return script


def withhold_then_finish_script():
    """An ambiguity-resolution runtime that withholds the first completion
    with a pending evidence request (earning the one repair continuation),
    then answers that continuation's observation with a finish action and no
    request: the continuation's further tool-only persistence can only be
    bounded by the finish-phase answer-only boundary, and the single
    answer-only continuation that also ends tool-only must terminate
    visibly — one terminal status, one abandon, no surviving session."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "amb-finish-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal")},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Clarify the phrase.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "amb-finish-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1"],
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/complete":
            return (
                200,
                {
                    "session_id": "amb-finish-1",
                    "status": "needs_evidence",
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-complete-1",
                            "capability": "reason",
                            "query": "One clarifying observation.",
                        },
                    },
                },
            )
        return 200, {"status": "ok"}

    return script
