"""Causal document-synthesis runtime scripts for the terminal tests.

Not a pytest module: these reproduce the bounded-completion signatures
around evidence sufficiency — preloaded-then-redundant discovery,
never-sufficient wandering, single sufficient batches, degenerate
repeated evidence, and permanently pending requests.
"""

from __future__ import annotations

from typing import Any

from pi_terminal_constants import CAUSAL_CERTIFIED, CAUSAL_PROMPT, EVIDENCE


def evidence_ready_wandering_script(completes: bool = True):
    """A causal runtime that preloads accepted evidence on start (one
    accepted observation batch), accepts the model's first observation
    (the second batch), then answers every further /v1/observe with no
    request and no finish: the model's discovery persistence past that
    point is redundant. /v1/complete certifies only when ``completes``."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "causal-term-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "accepted_evidence_ids": ["ev-1", "ev-2"],
                    "context": {"goal": body.get("goal"), "evidence": EVIDENCE},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "causal-term-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-3"],
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete" and completes:
            return (
                200,
                {
                    "session_id": "causal-term-1",
                    "status": "complete",
                    "answer": CAUSAL_CERTIFIED,
                },
            )
        return 200, {"status": "ok"}

    return script


def evidence_insufficient_wandering_script():
    """A causal runtime whose accepted evidence never carries two unique
    identities from two distinct sources (only ev-1 from one file is ever
    accepted): below the sufficiency threshold, so wandering discovery must
    stay admitted until the host tool budget."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "causal-insuf-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {
                        "goal": body.get("goal"),
                        "evidence": EVIDENCE[:1],
                    },
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "causal-insuf-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-3"],
                    "next_action": {"type": "reason"},
                },
            )
        return 200, {"status": "ok"}

    return script


def single_batch_sufficient_script(completes: bool = True):
    """A causal runtime whose only accepting observation batch carries two
    unique identities from two distinct clean sources in one response: that
    single batch can become sufficient. The start payload's evidence is
    id-less, so none of it can count before the batch."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "causal-single-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {
                        "goal": body.get("goal"),
                        "evidence": [
                            {
                                "source": "pi:read:facts/x.txt",
                                "content": "An observation with no runtime id.",
                            }
                        ],
                    },
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "causal-single-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1", "ev-2"],
                    "context": {"goal": CAUSAL_PROMPT, "evidence": EVIDENCE},
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete" and completes:
            return (
                200,
                {
                    "session_id": "causal-single-1",
                    "status": "complete",
                    "answer": CAUSAL_CERTIFIED,
                },
            )
        return 200, {"status": "ok"}

    return script


def repeated_evidence_wandering_script(mode: str):
    """A causal runtime that accepts two observation batches which never
    reach two unique identities from two distinct sources: ``same_source``
    accepts two distinct ids from one source record, ``repeated_identity``
    accepts the same id twice from two sources, and ``poisoned`` accepts
    only quarantined or failed entries. Sufficiency must never trigger."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": f"causal-{mode}-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal"), "evidence": []},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            if mode == "same_source":
                evidence = [
                    {
                        "evidence_id": "ev-1",
                        "source": "pi:read:facts/a.txt",
                        "content": EVIDENCE[0]["content"],
                    },
                    {
                        "evidence_id": "ev-2",
                        "source": "pi:read:facts/a.txt",
                        "content": EVIDENCE[1]["content"],
                    },
                ]
            elif mode == "repeated_identity":
                evidence = [
                    {
                        "evidence_id": "ev-1",
                        "source": "pi:read:facts/a.txt",
                        "content": EVIDENCE[0]["content"],
                    },
                    {
                        "evidence_id": "ev-1",
                        "source": "pi:read:facts/b.txt",
                        "content": EVIDENCE[0]["content"],
                    },
                ]
            else:
                evidence = [
                    {
                        "evidence_id": "ev-1",
                        "source": "pi:read:facts/a.txt",
                        "content": "A quarantined observation.",
                        "quarantine_flags": ["instruction_like_segment"],
                    },
                    {
                        "evidence_id": "ev-2",
                        "source": "pi:read:facts/b.txt",
                        "content": "A failed observation.",
                        "status": "failed",
                    },
                ]
            return (
                200,
                {
                    "session_id": f"causal-{mode}-1",
                    "status": "observing",
                    "accepted_evidence_ids": [entry["evidence_id"] for entry in evidence],
                    "context": {"goal": CAUSAL_PROMPT, "evidence": evidence},
                    "next_action": {"type": "reason"},
                },
            )
        return 200, {"status": "ok"}

    return script


def always_pending_request_script():
    """A causal runtime with two clean independent sources accepted, where
    every response carries a fresh pending runtime evidence request: a
    pending request must always reopen discovery, so the sufficiency guard
    can never force the answer and only the host tool budget bounds the
    run."""

    counter = {"n": 0}

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "causal-pending-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal"), "evidence": []},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Find what causes the clash.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            counter["n"] += 1
            return (
                200,
                {
                    "session_id": "causal-pending-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1", "ev-2"],
                    "context": {"goal": CAUSAL_PROMPT, "evidence": EVIDENCE},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": f"req-{counter['n']}",
                            "capability": "reason",
                            "query": "Another discriminating question.",
                        },
                    },
                },
            )
        return 200, {"status": "ok"}

    return script
