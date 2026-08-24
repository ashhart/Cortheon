"""The tool catalogue and JSON Schemas that tools/list publishes."""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_mcp_core.protocol import HOST_RECEIPT_OUTCOMES


def tool_definitions(*, advanced: bool = False) -> list[dict[str, Any]]:
    string = {"type": "string"}
    string_array = {"type": "array", "items": string}
    hypothesis = {
        "type": "object",
        "properties": {
            "statement": string,
            "falsification_test": string,
        },
        "required": ["statement", "falsification_test"],
        "additionalProperties": False,
    }
    hypothesis_update = {
        "type": "object",
        "properties": {
            "hypothesis_id": string,
            "status": {
                "type": "string",
                "enum": ["open", "supported", "refuted", "uncertain"],
            },
            "evidence_ids": string_array,
        },
        "required": ["hypothesis_id", "status"],
        "additionalProperties": False,
    }
    completion_hypothesis = {
        "type": "object",
        "properties": {
            "statement": string,
            "falsification_test": string,
            "status": {
                "type": "string",
                "enum": ["supported", "refuted", "uncertain"],
            },
            "evidence_ids": string_array,
        },
        "required": [
            "statement",
            "falsification_test",
            "status",
            "evidence_ids",
        ],
        "additionalProperties": False,
    }
    host_receipt = {
        "type": "object",
        "description": (
            "Exact provenance for a host tool that was actually run. Deterministic "
            "code, document, diff, command, and test checks require this receipt."
        ),
        "properties": {
            "tool": {
                "type": "string",
                "description": (
                    "Logical evidence operation requested by Cortheon, for example "
                    "grep, read, diff, or test."
                ),
            },
            "executor": {
                "type": "string",
                "description": (
                    "Optional actual harness tool that performed the operation, "
                    "for example shell or bash."
                ),
            },
            "outcome": {
                "type": "string",
                "enum": sorted(HOST_RECEIPT_OUTCOMES),
                "description": (
                    "Use match/no_match for grep, passed/failed for tests, changed "
                    "for diffs, and result for other successful calls."
                ),
            },
            "args": {
                "type": "object",
                "description": (
                    "Exact bounded host-tool arguments. Use pattern and path for grep; "
                    "use filePath for read."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["tool", "outcome", "args"],
        "additionalProperties": False,
    }
    observation = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "code",
                    "diff",
                    "test",
                    "command",
                    "documentation",
                    "web",
                    "user",
                    "artifact",
                    "analysis",
                    "other",
                ],
            },
            "content": string,
            "source": string,
            "url": string,
            "retrieved_at": string,
            "published_at": string,
            "purpose": {
                "type": "string",
                "enum": [
                    "discovery",
                    "corroboration",
                    "primary_fetch",
                    "contradiction_check",
                    "freshness_check",
                    "passive",
                ],
            },
            "status": {
                "type": "string",
                "enum": ["observed", "verified", "failed"],
                "default": "observed",
            },
            "supports": string_array,
            "contradicts": string_array,
            "host_receipt": host_receipt,
        },
        "required": ["kind", "content"],
        "additionalProperties": False,
    }
    claim = {
        "type": "object",
        "properties": {
            "claim": string,
            "evidence_ids": string_array,
        },
        "required": ["claim", "evidence_ids"],
        "additionalProperties": False,
    }
    tools = [
        {
            "name": "cortheon_start",
            "title": "Start Bounded Investigation",
            "description": (
                "Start an ephemeral evidence-driven investigation. Cortheon returns the "
                "next evidence request for the host to satisfy with its own tools."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": string,
                    "constraints": string_array,
                    "effort": {
                        "type": "string",
                        "enum": ["quick", "standard", "deep"],
                        "default": "quick",
                    },
                    "task_kind": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "code",
                            "research",
                            "documents",
                            "decision",
                            "general",
                        ],
                        "default": "auto",
                    },
                    "strictness": {
                        "type": "string",
                        "description": (
                            "How aggressively evidence requirements downgrade: "
                            "strict keeps the full contract, standard self-heals, "
                            "assist pre-arms downgrades for small local models."
                        ),
                        "enum": ["strict", "standard", "assist"],
                        "default": "standard",
                    },
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_step",
            "title": "Advance Investigation",
            "description": (
                "Submit public hypotheses, updates, questions, or a draft. Cortheon "
                "chooses the next bounded reasoning or host-evidence action."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "hypotheses": {"type": "array", "items": hypothesis},
                    "hypothesis_updates": {
                        "type": "array",
                        "items": hypothesis_update,
                    },
                    "open_questions": string_array,
                    "draft": string,
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_observe",
            "title": "Submit Live Host Evidence",
            "description": (
                "After actually running the requested host tool, return its focused result "
                "with the exact request_id. On the first observation omit supports and "
                "contradicts because hypothesis ids do not exist yet. Content is bounded, "
                "quarantined, memory-only, and never persisted. Non-web evidence must "
                "include host_receipt with the exact tool, actual arguments, and observed "
                "outcome. Cortheon returns ev* ids; never cite req* ids as evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "request_id": string,
                    "observations": {
                        "type": "array",
                        "minItems": 1,
                        "items": observation,
                    },
                },
                "required": ["session_id", "request_id", "observations"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_resume",
            "title": "Resume After Context Loss",
            "description": (
                "List the active investigations in this Cortheon process with their "
                "goals, pending next actions, and the working evidence context for "
                "the most recent one. Call this first when conversation context was "
                "compacted or lost, instead of asking the user to restate the task. "
                "It consumes no reasoning turn."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 3,
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_retract",
            "title": "Retract Mis-Marked Evidence",
            "description": (
                "Withdraw previously accepted ev* observations that turned out to be "
                "wrong or mis-linked. Retraction unlinks them from every hypothesis, "
                "frees their content for corrected resubmission, and never consumes a "
                "reasoning turn, so one bad observation cannot poison the session."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": string,
                    },
                    "reason": string,
                },
                "required": ["session_id", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_challenge",
            "title": "Challenge Draft",
            "description": (
                "Attack a draft for unsupported claims, confirmation bias, untested "
                "alternatives, and missing task-specific completion evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "draft": string,
                    "claims": {
                        "type": "array",
                        "minItems": 1,
                        "items": claim,
                    },
                },
                "required": ["session_id", "draft", "claims"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_verify",
            "title": "Verify Completion",
            "description": (
                "Fail closed unless claims are grounded, alternatives were tested, the "
                "draft was challenged, and live completion evidence satisfies the task."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "answer": string,
                    "claims": {
                        "type": "array",
                        "minItems": 1,
                        "items": claim,
                    },
                    "completion_evidence_ids": string_array,
                },
                "required": ["session_id", "answer", "claims"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_complete",
            "title": "Challenge, Verify, Complete, and Discard",
            "description": (
                "Preferred compact completion path. In one transaction, submit resolved "
                "public hypotheses, evidence-linked claims, completion evidence, and the "
                "answer. Cortheon challenges and verifies them, returns only an accepted "
                "answer, and discards the session. If a gate fails it withholds completion "
                "and returns the exact next action."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "answer": string,
                    "claims": {
                        "type": "array",
                        "minItems": 1,
                        "items": claim,
                    },
                    "hypotheses": {
                        "type": "array",
                        "minItems": 1,
                        "items": completion_hypothesis,
                    },
                    "completion_evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": string,
                    },
                },
                "required": [
                    "session_id",
                    "answer",
                    "claims",
                    "hypotheses",
                    "completion_evidence_ids",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_abandon",
            "title": "Abandon and Discard",
            "description": (
                "Explicitly abandon an inconclusive investigation and irreversibly "
                "discard its in-memory state. This never releases an answer."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"session_id": string},
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cortheon_finish",
            "title": "Finish and Discard",
            "description": (
                "Return the exact verified answer or abandon the investigation, then "
                "irreversibly discard all in-memory task evidence and state."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": string,
                    "mode": {
                        "type": "string",
                        "enum": ["complete", "abandon"],
                        "default": "complete",
                    },
                    "answer": string,
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
    ]
    preferred_order = {
        "cortheon_start": 0,
        "cortheon_observe": 1,
        "cortheon_complete": 2,
        "cortheon_retract": 3,
        "cortheon_abandon": 4,
        "cortheon_resume": 5,
        "cortheon_step": 6,
        "cortheon_challenge": 7,
        "cortheon_verify": 8,
        "cortheon_finish": 9,
    }
    if not advanced:
        compact_names = {
            "cortheon_start",
            "cortheon_observe",
            "cortheon_complete",
            "cortheon_retract",
            "cortheon_abandon",
            "cortheon_resume",
        }
        tools = [tool for tool in tools if tool["name"] in compact_names]
    tools.sort(key=lambda item: preferred_order[str(item["name"])])
    for tool in tools:
        tool["annotations"] = {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        }
    return tools
