"""Bounded model-facing results derived from evaluator-owned evidence."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_tools import ToolExecution


def evidence_brief(
    execution: ToolExecution,
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Expose accepted evidence without leaking Cortheon's internal state."""

    evidence_ids = observed.get("accepted_evidence_ids")
    accepted = evidence_ids if isinstance(evidence_ids, list) else []
    raw_action = observed.get("next_action")
    action = raw_action if isinstance(raw_action, dict) else {}
    required = action.get("required_fields")
    return {
        "status": execution.status,
        "accepted_evidence_ids": accepted,
        "evidence": [
            {
                "evidence_id": accepted[0] if len(accepted) == 1 else None,
                "content": execution.content,
            }
        ],
        "next_action": {
            "type": action.get("type"),
            "instruction": action.get("instruction"),
            "required_fields": (
                required
                if isinstance(required, list) and all(isinstance(item, str) for item in required)
                else []
            ),
        },
        "instruction": (
            "Solve the user's task from this evidence. If response_schema.fields is present, "
            "use every listed top-level and nested key exactly once and no other keys. The "
            "field_vocabulary map binds allowed scalar values to their exact fields; choose "
            "only from each field's list, without paraphrasing or moving values. The "
            "answer field must be the solved JSON object, never a string containing JSON, "
            "the schema, or these instructions. Claims must state the answer's material "
            "conclusions, not incidental raw numbers. Include every distinct hypothesis "
            "the task requests. Cite only exact accepted evidence IDs."
        ),
    }


def completion_brief(result: dict[str, Any]) -> dict[str, Any]:
    """Return only bounded verification feedback needed for one repair attempt."""

    next_action = result.get("next_action")
    action = next_action if isinstance(next_action, dict) else {}
    request = action.get("request")
    evidence_request = request if isinstance(request, dict) else {}
    return {
        "status": result.get("status"),
        "verification": result.get("verification"),
        "repair": {
            "type": action.get("type"),
            "instruction": action.get("instruction"),
            "reason": evidence_request.get("reason"),
        },
    }


def revision_brief(
    result: dict[str, Any],
    record: dict[str, Any],
    effect_status_map: dict[str, str],
) -> dict[str, Any]:
    """Echo one public model-authored revision record with a generic status rule."""

    return {
        **completion_brief(result),
        "revision_record": record,
        "effect_status_map": effect_status_map,
        "reasoning_binding": result.get("reasoning_binding"),
        "instruction": (
            "Construct the final answer from this public comparison. Set prior_status to "
            "the exact effect_status_map value for decisive_effect. Keep decisive_source "
            "and revised unchanged."
        ),
    }
