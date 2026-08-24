"""Codex lifecycle event handlers for the standalone hook facade."""

from __future__ import annotations

import json
from typing import Any

if __package__:
    from .hook_config import (
        CORTHEON_AUTO_CONTEXT,
        CORTHEON_COMPACT_AUTO_CONTEXT,
        CORTHEON_COMPACT_CONTEXT,
        CORTHEON_CONTEXT,
        CORTHEON_UNAVAILABLE_CONTEXT,
        MAX_HOST_ADAPTER_STEPS,
        SUBSTANTIVE_RE,
    )
    from .hook_transport import _facade
else:
    from hook_config import (
        CORTHEON_AUTO_CONTEXT,
        CORTHEON_COMPACT_AUTO_CONTEXT,
        CORTHEON_COMPACT_CONTEXT,
        CORTHEON_CONTEXT,
        CORTHEON_UNAVAILABLE_CONTEXT,
        MAX_HOST_ADAPTER_STEPS,
        SUBSTANTIVE_RE,
    )
    from hook_transport import _facade


def _user_prompt_submit(payload: dict[str, Any]) -> None:
    api = _facade()
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not (len(prompt) >= 180 or SUBSTANTIVE_RE.search(prompt)):
        return
    identity = api._identity(payload)
    if identity is not None and not api._ensure_runtime():
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": CORTHEON_UNAVAILABLE_CONTEXT,
                    }
                },
                separators=(",", ":"),
            )
        )
        return
    registered = None
    if identity is not None:
        strictness = api._configured_strictness()
        extra = {"strictness": strictness} if strictness else {}
        registered = api._post(
            "/v1/hooks/register",
            {**identity, **extra, "goal": prompt, "effort": "quick"},
        )
    if registered is None:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": CORTHEON_UNAVAILABLE_CONTEXT,
                    }
                },
                separators=(",", ":"),
            )
        )
        return
    if registered is not None and registered.get("degraded") is True:
        return
    automatic = bool(registered and registered.get("automatic") is True)
    if api._use_compact_context():
        context = CORTHEON_COMPACT_AUTO_CONTEXT if automatic else CORTHEON_COMPACT_CONTEXT
    else:
        context = CORTHEON_AUTO_CONTEXT if automatic else CORTHEON_CONTEXT
    if automatic and registered and registered.get("next_action"):
        context += "\n\nNEXT ACTION:\n" + json.dumps(
            registered["next_action"], separators=(",", ":"), ensure_ascii=False
        )
    if automatic and registered and registered.get("cognition"):
        context += "\n\nADAPTIVE COGNITION:\n" + json.dumps(
            registered["cognition"], separators=(",", ":"), ensure_ascii=False
        )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            },
            separators=(",", ":"),
        )
    )


def _pre_tool_use(payload: dict[str, Any]) -> None:
    api = _facade()
    identity = api._identity(payload)
    tool_name = payload.get("tool_name")
    if (
        identity is None
        or not isinstance(tool_name, str)
        or api._is_cortheon_skill_bootstrap(payload)
    ):
        return
    result = api._post(
        "/v1/hooks/pre-tool",
        {
            **identity,
            "tool_name": tool_name,
            "tool_input": (
                payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
            ),
        },
    )
    if result is None:
        return
    updated_input = result.get("updated_input")
    if result.get("allow") is True and isinstance(updated_input, dict):
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": updated_input,
                    }
                },
                separators=(",", ":"),
            )
        )
        return
    if result.get("allow") is not False:
        return
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "Cortheon must be started before this tool."
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            separators=(",", ":"),
        )
    )


def _post_tool_use(payload: dict[str, Any]) -> None:
    api = _facade()
    identity = api._identity(payload)
    tool_name = payload.get("tool_name")
    if (
        identity is None
        or not isinstance(tool_name, str)
        or api._is_cortheon_skill_bootstrap(payload)
    ):
        return
    response = payload.get("tool_response")
    succeeded = api._tool_succeeded(response)
    result = api._post(
        "/v1/hooks/post-tool",
        {
            **identity,
            "tool_name": tool_name,
            "succeeded": succeeded,
            "certified": api._contains_certified_completion(response),
            "tool_output": api._tool_output(response),
            "tool_metadata": api._tool_metadata(response),
        },
    )
    if result is None or not result.get("automatic") or not result.get("next_action"):
        return
    observation_error = result.get("observation_error")
    if not succeeded:
        prefix = (
            "The host action failed and Cortheon recorded that failed attempt. "
            "Inspect the real error and choose a recovery command; do not repeat "
            "the identical command merely to satisfy Cortheon. Current evidence goal: "
        )
    elif isinstance(observation_error, str) and observation_error:
        prefix = (
            "Cortheon did not bind that result to the pending request. The host "
            "action still ran and its real output remains available. Choose the "
            "next investigation command from that output; do not repeat a failed "
            "command merely to satisfy Cortheon. Current evidence goal: "
        )
    else:
        prefix = "Cortheon recorded the host result. Continue with this evidence goal: "
    cognition = result.get("cognition")
    cognition_context = (
        "\nAdaptive cognition: " + json.dumps(cognition, separators=(",", ":"), ensure_ascii=False)
        if isinstance(cognition, dict)
        else ""
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        prefix
                        + json.dumps(
                            result["next_action"], separators=(",", ":"), ensure_ascii=False
                        )
                        + cognition_context
                    ),
                }
            },
            separators=(",", ":"),
        )
    )


def _stop(payload: dict[str, Any]) -> None:
    api = _facade()
    identity = api._identity(payload)
    if identity is None:
        return
    answer = payload.get("last_assistant_message")
    stop_payload = {**identity, "answer": answer if isinstance(answer, str) else ""}
    result = api._post("/v1/hooks/stop", stop_payload)
    for _step in range(MAX_HOST_ADAPTER_STEPS):
        if (
            not isinstance(result, dict)
            or result.get("allow") is not False
            or result.get("terminal") is True
            or not api._run_host_adapter_step(payload, result)
        ):
            break
        result = api._post("/v1/hooks/stop", stop_payload)
    if result is None:
        print(
            json.dumps(
                {
                    "systemMessage": (
                        "Cortheon runtime became unavailable; this answer was not certified."
                    )
                },
                separators=(",", ":"),
            )
        )
        return
    if result.get("allow") is not False:
        caveat = result.get("caveat") if isinstance(result, dict) else None
        if (
            isinstance(caveat, str)
            and caveat
            and isinstance(result, dict)
            and result.get("certified") is not True
        ):
            print(json.dumps({"systemMessage": caveat}, separators=(",", ":")))
        return
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "Completion is withheld until Cortheon certifies this turn."
    if result.get("terminal") is True:
        output = {"continue": False, "stopReason": reason, "systemMessage": reason}
    else:
        output = {"decision": "block", "reason": reason}
    print(json.dumps(output, separators=(",", ":")))


def _session_end(payload: dict[str, Any]) -> None:
    api = _facade()
    identity = api._identity(payload, include_turn=False)
    if identity is not None:
        api._post("/v1/hooks/end", identity)
