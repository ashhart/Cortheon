"""Stdin dispatcher for the standalone Codex hook executable."""

from __future__ import annotations

if __package__:
    from .hook_transport import _facade
else:
    from hook_transport import _facade


def main() -> int:
    api = _facade()
    raw = api.sys.stdin.read(api.MAX_INPUT_CHARS + 1)
    if len(raw) > api.MAX_INPUT_CHARS:
        return 0
    try:
        payload = api._payload(raw)
    except (api.json.JSONDecodeError, ValueError):
        return 0
    event = payload.get("hook_event_name")
    handlers = {
        "UserPromptSubmit": api._user_prompt_submit,
        "PreToolUse": api._pre_tool_use,
        "PostToolUse": api._post_tool_use,
        "Stop": api._stop,
        "SessionEnd": api._session_end,
    }
    handler = handlers.get(event) if isinstance(event, str) else None
    if handler is not None:
        handler(payload)
    return 0
