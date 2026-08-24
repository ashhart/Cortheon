"""Event-stream parsing helpers for the terminal tests.

Extracts Cortheon custom entries and host-visible terminal-status
messages from Pi run output (JSON-lines stdout) or parsed event lists.
"""

from __future__ import annotations

import json
from typing import Any

from pi_terminal_constants import TERMINAL_STATUS_TYPE


def custom_entry_data(completed: Any, custom_type: str) -> list[dict[str, Any]]:
    events = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    data: list[dict[str, Any]] = []
    for line in events:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry = event.get("entry", {})
        if event.get("type") == "entry_appended" and entry.get("customType") == custom_type:
            data.append(entry.get("data", {}))
    return data


def terminal_status_messages(events_or_completed: Any) -> list[dict[str, Any]]:
    """Every host-visible Cortheon terminal-status custom message, in order.

    Accepts a CompletedProcess (JSON-lines stdout) or an already-parsed
    event list (RPC sessions). Pi delivers the extension's sendMessage
    custom message as a message_end event whose message role is "custom".
    """
    if isinstance(events_or_completed, list):
        events = events_or_completed
    else:
        events = [
            json.loads(line)
            for line in events_or_completed.stdout.splitlines()
            if line.strip().startswith("{")
        ]
    messages = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message", {})
        if (
            isinstance(message, dict)
            and message.get("role") == "custom"
            and message.get("customType") == TERMINAL_STATUS_TYPE
        ):
            messages.append(message)
    return messages
