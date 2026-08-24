"""Strict parser for Pi's host-visible Cortheon terminal status."""

from __future__ import annotations

from typing import Any

PI_TERMINAL_STATUS_TYPE = "cortheon-terminal-status-v1"
PI_TERMINAL_STATUS_VERSION = 1
PI_TERMINAL_REASON_MAX_CHARS = 512
PI_TERMINAL_MESSAGE_KEYS = frozenset(
    {"role", "customType", "content", "display", "details", "timestamp"}
)
PI_TERMINAL_DETAIL_KEYS = frozenset({"version", "status", "reason", "causal"})
PI_WITHHELD_HEADER = "[Cortheon withheld: completion was not certified]"
PI_WITHHELD_REASON_PREFIX = "The Cortheon investigation ended without a certified answer because "


def _pi_withheld_reason(text: Any) -> str | None:
    """Parse the one bounded withheld replacement emitted by the adapter."""

    if not isinstance(text, str):
        return None
    prefix = f"{PI_WITHHELD_HEADER}\n{PI_WITHHELD_REASON_PREFIX}"
    if not text.startswith(prefix) or not text.endswith("."):
        return None
    reason = text[len(prefix) : -1]
    if (
        not reason
        or reason != reason.strip()
        or len(reason) > PI_TERMINAL_REASON_MAX_CHARS
        or "\n" in reason
        or "\r" in reason
    ):
        return None
    return reason


def _pi_terminal_text(message: dict[str, Any]) -> str | None:
    """Return a terminal text only for Pi's exact bounded custom message."""
    if set(message) != PI_TERMINAL_MESSAGE_KEYS:
        return None
    timestamp = message["timestamp"]
    details = message["details"]
    if (
        message["role"] != "custom"
        or message["customType"] != PI_TERMINAL_STATUS_TYPE
        or message["display"] is not True
        or type(timestamp) is not int
        or not 0 <= timestamp < 2**63
        or not isinstance(details, dict)
        or set(details) != PI_TERMINAL_DETAIL_KEYS
        or type(details["version"]) is not int
        or details["version"] != PI_TERMINAL_STATUS_VERSION
        or details["status"] != "withheld"
        or type(details["causal"]) is not bool
    ):
        return None
    reason = details["reason"]
    if not isinstance(reason, str):
        return None
    content = message["content"]
    return content if _pi_withheld_reason(content) == reason else None
