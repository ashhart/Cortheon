"""Text normalization and bounded scalar parsing primitives."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from cortheon.sanitize import scan_text

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{2,}")


_SPACE_RE = re.compile(r"\s+")


_LOOKUP_TARGET_RE = re.compile(
    r"\b(?:imports?|contains?|includes?|mentions?|references?|uses?|defines?|has|"
    r"maps?\s+to)\s+"
    r"[`'\"]?([A-Za-z_](?:[A-Za-z0-9_.:-]*[A-Za-z0-9_:-])?)",
    flags=re.IGNORECASE,
)


_LOOKUP_PHRASE_RE = re.compile(
    r"\bcontains?\s+(?:the\s+)?phrase\s+[`'\"]?(.{1,300}?)[`'\"]?(?:[?.]|$)",
    flags=re.IGNORECASE,
)


_LOOKUP_STOP_TARGETS = frozenset(
    {"a", "an", "if", "it", "present", "that", "the", "this", "whether"}
)


def _text(
    value: Any,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{label} exceeds the {maximum}-character limit")
    return cleaned


def _optional_text(value: Any, label: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum=maximum)


def _optional_url(value: Any, label: str) -> str | None:
    if value is None:
        return None
    normalized = _text(value, label, maximum=2_000)
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not contain credentials")
    return normalized


def _optional_timestamp(
    value: Any,
    label: str,
    *,
    allow_date: bool = False,
) -> str | None:
    if value is None:
        return None
    normalized = _text(value, label, maximum=64)
    candidate = normalized
    if allow_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
        candidate += "T00:00:00+00:00"
    elif candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _string_list(
    values: Iterable[Any],
    label: str,
    *,
    maximum_items: int,
    maximum_chars: int,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{label} must be an array of strings")
    result: list[str] = []
    total = 0
    for value in values:
        if len(result) >= maximum_items:
            raise ValueError(f"{label} may contain at most {maximum_items} items")
        item = _text(value, label, maximum=maximum_chars)
        total += len(item)
        if total > maximum_chars:
            raise ValueError(f"{label} exceeds the {maximum_chars}-character limit")
        if item not in result:
            result.append(item)
    return result


def _normalized(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip().lower())


def _safe_public_label(value: str | None) -> str | None:
    """Keep instruction-shaped metadata out of model-visible context."""

    if value is None:
        return None
    scan = scan_text(value)
    return scan.clean_text if scan.clean_text else "[quarantined source label]"


def _lookup_target_match(value: str) -> re.Match[str] | None:
    matches = [
        match
        for match in _LOOKUP_TARGET_RE.finditer(value)
        if match.group(1).lower() not in _LOOKUP_STOP_TARGETS
    ]
    return matches[-1] if matches else None


def _lookup_phrase_target(value: str) -> str | None:
    match = _LOOKUP_PHRASE_RE.search(value)
    if match is None:
        return None
    target = match.group(1).strip().strip("`'\"").strip()
    return target or None
