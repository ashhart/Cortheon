"""Focused UTC timestamp parsing and ordering for parity regrading.

Regrading must be timeless: pack validity and the preregistration sequence
are evaluated at the HMAC-attested execution time recorded in the archived
artifacts, never at regrade wall-clock time. All timestamps that participate
in that ordering must be ISO-8601 with an explicit UTC (``Z`` or ``+00:00``)
offset so comparisons are never silently naive or timezone-shifted.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def parse_utc_timestamp(value: object, *, field: str) -> datetime:
    """Parse an ISO-8601 string that carries an explicit UTC offset."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or (parsed.utcoffset() or timedelta()).total_seconds() != 0:
        raise ValueError(f"{field} must carry a UTC (+00:00 / Z) offset")
    return parsed


def ordering_holds(first: object, middle: object, last: object) -> bool:
    """Return True when first <= middle <= last under UTC parsing.

    Any unparseable or non-UTC value fails closed (returns False) rather
    than raising, so callers can use it directly inside report checks.
    """

    try:
        first_at = parse_utc_timestamp(first, field="first")
        middle_at = parse_utc_timestamp(middle, field="middle")
        last_at = parse_utc_timestamp(last, field="last")
    except ValueError:
        return False
    return first_at <= middle_at <= last_at
