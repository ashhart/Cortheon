"""Value coercion shared by every gate.

A report is untrusted input: a gate must read a missing, mistyped, or
non-finite field as *absent* rather than crashing or silently coercing it into
a passing value. These helpers are the single place that decision is made.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _nested_number(payload: dict[str, Any], outer: str, inner: str) -> float | None:
    return _number(_mapping(payload.get(outer)).get(inner))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _after(left: str, right: str) -> bool:
    try:
        return datetime.fromisoformat(left.replace("Z", "+00:00")) > datetime.fromisoformat(
            right.replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False


def _stable_seed(*values: str) -> int:
    return int.from_bytes(
        hashlib.sha256(":".join(values).encode("utf-8")).digest()[:8],
        "big",
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
