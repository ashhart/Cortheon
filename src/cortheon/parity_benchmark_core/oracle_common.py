"""Bounded parsing and exact-value helpers for private structured oracles."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any


def answer_object(answer: str) -> dict[str, Any] | None:
    if not isinstance(answer, str) or not answer.strip() or len(answer.encode("utf-8")) > 65_536:
        return None
    fences = re.findall(r"```json\s*\n(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    candidate = fences[0] if len(fences) == 1 else answer if not fences else ""
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def closed_object(value: Any, required: set[str], optional: set[str] | None = None) -> bool:
    return isinstance(value, dict) and set(value) == required | (set(value) & (optional or set()))


def exact_records(value: Any, fields: tuple[str, ...]) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or len(value) > 64:
        return None
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != set(fields):
            return None
        records.append(item)
    return records


def record_map(value: Any, fields: tuple[str, ...], key: str = "id") -> dict[str, dict] | None:
    records = exact_records(value, fields)
    if records is None:
        return None
    mapped: dict[str, dict[str, Any]] = {}
    for item in records:
        identifier = item.get(key)
        if not isinstance(identifier, str) or identifier in mapped:
            return None
        mapped[identifier] = item
    return mapped


def string_set(value: Any, *, minimum: int = 0, maximum: int = 64) -> set[str] | None:
    if (
        not isinstance(value, list)
        or not minimum <= len(value) <= maximum
        or any(not isinstance(item, str) or not item or len(item) > 200 for item in value)
    ):
        return None
    result = set(value)
    return result if len(result) == len(value) else None


def decimal_value(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def evidence_digest(task_class: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"task_class": task_class, "answer": payload},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
