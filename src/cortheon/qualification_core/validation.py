"""Strict, content-free validation primitives for manifest documents."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cortheon.qualification_core.constants import (
    FORBIDDEN_CREDENTIAL_KEYS,
    MAX_MANIFEST_BYTES,
)
from cortheon.qualification_core.models import QualificationError


def _reject_unknown(
    value: dict[str, Any],
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise QualificationError(f"{location} contains unsupported field(s): {', '.join(unknown)}")


def _reject_embedded_credentials(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_CREDENTIAL_KEYS:
                raise QualificationError(
                    f"{location}.{key} is forbidden; reference an environment "
                    "variable with api_key_env instead"
                )
            _reject_embedded_credentials(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_credentials(child, f"{location}[{index}]")


def _bounded_text(
    value: Any,
    *,
    field: str,
    default: str | None = None,
    limit: int = 512,
) -> str:
    if value is None and default is not None:
        value = default
    if not isinstance(value, str) or not value or len(value) > limit or "\0" in value:
        raise QualificationError(f"{field} must be a non-empty bounded string")
    return value


def _bounded_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise QualificationError(f"{field} must be between {minimum} and {maximum}")
    return value


def _bounded_number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{field} must be a number")
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise QualificationError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def _http_url(value: Any, *, field: str, default: str) -> str:
    parsed_value = _bounded_text(value, field=field, default=default, limit=2_048)
    parsed = urlparse(parsed_value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise QualificationError(f"{field} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise QualificationError(f"{field} must not contain credentials, a query, or a fragment")
    return parsed_value.rstrip("/")


def _parse_document(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise QualificationError(f"cannot read manifest {path}: {exc}") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise QualificationError("manifest exceeds the 1 MB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError("manifest must be UTF-8") from exc
    try:
        value = tomllib.loads(text) if path.suffix.lower() == ".toml" else json.loads(text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise QualificationError(f"invalid manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError("manifest root must be an object")
    return value, hashlib.sha256(raw).hexdigest()
