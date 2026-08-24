"""Closed execution values for repository-only operator-lift runs."""

from __future__ import annotations

import math
import re
import urllib.parse
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    base_url: str
    provider_id: str
    model_id: str
    api_key: str
    timeout_seconds: float = 120.0
    context_tokens: int = 16_384
    output_tokens: int = 2_048
    max_steps: int = 8
    max_tool_calls: int = 12

    def validate(self) -> None:
        endpoint = urllib.parse.urlsplit(self.base_url)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or endpoint.path.rstrip("/") != "/v1"
        ):
            raise ValueError("operator-lift endpoint must be a credential-free loopback /v1 URL")
        for value, label in (
            (self.provider_id, "provider_id"),
            (self.model_id, "model_id"),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{label} is invalid")
        if not isinstance(self.api_key, str) or len(self.api_key) > 8_192:
            raise ValueError("api_key is invalid")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 1 <= self.timeout_seconds <= 300
        ):
            raise ValueError("timeout_seconds is invalid")
        for value, lower, upper, label in (
            (self.context_tokens, 1_024, 1_000_000, "context_tokens"),
            (self.output_tokens, 128, 100_000, "output_tokens"),
            (self.max_steps, 1, 32, "max_steps"),
            (self.max_tool_calls, 1, 128, "max_tool_calls"),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{label} is invalid")

    def public_identity(self) -> dict[str, str | int | float]:
        self.validate()
        return {
            "endpoint_scope": "loopback_openai_v1",
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "timeout_seconds": float(self.timeout_seconds),
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass(frozen=True, slots=True)
class ScheduledCell:
    sequence: int
    case_id: str
    operator: str
    condition_id: str
    repeat: int

    @property
    def cell_id(self) -> str:
        return f"{self.case_id}--{self.condition_id}--r{self.repeat}"


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    submission: dict[str, object]
    summary: dict[str, object]
