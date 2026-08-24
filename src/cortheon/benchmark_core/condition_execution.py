"""Evaluator-owned setup and attestation for qualification conditions."""

from __future__ import annotations

import json
import os
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

CONTROL_SCHEMA_VERSION = 1
CONTROL_FD_ENV = "CORTHEON_CONTROL_FD"
CONTROL_LIMIT = 16_384
CONTROL_ENV_KEYS = (
    CONTROL_FD_ENV,
    "CORTHEON_EVALUATOR_PROFILE",
    "CORTHEON_COGNITIVE_TOKEN",
    "CORTHEON_EVALUATOR_MAX_STEPS",
    "CORTHEON_AUTO_ENABLE",
    "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE",
    "CORTHEON_MAX_HOST_TOOL_CALLS",
)


@dataclass(frozen=True, slots=True)
class AppliedCondition:
    """One evaluator-selected condition bound to a fresh execution nonce."""

    profile: dict[str, Any]
    nonce: str


def prepare_condition(
    environment: dict[str, str],
    profile: dict[str, Any] | None,
    *,
    treatment: bool,
) -> AppliedCondition | None:
    """Bind a profile to one process without allowing ambient reuse."""

    for key in CONTROL_ENV_KEYS:
        environment.pop(key, None)
    if profile is None:
        return None
    nonce = secrets.token_hex(16)
    applied = {**profile, "nonce": nonce}
    return AppliedCondition(profile=applied, nonce=nonce)


def condition_control_payload(
    applied: AppliedCondition | None,
    *,
    token: str,
    host: str,
    treatment: bool,
    max_steps: int,
    max_host_tool_calls: int,
) -> bytes | None:
    """Encode one bounded evaluator-to-adapter control message."""

    if not treatment:
        return None
    if type(max_steps) is not int or not 1 <= max_steps <= 1_024:
        raise ValueError("evaluator step limit must be an integer from 1 to 1024")
    if type(max_host_tool_calls) is not int or not 1 <= max_host_tool_calls <= 64:
        raise ValueError("evaluator tool-call limit must be an integer from 1 to 64")
    payload = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "evaluation_profile": applied.profile if applied else None,
        "cognitive_token": token,
        "evaluator_max_steps": max_steps if host == "pi" else None,
        "auto_enable": host == "pi",
        "benchmark_capture_candidate": host == "pi",
        "max_host_tool_calls": max_host_tool_calls,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > CONTROL_LIMIT:
        raise ValueError("evaluator control payload exceeds its byte limit")
    return encoded


def fetch_runtime_receipt(
    url: str,
    nonce: str,
    *,
    token: str = "",
) -> dict[str, Any] | None:
    """Consume the memory-only runtime receipt for one evaluator nonce."""

    body = json.dumps({"nonce": nonce}, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/evaluation-receipt",
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def condition_receipt_valid(
    applied: AppliedCondition | None,
    receipt: dict[str, Any] | None,
    runtime_delta: dict[str, int] | None,
    *,
    treatment: bool,
    host: str,
) -> bool | None:
    """Validate observed application, including bare's absence of runtime use."""

    if applied is None:
        return None
    if not treatment:
        return bool(
            receipt is None
            and runtime_delta is not None
            and runtime_delta
            and all(value == 0 for value in runtime_delta.values())
        )
    config = applied.profile["config"]
    fixed = {
        "schema_version": 1,
        "config_sha256": applied.profile["config_sha256"],
        "implementation_sha256": applied.profile["implementation_sha256"],
        "intercepts_final": config["intercepts_final"],
        "cleanup_before_answer": config["cleanup_before_answer"],
        "runtime_profile_received": True,
        "adapter_receipt": {
            "schema_version": 1,
            "host": host,
            "control_transport": "fd",
            "config_sha256": applied.profile["config_sha256"],
            "nonce": applied.nonce,
            "operators": config["operators"],
        },
    }
    if not isinstance(receipt, dict) or set(receipt) != {*fixed, "operator_counts"}:
        return False
    if any(receipt.get(key) != value for key, value in fixed.items()):
        return False
    if runtime_delta is None:
        return False
    terminals = sum(
        runtime_delta.get(key, 0)
        for key in (
            "sessions_completed",
            "sessions_evidence_closed",
            "sessions_abandoned",
        )
    )
    if config["cleanup_before_answer"]:
        lifecycle_valid = bool(
            runtime_delta.get("sessions_started") == 1
            and runtime_delta.get("observations_accepted", 0) >= 1
            and runtime_delta.get("sessions_abandoned") == 1
            and runtime_delta.get("sessions_completed") == 0
            and runtime_delta.get("sessions_evidence_closed") == 0
            and runtime_delta.get("completion_withheld") == 0
        )
    else:
        lifecycle_valid = bool(runtime_delta.get("sessions_started") == 1 and terminals == 1)
    if not lifecycle_valid:
        return False
    counts = receipt.get("operator_counts")
    operators = config["operators"]
    return bool(
        isinstance(counts, dict)
        and set(counts) == set(operators)
        and all(
            type(counts[key]) is int
            and 0 <= counts[key] <= 10_000
            and (operators[key] is True or counts[key] == 0)
            for key in operators
        )
    )


def condition_token() -> str:
    """Read the runtime token without ever placing it in report state."""

    return os.environ.get("CORTHEON_COGNITIVE_TOKEN", "")
