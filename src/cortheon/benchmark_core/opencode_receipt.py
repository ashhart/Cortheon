"""Evaluator-captured, sanitized OpenCode execution receipts."""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class OpenCodeReceipt:
    data: dict[str, Any] | None
    session_id: str | None
    host_version: str | None
    error: str | None


def _identity(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 256 else None


def _session_identity(value: Any) -> str | None:
    if not isinstance(value, str) or re.fullmatch(r"ses_[A-Za-z0-9]{1,128}", value) is None:
        return None
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _token_metrics(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict) or not isinstance(value.get("cache"), dict):
        return None
    cache = value["cache"]
    metrics = {
        "input": _integer(value.get("input")),
        "output": _integer(value.get("output")),
        "reasoning": _integer(value.get("reasoning")),
        "cache_read": _integer(cache.get("read")),
        "cache_write": _integer(cache.get("write")),
    }
    if any(item is None for item in metrics.values()):
        return None
    return {key: cast(int, item) for key, item in metrics.items()}


def step_metrics(
    events: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, int], float, str]], bool]:
    metrics: list[tuple[dict[str, int], float, str]] = []
    invalid = False
    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part")
        tokens = _token_metrics(part.get("tokens")) if isinstance(part, dict) else None
        cost = _number(part.get("cost")) if isinstance(part, dict) else None
        reason = _identity(part.get("reason")) if isinstance(part, dict) else None
        if tokens is None or cost is None or reason is None:
            invalid = True
            continue
        metrics.append((tokens, cost, reason))
    return metrics, invalid


def _session_id(events: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    recognized = {"step_start", "text", "tool_use", "step_finish"}
    observed: list[str] = []
    active = False
    for event in events:
        event_type = event.get("type")
        if event_type not in recognized:
            continue
        session_id = _session_identity(event.get("sessionID"))
        if session_id is None:
            return None, "missing_top_level_session_id"
        part = event.get("part")
        if isinstance(part, dict) and "sessionID" in part and part.get("sessionID") != session_id:
            return None, "nested_session_id_mismatch"
        observed.append(session_id)
        if event_type == "step_start":
            if active:
                return None, "invalid_session_event_order"
            active = True
        elif event_type == "step_finish":
            if not active:
                return None, "invalid_session_event_order"
            active = False
        elif not active:
            return None, "invalid_session_event_order"
    if not observed:
        return None, "missing_session_events"
    if active:
        return None, "incomplete_session_event_order"
    if len(set(observed)) != 1:
        return None, "mixed_top_level_session_ids"
    return observed[0], None


def _bounded_command(
    command: list[str], *, cwd: Any, env: dict[str, str], timeout: float = 15.0
) -> tuple[int | None, bytes, str | None]:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None, b"", "receipt_command_unavailable"
    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    error: str | None = None
    while selector.get_map():
        if time.monotonic() >= deadline:
            error = "receipt_command_timeout"
            process.kill()
            break
        for key, _mask in selector.select(0.1):
            chunk = os.read(cast(Any, key.fileobj).fileno(), 65_536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            buffers[key.data].extend(chunk)
            if len(buffers[key.data]) > 5_000_000:
                error = "receipt_command_output_too_large"
                process.kill()
                break
        if error is not None:
            break
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return None, bytes(buffers["stdout"]), "receipt_command_did_not_exit"
    return process.returncode, bytes(buffers["stdout"]), error


def capture_opencode_receipt(
    executable: str,
    events: list[dict[str, Any]],
    *,
    cwd: Any,
    env: dict[str, str],
) -> OpenCodeReceipt:
    """Export one sanitized receipt in memory before the isolated config closes."""

    session_id, session_error = _session_id(events)
    if session_error is not None or session_id is None:
        return OpenCodeReceipt(None, None, None, session_error)
    version_rc, version_out, version_error = _bounded_command(
        [executable, "--version"], cwd=cwd, env=env
    )
    try:
        version = _identity(version_out.decode("utf-8", errors="strict").strip())
    except UnicodeDecodeError:
        version = None
    if version_error is not None or version_rc != 0 or version is None:
        return OpenCodeReceipt(None, session_id, None, "opencode_version_unavailable")
    export_rc, output, export_error = _bounded_command(
        [executable, "export", session_id, "--sanitize", "--pure"],
        cwd=cwd,
        env=env,
    )
    if export_error is not None or export_rc != 0:
        return OpenCodeReceipt(None, session_id, version, export_error or "sanitized_export_failed")
    try:
        data = json.loads(output.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return OpenCodeReceipt(None, session_id, version, "malformed_sanitized_export")
    if not isinstance(data, dict):
        return OpenCodeReceipt(None, session_id, version, "malformed_sanitized_export")
    return OpenCodeReceipt(data, session_id, version, None)


def receipt_identity(
    receipt: OpenCodeReceipt | None,
    event_metrics: list[tuple[dict[str, int], float, str]],
) -> tuple[str | None, str | None, str | None]:
    if receipt is None or receipt.error is not None or receipt.data is None:
        return None, None, receipt.error if receipt is not None else "missing_sanitized_export"
    if _session_identity(receipt.session_id) is None or _identity(receipt.host_version) is None:
        return None, None, "malformed_sanitized_export"
    info = receipt.data.get("info")
    messages = receipt.data.get("messages")
    if (
        not isinstance(info, dict)
        or not isinstance(messages, list)
        or not 0 < len(messages) <= 2_048
    ):
        return None, None, "malformed_sanitized_export"
    if info.get("id") != receipt.session_id or info.get("version") != receipt.host_version:
        return None, None, "export_session_or_version_mismatch"
    model = info.get("model")
    provider = _identity(model.get("providerID")) if isinstance(model, dict) else None
    model_id = _identity(model.get("id")) if isinstance(model, dict) else None
    assistants: list[tuple[dict[str, int], float, str]] = []
    for message in messages:
        message_info = message.get("info") if isinstance(message, dict) else None
        if (
            not isinstance(message_info, dict)
            or message_info.get("sessionID") != receipt.session_id
        ):
            return None, None, "export_message_session_mismatch"
        if message_info.get("role") != "assistant":
            continue
        if (
            _identity(message_info.get("providerID")) != provider
            or _identity(message_info.get("modelID")) != model_id
        ):
            return None, None, "export_assistant_identity_mismatch"
        tokens = _token_metrics(message_info.get("tokens"))
        cost = _number(message_info.get("cost"))
        finish = _identity(message_info.get("finish"))
        if tokens is None or cost is None or finish is None:
            return None, None, "invalid_export_assistant_measurements"
        assistants.append((tokens, cost, finish))
    if provider is None or model_id is None or not assistants or assistants != event_metrics:
        return None, None, "export_event_reconciliation_failed"
    totals = {key: sum(item[0][key] for item in assistants) for key in assistants[0][0]}
    session_cost = _number(info.get("cost"))
    if _token_metrics(info.get("tokens")) != totals or session_cost is None:
        return None, None, "export_session_totals_mismatch"
    if not math.isclose(session_cost, sum(item[1] for item in assistants), abs_tol=1e-12):
        return None, None, "export_session_totals_mismatch"
    return provider, model_id, None
