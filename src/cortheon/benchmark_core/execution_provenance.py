"""Evaluator-owned host execution identity, metering, and budget control."""

from __future__ import annotations

import json
import math
import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from typing import Any, cast

from cortheon.benchmark_core.condition_execution import CONTROL_FD_ENV
from cortheon.benchmark_core.opencode_receipt import (
    OpenCodeReceipt,
    receipt_identity,
    step_metrics,
)
from cortheon.benchmark_core.pi_terminal import _pi_terminal_text


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    max_steps: int
    max_tool_calls: int
    timeout_seconds: float
    context_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ProcessCapture:
    stdout: str
    stderr: str
    returncode: int | None
    latency_seconds: float
    timed_out: bool
    budget_reason: str | None


@dataclass(frozen=True, slots=True)
class ExecutionFacts:
    provider_id: str | None
    model_id: str | None
    identity_valid: bool
    identity_reason: str | None
    identity_provenance: str
    tokens: int | None
    cost_usd: float | None
    measurements_valid: bool
    measurement_reason: str | None
    steps: int
    tool_calls: int


def _bounded_identity(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 256 else None


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _nonnegative_integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _zero_pi_abort_usage(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "input",
        "output",
        "cacheRead",
        "cacheWrite",
        "totalTokens",
        "cost",
    }:
        return False
    cost = value["cost"]
    if not isinstance(cost, dict) or set(cost) != {
        "input",
        "output",
        "cacheRead",
        "cacheWrite",
        "total",
    }:
        return False
    return all(
        type(item) is int and item == 0
        for item in (*value.values(), *cost.values())
        if item is not cost
    )


def _exact_pi_abort_envelope(
    message: dict[str, Any], prior_identity: tuple[str, str] | None
) -> bool:
    return bool(
        prior_identity is not None
        and set(message)
        == {
            "role",
            "content",
            "api",
            "provider",
            "model",
            "usage",
            "stopReason",
            "errorMessage",
            "timestamp",
        }
        and message["role"] == "assistant"
        and message["content"] == []
        and message["api"] == "openai-completions"
        and (message["provider"], message["model"]) == prior_identity
        and _zero_pi_abort_usage(message["usage"])
        and message["stopReason"] == "error"
        and message["errorMessage"] == "This operation was aborted"
        and type(message["timestamp"]) is int
    )


def _pi_facts(events: list[dict[str, Any]]) -> ExecutionFacts:
    identities: list[tuple[str, str]] = []
    tokens = 0
    cost = 0.0
    steps = 0
    measurement_error = False
    custom_terminal_seen = False
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if isinstance(message, dict) and _pi_terminal_text(message) is not None:
            custom_terminal_seen = True
            continue
        if not isinstance(message, dict) or message.get("role") != "assistant":
            custom_terminal_seen = False
            continue
        prior_identity = identities[-1] if identities and identities[-1] != ("", "") else None
        if custom_terminal_seen and _exact_pi_abort_envelope(message, prior_identity):
            custom_terminal_seen = False
            continue
        custom_terminal_seen = False
        steps += 1
        provider = _bounded_identity(message.get("provider"))
        model = _bounded_identity(message.get("responseModel", message.get("model")))
        if provider is None or model is None:
            identities.append(("", ""))
        else:
            identities.append((provider, model))
        usage = message.get("usage")
        if not isinstance(usage, dict):
            measurement_error = True
            continue
        message_tokens = _nonnegative_integer(usage.get("totalTokens"))
        raw_cost = usage.get("cost")
        message_cost = (
            _nonnegative_number(raw_cost.get("total")) if isinstance(raw_cost, dict) else None
        )
        if message_tokens is None or message_cost is None:
            measurement_error = True
            continue
        tokens += message_tokens
        cost += message_cost
    unique = set(identities)
    identity_valid = len(unique) == 1 and ("", "") not in unique
    provider_id, model_id = next(iter(unique)) if identity_valid else (None, None)
    if not identities:
        identity_reason = "missing_assistant_identity"
    elif not identity_valid:
        identity_reason = "missing_or_mixed_assistant_identity"
    else:
        identity_reason = None
    measurements_valid = bool(identities) and not measurement_error
    return ExecutionFacts(
        provider_id,
        model_id,
        identity_valid,
        identity_reason,
        "pi_message_end" if identity_valid else "unavailable",
        tokens if measurements_valid else None,
        cost if measurements_valid else None,
        measurements_valid,
        None if measurements_valid else "missing_or_invalid_pi_usage",
        steps,
        sum(event.get("type") == "tool_execution_start" for event in events),
    )


def _opencode_facts(
    events: list[dict[str, Any]], receipt: OpenCodeReceipt | None
) -> ExecutionFacts:
    event_metrics, measurement_error = step_metrics(events)
    tokens = 0
    cost = 0.0
    for token_metrics, item_cost, _reason in event_metrics:
        tokens += sum(token_metrics.values())
        cost += item_cost
    provider, model_id, identity_reason = receipt_identity(receipt, event_metrics)
    measurements_valid = bool(event_metrics) and not measurement_error and identity_reason is None
    identity_valid = provider is not None and model_id is not None and identity_reason is None
    return ExecutionFacts(
        provider,
        model_id,
        identity_valid,
        identity_reason,
        "opencode_sanitized_export" if identity_valid else "unavailable",
        tokens if measurements_valid else None,
        cost if measurements_valid else None,
        measurements_valid,
        None if measurements_valid else "missing_or_invalid_opencode_usage",
        len(event_metrics),
        sum(event.get("type") == "tool_use" for event in events),
    )


def _generic_mcp_facts(events: list[dict[str, Any]]) -> ExecutionFacts:
    from cortheon.benchmark_core.generic_mcp_diagnostics import transcript_diagnostic
    from cortheon.benchmark_core.generic_mcp_validation import validate_transcript

    structurally_valid = validate_transcript(events)
    start = events[0] if events and isinstance(events[0], dict) else {}
    messages = [event for event in events if event.get("type") == "message"]
    valid = structurally_valid and bool(messages)
    tokens = sum(event["tokens"] for event in messages) if valid else None
    costs = [event.get("cost_usd") for event in messages]
    cost = (
        None if not valid or any(item is None for item in costs) else sum(cast(list[float], costs))
    )
    return ExecutionFacts(
        start.get("provider_requested") if valid else None,
        start.get("model_requested") if valid else None,
        valid,
        None if valid else transcript_diagnostic(events),
        "generic_mcp_evaluator_transcript" if valid else "unavailable",
        tokens,
        cost,
        valid and tokens is not None,
        None if valid and tokens is not None else transcript_diagnostic(events),
        len(messages),
        sum(event.get("type") == "tool_request" for event in events),
    )


def execution_facts(
    events: list[dict[str, Any]],
    *,
    host: str,
    opencode_receipt: OpenCodeReceipt | None = None,
) -> ExecutionFacts:
    if host == "pi":
        return _pi_facts(events)
    if host == "opencode":
        return _opencode_facts(events, opencode_receipt)
    if host == "generic_mcp":
        return _generic_mcp_facts(events)
    raise ValueError(f"unsupported execution host: {host!r}")


def _budget_signal(event: dict[str, Any], host: str, policy: ExecutionPolicy) -> str | None:
    if host == "pi":
        message = event.get("message")
        if (
            event.get("type") == "message_end"
            and isinstance(message, dict)
            and message.get("role") == "assistant"
        ):
            reason = message.get("stopReason")
            content = message.get("content")
            has_tool_call = isinstance(content, list) and any(
                isinstance(item, dict) and item.get("type") in {"toolCall", "tool_call", "tool_use"}
                for item in content
            )
            return "max_steps" if reason in {"toolUse", "tool_calls"} or has_tool_call else None
        return None
    part = event.get("part")
    if event.get("type") == "step_finish" and isinstance(part, dict):
        return "max_steps" if part.get("reason") in {"tool-calls", "tool_calls"} else None
    return None


def execute_host_process(
    command: list[str],
    *,
    cwd: Any,
    env: dict[str, str],
    host: str,
    policy: ExecutionPolicy,
    control_payload: bytes | None = None,
    stdin_payload: bytes | None = None,
) -> ProcessCapture:
    """Stream one JSONL host process and terminate at evaluator-owned bounds."""

    started = time.perf_counter()
    read_fd: int | None = None
    write_fd: int | None = None
    if control_payload is not None:
        if os.name != "posix":
            raise RuntimeError("evaluator control FD requires a POSIX host")
        read_fd, write_fd = os.pipe()
    child_env = env if read_fd is None else {**env, CONTROL_FD_ENV: str(read_fd)}
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_payload is not None else None,
            pass_fds=(read_fd,) if read_fd is not None else (),
        )
        if read_fd is not None:
            os.close(read_fd)
            read_fd = None
        if write_fd is not None:
            assert control_payload is not None
            view = memoryview(control_payload)
            while view:
                try:
                    written = os.write(write_fd, view)
                except BrokenPipeError:
                    break
                view = view[written:]
        if stdin_payload is not None:
            assert process.stdin is not None
            try:
                process.stdin.write(stdin_payload)
                process.stdin.close()
            except BrokenPipeError:
                pass
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    pending_stdout = bytearray()
    steps = tools = 0
    timed_out = False
    budget_reason: str | None = None
    stop_deadline: float | None = None
    forced_drain_deadline: float | None = None
    pi_step_cap_pending = False

    def request_stop() -> None:
        nonlocal stop_deadline
        if stop_deadline is not None or process.poll() is not None:
            return
        # Pi's print mode handles SIGTERM by disposing its runtime, which
        # emits session_shutdown to extensions. Give that bounded cleanup a
        # chance to abandon leases and emit a host-owned terminal receipt.
        process.terminate()
        stop_deadline = time.perf_counter() + 1.0

    while selector.get_map():
        now = time.perf_counter()
        if stop_deadline is None:
            remaining = policy.timeout_seconds - (now - started)
            if remaining <= 0:
                timed_out = True
                request_stop()
                continue
            wait_seconds = min(0.1, remaining)
        else:
            if process.poll() is None and now >= stop_deadline:
                process.kill()
                forced_drain_deadline = now + 0.25
            if forced_drain_deadline is not None and now >= forced_drain_deadline:
                break
            wait_seconds = 0.05
        for key, _mask in selector.select(wait_seconds):
            chunk = os.read(cast(Any, key.fileobj).fileno(), 65_536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            target = buffers[key.data]
            target.extend(chunk)
            if len(target) > 10_000_000:
                if budget_reason is None:
                    budget_reason = "output_bytes"
                request_stop()
            if key.data != "stdout":
                continue
            pending_stdout.extend(chunk)
            while b"\n" in pending_stdout:
                raw_line, _, remainder = pending_stdout.partition(b"\n")
                pending_stdout = bytearray(remainder)
                try:
                    event = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                if host == "pi":
                    message = event.get("message")
                    assistant_ended = bool(
                        event.get("type") == "message_end"
                        and isinstance(message, dict)
                        and message.get("role") == "assistant"
                    )
                    steps += int(assistant_ended)
                    tools += int(event.get("type") == "tool_execution_start")
                    continuation = _budget_signal(event, host, policy)
                    if (
                        assistant_ended
                        and steps == policy.max_steps
                        and continuation == "max_steps"
                    ):
                        # The final permitted assistant turn owns its tool
                        # boundary. Let those calls reach Pi/Cortheon, then
                        # stop at Pi's authoritative turn_end boundary.
                        pi_step_cap_pending = True
                    if event.get("type") == "turn_end" and pi_step_cap_pending:
                        if budget_reason is None:
                            budget_reason = "max_steps"
                        request_stop()
                elif host == "generic_mcp":
                    steps += int(event.get("type") == "message")
                    tools += int(event.get("type") == "tool_request")
                    continuation = None
                else:
                    steps += int(event.get("type") == "step_finish")
                    tools += int(event.get("type") == "tool_use")
                    continuation = _budget_signal(event, host, policy)
                if steps > policy.max_steps or (
                    host != "pi" and steps == policy.max_steps and continuation == "max_steps"
                ):
                    if budget_reason is None:
                        budget_reason = "max_steps"
                    request_stop()
                elif tools > policy.max_tool_calls:
                    if budget_reason is None:
                        budget_reason = "max_tool_calls"
                    request_stop()
        if process.poll() is not None and not selector.get_map():
            break
    if process.poll() is None:
        request_stop()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=5)
    if host == "pi" and pi_step_cap_pending and budget_reason is None:
        budget_reason = "max_steps"
    return ProcessCapture(
        bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        process.returncode,
        time.perf_counter() - started,
        timed_out,
        budget_reason,
    )
