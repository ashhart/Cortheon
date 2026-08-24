"""Provider configs, event parsing, and telemetry helpers for the runners."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.models import (
    BenchmarkCase,
    DiagnosticCase,
    JoinCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    SemanticCase,
)
from cortheon.benchmark_core.outcomes import EvaluationOutcome, is_exact_terminal_success
from cortheon.benchmark_core.pi_terminal import (
    PI_TERMINAL_REASON_MAX_CHARS as PI_TERMINAL_REASON_MAX_CHARS,
)
from cortheon.benchmark_core.pi_terminal import (
    PI_TERMINAL_STATUS_TYPE as PI_TERMINAL_STATUS_TYPE,
)
from cortheon.benchmark_core.pi_terminal import (
    _pi_terminal_text,
)
from cortheon.benchmark_core.transport_outcomes import (
    CANDIDATE_ENTRY_TYPE as CANDIDATE_ENTRY_TYPE,
)
from cortheon.benchmark_core.transport_outcomes import (
    CANDIDATE_MAX_CHARS as CANDIDATE_MAX_CHARS,
)
from cortheon.benchmark_core.transport_outcomes import (
    captured_candidate,
)


def _provider_config(args: argparse.Namespace, *, treatment: bool) -> str:
    options: dict[str, Any] = {"baseURL": args.base_url}
    if args.api_key:
        options["apiKey"] = args.api_key
    provider = {
        "options": options,
        "models": {
            args.model_id: {
                "name": args.model_id,
                "limit": {
                    "context": args.context_tokens,
                    "output": args.output_tokens,
                },
                # Greedy decoding for BOTH paired arms: sampling noise is not
                # part of the substrate-versus-baseline comparison.
                "options": {"temperature": 0},
            }
        },
    }
    plugin: list[str] = []
    if treatment:
        override = getattr(args, "evaluation_plugin_path", None)
        plugin_path = (
            Path(override)
            if override is not None
            else Path(str(files("cortheon").joinpath("opencode_plugin.js")))
        )
        plugin = [plugin_path.resolve().as_uri()]
    return json.dumps(
        {
            "provider": {args.provider: provider},
            "model": f"{args.provider}/{args.model_id}",
            "small_model": f"{args.provider}/{args.model_id}",
            "agent": {
                "build": {
                    "steps": int(getattr(args, "max_steps", 4)),
                }
            },
            "plugin": plugin,
            "mcp": {},
        },
        separators=(",", ":"),
    )


def _pi_provider_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build an isolated Pi provider catalog for a paired benchmark."""

    return {
        "providers": {
            args.provider: {
                "baseUrl": args.base_url,
                "api": "openai-completions",
                "apiKey": args.api_key or "not-required",
                "authHeader": bool(args.api_key),
                "models": [
                    {
                        "id": args.model_id,
                        "name": args.model_id,
                        "reasoning": bool(args.reasoning),
                        "input": ["text"],
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                        "contextWindow": args.context_tokens,
                        "maxTokens": args.output_tokens,
                    }
                ],
            }
        }
    }


def _captured_candidate(events: Iterable[dict[str, Any]]) -> str | None:
    return captured_candidate(events)


# Bounded benchmark-only code for why a causal candidate was not certified.
# These values match pi_core/candidate_capture.ts.
STAGE_ENTRY_TYPE = "cortheon-benchmark-causal-stage-v1"
STAGE_ENTRY_VERSION = 1
STAGE_ENTRY_STAGE = "causal_synthesis"
STAGE_ENTRY_DATA_KEYS = frozenset({"version", "stage", "reason"})
CAUSAL_STAGE_REASONS = frozenset(
    {
        "deliberation_empty",
        "validation_failed",
        "mapping_failed",
        "transport_failed",
        "runtime_withheld",
        "terminated_before_deliberation",
    }
)


def _captured_stage_reason(events: Iterable[dict[str, Any]]) -> str | None:
    """Return the last bounded causal stage code; malformed latest wins."""

    last_entry: dict[str, Any] | None = None
    for event in events:
        if event.get("type") != "entry_appended":
            continue
        entry = event.get("entry")
        if not isinstance(entry, dict) or entry.get("type") != "custom":
            continue
        if entry.get("customType") != STAGE_ENTRY_TYPE:
            continue
        last_entry = entry
    if last_entry is None:
        return None
    if not isinstance(last_entry.get("id"), str) or not isinstance(
        last_entry.get("timestamp"), str
    ):
        return None
    data = last_entry.get("data")
    if not isinstance(data, dict) or set(data) != STAGE_ENTRY_DATA_KEYS:
        return None
    if type(data["version"]) is not int or data["version"] != STAGE_ENTRY_VERSION:
        return None
    if data["stage"] != STAGE_ENTRY_STAGE:
        return None
    reason = data["reason"]
    if not isinstance(reason, str) or reason not in CAUSAL_STAGE_REASONS:
        return None
    return reason


def _final_text(
    events: Iterable[dict[str, Any]],
    *,
    host: str = "opencode",
) -> str:
    if host == "pi":
        final = ""
        for event in events:
            if event.get("type") != "message_end":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            terminal = _pi_terminal_text(message)
            if terminal is not None:
                final = terminal
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            text = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
            if text:
                final = text
        return final
    texts = [
        str(event.get("part", {}).get("text", ""))
        for event in events
        if event.get("type") == "text"
    ]
    return texts[-1].strip() if texts else ""


# Result texts the Cortheon Pi adapter substitutes when it blocks a tool
# call (its block results also carry terminate: true).
PI_BLOCKED_TOOL_HINTS = (
    "Cortheon has all the evidence",
    "Cortheon already certified",
    "Cortheon reached its host tool budget",
)
# Match Pi's full unavailable-tool response, not host errors that happen to
# contain "not found".
PI_UNAVAILABLE_TOOL_PATTERN = re.compile(r"^Tool \S+ not found$")


def _pi_result_text(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    return " ".join(
        str(block.get("text", "")) for block in result.get("content", []) if isinstance(block, dict)
    )


def _pi_tool_statistics(
    events: list[dict[str, Any]],
) -> tuple[int, int, int, int, int]:
    """Separate Pi attempts, executions, blocks, unavailable tools and errors."""
    attempts = sum(event.get("type") == "tool_execution_start" for event in events)
    executed = blocked = unavailable = errors = 0
    for event in events:
        if event.get("type") != "tool_execution_end":
            continue
        result = event.get("result")
        text = _pi_result_text(result)
        if PI_UNAVAILABLE_TOOL_PATTERN.match(text.strip()):
            unavailable += 1
        elif (isinstance(result, dict) and result.get("terminate") is True) or text.startswith(
            PI_BLOCKED_TOOL_HINTS
        ):
            blocked += 1
        else:
            executed += 1
            if event.get("isError") is True:
                errors += 1
    return attempts, executed, blocked, unavailable, errors


def _event_statistics(
    events: list[dict[str, Any]],
    *,
    host: str,
) -> tuple[int, int, int, int, int, int]:
    """Return (tokens, tool_calls, tool_errors, host_tool_executions,
    blocked_tool_calls, unavailable_tool_calls). Tool calls are model attempts.
    """
    if host == "pi":
        attempts, executed, blocked, unavailable, errors = _pi_tool_statistics(events)
        tokens = sum(
            int(usage.get("totalTokens", 0) or 0)
            for event in events
            if event.get("type") == "message_end"
            and isinstance(event.get("message"), dict)
            and isinstance((usage := event["message"].get("usage")), dict)
        )
        return tokens, attempts, errors, executed, blocked, unavailable

    if host == "generic_mcp":
        messages = [event for event in events if event.get("type") == "message"]
        results = [event for event in events if event.get("type") == "tool_result"]
        host_results = [event for event in results if event.get("origin") == "host"]
        tokens = sum(
            event.get("tokens", 0) for event in messages if type(event.get("tokens")) is int
        )
        errors = sum(event.get("status") in {"error", "failed"} for event in host_results)
        return tokens, len(results), errors, len(host_results), 0, 0

    tool_events = [event for event in events if event.get("type") == "tool_use"]
    tool_errors = 0
    for event in tool_events:
        part = event.get("part")
        state = part.get("state") if isinstance(part, dict) else None
        tool_errors += int(isinstance(state, dict) and state.get("status") == "error")
    cumulative = [
        int(tokens.get("total", 0) or 0)
        for event in events
        if event.get("type") == "step_finish"
        and isinstance(event.get("part"), dict)
        and isinstance((tokens := event["part"].get("tokens")), dict)
    ]
    tokens = max(cumulative, default=0)
    return tokens, len(tool_events), tool_errors, 0, 0, 0


def _delivery_succeeded(
    final: str,
    *,
    timed_out: bool,
    process_error: str | None,
    evaluator_outcome: EvaluationOutcome,
) -> bool:
    """Fail closed when a harness run did not terminate successfully."""

    return bool(
        not timed_out
        and process_error is None
        and final
        and is_exact_terminal_success(evaluator_outcome)
    )


def _opencode_step_budget_exhausted(final: str) -> bool:
    """Recognize OpenCode's terminal step-limit response, not arbitrary mentions."""

    normalized = re.sub(r"^[#*_`\\s]+", "", final).upper()
    return normalized.startswith("CRITICAL - MAXIMUM STEPS REACHED")


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            events.append(value)
    return events


def _runtime_metric_snapshot(url: str) -> dict[str, int] | None:
    try:
        with urllib.request.urlopen(
            url.rstrip("/") + "/metrics",
            timeout=2,
        ) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    if payload.get("storage") != "memory_only":
        return None
    required = (
        "sessions_started",
        "observations_accepted",
        "sessions_completed",
        "completion_withheld",
    )
    optional = (
        "sessions_evidence_closed",
        "sessions_abandoned",
        "controller_decisions",
        "controller_alternatives_considered",
    )
    try:
        return {
            **{name: int(payload[name]) for name in required},
            **{name: int(payload.get(name, 0)) for name in optional},
        }
    except (KeyError, TypeError, ValueError):
        return None


def _substrate_telemetry_valid(runtime_delta: dict[str, int] | None) -> bool:
    """Require one engaged session and exactly one terminal outcome."""

    if runtime_delta is None:
        return False
    terminals = (
        runtime_delta.get("sessions_completed", 0)
        + runtime_delta.get("sessions_evidence_closed", 0)
        + runtime_delta.get("sessions_abandoned", 0)
    )
    return (
        runtime_delta.get("sessions_started", 0) == 1
        and runtime_delta.get("observations_accepted", 0) >= 1
        and terminals == 1
    )


def _runtime_metric_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None or set(before) != set(after):
        return None
    delta = {name: after[name] - before[name] for name in before}
    return delta if all(value >= 0 for value in delta.values()) else None


def _task_type(case: BenchmarkCase) -> str:
    if isinstance(case, ReasoningCase):
        return (
            "novel_abductive_synthesis"
            if case.mode == "novel_synthesis"
            else "ambiguity_resolution"
        )
    if isinstance(case, LongHorizonCase):
        return "long_horizon_execution"
    if isinstance(case, PatchCase):
        return "repository_patch"
    if isinstance(case, DiagnosticCase):
        return "evidence_bound_debugging"
    if isinstance(case, PlanningCase):
        return "constraint_bound_planning"
    if isinstance(case, JoinCase):
        return "cross_file_numeric_join"
    if isinstance(case, SemanticCase):
        return "semantic_cross_document"
    if isinstance(case, ResearchCase):
        return "current_web_research"
    return "import_lookup"
