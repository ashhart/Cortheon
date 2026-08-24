"""Real generic-MCP execution for one sealed operator-lift cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.condition_execution import prepare_condition
from cortheon.benchmark_core.execution_provenance import (
    ExecutionPolicy,
    execute_host_process,
    execution_facts,
)
from cortheon.benchmark_core.generic_mcp_runner import (
    generic_transcript_valid,
    prepare_generic_process,
)
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome
from cortheon.operator_lift.execution_models import ExecutionConfig, ExecutionOutcome, ScheduledCell
from cortheon.operator_lift.execution_schedule import canonical_bytes, case_goal
from cortheon.operator_lift.models import LiftCase, LiftManifest
from cortheon.operator_lift.sealing import public_case, public_projection_sha256
from cortheon.qualification_core.conditions import execution_profile


@dataclass(frozen=True, slots=True)
class _HostCase:
    case_id: str
    prompt: str
    task_kind: str = "general"
    resource_paths: tuple[str, ...] = ("public-projection.json",)


def _goal(case: LiftCase) -> str:
    return case_goal(case)


def _projection_lines(case: LiftCase) -> str:
    return canonical_bytes(public_case(case)).decode() + "\n"


def _materialize_public_workspace(root: Path, case: LiftCase) -> None:
    (root / "evidence").mkdir()
    (root / "public-projection.json").write_text(_projection_lines(case), encoding="utf-8")
    for source_id, content in case.evidence:
        (root / "evidence" / f"{source_id}.txt").write_text(content, encoding="utf-8")
    if case.operator == "adaptive_stopping":
        actions = root / "actions"
        actions.mkdir()
        for action_id, observation in case.oracle["observations"]:
            (actions / f"{action_id}.txt").write_text(observation, encoding="utf-8")


def _events(stdout: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def _response(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _action_trace(case: LiftCase, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if case.operator != "adaptive_stopping":
        return []
    costs = {action_id: cost for action_id, _description, cost in case.action_catalog}
    observations = dict(case.oracle["observations"])
    requests: dict[str, list[str]] = {}
    trace: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "tool_request" and event.get("origin") == "host":
            arguments = event.get("arguments")
            paths: list[str] = []
            if event.get("name") == "host_read" and isinstance(arguments, dict):
                paths = [str(arguments.get("path", ""))]
            elif event.get("name") == "host_read_many" and isinstance(arguments, dict):
                raw = arguments.get("paths")
                paths = [str(path) for path in raw] if isinstance(raw, list) else []
            elif event.get("name") == "host_search" and isinstance(arguments, dict):
                paths = [str(arguments.get("path", ""))]
            action_ids = [
                path.removeprefix("actions/").removesuffix(".txt")
                for path in paths
                if path.startswith("actions/") and path.endswith(".txt")
            ]
            if isinstance(event.get("call_id"), str) and action_ids:
                requests[event["call_id"]] = action_ids
        if (
            event.get("type") == "tool_result"
            and event.get("origin") == "host"
            and event.get("status") == "result"
            and isinstance(event.get("call_id"), str)
        ):
            for action_id in requests.pop(event["call_id"], []):
                if action_id not in costs or action_id not in observations:
                    continue
                trace.append(
                    {
                        "sequence": len(trace) + 1,
                        "action_id": action_id,
                        "observation_sha256": hashlib.sha256(
                            observations[action_id].encode()
                        ).hexdigest(),
                        "cost": costs[action_id],
                    }
                )
    return trace


def run_cell(
    config: ExecutionConfig,
    manifest: LiftManifest,
    case: LiftCase,
    cell: ScheduledCell,
) -> ExecutionOutcome:
    config.validate()
    binding = (
        manifest.full_condition
        if cell.condition_id == "full"
        else (
            manifest.placebo_condition
            if cell.condition_id == manifest.placebo_condition.condition_id
            else manifest.ablation_conditions[case.operator]
        )
    )
    profile = execution_profile(cell.condition_id, binding.implementation_sha256)
    environment = os.environ.copy()
    applied = prepare_condition(environment, profile, treatment=True)
    if applied is None:
        raise RuntimeError("operator-lift condition was not applied")
    args = argparse.Namespace(
        base_url=config.base_url,
        api_key=config.api_key,
        provider=config.provider_id,
        model_id=config.model_id,
        timeout_seconds=config.timeout_seconds,
        output_tokens=config.output_tokens,
        generic_web_command=None,
    )
    policy = ExecutionPolicy(
        config.max_steps,
        config.max_tool_calls,
        config.timeout_seconds,
        config.context_tokens,
        config.output_tokens,
    )
    with tempfile.TemporaryDirectory(prefix="cortheon-operator-public-") as temporary:
        workspace = Path(temporary)
        _materialize_public_workspace(workspace, case)
        host_case = _HostCase(cell.cell_id, _goal(case))
        spec = prepare_generic_process(
            args,
            host_case,
            repeat=cell.repeat,
            condition=cell.condition_id,
            applied=applied,
            workspace=workspace,
            environment=environment,
            token="",
            max_steps=config.max_steps,
            max_tool_calls=config.max_tool_calls,
            control_payload=None,
        )
        capture = execute_host_process(
            spec.command,
            cwd=workspace,
            env=spec.environment,
            host="generic_mcp",
            policy=policy,
            control_payload=spec.control_payload,
            stdin_payload=spec.stdin_payload,
        )
        events = _events(capture.stdout)
    transcript_valid = generic_transcript_valid(
        events,
        applied=applied,
        spec=spec,
        args=args,
        require_web=False,
    )
    facts = execution_facts(events, host="generic_mcp")
    transport = parse_transport_outcome(events, host="generic_mcp")
    identity_valid = bool(
        transcript_valid
        and facts.identity_valid
        and facts.provider_id == config.provider_id
        and facts.model_id == config.model_id
    )
    trace = _action_trace(case, events)
    delivered = transport.outcome.terminal_status == "success"
    safe = bool(
        identity_valid
        and facts.measurements_valid
        and not capture.timed_out
        and capture.budget_reason is None
        and capture.returncode == 0
    )
    submission: dict[str, object] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "case_commitment": manifest.case_commitments[case.case_id],
        "condition_id": binding.condition_id,
        "condition_config_sha256": binding.config_sha256,
        "implementation_sha256": binding.implementation_sha256,
        "repeat": cell.repeat,
        "delivered": delivered,
        "safe": safe,
        "evaluator_provenance": {
            "schema_version": 1,
            "producer": "evaluator",
            "candidate_supplied": False,
            "evaluator_id": manifest.evaluator_id,
            "evaluator_implementation_sha256": manifest.evaluator_implementation_sha256,
            "public_projection_sha256": public_projection_sha256(case),
            "oracle_access_blocked": True,
            "trace": trace,
            "terminal_after_sequence": len(trace),
            "terminal_reason": (
                "sufficient" if case.operator == "adaptive_stopping" else "not_applicable"
            ),
        },
        "response": _response(transport.final_text),
    }
    summary: dict[str, object] = {
        "condition_id": binding.condition_id,
        "transcript_sha256": hashlib.sha256(canonical_bytes(events)).hexdigest(),
        "transcript_valid": transcript_valid,
        "identity_valid": identity_valid,
        "identity_provenance": facts.identity_provenance,
        "identity_reason": facts.identity_reason,
        "measurements_valid": facts.measurements_valid,
        "measurement_reason": facts.measurement_reason,
        "tokens": facts.tokens,
        "cost_usd": facts.cost_usd,
        "steps": facts.steps,
        "inference_calls": facts.steps,
        "tool_calls": facts.tool_calls,
        "latency_seconds": capture.latency_seconds,
        "timed_out": capture.timed_out,
        "budget_reason": capture.budget_reason,
        "returncode": capture.returncode,
        "terminal_status": transport.outcome.terminal_status,
    }
    return ExecutionOutcome(submission, summary)
