"""Evaluator process specification and attestation for generic MCP cells."""

from __future__ import annotations

import hashlib
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.condition_execution import AppliedCondition
from cortheon.benchmark_core.generic_mcp_claim_validation import validate_claim_transcript
from cortheon.benchmark_core.generic_mcp_process import _web_provider
from cortheon.benchmark_core.generic_mcp_protocol import canonical_json, payload_sha256
from cortheon.benchmark_core.generic_mcp_source import (
    EXPECTED_DIGEST_ENV,
    generic_source_sha256,
    resource_records,
)
from cortheon.benchmark_core.models import LongHorizonCase, PatchCase, ResearchCase
from cortheon.cognitive_protocol import CORTHEON_PROTOCOL_VERSION
from cortheon.qualification_core.conditions import execution_profile, implementation_digest


@dataclass(frozen=True, slots=True)
class GenericProcessSpec:
    command: list[str]
    environment: dict[str, str]
    control_payload: bytes
    stdin_payload: bytes
    wrapper_source_sha256: str
    web_identity: dict[str, str] | None
    task_kind: str
    resource_records: tuple[dict[str, Any], ...]


def generic_evaluation_profile(
    current: dict[str, Any] | None,
    *,
    treatment: bool,
) -> dict[str, Any]:
    return current or execution_profile(
        "full" if treatment else "bare",
        implementation_digest(),
    )


def _control(
    applied: AppliedCondition,
    *,
    token: str,
    max_tool_calls: int,
) -> bytes:
    value = {
        "schema_version": 1,
        "evaluation_profile": applied.profile,
        "cognitive_token": token,
        "evaluator_max_steps": None,
        "auto_enable": False,
        "benchmark_capture_candidate": False,
        "max_host_tool_calls": max_tool_calls,
    }
    encoded = canonical_json(value).encode()
    if len(encoded) > 16_384:
        raise ValueError("generic MCP control payload exceeded its bound")
    return encoded


def _test_catalogue(case: Any) -> dict[str, list[str]]:
    if not isinstance(case, (PatchCase, LongHorizonCase)):
        return {}
    command = shlex.split(case.test_command)
    if not command:
        raise ValueError("generic MCP patch test command is empty")
    return {"case-test": command}


def _task_kind(case: Any) -> str:
    explicit = getattr(case, "task_kind", None)
    if explicit is not None:
        if explicit not in {"auto", "code", "research", "documents", "decision", "general"}:
            raise ValueError("generic MCP case task_kind is invalid")
        return explicit
    if isinstance(case, ResearchCase):
        return "research"
    if isinstance(case, (PatchCase, LongHorizonCase)):
        return "code"
    return "auto"


def _resource_paths(case: Any) -> list[str]:
    explicit = getattr(case, "resource_paths", ())
    if not isinstance(explicit, (list, tuple)):
        raise ValueError("generic MCP case resource paths are invalid")
    return list(explicit)


def prepare_generic_process(
    args: Any,
    case: Any,
    *,
    repeat: int,
    condition: str,
    applied: AppliedCondition,
    workspace: Path,
    environment: dict[str, str],
    token: str,
    max_steps: int,
    max_tool_calls: int,
    control_payload: bytes | None,
) -> GenericProcessSpec:
    marker = applied.nonce
    (workspace / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    source_sha256 = generic_source_sha256()
    child_environment = {key: value for key, value in environment.items() if key != "PYTHONPATH"}
    child_environment[EXPECTED_DIGEST_ENV] = source_sha256
    raw_web = getattr(args, "generic_web_command", None)
    web_command = list(raw_web) if isinstance(raw_web, (list, tuple)) and raw_web else None
    web_identity = _web_provider(web_command)[1] if web_command is not None else None
    task_kind = _task_kind(case)
    resources = tuple(_resource_paths(case))
    records = resource_records(workspace, resources)
    payload = {
        "schema_version": 1,
        "task_id": f"{case.case_id}-{repeat}-{condition}",
        "goal": case.prompt,
        "task_kind": task_kind,
        "require_web": isinstance(case, ResearchCase),
        "workspace": str(workspace),
        "workspace_nonce": marker,
        "resource_paths": list(resources),
        "base_url": args.base_url,
        "api_key": args.api_key,
        "provider_id": args.provider,
        "model_id": args.model_id,
        "timeout_seconds": float(args.timeout_seconds),
        "output_tokens": int(args.output_tokens),
        "max_steps": max_steps,
        "max_tool_calls": max_tool_calls,
        "tests": _test_catalogue(case),
        "web_command": web_command,
    }
    launcher = Path(__file__).resolve().with_name("generic_mcp_launcher.py")
    return GenericProcessSpec(
        [sys.executable, "-I", str(launcher)],
        child_environment,
        control_payload or _control(applied, token=token, max_tool_calls=max_tool_calls),
        canonical_json(payload).encode(),
        source_sha256,
        web_identity,
        task_kind,
        records,
    )


def generic_transcript_valid(
    events: list[dict[str, Any]],
    *,
    applied: AppliedCondition,
    spec: GenericProcessSpec,
    args: Any,
    require_web: bool,
) -> bool:
    endpoint_sha256 = hashlib.sha256(str(args.base_url).rstrip("/").encode()).hexdigest()
    return validate_claim_transcript(
        events,
        expected_config_sha256=applied.profile["config_sha256"],
        expected_implementation_sha256=applied.profile["implementation_sha256"],
        expected_endpoint_sha256=endpoint_sha256,
        expected_wrapper_source_sha256=spec.wrapper_source_sha256,
        expected_web_identity=spec.web_identity,
        expected_task_kind=spec.task_kind,
        expected_resource_records=spec.resource_records,
        require_web=require_web,
    )


def generic_transcript_sha256(events: list[dict[str, Any]]) -> str:
    return payload_sha256(events)


def generic_result_metadata(
    spec: GenericProcessSpec | None,
    events: list[dict[str, Any]],
    transcript_valid: bool | None,
) -> dict[str, Any]:
    return {
        "host_assurance": "evaluator_wrapped" if spec is not None else "native_enforced",
        "host_transcript_valid": transcript_valid,
        "host_transcript_sha256": (
            generic_transcript_sha256(events) if spec is not None and events else None
        ),
        "host_identity_sha256": (
            generic_identity_digest(spec.wrapper_source_sha256, spec.web_identity)
            if spec is not None
            else None
        ),
    }


def generic_runtime_facts(events: list[dict[str, Any]]) -> dict[str, int]:
    transitions = [event for event in events if event.get("type") == "runtime_transition"]
    return {
        "sessions_started": sum(event.get("transition") == "start" for event in transitions),
        "observations_accepted": sum(event.get("transition") == "observe" for event in transitions),
        "sessions_completed": sum(event.get("transition") == "complete" for event in transitions),
        "sessions_evidence_closed": 0,
        "sessions_abandoned": sum(event.get("transition") == "abandon" for event in transitions),
        "completion_withheld": sum(
            event.get("type") == "terminal" and event.get("disposition") == "withhold"
            for event in events
        ),
        "controller_decisions": 0,
        "controller_alternatives_considered": 0,
    }


def generic_receipt(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    receipts = [
        event.get("receipt") for event in events if event.get("type") == "evaluation_receipt"
    ]
    return receipts[0] if len(receipts) == 1 and isinstance(receipts[0], dict) else None


def generic_identity_digest(source_sha256: str, web_identity: dict[str, str] | None) -> str:
    return payload_sha256({"wrapper_source_sha256": source_sha256, "web_provider": web_identity})


def generic_implementation_snapshot(
    web_command: list[str] | None,
) -> tuple[dict[str, str], dict[str, str] | None]:
    source = generic_source_sha256()
    web_identity = _web_provider(web_command)[1] if web_command else None
    runtime = implementation_digest()
    return (
        {
            "wrapper_sha256": source,
            "runtime_sha256": runtime,
            "condition_sha256": runtime,
            "web_provider_sha256": payload_sha256(web_identity),
            "host_identity_sha256": generic_identity_digest(source, web_identity),
        },
        web_identity,
    )


def generic_embedded_health(snapshot: dict[str, str]) -> dict[str, Any]:
    return {
        "ok": True,
        "service": "cortheon-generic-mcp-evaluator",
        "version": "1",
        "protocol_version": CORTHEON_PROTOCOL_VERSION,
        "source_fingerprint": snapshot["wrapper_sha256"],
        "storage": "memory_only",
    }
