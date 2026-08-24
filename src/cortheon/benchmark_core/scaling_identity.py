"""Evaluator-owned identity for diagnostic scaling experiments."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

SCALING_REPORT_SCHEMA = 14
SCALING_IDENTITY_SCHEMA = 1

_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_report_schema",
        "repository",
        "case_bank",
        "schedule",
        "host",
        "cortheon_runtime",
        "inference",
        "limits",
        "frontier",
        "max_steps",
    }
)
_NESTED_KEYS = {
    "repository": frozenset({"name", "snapshot_sha256"}),
    "case_bank": frozenset({"suite", "selection_sha256", "case_count"}),
    "schedule": frozenset({"seed", "repeats", "conditions", "schedule_sha256"}),
    "host": frozenset({"kind", "configured_command", "executable_sha256", "version"}),
    "cortheon_runtime": frozenset(
        {
            "endpoint_sha256",
            "artifact_sha256",
            "adapter_sha256",
            "observed_service",
            "observed_version",
            "observed_protocol_version",
            "observed_source_fingerprint",
        }
    ),
    "inference": frozenset(
        {
            "provider",
            "model_id",
            "observed_model_id",
            "endpoint_sha256",
            "registered_artifact_sha256",
            "reasoning",
        }
    ),
    "limits": frozenset({"timeout_seconds", "context_tokens", "output_tokens", "max_tool_calls"}),
}
_FRONTIER_KEYS = frozenset(
    {
        "kind",
        "provider",
        "model_id",
        "observed_model_id",
        "endpoint_sha256",
        "registered_artifact_sha256",
        "executable_sha256",
        "version",
        "max_budget_usd",
    }
)
_GENERIC_IMPLEMENTATION_KEYS = frozenset(
    {
        "wrapper_pre_sha256",
        "wrapper_post_sha256",
        "runtime_pre_sha256",
        "runtime_post_sha256",
        "condition_pre_sha256",
        "condition_post_sha256",
        "web_provider_pre_sha256",
        "web_provider_post_sha256",
        "host_identity_pre_sha256",
        "host_identity_post_sha256",
    }
)


def _scaling_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _scaling_digest(value: Any) -> str:
    return hashlib.sha256(_scaling_json(value)).hexdigest()


def _scaling_file_digest(path: Traversable) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, FileNotFoundError, IsADirectoryError):
        return None


def _scaling_tree_digest(root: Traversable) -> str | None:
    entries: list[tuple[str, bytes]] = []

    def walk(directory: Traversable, prefix: str) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except (OSError, FileNotFoundError, NotADirectoryError):
            return
        for child in children:
            relative = f"{prefix}/{child.name}" if prefix else child.name
            if child.name == "__pycache__" or child.name.endswith((".pyc", ".pyo")):
                continue
            if child.is_dir():
                walk(child, relative)
            elif child.is_file():
                entries.append((relative, child.read_bytes()))

    try:
        walk(root, "")
    except OSError:
        return None
    if not entries:
        return None
    digest = hashlib.sha256()
    for relative, content in entries:
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _scaling_adapter_digest(root: Traversable, host: str) -> str | None:
    facade_name, core_name, suffix = (
        ("opencode_plugin.js", "opencode_core", ".js")
        if host == "opencode"
        else ("pi_extension.ts", "pi_core", ".ts")
    )
    members: list[tuple[str, bytes]] = []

    def add(path: Traversable, relative: str) -> None:
        if path.is_dir():
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                add(child, f"{relative}/{child.name}")
        elif path.is_file() and (relative == facade_name or relative.endswith(suffix)):
            members.append((relative, path.read_bytes()))

    try:
        add(root.joinpath(facade_name), facade_name)
        add(root.joinpath(core_name), core_name)
    except (OSError, FileNotFoundError, NotADirectoryError):
        return None
    if not members or members[0][0] != facade_name:
        return None
    digest = hashlib.sha256()
    for relative, content in members:
        digest.update(_scaling_json([relative, hashlib.sha256(content).hexdigest()]))
    return digest.hexdigest()


def _scaling_command_identity(command: str) -> dict[str, Any]:
    resolved = shutil.which(command)
    if resolved is None:
        candidate = Path(command)
        resolved = str(candidate.resolve()) if candidate.is_file() else None
    executable_sha256 = None
    version = None
    if resolved is not None:
        executable_sha256 = _scaling_file_digest(Path(resolved))
        try:
            completed = subprocess.run(
                [resolved, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            lines = (completed.stdout or completed.stderr).strip().splitlines()
            version = lines[0][:200] if lines else None
    return {
        "configured_command": command,
        "executable_sha256": executable_sha256,
        "version": version,
    }


def _scaling_schedule_cells(
    jobs: Iterable[tuple[Any, int, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": str(case.case_id),
            "repeat": repeat,
            "condition": condition,
        }
        for case, repeat, condition in jobs
    ]


def _scaling_experiment_identity(
    args: Any,
    *,
    health: dict[str, Any],
    inference: dict[str, Any],
    frontier_inference: dict[str, Any] | None,
    repository_snapshot: str,
    blinded_cases: list[dict[str, Any]],
    jobs: list[tuple[Any, int, str]],
) -> dict[str, Any]:
    package = files("cortheon")
    host_command = args.opencode if args.host == "opencode" else args.pi
    if args.host == "generic_mcp":
        launcher = Path(__file__).resolve().with_name("generic_mcp_launcher.py")
        host_identity = {
            "configured_command": f"{sys.executable} -I {launcher}",
            "executable_sha256": _scaling_file_digest(launcher),
            "version": "generic-mcp-transcript-v1",
        }
        generic_pre = args.generic_implementation_pre
        generic_post = args.generic_implementation_post
    else:
        host_identity = _scaling_command_identity(host_command)
        generic_pre = generic_post = None
    frontier = None
    if args.frontier_cli:
        frontier_command = _scaling_command_identity(args.frontier_cli)
        frontier = {
            "kind": "cli",
            "provider": "cli",
            "model_id": args.frontier_cli_model,
            "observed_model_id": (frontier_inference or {}).get("model_id"),
            "endpoint_sha256": None,
            "registered_artifact_sha256": args.frontier_inference_artifact_sha256 or None,
            "executable_sha256": frontier_command["executable_sha256"],
            "version": frontier_command["version"],
            "max_budget_usd": args.frontier_max_budget_usd,
        }
    elif args.frontier_model_id:
        frontier = {
            "kind": "endpoint",
            "provider": args.frontier_provider,
            "model_id": args.frontier_model_id,
            "observed_model_id": (frontier_inference or {}).get("model_id"),
            "endpoint_sha256": _scaling_digest(
                (args.frontier_base_url or args.base_url).rstrip("/")
            ),
            "registered_artifact_sha256": args.frontier_inference_artifact_sha256 or None,
            "executable_sha256": None,
            "version": None,
            "max_budget_usd": args.frontier_max_budget_usd,
        }
    schedule_cells = _scaling_schedule_cells(jobs)
    identity = {
        "schema_version": SCALING_IDENTITY_SCHEMA,
        "benchmark_report_schema": SCALING_REPORT_SCHEMA,
        "repository": {
            "name": args.repository.name,
            "snapshot_sha256": repository_snapshot,
        },
        "case_bank": {
            "suite": args.suite,
            "selection_sha256": _scaling_digest(blinded_cases),
            "case_count": len(blinded_cases),
        },
        "schedule": {
            "seed": args.seed,
            "repeats": args.repeats,
            "conditions": sorted({cell["condition"] for cell in schedule_cells}),
            "schedule_sha256": _scaling_digest(schedule_cells),
        },
        "host": {"kind": args.host, **host_identity},
        "cortheon_runtime": {
            "endpoint_sha256": _scaling_digest(
                "embedded-generic-mcp"
                if args.host == "generic_mcp"
                else args.runtime_url.rstrip("/")
            ),
            "artifact_sha256": _scaling_tree_digest(package),
            "adapter_sha256": (
                generic_pre["host_identity_sha256"]
                if generic_pre is not None
                else _scaling_adapter_digest(package, args.host)
            ),
            "observed_service": health.get("service"),
            "observed_version": health.get("version"),
            "observed_protocol_version": health.get("protocol_version"),
            "observed_source_fingerprint": health.get("source_fingerprint"),
        },
        "inference": {
            "provider": args.provider,
            "model_id": args.model_id,
            "observed_model_id": inference.get("model_id"),
            "endpoint_sha256": _scaling_digest(args.base_url.rstrip("/")),
            "registered_artifact_sha256": args.inference_artifact_sha256 or None,
            "reasoning": bool(args.reasoning),
        },
        "limits": {
            "timeout_seconds": args.timeout_seconds,
            "context_tokens": args.context_tokens,
            "output_tokens": args.output_tokens,
            "max_tool_calls": args.max_tool_calls,
        },
        "frontier": frontier,
        "max_steps": args.max_steps,
    }
    if generic_pre is not None and generic_post is not None:
        identity["schema_version"] = 2
        identity["generic_implementation"] = {
            "wrapper_pre_sha256": generic_pre["wrapper_sha256"],
            "wrapper_post_sha256": generic_post["wrapper_sha256"],
            "runtime_pre_sha256": generic_pre["runtime_sha256"],
            "runtime_post_sha256": generic_post["runtime_sha256"],
            "condition_pre_sha256": generic_pre["condition_sha256"],
            "condition_post_sha256": generic_post["condition_sha256"],
            "web_provider_pre_sha256": generic_pre["web_provider_sha256"],
            "web_provider_post_sha256": generic_post["web_provider_sha256"],
            "host_identity_pre_sha256": generic_pre["host_identity_sha256"],
            "host_identity_post_sha256": generic_post["host_identity_sha256"],
        }
    return identity


def _scaling_valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scaling_valid_text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256


def _scaling_identity_valid(identity: Any) -> bool:
    if not isinstance(identity, dict):
        return False
    generic = identity.get("schema_version") == 2
    expected_keys = _IDENTITY_KEYS | ({"generic_implementation"} if generic else set())
    if set(identity) != expected_keys:
        return False
    if identity["schema_version"] not in {1, 2} or identity["benchmark_report_schema"] != 14:
        return False
    for key, expected in _NESTED_KEYS.items():
        if not isinstance(identity[key], dict) or set(identity[key]) != expected:
            return False
    repository = identity["repository"]
    case_bank = identity["case_bank"]
    schedule = identity["schedule"]
    host = identity["host"]
    runtime = identity["cortheon_runtime"]
    inference = identity["inference"]
    limits = identity["limits"]
    scalar_text = [
        repository["name"],
        case_bank["suite"],
        host["kind"],
        host["configured_command"],
        host["version"],
        runtime["observed_service"],
        runtime["observed_version"],
        runtime["observed_protocol_version"],
        inference["provider"],
        inference["model_id"],
        inference["observed_model_id"],
    ]
    digests = [
        repository["snapshot_sha256"],
        case_bank["selection_sha256"],
        schedule["schedule_sha256"],
        host["executable_sha256"],
        runtime["endpoint_sha256"],
        runtime["artifact_sha256"],
        runtime["adapter_sha256"],
        runtime["observed_source_fingerprint"],
        inference["endpoint_sha256"],
        inference["registered_artifact_sha256"],
    ]
    if not all(_scaling_valid_text(value) for value in scalar_text) or not all(
        _scaling_valid_sha(value) for value in digests
    ):
        return False
    if host["kind"] not in {"opencode", "pi", "generic_mcp"}:
        return False
    if generic != (host["kind"] == "generic_mcp"):
        return False
    if generic:
        implementation = identity.get("generic_implementation")
        if (
            not isinstance(implementation, dict)
            or set(implementation) != _GENERIC_IMPLEMENTATION_KEYS
            or not all(_scaling_valid_sha(value) for value in implementation.values())
            or implementation["wrapper_pre_sha256"] != implementation["wrapper_post_sha256"]
            or implementation["runtime_pre_sha256"] != implementation["runtime_post_sha256"]
            or implementation["condition_pre_sha256"] != implementation["condition_post_sha256"]
            or implementation["web_provider_pre_sha256"]
            != implementation["web_provider_post_sha256"]
            or implementation["host_identity_pre_sha256"]
            != implementation["host_identity_post_sha256"]
            or runtime["adapter_sha256"] != implementation["host_identity_pre_sha256"]
            or runtime["observed_source_fingerprint"] != implementation["wrapper_pre_sha256"]
        ):
            return False
    if type(inference["reasoning"]) is not bool:
        return False
    if inference["observed_model_id"] != inference["model_id"]:
        return False
    if not all(
        type(value) is int and value > 0
        for value in (
            case_bank["case_count"],
            schedule["repeats"],
            limits["context_tokens"],
            limits["output_tokens"],
            identity["max_steps"],
        )
    ):
        return False
    if type(limits["max_tool_calls"]) is not int or limits["max_tool_calls"] < 0:
        return False
    if type(schedule["seed"]) is not int:
        return False
    if (
        not isinstance(schedule["conditions"], list)
        or schedule["conditions"] != sorted(set(schedule["conditions"]))
        or not all(_scaling_valid_text(value) for value in schedule["conditions"])
    ):
        return False
    timeout = limits["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        return False
    return _scaling_frontier_valid(identity["frontier"])


def _scaling_frontier_valid(frontier: Any) -> bool:
    if frontier is None:
        return True
    if not isinstance(frontier, dict) or set(frontier) != _FRONTIER_KEYS:
        return False
    if frontier["kind"] not in {"cli", "endpoint"}:
        return False
    if not all(
        _scaling_valid_text(frontier[key]) for key in ("provider", "model_id", "observed_model_id")
    ) or not _scaling_valid_sha(frontier["registered_artifact_sha256"]):
        return False
    budget = frontier["max_budget_usd"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget <= 0
    ):
        return False
    if frontier["kind"] == "cli":
        return (
            frontier["endpoint_sha256"] is None
            and _scaling_valid_sha(frontier["executable_sha256"])
            and _scaling_valid_text(frontier["version"])
            and frontier["observed_model_id"] == frontier["model_id"]
        )
    return (
        _scaling_valid_sha(frontier["endpoint_sha256"])
        and frontier["executable_sha256"] is None
        and frontier["version"] is None
        and frontier["observed_model_id"] == frontier["model_id"]
    )


def _scaling_family_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key != "max_steps"}
