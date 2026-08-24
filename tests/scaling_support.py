"""Closed schema-14 scaling report fixtures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cortheon.benchmark_core.audit import _audit_manifest, _canonical_json

SHA = "a" * 64


def _digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _schedule_cells(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": run["case_id"],
            "repeat": run.get("repeat", 0),
            "condition": run["condition"],
        }
        for run in runs
    ]


def identity(
    *,
    budget: int,
    runs: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    has_frontier = any(run["condition"] == "frontier" for run in runs)
    return {
        "schema_version": 1,
        "benchmark_report_schema": 14,
        "repository": {"name": "demo", "snapshot_sha256": "1" * 64},
        "case_bank": {
            "suite": "semantic",
            "selection_sha256": _digest(cases),
            "case_count": len(cases),
        },
        "schedule": {
            "seed": 7,
            "repeats": len({int(run.get("repeat", 0)) for run in runs}),
            "conditions": sorted({str(run["condition"]) for run in runs}),
            "schedule_sha256": _digest(_schedule_cells(runs)),
        },
        "host": {
            "kind": "pi",
            "configured_command": "pi",
            "executable_sha256": "2" * 64,
            "version": "pi 1.0",
        },
        "cortheon_runtime": {
            "endpoint_sha256": "3" * 64,
            "artifact_sha256": "4" * 64,
            "adapter_sha256": "5" * 64,
            "observed_service": "cortheon-cognitive",
            "observed_version": "1.0",
            "observed_protocol_version": "1.0.0",
            "observed_source_fingerprint": "6" * 64,
        },
        "inference": {
            "provider": "local",
            "model_id": "demo",
            "observed_model_id": "demo",
            "endpoint_sha256": "7" * 64,
            "registered_artifact_sha256": "8" * 64,
            "reasoning": False,
        },
        "limits": {
            "timeout_seconds": 60.0,
            "context_tokens": 8192,
            "output_tokens": 512,
            "max_tool_calls": 16,
        },
        "frontier": (
            {
                "kind": "endpoint",
                "provider": "frontier",
                "model_id": "frontier",
                "observed_model_id": "frontier",
                "endpoint_sha256": "9" * 64,
                "registered_artifact_sha256": "b" * 64,
                "executable_sha256": None,
                "version": None,
                "max_budget_usd": 1.0,
            }
            if has_frontier
            else None
        ),
        "max_steps": budget,
    }


def report(
    runs: list[dict[str, Any]],
    *,
    budget: int = 4,
) -> dict[str, Any]:
    normalized = deepcopy(runs)
    for run in normalized:
        frontier = run["condition"] == "frontier"
        run.setdefault("tool_errors", 0)
        run.setdefault("expected_verdict", "allow")
        outcome = run.get("evaluator_outcome")
        terminal_status = outcome.get("terminal_status") if isinstance(outcome, dict) else None
        run.setdefault(
            "failure_owner",
            None if terminal_status in {"success", "withheld"} else "candidate",
        )
        run.setdefault("host_tool_executions", 0)
        run.setdefault("blocked_tool_calls", 0)
        run.setdefault("unavailable_tool_calls", 0)
        run.setdefault("tokens", 0)
        run.setdefault("inference_provider_id", "frontier" if frontier else "local")
        run.setdefault("execution_identity_valid", True)
        run.setdefault("execution_identity_provenance", "pi_message_end")
        run.setdefault("execution_measurements_valid", True)
        run.setdefault("observed_steps", 1)
        run.setdefault("policy_timeout_seconds", 60.0)
        run.setdefault("policy_max_steps", budget)
        run.setdefault("policy_max_tool_calls", 16)
        run.setdefault("policy_context_tokens", 8192)
        run.setdefault("policy_output_tokens", 512)
        run.setdefault("model_text_chars", 0)
        run.setdefault("deliverable_chars", 0)
        run.setdefault("runtime_sessions_started", 0)
        run.setdefault("runtime_observations_accepted", 0)
        run.setdefault("runtime_completion_withheld", 0)
        run.setdefault("runtime_controller_decisions", 0)
        run.setdefault("runtime_controller_alternatives_considered", 0)
        run.setdefault("retry_count", 0)
        run.setdefault("retry_reason", None)
        run.setdefault("prior_attempts", [])
    verdicts = {
        case_id: next(run["expected_verdict"] for run in normalized if run["case_id"] == case_id)
        for case_id in sorted({r["case_id"] for r in normalized})
    }
    cases = [
        {"case_id": case_id, "expected_verdict": verdict} for case_id, verdict in verdicts.items()
    ]
    value = {
        "schema_version": 14,
        "model": "local/demo",
        "host": "pi",
        "max_steps": budget,
        "repository": {"name": "demo", "snapshot_sha256": "1" * 64},
        "seed": 7,
        "suite": "semantic",
        "cases": cases,
        "experiment_identity": identity(budget=budget, runs=normalized, cases=cases),
        "runs": normalized,
    }
    return reseal(value)


def reseal(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("audit", None)
    value["audit"] = _audit_manifest(value)
    return value


def refresh_bindings(value: dict[str, Any]) -> dict[str, Any]:
    identity = value["experiment_identity"]
    identity["case_bank"]["selection_sha256"] = _digest(value["cases"])
    identity["case_bank"]["case_count"] = len(value["cases"])
    identity["schedule"]["schedule_sha256"] = _digest(_schedule_cells(value["runs"]))
    return reseal(value)
