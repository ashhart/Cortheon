"""Shared fixtures for the split qualification-factory test modules.

Deliberately outside the ``test_*`` namespace so pytest imports it as a
helper rather than collecting it as a test module.
"""

import json
from typing import Literal

from cortheon.cognitive_benchmark import WITHHELD_PREFIX, EvaluationOutcome, RunResult
from cortheon.qualification_core.conditions import (
    AVAILABLE_CONDITIONS,
    CONDITION_REGISTRY_VERSION,
    CONDITIONS,
    HISTORICAL_CONDITIONS,
    OPERATOR_KEYS,
    condition_record,
    implementation_digest,
)
from cortheon.qualification_factory import Cell


def _condition_entries():
    implementation = implementation_digest()
    return [
        {
            "id": condition,
            "config_sha256": condition_record(
                condition,
                implementation_sha256=implementation,
            )["config_sha256"],
            "implementation_sha256": implementation,
        }
        for condition in AVAILABLE_CONDITIONS
    ]


def _historical_condition_entries():
    implementation = implementation_digest()
    return [
        {
            "id": condition,
            "config_sha256": condition_record(
                condition,
                implementation_sha256=implementation,
                host="opencode",
            )["config_sha256"],
            "implementation_sha256": condition_record(
                condition,
                implementation_sha256=implementation,
                host="opencode",
            )["implementation_sha256"],
        }
        for condition in HISTORICAL_CONDITIONS
    ]


# The terminal a treatment run ends on when completion was not certified.
# Only this text makes an undelivered run a block rather than a delivery
# failure, so block fixtures must carry it.
WITHHELD_TERMINAL = (
    "[Cortheon withheld: completion was not certified]\n"
    "The Cortheon investigation ended without a certified answer because "
    "the evaluator observed an authenticated test terminal."
)
assert WITHHELD_TERMINAL.startswith(WITHHELD_PREFIX)


def _cell(**overrides):
    values = {
        "cell_id": "local-semantic",
        "suite": "semantic",
        "host": "opencode",
        "provider": "Local",
        "base_url": "http://127.0.0.1:18081/v1",
        "api_key_env": None,
        "model_id": "small-model",
        "runtime_url": "http://127.0.0.1:8743",
        "cases": 2,
        "repeats": 1,
        "seed": 7,
        "timeout_seconds": 60.0,
        "context_tokens": 8_192,
        "output_tokens": 512,
        "max_steps": 4,
        "reasoning": False,
        "opencode": "opencode",
        "pi": "pi",
        "condition_ids": AVAILABLE_CONDITIONS,
        "condition_implementation_sha256": implementation_digest(),
        "historical_comparison": False,
    }
    values.update(overrides)
    return Cell(**values)


def _result(
    case_id,
    repeat,
    condition,
    correct,
    *,
    delivered=True,
    process_error=None,
    telemetry=None,
    final_text="private model output",
    artifact_correct=None,
    candidate_correct=None,
    timed_out=False,
    expected_verdict: Literal["allow", "block"] | None = "allow",
    failure_owner: Literal["candidate", "external_infrastructure"] | None = None,
):
    if timed_out or process_error is not None:
        outcome = EvaluationOutcome(
            "pi", "transport_error", "pi_assistant", "timeout" if timed_out else "process_error"
        )
    elif delivered:
        outcome = EvaluationOutcome("pi", "success", "pi_assistant", "stop")
    elif final_text == WITHHELD_TERMINAL:
        outcome = EvaluationOutcome("pi", "withheld", "pi_custom_terminal", "withheld")
    else:
        outcome = EvaluationOutcome("pi", "missing", "none", None)
    registered = (
        condition in CONDITIONS
        and condition_record(
            condition,
            implementation_sha256=implementation_digest(),
            host="opencode",
        )["available"]
    )
    record = (
        condition_record(
            condition,
            implementation_sha256=implementation_digest(),
            host="opencode",
        )
        if registered
        else None
    )
    if registered:
        assert record is not None
    bare = condition == "bare"
    return RunResult(
        case_id=case_id,
        repeat=repeat,
        condition=condition,
        expected="private expected answer",
        final_text=final_text,
        delivered=delivered,
        correct=correct,
        latency_seconds=1.0,
        tokens=10,
        tool_calls=1,
        tool_errors=0,
        timed_out=timed_out,
        process_error=process_error,
        expected_verdict=expected_verdict,
        failure_owner=(
            failure_owner
            if failure_owner is not None
            else "candidate"
            if timed_out
            or process_error is not None
            or (not delivered and final_text != WITHHELD_TERMINAL)
            else None
        ),
        evaluator_outcome=outcome,
        task_type="semantic_cross_document",
        artifact_correct=artifact_correct,
        candidate_correct=candidate_correct,
        substrate_telemetry_valid=telemetry,
        runtime_sessions_started=1 if telemetry else 0,
        runtime_observations_accepted=1 if telemetry else 0,
        runtime_sessions_completed=(1 if telemetry and condition != "retrieval_only" else 0),
        runtime_sessions_abandoned=(1 if telemetry and condition == "retrieval_only" else 0),
        condition_registry_version=CONDITION_REGISTRY_VERSION if registered else None,
        condition_config_sha256=record["config_sha256"] if record else None,
        condition_implementation_sha256=(record["implementation_sha256"] if record else None),
        condition_requires_runtime_completion=(
            CONDITIONS[condition].intercepts_final if registered else None
        ),
        condition_profile_receipt_valid=True if registered else None,
        condition_observed_config_sha256=(
            record["config_sha256"] if registered and not bare and record is not None else None
        ),
        condition_observed_implementation_sha256=(
            record["implementation_sha256"]
            if registered and not bare and record is not None
            else None
        ),
        condition_adapter_receipt_valid=True if registered and not bare else None,
        condition_operator_counts=(
            dict.fromkeys(OPERATOR_KEYS, 0)
            if registered and not bare and condition != "old_planner"
            else None
        ),
    )


def _write_manifest(tmp_path, **updates):
    value = {
        "schema_version": 3,
        "tier": "pr",
        "repository": ".",
        "seed": 7,
        "cells": [
            {
                "id": "local-semantic",
                "suite": "semantic",
                "host": "opencode",
                "provider": "Local",
                "model_id": "small-model",
                "cases": 2,
                "repeats": 1,
                "conditions": _condition_entries(),
            }
        ],
    }
    value.update(updates)
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(value))
    return path
