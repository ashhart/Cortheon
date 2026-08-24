"""Hostile state changes cannot detach revision reasoning from completion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_runtime import EvaluatorMcpRuntime
from cortheon.cognitive_core.models import CognitiveRuntimeError
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal
from cortheon.qualification_core.conditions import execution_profile

_RECORD = {
    "prior": "h_alex_oncall",
    "original_source": "source_a",
    "decisive_source": "source_b",
    "decisive_effect": "refutes",
    "revised": "h_priya_oncall",
}
_ANSWER = {
    "prior": "h_alex_oncall",
    "prior_status": "refuted",
    "revised": "h_priya_oncall",
    "decisive_source": "source_b",
}


def _bound_runtime(tmp_path: Path) -> tuple[EvaluatorMcpRuntime, str]:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    marker = "revision-binding"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    runtime = EvaluatorMcpRuntime(profile, resource_paths=("public-projection.json",))
    runtime.start(_goal(case), task_kind="general")
    runtime.observe(
        IsolatedExecutor(tmp_path, marker_nonce=marker).execute(
            "read-record",
            "host_read",
            {"path": "public-projection.json"},
        )
    )
    runtime.lifecycle_call("cortheon_step", {"draft": json.dumps(_RECORD)})
    assert runtime.session_id is not None
    return runtime, runtime.session_id


def _verify(runtime: EvaluatorMcpRuntime, session_id: str) -> None:
    runtime.server.runtime.verify(
        session_id,
        answer=json.dumps(_ANSWER),
        claims=[
            {
                "claim": "source_b refutes h_alex_oncall and supports h_priya_oncall.",
                "evidence_ids": ["ev1"],
            }
        ],
        completion_evidence_ids=["ev1"],
    )


def test_retracted_contract_invalidates_the_accepted_reasoning_record(tmp_path: Path) -> None:
    runtime, session_id = _bound_runtime(tmp_path)
    runtime.server.runtime.retract(session_id, ["ev1"], reason="bad source")

    with pytest.raises(
        CognitiveRuntimeError,
        match="exactly one public revision contract is required",
    ):
        _verify(runtime, session_id)


def test_in_memory_record_mutation_is_detected_before_verification(tmp_path: Path) -> None:
    runtime, session_id = _bound_runtime(tmp_path)
    runtime.server.runtime._sessions[session_id].revision_record = {
        **_RECORD,
        "decisive_source": "source_a",
    }

    with pytest.raises(CognitiveRuntimeError, match="accepted revision binding changed"):
        _verify(runtime, session_id)
