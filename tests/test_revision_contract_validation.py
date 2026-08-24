"""The runtime and generic host share one closed public revision contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_projection import completion_answer_schema
from cortheon.benchmark_core.generic_mcp_runtime import EvaluatorMcpRuntime
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


def _profile() -> dict[str, Any]:
    value = execution_profile("full", "a" * 64)
    value["nonce"] = "7" * 32
    return value


def _case_payload() -> tuple[Any, dict[str, Any]]:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    return case, copy.deepcopy(public_case(case))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["response_schema"].__setitem__("fields", ["wrong"]),
        lambda payload: payload["response_schema"].__setitem__(
            "hypothesis_vocabulary", ["h_alex_oncall", "h_alex_oncall"]
        ),
        lambda payload: payload["response_schema"]["effect_status_map"].__setitem__(
            "refutes", "outside-vocabulary"
        ),
        lambda payload: payload["response_schema"].__setitem__(
            "effect_changes_hypothesis", {"supports": False}
        ),
    ],
    ids=("fields", "duplicate-hypotheses", "status-map", "change-map"),
)
def test_projection_and_runtime_reject_the_same_malformed_contract(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    case, payload = _case_payload()
    mutate(payload)
    path = tmp_path / "public-projection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert completion_answer_schema(tmp_path, (path.name,)) is None
    marker = "revision-contract"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    runtime = EvaluatorMcpRuntime(_profile(), resource_paths=(path.name,))
    runtime.start(_goal(case), task_kind="general")
    runtime.observe(
        IsolatedExecutor(tmp_path, marker_nonce=marker).execute(
            "read-contract", "host_read", {"path": path.name}
        )
    )

    with pytest.raises(RuntimeError, match="exactly one public revision contract is required"):
        runtime.lifecycle_call("cortheon_step", {"draft": json.dumps(_RECORD)})


def test_revision_contract_requires_artifact_read_provenance() -> None:
    case, payload = _case_payload()
    runtime = EvaluatorMcpRuntime(_profile(), resource_paths=("public-projection.json",))
    runtime.start(_goal(case), task_kind="general")
    request = runtime.next_action["request"]  # type: ignore[index]
    runtime._call(
        "cortheon_observe",
        {
            "session_id": runtime.session_id,
            "request_id": request["request_id"],
            "observations": [
                {
                    "kind": "code",
                    "source": "public-projection.json",
                    "content": json.dumps(payload),
                    "host_receipt": {
                        "tool": "read",
                        "outcome": "result",
                        "args": {"path": "public-projection.json"},
                    },
                }
            ],
        },
    )

    with pytest.raises(RuntimeError, match="exactly one public revision contract is required"):
        runtime.lifecycle_call("cortheon_step", {"draft": json.dumps(_RECORD)})
