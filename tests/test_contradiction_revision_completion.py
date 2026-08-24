"""Ablation-faithful contradiction revision through the generic evaluator host."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_runtime import EvaluatorMcpRuntime
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.cognitive_core.models import CognitiveRuntimeError
from cortheon.cognitive_core.tasks import _is_contradiction_revision_goal
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal
from cortheon.qualification_core.conditions import execution_profile

_ANSWER = {
    "prior": "h_alex_oncall",
    "prior_status": "refuted",
    "revised": "h_priya_oncall",
    "decisive_source": "source_b",
}


def _executor(root: Path) -> IsolatedExecutor:
    marker = "revision-workspace"
    (root / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    return IsolatedExecutor(root, marker_nonce=marker)


def _rejected_completion_error(result: Any) -> str:
    event = next(
        item
        for item in result.events
        if item["type"] == "tool_result"
        and item["origin"] == "mcp"
        and json.loads(item["content"]).get("status") == "rejected"
    )
    return str(json.loads(event["content"])["error"])


class _RevisionModel:
    provider_id = "local"
    model_id = "same-model"
    endpoint_sha256 = "e" * 64

    def __init__(
        self,
        *,
        invalid_revision_arguments: bool = False,
        revision_record: dict[str, str] | None = None,
        answer: dict[str, str] | None = None,
        binding_mode: str = "valid",
        claims: list[dict[str, Any]] | None = None,
        hypotheses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.offers: list[tuple[list[str], str]] = []
        self.parameters: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.invalid_revision_arguments = invalid_revision_arguments
        self.revision_record = revision_record
        self.answer = answer
        self.binding_mode = binding_mode
        self.claims = claims
        self.hypotheses = hypotheses

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        self.offers.append(([tool["function"]["name"] for tool in tools], tool_choice))
        self.parameters.append(tools[0]["function"]["parameters"])
        if messages[-1]["role"] == "tool":
            self.tool_results.append(json.loads(messages[-1]["content"]))
        if tool_choice == "host_read":
            call = ModelToolCall(
                "read-record",
                "host_read",
                {"path": "public-projection.json"},
            )
        elif tool_choice == "host_reason":
            arguments = self.revision_record or {
                "prior": "h_alex_oncall",
                "original_source": "source_a",
                "decisive_source": "source_b",
                "decisive_effect": "refutes",
                "revised": "h_priya_oncall",
            }
            call = ModelToolCall(
                "revise",
                "host_reason",
                {"draft": arguments} if self.invalid_revision_arguments else arguments,
            )
        else:
            reasoning_binding = self.tool_results[-1].get("reasoning_binding")
            if isinstance(reasoning_binding, dict) and self.binding_mode == "tampered":
                reasoning_binding = {
                    **reasoning_binding,
                    "reasoning_binding_sha256": "0" * 64,
                }
            if self.binding_mode == "missing":
                reasoning_binding = None
            call = ModelToolCall(
                "complete",
                "host_complete",
                {
                    "answer": self.answer or _ANSWER,
                    **(
                        {"reasoning_binding": reasoning_binding}
                        if reasoning_binding is not None
                        else {}
                    ),
                    "claims": self.claims
                    or [
                        {
                            "claim": (
                                "The task record says source_b refutes h_alex_oncall and "
                                "supports h_priya_oncall as the revised hypothesis."
                            ),
                            "evidence_ids": ["ev1"],
                        }
                    ],
                    "hypotheses": self.hypotheses
                    or [
                        {
                            "statement": "source_b supersedes h_alex_oncall.",
                            "falsification_test": "Check the signed current rota in source_b.",
                            "status": "supported",
                            "evidence_ids": ["ev1"],
                        },
                        {
                            "statement": "source_b supports h_priya_oncall as current.",
                            "falsification_test": "Check for a newer signed rota.",
                            "status": "supported",
                            "evidence_ids": ["ev1"],
                        },
                    ],
                    "completion_evidence_ids": ["ev1"],
                },
            )
        return ModelTurn(
            self.provider_id,
            self.model_id,
            "",
            (call,),
            "tool_calls",
            3,
        )


def test_revision_goal_detection_is_narrow() -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    assert _is_contradiction_revision_goal(_goal(case))
    assert not _is_contradiction_revision_goal(
        "Summarize the evidence and state the strongest current hypothesis."
    )


def test_same_model_gets_only_the_registered_revision_step(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    goal = _goal(case)
    results: dict[str, tuple[_RevisionModel, Any]] = {}
    for condition in ("full", "without_contradiction_revision"):
        root = tmp_path / condition
        root.mkdir()
        (root / "public-projection.json").write_text(
            json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        profile = execution_profile(condition, "a" * 64)
        profile["nonce"] = "7" * 32
        model = _RevisionModel()
        result = GenericMcpHost(
            task_id=f"revision-{condition}",
            evaluation_profile=profile,
            model=model,  # type: ignore[arg-type]
            executor=_executor(root),
            max_steps=3,
            resource_paths=("public-projection.json",),
        ).run(goal, task_kind="general")
        results[condition] = model, result

    full_model, full_result = results["full"]
    reduced_model, reduced_result = results["without_contradiction_revision"]
    assert full_model.offers == [
        (["host_read"], "host_read"),
        (["host_reason"], "host_reason"),
        (["host_complete"], "host_complete"),
    ]
    assert reduced_model.offers == [
        (["host_read"], "host_read"),
        (["host_complete"], "host_complete"),
    ]
    assert full_model.tool_results[0]["evidence"] == reduced_model.tool_results[0]["evidence"]
    draft_schema = full_model.parameters[1]
    full_answer_schema = dict(full_model.parameters[2]["properties"]["answer"])
    reduced_answer_schema = dict(reduced_model.parameters[1]["properties"]["answer"])
    full_answer_schema.pop("description")
    reduced_answer_schema.pop("description")
    assert full_answer_schema == reduced_answer_schema
    assert draft_schema["required"] == [
        "prior",
        "original_source",
        "decisive_source",
        "decisive_effect",
        "revised",
    ]
    assert set(draft_schema["properties"]["decisive_effect"]["enum"]) == {
        "supports",
        "refutes",
    }
    binding = full_model.tool_results[1]["reasoning_binding"]
    assert full_model.parameters[2]["properties"]["reasoning_binding"] == {"const": binding}
    assert full_model.tool_results[1]["revision_record"]["decisive_effect"] == "refutes"
    assert "effect_status_map value" in full_model.tool_results[1]["instruction"]
    assert full_result.delivered and reduced_result.delivered
    complete_request = next(
        event
        for event in full_result.events
        if event["type"] == "tool_request" and event["name"] == "host_complete"
    )
    assert complete_request["arguments"]["reasoning_binding"] == binding
    complete_result = next(
        event
        for event in full_result.events
        if event["type"] == "tool_result" and event["call_id"] == complete_request["call_id"]
    )
    assert json.loads(complete_result["content"])["reasoning_binding"] == binding
    assert validate_transcript(list(full_result.events))
    assert validate_transcript(list(reduced_result.events))
    full_receipt = next(
        event["receipt"] for event in full_result.events if event["type"] == "evaluation_receipt"
    )
    reduced_receipt = next(
        event["receipt"] for event in reduced_result.events if event["type"] == "evaluation_receipt"
    )
    assert full_receipt["operator_counts"]["contradiction_revision"] > 0
    assert reduced_receipt["operator_counts"]["contradiction_revision"] == 0


def test_schema_invalid_revision_is_measured_without_execution(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    model = _RevisionModel(invalid_revision_arguments=True)
    host = GenericMcpHost(
        task_id="invalid-revision",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
        resource_paths=("public-projection.json",),
    )

    result = host.run(_goal(case), task_kind="general")

    assert not result.delivered
    assert result.process_error is None
    assert result.events[-1]["disposition"] == "withhold"
    rejected = next(
        event
        for event in result.events
        if event["type"] == "tool_result" and event["origin"] == "mcp"
    )
    assert "violated the offered schema" in rejected["content"]
    assert host.runtime is not None
    assert host.runtime.server.runtime.active_sessions == 0
    assert validate_transcript(list(result.events))


def test_completion_rejects_answer_that_contradicts_accepted_revision_record(
    tmp_path: Path,
) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    model = _RevisionModel(
        revision_record={
            "prior": "h_alex_oncall",
            "original_source": "source_a",
            "decisive_source": "source_b",
            "decisive_effect": "supports",
            "revised": "h_alex_oncall",
        },
        answer={
            "prior": "h_alex_oncall",
            "prior_status": "refuted",
            "revised": "h_alex_oncall",
            "decisive_source": "source_b",
        },
    )
    result = GenericMcpHost(
        task_id="contradictory-revision",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
        resource_paths=("public-projection.json",),
    ).run(_goal(case), task_kind="general")

    assert not result.delivered
    assert "final answer contradicts the public revision record" in _rejected_completion_error(
        result
    )
    assert result.events[-1]["disposition"] == "withhold"


@pytest.mark.parametrize(
    "answer",
    [
        {
            "prior": "h_priya_oncall",
            "prior_status": "refuted",
            "revised": "h_priya_oncall",
            "decisive_source": "source_b",
        },
        {
            "prior": "h_alex_oncall",
            "prior_status": "refuted",
            "revised": "h_alex_oncall",
            "decisive_source": "source_b",
        },
        {
            "prior": "h_alex_oncall",
            "prior_status": "refuted",
            "revised": "h_priya_oncall",
            "decisive_source": "source_a",
        },
        {
            "prior": "h_alex_oncall",
            "prior_status": "supported",
            "revised": "h_priya_oncall",
            "decisive_source": "source_b",
        },
    ],
    ids=("prior", "revised", "source", "status"),
)
def test_every_final_revision_field_is_bound_to_the_accepted_record(
    tmp_path: Path,
    answer: dict[str, str],
) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    result = GenericMcpHost(
        task_id="field-binding",
        evaluation_profile=profile,
        model=_RevisionModel(answer=answer),  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
        resource_paths=("public-projection.json",),
    ).run(_goal(case), task_kind="general")

    assert not result.delivered
    assert "final answer contradicts the public revision record" in _rejected_completion_error(
        result
    )


@pytest.mark.parametrize("binding_mode", ["missing", "tampered"])
def test_completion_call_must_carry_the_exact_runtime_reasoning_binding(
    tmp_path: Path,
    binding_mode: str,
) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    result = GenericMcpHost(
        task_id=f"{binding_mode}-binding",
        evaluation_profile=profile,
        model=_RevisionModel(binding_mode=binding_mode),  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
        resource_paths=("public-projection.json",),
    ).run(_goal(case), task_kind="general")

    assert not result.delivered
    assert result.events[-1]["disposition"] == "withhold"
    assert "model tool arguments violated the offered schema" in result.events[-1]["text"]
    assert validate_transcript(list(result.events))


def test_verify_then_finish_path_cannot_bypass_the_revision_binding(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    runtime = EvaluatorMcpRuntime(profile, resource_paths=("public-projection.json",))
    runtime.start(_goal(case), task_kind="general")
    runtime.observe(
        _executor(tmp_path).execute(
            "read-record",
            "host_read",
            {"path": "public-projection.json"},
        )
    )
    runtime.lifecycle_call(
        "cortheon_step",
        {
            "draft": json.dumps(
                {
                    "prior": "h_alex_oncall",
                    "original_source": "source_a",
                    "decisive_source": "source_b",
                    "decisive_effect": "supports",
                    "revised": "h_alex_oncall",
                }
            )
        },
    )
    assert runtime.session_id is not None

    with pytest.raises(
        CognitiveRuntimeError,
        match="final answer contradicts the public revision record",
    ):
        runtime.server.runtime.verify(
            runtime.session_id,
            answer=json.dumps(
                {
                    "prior": "h_alex_oncall",
                    "prior_status": "refuted",
                    "revised": "h_alex_oncall",
                    "decisive_source": "source_b",
                }
            ),
            claims=[
                {
                    "claim": "source_b determines the current status of h_alex_oncall.",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )
