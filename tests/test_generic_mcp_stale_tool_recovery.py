from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_projection import (
    completion_answer_schema,
    host_discrimination_tool,
)
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal
from cortheon.qualification_core.conditions import execution_profile


class _StaleThenReasoningModel:
    evaluator_executes_exact_tools = True
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self, *, recover: bool) -> None:
        self.recover = recover
        self.calls = 0
        self.offers: list[str] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        self.calls += 1
        self.offers.append(tool_choice)
        if self.calls == 1 or (self.calls == 2 and not self.recover):
            return _turn(f"stale-{self.calls}", "host_read", {})
        if tool_choice == "host_reason":
            assert messages[-1]["role"] == "system"
            assert "host_reason" in messages[-1]["content"]
            return _turn(
                "reason",
                "host_reason",
                {
                    "hypotheses": [
                        {
                            "statement": (
                                "legacy_broker_overload causes activation_drop on weekend"
                            ),
                            "falsification_test": (
                                "route_new_broker; drop_persists refutes the leader"
                            ),
                        },
                        {
                            "statement": "cohort_selection_bias causes activation_drop on weekend",
                            "falsification_test": "Hold selection fixed and compare activation",
                        },
                    ]
                },
            )
        return _turn("complete", "host_complete", _completion())


def _turn(call_id: str, name: str, arguments: dict[str, Any]) -> ModelTurn:
    return ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall(call_id, name, arguments),),
        "tool_calls",
        5,
    )


def _completion() -> dict[str, Any]:
    evidence_ids = ["ev1"]
    return {
        "answer": {
            "leading": {
                "cause": "legacy_broker_overload",
                "outcome": "activation_drop",
                "scope": "weekend",
            },
            "rival": {
                "cause": "cohort_selection_bias",
                "outcome": "activation_drop",
                "scope": "weekend",
            },
            "falsification": {
                "intervention": "route_new_broker",
                "result": "drop_persists",
                "refutes": "legacy_broker_overload",
            },
        },
        "claims": [
            {
                "claim": (
                    "The record supports legacy_broker_overload as leader and "
                    "cohort_selection_bias as rival."
                ),
                "evidence_ids": evidence_ids,
            }
        ],
        "hypotheses": [
            {
                "statement": "legacy_broker_overload causes activation_drop on weekend",
                "falsification_test": "route_new_broker and check whether drop_persists",
                "status": "supported",
                "evidence_ids": evidence_ids,
            },
            {
                "statement": "cohort_selection_bias causes activation_drop on weekend",
                "falsification_test": "Hold selection fixed and compare activation",
                "status": "uncertain",
                "evidence_ids": evidence_ids,
            },
        ],
        "completion_evidence_ids": evidence_ids,
    }


def _host(tmp_path: Path, model: _StaleThenReasoningModel) -> GenericMcpHost:
    marker = "stale-tool-workspace"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    case = next(item for item in development_cases() if item.case_id == "hypothesis_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    return GenericMcpHost(
        task_id="stale-tool-recovery",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=4,
        resource_paths=("public-projection.json",),
    )


def test_one_stale_host_tool_call_gets_one_bounded_retry(tmp_path: Path) -> None:
    model = _StaleThenReasoningModel(recover=True)
    case = next(item for item in development_cases() if item.case_id == "hypothesis_01")

    result = _host(tmp_path, model).run(_goal(case), task_kind="general")

    assert result.delivered
    assert model.offers == ["host_reason", "host_reason", "host_complete"]
    rejected = [
        event
        for event in result.events
        if event["type"] == "tool_result" and "stale host tool" in event.get("content", "")
    ]
    assert len(rejected) == 1
    assert validate_transcript(list(result.events))


def test_repeated_stale_host_tool_call_is_withheld(tmp_path: Path) -> None:
    model = _StaleThenReasoningModel(recover=False)
    case = next(item for item in development_cases() if item.case_id == "hypothesis_01")

    result = _host(tmp_path, model).run(_goal(case), task_kind="general")

    assert not result.delivered
    assert model.calls == 2
    assert result.events[-1]["disposition"] == "withhold"
    assert validate_transcript(list(result.events))


class _StaleOncePerBridgeModel(_StaleThenReasoningModel):
    def __init__(self) -> None:
        super().__init__(recover=True)
        self.seen_choices: set[str] = set()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        if tool_choice not in self.seen_choices:
            self.seen_choices.add(tool_choice)
            self.calls += 1
            self.offers.append(tool_choice)
            return _turn(f"stale-{tool_choice}", "host_read", {})
        return super().complete(messages, tools, tool_choice=tool_choice)


def test_each_distinct_reasoning_bridge_gets_one_bounded_stale_retry(tmp_path: Path) -> None:
    model = _StaleOncePerBridgeModel()
    case = next(item for item in development_cases() if item.case_id == "hypothesis_01")

    result = _host(tmp_path, model).run(_goal(case), task_kind="general")

    rejected = [
        event
        for event in result.events
        if event["type"] == "tool_result" and "stale host tool" in event.get("content", "")
    ]
    assert len(rejected) == 2
    assert result.delivered
    assert validate_transcript(list(result.events))


class _ReachableModel:
    evaluator_executes_exact_tools = True
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self) -> None:
        self.calls = 0
        self.choices: list[str] = []

    def complete(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        self.calls += 1
        self.choices.append(tool_choice)
        return ModelTurn("local", "small", "bounded candidate", (), "stop", 3)


def test_duplicate_bound_evidence_cannot_create_an_exact_tool_loop(tmp_path: Path) -> None:
    marker = "duplicate-evidence-workspace"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    case = next(item for item in development_cases() if item.case_id == "discriminate_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "8" * 32
    model = _ReachableModel()
    result = GenericMcpHost(
        task_id="duplicate-evidence-loop",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=1,
        resource_paths=("public-projection.json",),
    ).run(_goal(case), task_kind="general")

    decisions = [event for event in result.events if event["type"] == "runtime_tool_decision"]
    assert model.calls == 1
    assert model.choices == ["host_reason"]
    assert len(decisions) == 1
    assert result.process_error is None
    assert validate_transcript(list(result.events))


def test_discrimination_projection_exposes_choices_without_the_answer(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "discriminate_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    schema = completion_answer_schema(tmp_path, ("public-projection.json",))

    assert schema is not None
    properties = schema["properties"]
    assert properties["probe_id"]["enum"] == ["warm_replay", "ping_region"]
    assert set(properties["positive_supports"]["enum"]) == {"h_cache", "h_network"}
    assert properties["positive_supports"] == properties["negative_supports"]
    assert schema["additionalProperties"] is False
    tool = host_discrimination_tool(schema)
    assert tool["function"]["name"] == "host_reason"
    assert tool["function"]["parameters"] == schema


class _DiscriminationModel:
    evaluator_executes_exact_tools = True
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        self.calls += 1
        answer = {
            "probe_id": "warm_replay",
            "positive_supports": "h_cache",
            "negative_supports": "h_network",
        }
        if tool_choice == "host_reason":
            return _turn("discriminate", "host_reason", answer)
        return _turn(
            "complete-discrimination",
            "host_complete",
            {
                "answer": answer,
                "claims": [
                    {
                        "claim": "warm_replay separates h_cache from h_network",
                        "evidence_ids": ["ev1"],
                    }
                ],
                "hypotheses": [
                    {
                        "statement": "warm_replay supports h_cache when replay is fast",
                        "falsification_test": "observe a slow warm replay",
                        "status": "uncertain",
                        "evidence_ids": ["ev1"],
                    },
                    {
                        "statement": "warm_replay supports h_network when replay stays slow",
                        "falsification_test": "observe a fast warm replay",
                        "status": "uncertain",
                        "evidence_ids": ["ev1"],
                    },
                ],
                "completion_evidence_ids": ["ev1"],
            },
        )


def test_discrimination_reasoning_can_advance_through_hypothesis_pass(
    tmp_path: Path,
) -> None:
    marker = "discrimination-transcript-workspace"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    case = next(item for item in development_cases() if item.case_id == "discriminate_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "6" * 32

    result = GenericMcpHost(
        task_id="discrimination-reasoning-pass",
        evaluation_profile=profile,
        model=_DiscriminationModel(),  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=3,
        resource_paths=("public-projection.json",),
    ).run(_goal(case), task_kind="general")

    assert result.delivered
    assert validate_transcript(list(result.events))


def test_cross_source_projection_closes_the_full_premise_path(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "derivation_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    schema = completion_answer_schema(tmp_path, ("public-projection.json",))

    assert schema is not None
    assert schema["required"] == ["subject", "relation", "object", "premises"]
    assert schema["properties"]["relation"]["enum"] == ["steward"]
    premises = schema["properties"]["premises"]
    assert premises["minItems"] == premises["maxItems"] == 3
    assert premises["items"]["properties"]["source_id"]["enum"] == [
        "source_a",
        "source_b",
        "source_c",
    ]
    assert schema["additionalProperties"] is False
