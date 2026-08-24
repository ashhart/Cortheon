"""Small-model protocol adherence without weakening Cortheon verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_runner import _task_kind
from cortheon.benchmark_core.generic_mcp_runtime import EvaluatorMcpRuntime
from cortheon.benchmark_core.generic_mcp_tools import ToolExecution, ToolRequest
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal
from cortheon.qualification_core.conditions import execution_profile


def _profile() -> dict[str, Any]:
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    return profile


def _executor(root: Path) -> IsolatedExecutor:
    marker = "microstep-workspace"
    (root / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    (root / "example.py").write_text("import json\n", encoding="utf-8")
    return IsolatedExecutor(root, marker_nonce=marker)


class _AdherentSmallModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(
        self,
        *,
        evidence_id: str = "ev1",
        first_name: str = "host_search",
        first_arguments: dict[str, Any] | None = None,
    ) -> None:
        self.evidence_id = evidence_id
        self.first_name = first_name
        self.first_arguments = first_arguments or {"pattern": "pathlib", "path": "example.py"}
        self.offers: list[tuple[list[str], str]] = []
        self.parameters: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        names = [tool["function"]["name"] for tool in tools]
        self.offers.append((names, tool_choice))
        self.parameters.append(tools[0]["function"]["parameters"])
        if len(self.offers) == 1:
            return ModelTurn(
                "local",
                "small",
                "",
                (
                    ModelToolCall(
                        "read-1",
                        self.first_name,
                        self.first_arguments,
                    ),
                ),
                "tool_calls",
                3,
            )
        start = next(
            json.loads(message["content"])
            for message in messages
            if message["role"] == "system"
            and isinstance(message["content"], str)
            and message["content"].startswith("{")
            and "session" in json.loads(message["content"])
        )
        assert start["session"]["session_id"]
        return ModelTurn(
            "local",
            "small",
            "",
            (
                ModelToolCall(
                    "complete-1",
                    "host_complete",
                    {
                        "answer": "No.",
                        "claims": [
                            {
                                "claim": "example.py does not import pathlib.",
                                "evidence_ids": [self.evidence_id],
                            }
                        ],
                        "hypotheses": [
                            {
                                "statement": "The pathlib import is absent.",
                                "falsification_test": "Search example.py for pathlib.",
                                "status": "supported",
                                "evidence_ids": [self.evidence_id],
                            }
                        ],
                        "completion_evidence_ids": [self.evidence_id],
                    },
                ),
            ),
            "tool_calls",
            4,
        )


def test_runtime_projects_one_forced_microstep_then_one_completion(tmp_path: Path) -> None:
    model = _AdherentSmallModel()
    host = GenericMcpHost(
        task_id="microstep-task",
        evaluation_profile=_profile(),
        model=model,  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
    )

    result = host.run("Does example.py import pathlib?", task_kind="code")

    assert result.delivered and result.final_text == "No."
    assert model.offers == [(["host_search"], "host_search"), (["host_complete"], "host_complete")]
    messages = [event for event in result.events if event["type"] == "message"]
    assert [event["available_tools"] for event in messages] == [
        ["host_search"],
        ["host_complete"],
    ]
    assert [event["tool_choice"] for event in messages] == ["host_search", "host_complete"]
    assert model.parameters[0]["properties"] == {
        "pattern": {"const": "pathlib"},
        "path": {"const": "example.py"},
    }
    assert validate_transcript(list(result.events))


def test_host_complete_never_repairs_false_evidence_links(tmp_path: Path) -> None:
    model = _AdherentSmallModel(evidence_id="ev999")
    host = GenericMcpHost(
        task_id="unsupported-evidence",
        evaluation_profile=_profile(),
        model=model,  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=2,
    )

    result = host.run("Does example.py import pathlib?", task_kind="code")

    assert not result.delivered
    assert result.final_text.startswith("[Cortheon withheld:")
    assert result.events[-1]["provenance"] == "generic_mcp_wrapper"
    assert result.process_error is None
    assert validate_transcript(list(result.events))


def test_explicit_operator_case_kind_is_general() -> None:
    class Case:
        task_kind = "general"

    assert _task_kind(Case()) == "general"


def test_pathless_general_inspection_is_bound_to_evaluator_resource() -> None:
    runtime = EvaluatorMcpRuntime(
        _profile(),
        resource_paths=("public-projection.json",),
    )

    runtime.start("Frame competing causal hypotheses from the sealed input.", task_kind="general")

    assert runtime.projected_host_tool() == "host_read"
    assert runtime.projected_arguments("host_read") == {
        "path": "public-projection.json",
    }
    assert runtime.validate_host_arguments("host_read", {"path": "public-projection.json"})
    assert not runtime.validate_host_arguments("host_read", {"path": "private-oracle.json"})


def test_completion_schema_tells_small_models_to_localize_uncertainty() -> None:
    from cortheon.benchmark_core.generic_mcp_projection import host_complete_tool

    description = host_complete_tool()["function"]["parameters"]["properties"]["answer"][
        "description"
    ]
    assert "clause that explicitly says it remains uncertain" in description
    hypotheses = host_complete_tool()["function"]["parameters"]["properties"]["hypotheses"]
    statement = hypotheses["items"]["properties"]["statement"]["description"]
    assert "Name the actual component or mechanism" in statement


def test_repair_tool_requires_one_complete_replacement_draft() -> None:
    from cortheon.benchmark_core.generic_mcp_projection import host_repair_tool

    function = host_repair_tool()["function"]
    assert function["name"] == "host_reason"
    assert function["parameters"]["required"] == ["draft"]
    assert function["parameters"]["additionalProperties"] is False


def test_bound_resource_is_a_sourced_task_record_for_numeric_claims() -> None:
    runtime = EvaluatorMcpRuntime(
        _profile(),
        resource_paths=("public-projection.json",),
    )
    runtime.start("Identify the broker threshold.", task_kind="general")
    request = ToolRequest.create(
        "read-record",
        "host_read",
        {"path": "public-projection.json"},
    )
    execution = ToolExecution(
        request,
        "result",
        "The broker threshold is 500 requests.",
        {
            "tool": "read",
            "executor": "generic_mcp_wrapper",
            "outcome": "result",
            "args": {"path": "public-projection.json"},
        },
    )

    observed = runtime.observe(execution)
    evidence = observed["context"]["evidence"][0]
    assert evidence["kind"] == "artifact"
    assert evidence["source"] == "public-projection.json"

    completed = runtime.lifecycle_call(
        "cortheon_complete",
        {
            "answer": "The broker threshold is 500 requests.",
            "claims": [
                {
                    "claim": "The broker threshold is 500 requests.",
                    "evidence_ids": ["ev1"],
                }
            ],
            "hypotheses": [
                {
                    "statement": "The broker threshold is 500 requests.",
                    "falsification_test": "Read the task record.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            "completion_evidence_ids": ["ev1"],
        },
    )
    profiles = (
        completed.get("claim_verification") or completed["verification"]["claim_verification"]
    )
    profile = profiles[0]
    assert profile["claim_type"] == "quantitative"
    assert not any("directly read task record" in gap for gap in profile["gaps"])


def test_read_many_projects_one_exact_receipt_per_file(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    executor = _executor(root)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    execution = executor.execute(
        "read-many",
        "host_read_many",
        {"paths": ["a.txt", "b.txt"]},
    )

    observations = EvaluatorMcpRuntime._read_many_observations(execution)

    assert [item["content"] for item in observations] == ["alpha", "beta"]
    assert [item["source"] for item in observations] == ["a.txt", "b.txt"]
    assert [item["host_receipt"]["args"] for item in observations] == [
        {"filePath": "a.txt"},
        {"filePath": "b.txt"},
    ]


def test_single_resource_result_gives_model_evidence_not_runtime_state(tmp_path: Path) -> None:
    model = _AdherentSmallModel()
    host = GenericMcpHost(
        task_id="single-resource",
        evaluation_profile=_profile(),
        model=model,  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=3,
        resource_paths=("example.py",),
    )

    result = host.run("Does example.py import pathlib?", task_kind="code")

    assert model.offers == [
        (["host_search"], "host_search"),
        (["host_complete"], "host_complete"),
    ]
    tool_messages = [event for event in result.events if event["type"] == "tool_result"]
    assert tool_messages[0]["content"] == "No matches."
    assert "context" not in tool_messages[0]


class _MediatedModel:
    provider_id = "local"
    model_id = "same-model"
    endpoint_sha256 = "e" * 64

    def __init__(self) -> None:
        self.calls = 0
        self.tool_results: list[dict[str, Any]] = []
        self.offers: list[tuple[list[str], str]] = []
        self.parameters: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        self.calls += 1
        self.offers.append(([item["function"]["name"] for item in tools], tool_choice))
        self.parameters.append(tools[0]["function"]["parameters"])
        if messages[-1]["role"] == "tool":
            self.tool_results.append(json.loads(messages[-1]["content"]))
        if self.calls == 1:
            return ModelTurn(
                self.provider_id,
                self.model_id,
                "",
                (ModelToolCall("read", "host_read", {"path": "public-projection.json"}),),
                "tool_calls",
                2,
            )
        if tool_choice == "host_reason":
            return ModelTurn(
                self.provider_id,
                self.model_id,
                "",
                (
                    ModelToolCall(
                        "reason",
                        "host_reason",
                        {
                            "hypotheses": [
                                {
                                    "statement": (
                                        "legacy_broker_overload causes activation_drop "
                                        "in weekend accounts"
                                    ),
                                    "falsification_test": (
                                        "route_new_broker; drop_persists refutes the leader"
                                    ),
                                },
                                {
                                    "statement": (
                                        "cohort_selection_bias causes activation_drop "
                                        "in weekend accounts"
                                    ),
                                    "falsification_test": (
                                        "hold cohort selection fixed and compare activation"
                                    ),
                                },
                            ]
                        },
                    ),
                ),
                "tool_calls",
                2,
            )
        answer = {
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
        }
        return ModelTurn(
            self.provider_id,
            self.model_id,
            "",
            (
                ModelToolCall(
                    "complete",
                    "host_complete",
                    {
                        "answer": answer,
                        "claims": [
                            {
                                "claim": (
                                    "The task record supports legacy_broker_overload as "
                                    "the leader and cohort_selection_bias as the rival."
                                ),
                                "evidence_ids": ["ev1"],
                            }
                        ],
                        "hypotheses": [
                            {
                                "statement": (
                                    "legacy_broker_overload causes activation_drop in weekend"
                                ),
                                "falsification_test": (
                                    "route_new_broker and observe whether drop_persists"
                                ),
                                "status": "supported",
                                "evidence_ids": ["ev1"],
                            },
                            {
                                "statement": (
                                    "cohort_selection_bias causes activation_drop in weekend"
                                ),
                                "falsification_test": (
                                    "hold cohort selection fixed and compare activation"
                                ),
                                "status": "uncertain",
                                "evidence_ids": ["ev1"],
                            },
                        ],
                        "completion_evidence_ids": ["ev1"],
                    },
                ),
            ),
            "tool_calls",
            2,
        )


def test_same_model_sees_only_registered_hypothesis_intervention(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "hypothesis_01")
    goal = _goal(case)
    results: dict[str, tuple[_MediatedModel, Any]] = {}
    for condition in ("full", "without_hypothesis_framing"):
        root = tmp_path / condition
        root.mkdir()
        executor = _executor(root)
        (root / "public-projection.json").write_text(
            json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        profile = execution_profile(condition, "a" * 64)
        profile["nonce"] = "7" * 32
        model = _MediatedModel()
        result = GenericMcpHost(
            task_id=f"mediation-{condition}",
            evaluation_profile=profile,
            model=model,  # type: ignore[arg-type]
            executor=executor,
            max_steps=3,
            resource_paths=("public-projection.json",),
        ).run(goal, task_kind="general")
        results[condition] = model, result

    full_model, full_result = results["full"]
    reduced_model, reduced_result = results["without_hypothesis_framing"]
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
    assert full_model.tool_results[0]["accepted_evidence_ids"] == ["ev1"]
    assert reduced_model.tool_results[0]["accepted_evidence_ids"] == ["ev1"]
    assert "exactly 2 evidence-grounded" in full_model.tool_results[0]["next_action"]["instruction"]
    assert (
        "Synthesize a compact draft" in reduced_model.tool_results[0]["next_action"]["instruction"]
    )
    assert full_result.model_steps == 3
    assert reduced_result.model_steps == 2
    assert full_result.tool_calls == 3
    assert reduced_result.tool_calls == 2
    assert full_result.delivered and reduced_result.delivered
    full_answer = full_model.parameters[-1]["properties"]["answer"]
    reduced_answer = reduced_model.parameters[-1]["properties"]["answer"]
    assert full_answer == reduced_answer
    assert full_answer["properties"]["falsification"]["properties"]["refutes"] == {
        "type": "string",
        "enum": ["cohort_selection_bias", "legacy_broker_overload"],
    }
