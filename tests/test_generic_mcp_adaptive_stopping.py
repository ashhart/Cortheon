from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_projection import completion_answer_schema
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal
from cortheon.qualification_core.conditions import execution_profile


def _turn(call_id: str, name: str, arguments: dict[str, Any]) -> ModelTurn:
    return ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall(call_id, name, arguments),),
        "tool_calls",
        5,
    )


def test_stopping_projection_exposes_actions_cost_shape_without_observations(
    tmp_path: Path,
) -> None:
    case = next(item for item in development_cases() if item.case_id == "stopping_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    schema = completion_answer_schema(tmp_path, ("public-projection.json",))

    assert schema is not None
    assert schema["properties"]["actions"]["items"]["enum"] == [
        "hash_binary",
        "read_changelog",
        "scan_all_logs",
    ]
    assert schema["properties"]["decision"]["enum"] == ["build_cobalt", "undetermined"]
    assert schema["properties"]["stop_reason"]["enum"] == ["sufficient"]
    assert "hash_matches_cobalt" not in json.dumps(schema)


class _StoppingModel:
    evaluator_executes_exact_tools = True
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def complete(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        if tool_choice == "host_reason":
            return _turn(
                "stopping-hypotheses",
                "host_reason",
                {
                    "hypotheses": [
                        {
                            "statement": "the signed hash identifies build_cobalt",
                            "falsification_test": "observe a non-cobalt signed hash",
                        },
                        {
                            "statement": "the signed hash identifies build_amber",
                            "falsification_test": "observe hash_matches_cobalt",
                        },
                    ]
                },
            )
        return _turn(
            "stopping-complete",
            "host_complete",
            {
                "answer": {
                    "actions": ["hash_binary"],
                    "decision": "build_cobalt",
                    "total_cost": 1,
                    "stop_reason": "sufficient",
                },
                "claims": [
                    {
                        "claim": "hash_matches_cobalt identifies build_cobalt",
                        "evidence_ids": ["ev2"],
                    },
                    {
                        "claim": "The public task requires one signed binary hash probe.",
                        "evidence_ids": ["ev1"],
                    },
                ],
                "hypotheses": [
                    {
                        "statement": "the signed hash identifies build_cobalt",
                        "falsification_test": "observe a non-cobalt signed hash",
                        "status": "supported",
                        "evidence_ids": ["ev2"],
                    },
                    {
                        "statement": "the signed hash identifies build_amber",
                        "falsification_test": "observe hash_matches_cobalt",
                        "status": "refuted",
                        "evidence_ids": ["ev2"],
                    },
                ],
                "completion_evidence_ids": ["ev1", "ev2"],
            },
        )


def _host(tmp_path: Path, condition: str) -> GenericMcpHost:
    marker = f"adaptive-stopping-{condition}"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    case = next(item for item in development_cases() if item.case_id == "stopping_01")
    (tmp_path / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (tmp_path / "actions").mkdir()
    for action_id, observation in case.oracle["observations"]:
        (tmp_path / "actions" / f"{action_id}.txt").write_text(observation, encoding="utf-8")
    profile = execution_profile(condition, "a" * 64)
    profile["nonce"] = ("3" if condition == "full" else "1") * 32
    return GenericMcpHost(
        task_id=f"adaptive-stopping-{condition}",
        evaluation_profile=profile,
        model=_StoppingModel(),  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=3,
        resource_paths=("public-projection.json",),
    )


def test_adaptive_stopping_executes_only_the_sufficient_public_probe(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "stopping_01")
    result = _host(tmp_path, "full").run(_goal(case), task_kind="general")
    action_reads = [
        event["arguments"]["path"]
        for event in result.events
        if event.get("type") == "tool_request"
        and event.get("name") == "host_read"
        and str(event.get("arguments", {}).get("path", "")).startswith("actions/")
    ]
    assert action_reads == ["actions/hash_binary.txt"]
    assert result.delivered
    assert validate_transcript(list(result.events))


def test_adaptive_completion_cannot_claim_an_action_that_never_ran(tmp_path: Path) -> None:
    case = next(item for item in development_cases() if item.case_id == "stopping_01")
    result = _host(tmp_path, "without_adaptive_stopping").run(_goal(case), task_kind="general")

    assert not result.delivered
    assert "withheld" in result.final_text
    assert validate_transcript(list(result.events))
