"""Hostile checks for generic MCP identity, cleanup, and closed capabilities."""

from __future__ import annotations

import copy
import hashlib
import itertools
from pathlib import Path
from typing import Any

import pytest

from cortheon.benchmark_core.generic_mcp_diagnostics import transcript_diagnostic
from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import (
    ModelToolCall,
    ModelTurn,
    OpenAiModelClient,
)
from cortheon.benchmark_core.generic_mcp_order_validation import host_request_matches_action
from cortheon.benchmark_core.generic_mcp_process import _web_provider
from cortheon.benchmark_core.generic_mcp_protocol import canonical_json
from cortheon.benchmark_core.generic_mcp_runtime import EvaluatorMcpRuntime
from cortheon.benchmark_core.generic_mcp_search_projection import discovery_pattern
from cortheon.benchmark_core.generic_mcp_tools import host_tool_definitions
from cortheon.benchmark_core.generic_mcp_turns import bind_forced_arguments, bind_forced_turn
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.qualification_core.conditions import execution_profile


class _Model:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self, turn: ModelTurn) -> None:
        self.turn = turn

    def complete(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        assert tool_choice == "auto" or any(
            tool["function"]["name"] == tool_choice for tool in _tools
        )
        return self.turn


def _profile(condition: str) -> dict[str, Any]:
    profile = execution_profile(condition, "a" * 64)
    profile["nonce"] = "5" * 32
    return profile


def _executor(root: Path) -> IsolatedExecutor:
    marker = "hardening-workspace"
    (root / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    return IsolatedExecutor(root, marker_nonce=marker)


def _valid_bare_events(root: Path) -> list[dict[str, Any]]:
    turn = ModelTurn("local", "small", "answer", (), "stop", 1)
    host = GenericMcpHost(
        task_id="bounded-task",
        evaluation_profile=_profile("bare"),
        model=_Model(turn),  # type: ignore[arg-type]
        executor=_executor(root),
        max_steps=1,
    )
    return list(host.run("Question").events)


def test_terminal_disposition_provenance_cross_products_are_rejected(tmp_path: Path) -> None:
    valid = _valid_bare_events(tmp_path)
    assert validate_transcript(valid)
    combinations = itertools.product(
        ("release", "withhold", "fail_open"),
        ("cortheon_complete", "generic_mcp_model", "generic_mcp_wrapper"),
        ("certified", "stop", "bounded_incomplete", "degraded_runtimeerror"),
    )
    for disposition, provenance, reason in combinations:
        if (disposition, provenance, reason) == (
            "release",
            "generic_mcp_model",
            "stop",
        ):
            continue
        events = copy.deepcopy(valid)
        events[-1].update(
            disposition=disposition,
            provenance=provenance,
            finish_reason=reason,
        )
        assert not validate_transcript(events)


def test_transcript_diagnostic_is_content_free_and_specific(tmp_path: Path) -> None:
    valid = _valid_bare_events(tmp_path)
    assert transcript_diagnostic(valid) is None
    assert transcript_diagnostic(valid[:-1]) == "missing_terminal"
    broken = copy.deepcopy(valid)
    message = next(event for event in broken if event["type"] == "message")
    message["tool_call_ids"] = ["unresolved"]
    assert transcript_diagnostic(broken) == "unresolved_announced_tool_call"


def test_validator_rejects_unbounded_identity_calls_and_false_capabilities(
    tmp_path: Path,
) -> None:
    valid = _valid_bare_events(tmp_path)
    variants = []
    oversized_task = copy.deepcopy(valid)
    for event in oversized_task:
        event["task_id"] = "x" * 129
    variants.append(oversized_task)
    invalid_nonce = copy.deepcopy(valid)
    for event in invalid_nonce:
        event["nonce"] = "not valid"
    variants.append(invalid_nonce)
    false_isolation = copy.deepcopy(valid)
    false_isolation[0]["capabilities"]["isolated_workspace"] = False
    variants.append(false_isolation)
    oversized_finish = copy.deepcopy(valid)
    next(event for event in oversized_finish if event["type"] == "message")["finish_reason"] = (
        "x" * 129
    )
    variants.append(oversized_finish)
    assert all(not validate_transcript(events) for events in variants)


def test_validator_binds_the_exact_model_tool_catalogue(tmp_path: Path) -> None:
    valid = _valid_bare_events(tmp_path)
    assert validate_transcript(valid)
    message = next(event for event in valid if event["type"] == "message")

    mutated_schema = copy.deepcopy(valid)
    changed = next(event for event in mutated_schema if event["type"] == "message")
    changed["tool_catalogue"][0]["function"]["description"] = "mutated"
    assert not validate_transcript(mutated_schema)

    mutated_digest = copy.deepcopy(valid)
    changed = next(event for event in mutated_digest if event["type"] == "message")
    changed["tool_catalogue_sha256"] = "0" * 64
    assert not validate_transcript(mutated_digest)
    assert message["tool_catalogue_sha256"] != "0" * 64


@pytest.mark.parametrize("bad_cost", [True, "0.01"])
def test_model_cost_rejects_coercible_non_numbers(bad_cost: object) -> None:
    client = OpenAiModelClient(
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        provider_id="local",
        model_id="small",
        timeout_seconds=1,
        output_tokens=10,
    )
    payload = {
        "model": "small",
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 1, "cost": bad_cost},
    }
    with pytest.raises(RuntimeError, match="cost"):
        client._turn(payload)


@pytest.mark.parametrize("bad_duration", [True, "1.0", -1.0, float("inf")])
def test_model_rejects_invalid_omlx_load_duration(bad_duration: object) -> None:
    client = OpenAiModelClient(
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        provider_id="omlx",
        model_id="small",
        timeout_seconds=1,
        output_tokens=10,
    )
    payload = {
        "model": "small",
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 1, "model_load_duration": bad_duration},
    }
    with pytest.raises(RuntimeError, match="model_load_duration"):
        client._turn(payload)


def test_model_accepts_valid_omlx_load_duration() -> None:
    client = OpenAiModelClient(
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        provider_id="omlx",
        model_id="small",
        timeout_seconds=1,
        output_tokens=10,
    )
    turn = client._turn(
        {
            "model": "small",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 1, "model_load_duration": 0.25},
        }
    )
    assert turn.tokens == 1


@pytest.mark.parametrize(
    ("action", "arguments", "allowed"),
    [
        (
            {
                "type": "reason",
                "submit_via": "cortheon_challenge",
                "required_fields": ["draft", "claims"],
            },
            {"answer": {}, "claims": []},
            True,
        ),
        (
            {
                "type": "reason",
                "submit_via": "cortheon_step",
                "required_fields": ["hypotheses"],
            },
            {"hypotheses": []},
            True,
        ),
        (
            {
                "type": "reason",
                "submit_via": "cortheon_challenge",
                "required_fields": ["draft", "claims"],
            },
            {"answer": {}},
            False,
        ),
        (
            {
                "type": "reason",
                "submit_via": "cortheon_challenge",
                "required_fields": ["unknown"],
            },
            {"unknown": "value"},
            False,
        ),
    ],
)
def test_fused_reason_completion_requires_every_projected_field(
    action: dict[str, object],
    arguments: dict[str, object],
    allowed: bool,
) -> None:
    request = {"name": "host_complete", "arguments": arguments}
    assert host_request_matches_action(action, request, []) is allowed


def test_forced_evaluator_arguments_replace_model_serialization_errors() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "host_read_many",
            "description": "Read fixed files.",
            "parameters": {
                "type": "object",
                "properties": {"paths": {"const": ["a.txt", "b.txt"]}},
                "required": ["paths"],
                "additionalProperties": False,
            },
        },
    }
    turn = ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall("read", "host_read_many", {"paths": '["a.txt","b.txt"]'}),),
        "tool_calls",
        1,
    )
    bound = bind_forced_arguments(turn, [tool], "host_read_many")
    assert bound.tool_calls[0].arguments == {"paths": ["a.txt", "b.txt"]}
    assert turn.tool_calls[0].arguments == {"paths": '["a.txt","b.txt"]'}
    assert bind_forced_arguments(turn, [tool], "auto") is turn

    wrong_tool = ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall("read", "host_search", {"pattern": "stale", "path": "."}),),
        "tool_calls",
        1,
    )
    rebound, binding = bind_forced_turn(wrong_tool, [tool], "host_read_many")
    assert rebound.tool_calls[0].name == "host_read_many"
    assert rebound.tool_calls[0].arguments == {"paths": ["a.txt", "b.txt"]}
    assert binding == "tool_and_arguments"

    repair_tool = {
        "type": "function",
        "function": {
            "name": "host_reason",
            "parameters": {
                "type": "object",
                "properties": {"draft": {"type": "string"}},
                "required": ["draft"],
                "additionalProperties": False,
            },
        },
    }
    completion_shaped = ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall("repair", "host_reason", {"answer": "Revised.", "claims": []}),),
        "tool_calls",
        1,
    )
    repaired, binding = bind_forced_turn(completion_shaped, [repair_tool], "host_reason")
    assert repaired.tool_calls[0].arguments == {"draft": "Revised."}
    assert binding == "repair_projection"


def test_replace_is_not_exposed_without_a_runtime_mutation_request(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    names = {tool["function"]["name"] for tool in host_tool_definitions(web_enabled=False)}
    assert "host_replace" not in names
    with pytest.raises(ValueError, match="closed catalogue"):
        executor.execute(
            "replace-1",
            "host_replace",
            {"path": "file.py", "old": "a", "new": "b"},
        )


def test_search_supports_bounded_literal_alternatives(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    (tmp_path / "index.md").write_text("Deploy Atlas Portal.\n", encoding="utf-8")

    execution = executor.execute(
        "search-alternatives",
        "host_search",
        {"pattern": "Atlas|Pipeline", "path": "."},
    )

    assert execution.status == "match"
    assert "Deploy Atlas Portal" in execution.content


def test_document_discovery_prefers_quoted_task_terms() -> None:
    query = "Search the live project for document paths for: decide what 'deploy Atlas' means."

    assert discovery_pattern(query) == "deploy|Atlas"


def test_failed_mcp_abandon_falls_back_to_direct_in_memory_cleanup() -> None:
    runtime = EvaluatorMcpRuntime(_profile("full"))
    runtime.start("Does example.py import pathlib?")

    def fail_lifecycle(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("MCP transport died")

    runtime.lifecycle_call = fail_lifecycle  # type: ignore[method-assign]
    assert runtime.abandon()
    assert runtime.closed
    assert runtime.server.runtime.active_sessions == 0


def test_host_emits_claim_ineligible_terminal_after_transport_cleanup(
    tmp_path: Path,
) -> None:
    turn = ModelTurn(
        "local",
        "small",
        "Candidate",
        (ModelToolCall("bad-call", "cortheon_observe", {}),),
        "tool_calls",
        1,
    )
    host = GenericMcpHost(
        task_id="cleanup-task",
        evaluation_profile=_profile("full"),
        model=_Model(turn),  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=1,
    )
    result = host.run("Does example.py import pathlib?")
    assert result.process_error is None
    assert not result.delivered
    assert result.events[-1]["disposition"] == "withhold"
    assert validate_transcript(list(result.events))
    assert host.runtime is not None
    assert host.runtime.server.runtime.active_sessions == 0


def test_mismatched_host_arguments_are_bounded_candidate_errors(tmp_path: Path) -> None:
    turn = ModelTurn(
        "local",
        "small",
        "",
        (
            ModelToolCall(
                "wrong-read",
                "host_search",
                {"pattern": "pathlib", "path": "unrelated.txt"},
            ),
        ),
        "tool_calls",
        1,
    )
    host = GenericMcpHost(
        task_id="argument-mismatch",
        evaluation_profile=_profile("full"),
        model=_Model(turn),  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=1,
    )
    result = host.run("Does example.py import pathlib?")
    assert result.process_error is None
    assert not result.delivered
    rejected = next(event for event in result.events if event["type"] == "tool_result")
    assert rejected["status"] == "error"
    assert validate_transcript(list(result.events))
    assert host.runtime is not None
    assert host.runtime.server.runtime.active_sessions == 0


def test_duplicate_forced_calls_are_resolved_once_and_withheld(tmp_path: Path) -> None:
    arguments = {"pattern": "pathlib", "path": "example.py"}
    turn = ModelTurn(
        "local",
        "small",
        "",
        (
            ModelToolCall("duplicate-a", "host_search", arguments),
            ModelToolCall("duplicate-b", "host_search", arguments),
        ),
        "tool_calls",
        2,
    )
    host = GenericMcpHost(
        task_id="duplicate-forced",
        evaluation_profile=_profile("full"),
        model=_Model(turn),  # type: ignore[arg-type]
        executor=_executor(tmp_path),
        max_steps=1,
    )

    result = host.run("Does example.py import pathlib?")

    assert not result.delivered and result.process_error is None
    assert result.tool_calls == 2
    assert [event["status"] for event in result.events if event["type"] == "tool_result"] == [
        "error",
        "error",
    ]
    assert validate_transcript(list(result.events))
    assert transcript_diagnostic(list(result.events)) is None
    assert host.runtime is not None and host.runtime.server.runtime.active_sessions == 0


def test_web_provider_executes_resolved_hashed_path_and_detects_change(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "provider"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo provider-1; exit 0; fi\n'
        "echo '{\"results\":[]}'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    link = tmp_path / "provider-link"
    link.symlink_to(executable)
    provider, identity = _web_provider([str(link)])
    assert identity["executable_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert (
        identity["config_sha256"]
        == hashlib.sha256(canonical_json([str(executable.resolve())]).encode()).hexdigest()
    )
    executable.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after handshake"):
        provider("host_web_search", {"query": "release"})
