"""Conservative small-model schema projection for forced bridge calls."""

import json
from pathlib import Path

from cortheon.benchmark_core.generic_mcp_diagnostics import transcript_diagnostic
from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_projection import (
    host_complete_tool,
    host_completion_repair_tool,
)
from cortheon.benchmark_core.generic_mcp_turns import bind_forced_turn
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.qualification_core.conditions import execution_profile


def test_closed_projection_only_removes_unknown_nested_fields() -> None:
    arguments = {
        "answer": "Answer.",
        "claims": [{"claim": "Fact.", "evidence_ids": ["ev1"], "description": "extra"}],
        "hypotheses": [
            {
                "statement": "Possibility.",
                "falsification_test": "Check it.",
                "status": "uncertain",
                "evidence_ids": ["ev1"],
                "description": "extra",
            }
        ],
        "completion_evidence_ids": ["ev1"],
    }
    turn = ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall("complete", "host_complete", arguments),),
        "tool_calls",
        1,
    )

    projected, binding = bind_forced_turn(turn, [host_complete_tool()], "host_complete")

    assert projected.tool_calls[0].arguments["claims"] == [
        {"claim": "Fact.", "evidence_ids": ["ev1"]}
    ]
    assert "description" not in projected.tool_calls[0].arguments["hypotheses"][0]
    assert binding == "schema_projection"
    assert turn.tool_calls[0].arguments == arguments


def test_projection_never_invents_a_missing_required_field() -> None:
    turn = ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall("complete", "host_complete", {"answer": "Incomplete."}),),
        "tool_calls",
        1,
    )

    projected, binding = bind_forced_turn(turn, [host_complete_tool()], "host_complete")

    assert projected is turn
    assert binding == "none"


def test_projection_decodes_an_exact_model_echoed_reasoning_binding() -> None:
    reasoning_binding = {
        "schema_version": "1",
        "reasoning_binding_sha256": "a" * 64,
    }
    tool = host_complete_tool(None, reasoning_binding)
    arguments = {
        "answer": "Answer.",
        "claims": [{"claim": "Fact.", "evidence_ids": ["ev1"]}],
        "hypotheses": [
            {
                "statement": "Possibility.",
                "falsification_test": "Check it.",
                "status": "supported",
                "evidence_ids": ["ev1"],
            }
        ],
        "completion_evidence_ids": ["ev1"],
        "reasoning_binding": json.dumps(reasoning_binding),
    }
    turn = ModelTurn(
        "local",
        "small",
        "",
        (ModelToolCall("complete", "host_complete", arguments),),
        "tool_calls",
        1,
    )

    projected, binding = bind_forced_turn(turn, [tool], "host_complete")

    assert projected.tool_calls[0].arguments["reasoning_binding"] == reasoning_binding
    assert binding == "arguments"


def test_answer_repair_freezes_prior_model_authored_support_fields() -> None:
    previous = {
        "answer": "Old answer.",
        "claims": [{"claim": "Fact.", "evidence_ids": ["ev1"]}],
        "hypotheses": [
            {
                "statement": "Cache compaction is possible.",
                "falsification_test": "Inspect compaction state.",
                "status": "uncertain",
                "evidence_ids": ["ev1"],
            }
        ],
        "completion_evidence_ids": ["ev1"],
    }
    tool = host_completion_repair_tool(previous, None, None)
    turn = ModelTurn(
        "local",
        "small",
        "",
        (
            ModelToolCall(
                "repair",
                "host_complete",
                {
                    "answer": "Cache compaction remains uncertain.",
                    "claims": [{"claim": "FORGED", "evidence_ids": ["ev999"]}],
                },
            ),
        ),
        "tool_calls",
        1,
    )

    repaired, binding = bind_forced_turn(turn, [tool], "host_complete")
    arguments = repaired.tool_calls[0].arguments

    assert arguments["answer"] == "Cache compaction remains uncertain."
    assert arguments["claims"] == previous["claims"]
    assert arguments["hypotheses"] == previous["hypotheses"]
    assert arguments["completion_evidence_ids"] == ["ev1"]
    assert binding == "arguments"


class _TwiceWrongModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            call = ModelToolCall(
                "search",
                "host_search",
                {"pattern": "pathlib", "path": "example.py"},
            )
        else:
            call = ModelToolCall(
                f"complete-{self.calls}",
                "host_complete",
                {
                    "answer": "Yes." if self.calls == 2 else "Definitely yes.",
                    "claims": [{"claim": "example.py imports pathlib.", "evidence_ids": ["ev1"]}],
                    "hypotheses": [
                        {
                            "statement": "The pathlib import is present.",
                            "falsification_test": "Search example.py for pathlib.",
                            "status": "supported",
                            "evidence_ids": ["ev1"],
                        }
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
            1,
        )


def test_changed_completion_is_withheld_after_one_bounded_retry(tmp_path: Path) -> None:
    marker = "bounded-retry"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    (tmp_path / "example.py").write_text("import json\n", encoding="utf-8")
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    model = _TwiceWrongModel()
    host = GenericMcpHost(
        task_id="bounded-retry",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=8,
    )

    result = host.run("Does example.py import pathlib?", task_kind="code")

    assert not result.delivered
    assert result.process_error is None
    assert result.model_steps == 3
    assert result.final_text == (
        "[Cortheon withheld: completion remained unverified after one bounded retry]"
    )
    assert validate_transcript(list(result.events))


class _DuplicateIdModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        call = ModelToolCall(
            "same-id",
            "host_search",
            {"pattern": "pathlib", "path": "example.py"},
        )
        return ModelTurn(
            self.provider_id,
            self.model_id,
            "",
            (call, call),
            "tool_calls",
            1,
        )


def test_repeated_call_id_is_a_bounded_model_failure_not_a_host_crash(tmp_path: Path) -> None:
    marker = "duplicate-id"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    (tmp_path / "example.py").write_text("import json\n", encoding="utf-8")
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    host = GenericMcpHost(
        task_id="duplicate-id",
        evaluation_profile=profile,
        model=_DuplicateIdModel(),  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=1,
    )

    result = host.run("Does example.py import pathlib?", task_kind="code")

    assert result.process_error is None
    assert not result.delivered
    assert result.final_text == "[Cortheon withheld: duplicate tool call ids]"
    assert transcript_diagnostic(list(result.events)) == "duplicate_announced_tool_call"
    assert host.runtime is not None
    assert host.runtime.server.runtime.active_sessions == 0


class _AutomaticOnlyModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64
    evaluator_executes_exact_tools = True

    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        raise AssertionError("a repeated exact runtime request must stop before model inference")


def test_repeated_automatic_request_exhaustion_is_a_bounded_withhold(tmp_path: Path) -> None:
    marker = "automatic-budget"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    (tmp_path / "example.py").write_text("import json\n", encoding="utf-8")
    profile = execution_profile("full", "a" * 64)
    profile["nonce"] = "7" * 32
    executor = IsolatedExecutor(tmp_path, marker_nonce=marker, maximum_calls=3)
    host = GenericMcpHost(
        task_id="automatic-budget",
        evaluation_profile=profile,
        model=_AutomaticOnlyModel(),  # type: ignore[arg-type]
        executor=executor,
        max_steps=1,
    )
    assert host.runtime is not None
    runtime = host.runtime
    runtime.projected_host_tool = lambda: "host_search"  # type: ignore[method-assign]
    runtime.projected_arguments = lambda name: {  # type: ignore[method-assign]
        "pattern": "pathlib",
        "path": "example.py",
    }
    runtime.validate_host_arguments = lambda *args, **kwargs: True  # type: ignore[method-assign]
    runtime.observe = lambda execution: {  # type: ignore[method-assign]
        "accepted_evidence_ids": [],
        "next_action": runtime.next_action,
    }

    result = host.run("Does example.py import pathlib?", task_kind="code")

    assert result.process_error is None
    assert not result.delivered
    assert result.tool_calls == 3
    assert result.model_steps == 0
    assert result.final_text == "[Cortheon withheld: host tool budget exhausted]"
    assert validate_transcript(list(result.events))
    assert runtime.server.runtime.active_sessions == 0


class _BareDuplicateIdModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        return ModelTurn(
            self.provider_id,
            self.model_id,
            "",
            (
                ModelToolCall(
                    "same-id",
                    "host_search",
                    {"pattern": "json", "path": "example.py"},
                ),
                ModelToolCall(
                    "same-id",
                    "host_search",
                    {"pattern": "pathlib", "path": "example.py"},
                ),
            ),
            "tool_calls",
            1,
        )


def test_bare_duplicate_call_id_is_rejected_before_any_execution(tmp_path: Path) -> None:
    marker = "bare-duplicate-id"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    (tmp_path / "example.py").write_text("import json\n", encoding="utf-8")
    profile = execution_profile("bare", "a" * 64)
    profile["nonce"] = "7" * 32
    host = GenericMcpHost(
        task_id="bare-duplicate-id",
        evaluation_profile=profile,
        model=_BareDuplicateIdModel(),  # type: ignore[arg-type]
        executor=IsolatedExecutor(tmp_path, marker_nonce=marker),
        max_steps=1,
    )

    result = host.run("Inspect example.py.", task_kind="code")

    assert result.process_error is None
    assert not result.delivered
    assert result.final_text == "[Cortheon withheld: duplicate tool call ids]"
    assert transcript_diagnostic(list(result.events)) == "duplicate_announced_tool_call"
    assert not executor_observed_paths(host)


def executor_observed_paths(host: GenericMcpHost) -> set[str]:
    return set(host.executor._observed_paths)


class _InvalidIdModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id

    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        return ModelTurn(
            self.provider_id,
            self.model_id,
            "",
            (
                ModelToolCall(
                    self.call_id,
                    "host_search",
                    {"pattern": "pathlib", "path": "example.py"},
                ),
            ),
            "tool_calls",
            1,
        )


def test_invalid_call_ids_are_bounded_before_the_ledger(tmp_path: Path) -> None:
    for index, call_id in enumerate(("", "bad id", "slash/id", "x" * 129)):
        root = tmp_path / str(index)
        root.mkdir()
        marker = f"invalid-id-{index}"
        (root / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
        (root / "example.py").write_text("import json\n", encoding="utf-8")
        profile = execution_profile("full", "a" * 64)
        profile["nonce"] = "7" * 32
        host = GenericMcpHost(
            task_id=f"invalid-id-{index}",
            evaluation_profile=profile,
            model=_InvalidIdModel(call_id),  # type: ignore[arg-type]
            executor=IsolatedExecutor(root, marker_nonce=marker),
            max_steps=1,
        )

        result = host.run("Does example.py import pathlib?", task_kind="code")

        assert result.process_error is None
        assert result.final_text == "[Cortheon withheld: invalid tool call ids]"
        assert not host.executor._observed_paths
        assert host.runtime is not None
        assert host.runtime.server.runtime.active_sessions == 0


class _InvalidToolModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        return ModelTurn(
            self.provider_id,
            self.model_id,
            "",
            (ModelToolCall("call-1", self.name, {}),),
            "tool_calls",
            1,
        )


def test_unregistrable_tool_names_are_bounded_before_the_ledger(tmp_path: Path) -> None:
    names = ("", "totally_fake", "host_fake", "cortheon_" + "x" * 129)
    for index, name in enumerate(names):
        root = tmp_path / str(index)
        root.mkdir()
        marker = f"invalid-tool-{index}"
        (root / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
        (root / "example.py").write_text("import json\n", encoding="utf-8")
        profile = execution_profile("full", "a" * 64)
        profile["nonce"] = "7" * 32
        host = GenericMcpHost(
            task_id=f"invalid-tool-{index}",
            evaluation_profile=profile,
            model=_InvalidToolModel(name),  # type: ignore[arg-type]
            executor=IsolatedExecutor(root, marker_nonce=marker),
            max_steps=1,
        )

        result = host.run("Does example.py import pathlib?", task_kind="code")

        assert result.process_error is None
        assert result.final_text == "[Cortheon withheld: invalid tool names]"
        assert not host.executor._observed_paths
        assert host.runtime is not None
        assert host.runtime.server.runtime.active_sessions == 0
