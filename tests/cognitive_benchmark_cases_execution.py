# ruff: noqa: F401

import argparse
import json
import subprocess
from dataclasses import asdict

import pytest
from scaling_support import report as _sealed_scaling_report

from cortheon.benchmark_core.execution_provenance import ProcessCapture
from cortheon.cognitive_benchmark import (
    DiagnosticCase,
    EvaluationOutcome,
    ImportCase,
    JoinCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    RunResult,
    SemanticCase,
    _audit_manifest,
    _blinded_case,
    _condition_summary,
    _delivery_succeeded,
    _event_statistics,
    _final_text,
    _frontier_comparison,
    _grade,
    _grade_patch_workspace,
    _integer_constants,
    _model_endpoint_health,
    _north_star_coverage,
    _paired_summary,
    _pi_provider_config,
    _postflight_probe,
    _provider_config,
    _workspace_environment,
    discover_benchmark_cases,
    discover_cases,
    discover_diagnostic_cases,
    discover_join_cases,
    discover_long_horizon_cases,
    discover_patch_cases,
    discover_planning_cases,
    discover_reasoning_cases,
    discover_semantic_cases,
    isolated_repository,
    run_frontier_cli_job,
    run_job,
    scaling_curve,
    verify_audit_bundle,
)
from cortheon.cognitive_benchmark import (
    main as cognitive_benchmark_main,
)


def _process_capture(
    stdout: str,
    *,
    timed_out: bool = False,
    budget_reason: str | None = None,
) -> ProcessCapture:
    return ProcessCapture(stdout, "", None if timed_out else 0, 0.1, timed_out, budget_reason)


def test_discovery_builds_balanced_blinded_live_cases(tmp_path):
    package = tmp_path / "src" / "demo"
    package.mkdir(parents=True)
    (package / "positive.py").write_text("from pathlib import Path\n")
    (package / "negative.py").write_text("import json\n")

    cases = discover_cases(tmp_path, count=2, seed=7)

    assert len(cases) == 2
    assert {case.expected for case in cases} == {True, False}
    assert all(case.case_id.startswith("case_") for case in cases)
    assert all("Do not modify files" in case.prompt for case in cases)


def test_import_grader_requires_polarity_and_positive_import_name():
    positive = ImportCase("p", "src/a.py", "pathlib", True, "")
    negative = ImportCase("n", "src/b.py", "pathlib", False, "")

    assert _grade(positive, "Yes — from pathlib import Path")
    assert _grade(positive, "The file imports the module pathlib. Yes.")
    assert not _grade(positive, "Yes.")
    assert not _grade(positive, "The file does not import pathlib.")
    assert _grade(negative, "The file does not import pathlib.")
    assert not _grade(negative, "Yes — import pathlib")
    assert not _grade(negative, "[Cortheon withheld this output]")


def test_research_grader_requires_exact_version_origins_and_conflict_check():
    case = ResearchCase(
        "r",
        "uv",
        "0.11.32",
        "https://github.com/astral-sh/uv/releases/latest",
        "https://pypi.org/project/uv/",
        "",
    )

    assert _grade(
        case,
        "Latest: 0.11.32. The sources agree. "
        "https://github.com/astral-sh/uv/releases/latest "
        "https://pypi.org/project/uv/",
    )
    assert not _grade(
        case,
        "Latest: 0.11.31. The sources agree. "
        "https://github.com/astral-sh/uv/releases/latest "
        "https://pypi.org/project/uv/",
    )
    assert not _grade(case, "Latest: 0.11.32. The sources agree.")


def test_join_discovery_and_grader_use_live_cross_file_constants(tmp_path):
    package = tmp_path / "src" / "demo"
    package.mkdir(parents=True)
    (package / "left.py").write_text("LEFT_LIMIT = 8_899\n")
    (package / "right.py").write_text("RIGHT_LIMIT: int = 1_000_000\n")

    cases = discover_join_cases(tmp_path, count=1, seed=9)

    assert len(cases) == 1
    case = cases[0]
    assert case.expected == 1_008_899
    assert case.paths[0] != case.paths[1]
    assert _grade(case, "8899 + 1000000 = 1008899.")
    assert _grade(case, "8899 + 1,000,000 = **1,008,899**.")
    assert not _grade(case, "8899 + 1000000 = 1088999.")
    assert not _grade(
        case,
        "The sum is 1088999. 8899 + 1000000 = 1008899.",
    )
    assert not _grade(case, "[Cortheon withheld this output]")


def test_constant_discovery_rejects_booleans_and_expressions():
    found = _integer_constants(
        "GOOD_LIMIT = 12\n"
        "TYPED_LIMIT: int = -4\n"
        "BOOL_LIMIT = True\n"
        "EXPR_LIMIT = 6 * 7\n"
        "lower_limit = 3\n"
    )

    assert found == {"GOOD_LIMIT": 12, "TYPED_LIMIT": -4}


def test_isolated_repository_discards_model_mutations(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    program = source / "program.py"
    program.write_text("VALUE = 1\n")
    ignored = source / ".cortheon"
    ignored.mkdir()
    (ignored / "large.bin").write_bytes(b"x" * 100)
    claude_config = source / ".claude"
    claude_config.mkdir()
    (claude_config / "settings.json").write_text('{"hooks": {"PreToolUse": []}}\n')

    with isolated_repository(source) as workspace:
        assert not (workspace / ".cortheon").exists()
        assert not (workspace / ".claude").exists()
        (workspace / "program.py").write_text("VALUE = 2\n")

    assert program.read_text() == "VALUE = 1\n"


def test_frontier_cli_job_is_isolated_bounded_and_graded(monkeypatch, tmp_path):
    observed: dict[str, object] = {}

    def complete(command, **kwargs):
        workspace = kwargs["cwd"]
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        observed["evidence"] = (workspace / "evidence" / "service.md").read_text()
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "Kepler requires approval from Noor Patel.",
                    "usage": {
                        "input_tokens": 20,
                        "cache_creation_input_tokens": 30,
                        "cache_read_input_tokens": 40,
                        "output_tokens": 10,
                    },
                    "num_turns": 3,
                    "permission_denials": [],
                    "total_cost_usd": 0.012,
                    "modelUsage": {"claude-sonnet-5": {"costUSD": 0.012}},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "cortheon.cognitive_benchmark.subprocess.run",
        complete,
    )
    case = SemanticCase(
        "semantic_rule",
        (("evidence/service.md", "Kepler requires approval from Noor Patel.\n"),),
        ("Kepler", "Noor Patel"),
        ("Morgan Vale",),
        "Who must approve Kepler? Inspect the repository evidence.",
    )
    args = argparse.Namespace(
        repository=tmp_path,
        frontier_cli="claude",
        frontier_cli_model="sonnet",
        frontier_max_budget_usd=0.25,
        timeout_seconds=30,
    )

    result = run_frontier_cli_job(args, case, repeat=1)

    command = observed["command"]
    assert isinstance(command, list)
    assert "--safe-mode" in command
    assert "--strict-mcp-config" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--max-budget-usd") + 1] == "0.25"
    assert observed["evidence"] == "Kepler requires approval from Noor Patel.\n"
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert "CORTHEON_RUNTIME_URL" not in environment
    assert result.condition == "frontier"
    assert result.correct
    assert result.delivered
    assert result.tokens == 100
    assert result.tool_calls == 2
    assert result.cost_usd == 0.012
    assert result.inference_model_id == "claude-sonnet-5"


def test_workspace_environment_cannot_point_host_at_live_checkout(tmp_path):
    isolated = _workspace_environment(
        {
            "PWD": "/live/repository",
            "INIT_CWD": "/live/repository",
            "OLDPWD": "/other",
        },
        tmp_path,
    )

    assert isolated["PWD"] == str(tmp_path)
    assert isolated["INIT_CWD"] == str(tmp_path)
    assert "OLDPWD" not in isolated


def test_pi_events_extract_only_the_final_assistant_answer_and_usage():
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I will inspect it."}],
                "usage": {"totalTokens": 11},
            },
        },
        {"type": "tool_execution_start"},
        {"type": "tool_execution_end", "isError": False, "result": {}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "No — there is no import."}],
                "usage": {"totalTokens": 7},
            },
        },
    ]

    assert _final_text(events, host="pi") == "No — there is no import."
    # tool_calls counts model attempts; the single end event executed on the
    # host with no error, block, or unavailable classification.
    assert _event_statistics(events, host="pi") == (18, 1, 0, 1, 0, 0)


def test_opencode_events_treat_step_usage_as_cumulative():
    events = [
        {"type": "step_finish", "part": {"tokens": {"total": 120}}},
        {"type": "tool_use", "part": {"state": {"status": "completed"}}},
        {"type": "step_finish", "part": {"tokens": {"total": 175}}},
        {"type": "tool_use", "part": {"state": {"status": "error"}}},
        {"type": "step_finish", "part": {"tokens": {"total": 230}}},
    ]

    # Other hosts keep zero defaults for the Pi-only classifications.
    assert _event_statistics(events, host="opencode") == (230, 2, 1, 0, 0, 0)


def _pi_end_event(tool: str, result: dict, *, is_error: bool = False) -> dict:
    return {
        "type": "tool_execution_end",
        "toolName": tool,
        "isError": is_error,
        "result": result,
    }


def _pi_start_event(tool: str) -> dict:
    return {"type": "tool_execution_start", "toolName": tool}


def test_pi_tool_traffic_fixture_normal_success():
    events = [
        _pi_start_event("read"),
        _pi_end_event(
            "read",
            {"content": [{"type": "text", "text": "alpha key amber"}]},
        ),
        _pi_start_event("bash"),
        _pi_end_event(
            "bash",
            {"content": [{"type": "text", "text": "ok"}]},
        ),
    ]
    assert _event_statistics(events, host="pi") == (0, 2, 0, 2, 0, 0)


def test_pi_tool_traffic_fixture_legitimate_host_error_still_executes():
    events = [
        _pi_start_event("bash"),
        _pi_end_event(
            "bash",
            {"content": [{"type": "text", "text": "cat: missing.txt: No such file"}]},
            is_error=True,
        ),
    ]
    # A host error is still a host execution and is counted in tool_errors;
    # isError lives on the event, not nested inside result.
    assert _event_statistics(events, host="pi") == (0, 1, 1, 1, 0, 0)


def test_pi_tool_traffic_fixture_host_error_containing_not_found_still_executes():
    events = [
        _pi_start_event("grep"),
        _pi_end_event(
            "grep",
            {"content": [{"type": "text", "text": "grep: pattern not found"}]},
            is_error=True,
        ),
    ]
    # The words "not found" inside a real host error are not Pi's exact
    # "Tool <name> not found" unavailable shape: this is one host execution
    # and one tool error, never an unavailable tool.
    assert _event_statistics(events, host="pi") == (0, 1, 1, 1, 0, 0)


def test_pi_tool_traffic_fixture_cortheon_block_counts_as_blocked():
    events = [
        _pi_start_event("read"),
        _pi_end_event(
            "read",
            {
                "terminate": True,
                "content": [
                    {
                        "type": "text",
                        "text": "Cortheon has all the evidence it needs for this "
                        "investigation. Do not call any tool.",
                    }
                ],
            },
        ),
        _pi_start_event("grep"),
        _pi_end_event(
            "grep",
            {"terminate": True, "content": [{"type": "text", "text": "blocked"}]},
        ),
    ]
    assert _event_statistics(events, host="pi") == (0, 2, 0, 0, 2, 0)


def test_pi_tool_traffic_fixture_unavailable_tool_counts_separately():
    events = [
        _pi_start_event("totally_unavailable_probe"),
        _pi_end_event(
            "totally_unavailable_probe",
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Tool totally_unavailable_probe not found",
                    }
                ]
            },
        ),
    ]
    assert _event_statistics(events, host="pi") == (0, 1, 0, 0, 0, 1)


def test_pi_run_job_populates_and_aggregates_tool_classifications(
    monkeypatch,
    tmp_path,
):
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "provider": "Local",
                        "model": "small-model",
                        "content": [{"type": "text", "text": "Counting."}],
                        "usage": {"totalTokens": 9, "cost": {"total": 0.0}},
                        "stopReason": "stop",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "isError": True,
                    "result": {"content": [{"type": "text", "text": "boom"}]},
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "result": {
                        "terminate": True,
                        "content": [
                            {
                                "type": "text",
                                "text": "Cortheon has all the evidence it needs.",
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "result": {"content": [{"type": "text", "text": "Tool probe not found"}]},
                }
            ),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "provider": "Local",
                        "model": "small-model",
                        "content": [{"type": "text", "text": "Yes — three keys"}],
                        "usage": {"totalTokens": 9, "cost": {"total": 0.0}},
                        "stopReason": "stop",
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "cortheon.benchmark_core.runner_local.execute_host_process",
        lambda *_args, **_kwargs: _process_capture(stdout),
    )
    args = argparse.Namespace(
        host="pi",
        pi="pi",
        provider="Local",
        model_id="small-model",
        base_url="http://127.0.0.1:9000/v1",
        api_key="",
        context_tokens=8_192,
        output_tokens=512,
        reasoning=False,
        repository=tmp_path,
        timeout_seconds=10,
        runtime_url="http://127.0.0.1:8743",
    )
    case = ImportCase("case", "src/example.py", "pathlib", True, "Inspect it.")

    result = run_job(args, case, repeat=0, treatment=False)

    assert result.tool_calls == 0  # no tool_execution_start events at all
    assert result.tool_errors == 1
    assert result.host_tool_executions == 1
    assert result.blocked_tool_calls == 1
    assert result.unavailable_tool_calls == 1
    summary = _condition_summary([result], "baseline")
    assert summary["host_tool_executions"] == 1
    assert summary["blocked_tool_calls"] == 1
