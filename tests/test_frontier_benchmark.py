import argparse
import json
import subprocess

from cortheon.cognitive_benchmark import EvaluationOutcome, ImportCase, RunResult
from cortheon.frontier_benchmark import (
    _decode_frontier_result,
    _frontier_command,
    _paired_frontier_summary,
    build_parser,
    run_frontier_job,
)


def _args(tmp_path):
    return argparse.Namespace(
        frontier_cli="claude",
        frontier_model="sonnet",
        frontier_effort="high",
        frontier_timeout_seconds=30,
        repository=tmp_path,
    )


def test_frontier_parser_reuses_the_shared_cli_option_without_conflict():
    args = build_parser().parse_args([])
    assert args.frontier_cli == "claude"
    assert args.frontier_model == "sonnet"


def test_frontier_command_is_ephemeral_blind_and_prompt_bounded(tmp_path):
    case = ImportCase(
        "case",
        "src/example.py",
        "pathlib",
        True,
        "Inspect the live file.",
    )

    command = _frontier_command(_args(tmp_path), case)

    assert command[:3] == ["claude", "-p", case.prompt]
    assert "--safe-mode" in command
    assert "--no-session-persistence" in command
    assert command[-2:] == ["--tools", "Read,Grep,Glob"]


def test_frontier_result_decoder_preserves_usage_cost_and_failure():
    payload = {
        "result": "Yes — import pathlib",
        "is_error": False,
        "num_turns": 3,
        "total_cost_usd": 0.123,
        "usage": {
            "input_tokens": 5,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 11,
            "output_tokens": 13,
        },
    }

    final, tokens, tools, cost, error = _decode_frontier_result(json.dumps(payload))

    assert final == "Yes — import pathlib"
    assert tokens == 36
    assert tools == 2
    assert cost == 0.123
    assert error is None


def test_frontier_job_uses_an_isolated_workspace_and_shared_grader(
    monkeypatch,
    tmp_path,
):
    package = tmp_path / "src"
    package.mkdir()
    source = package / "example.py"
    source.write_text("import pathlib\n")
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        assert kwargs["cwd"] != tmp_path
        assert (kwargs["cwd"] / "src" / "example.py").read_text() == ("import pathlib\n")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "result": "Yes — import pathlib",
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "num_turns": 1,
                    "total_cost_usd": 0.02,
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "cortheon.frontier_benchmark.subprocess.run",
        run,
    )
    case = ImportCase(
        "case",
        "src/example.py",
        "pathlib",
        True,
        "Inspect src/example.py.",
    )

    result = run_frontier_job(_args(tmp_path), case, repeat=0)

    assert result.correct
    assert result.condition == "frontier"
    assert result.cost_usd == 0.02
    assert source.read_text() == "import pathlib\n"


def test_frontier_deadline_is_a_task_failure_not_corrupt_infrastructure(
    monkeypatch,
    tmp_path,
):
    package = tmp_path / "src"
    package.mkdir()
    (package / "example.py").write_text("import pathlib\n")

    def time_out(command, **_kwargs):
        raise subprocess.TimeoutExpired(
            command,
            30,
            output='{"type":"result","result":"partial',
        )

    monkeypatch.setattr(
        "cortheon.frontier_benchmark.subprocess.run",
        time_out,
    )
    case = ImportCase(
        "case",
        "src/example.py",
        "pathlib",
        True,
        "Inspect src/example.py.",
    )

    result = run_frontier_job(_args(tmp_path), case, repeat=0)

    assert result.timed_out
    assert result.process_error is None
    assert not result.delivered
    assert not result.correct


def test_frontier_pairing_reports_cortheon_and_frontier_wins():
    common = {
        "expected": True,
        "delivered": True,
        "latency_seconds": 1.0,
        "tokens": 10,
        "tool_calls": 0,
        "tool_errors": 0,
        "timed_out": False,
        "process_error": None,
        "evaluator_outcome": EvaluationOutcome(
            "frontier_cli", "success", "frontier_result", "success"
        ),
    }
    results = [
        RunResult(
            case_id="cortheon_win",
            repeat=0,
            condition="cortheon",
            final_text="correct",
            correct=True,
            substrate_telemetry_valid=True,
            runtime_sessions_completed=1,
            **common,
        ),
        RunResult(
            case_id="cortheon_win",
            repeat=0,
            condition="frontier",
            final_text="wrong",
            correct=False,
            substrate_telemetry_valid=True,
            runtime_sessions_completed=1,
            **common,
        ),
        RunResult(
            case_id="frontier_win",
            repeat=0,
            condition="cortheon",
            final_text="wrong",
            correct=False,
            **common,
        ),
        RunResult(
            case_id="frontier_win",
            repeat=0,
            condition="frontier",
            final_text="correct",
            correct=True,
            **common,
        ),
    ]

    paired = _paired_frontier_summary(results, seed=7)

    assert paired["pairs"] == 2
    assert paired["invalid_pairs"] == 0
    assert paired["cortheon_wins"] == 1
    assert paired["frontier_wins"] == 1
    assert paired["ties"] == 0


def test_a_timed_out_frontier_control_cannot_hand_cortheon_a_paired_win():
    # A control that ran out of wall clock observed no outcome, so it is not a
    # beaten control. Scoring it would flatter the substrate against a
    # frontier agent that never answered.
    common = {
        "expected": True,
        "latency_seconds": 1.0,
        "tokens": 10,
        "tool_calls": 0,
        "tool_errors": 0,
        "process_error": None,
    }
    results = [
        RunResult(
            case_id="case",
            repeat=0,
            condition="cortheon",
            final_text="correct",
            delivered=True,
            correct=True,
            timed_out=False,
            **common,
        ),
        RunResult(
            case_id="case",
            repeat=0,
            condition="frontier",
            final_text="",
            delivered=False,
            correct=False,
            timed_out=True,
            **common,
        ),
    ]

    paired = _paired_frontier_summary(results, seed=7)

    assert paired["pairs"] == 0
    assert paired["invalid_pairs"] == 1
    assert paired["cortheon_wins"] == 0
    assert paired["cortheon_accuracy_delta"] == 0.0
    assert paired["mcnemar_exact_p"] == 1.0
