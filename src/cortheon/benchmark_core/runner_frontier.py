"""Frontier control execution through the authenticated CLI transport."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any

from cortheon.benchmark_core.grading import _grade
from cortheon.benchmark_core.models import (
    BenchmarkCase,
    DiagnosticCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    RunResult,
    SemanticCase,
)
from cortheon.benchmark_core.outcomes import is_exact_terminal_success
from cortheon.benchmark_core.run_support import _task_type
from cortheon.benchmark_core.transport_outcomes import (
    failed_transport_outcome,
    frontier_result_outcome,
)
from cortheon.benchmark_core.workspace import (
    _grade_patch_workspace,
    _prepare_patch_case,
    _prepare_semantic_case,
    _workspace_environment,
    isolated_repository,
)


def run_frontier_cli_job(
    args: argparse.Namespace,
    case: BenchmarkCase,
    *,
    repeat: int,
) -> RunResult:
    """Run a genuine tool-using frontier CLI in the same disposable grader."""

    environment = os.environ.copy()
    environment.pop("CORTHEON_RUNTIME_URL", None)
    environment.pop("CORTHEON_PLUGIN_DEBUG", None)
    environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    timed_out = False
    process_error: str | None = None
    final = ""
    tokens = 0
    tool_calls = 0
    tool_errors = 0
    cost = 0.0
    inference_model_id = args.frontier_cli_model
    artifact_correct: bool | None = None
    artifact_failure: str | None = None
    latency = 0.0
    payload: dict[str, Any] = {}

    with isolated_repository(
        args.repository,
        minimal=isinstance(
            case,
            (
                PatchCase,
                SemanticCase,
                ResearchCase,
                DiagnosticCase,
                PlanningCase,
                LongHorizonCase,
                ReasoningCase,
            ),
        ),
    ) as workspace:
        if isinstance(case, (PatchCase, LongHorizonCase)):
            _prepare_patch_case(case, workspace)
        elif isinstance(
            case,
            (SemanticCase, DiagnosticCase, PlanningCase, ReasoningCase),
        ):
            _prepare_semantic_case(case, workspace)
        command = [
            args.frontier_cli,
            "--safe-mode",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--no-session-persistence",
            "--permission-mode",
            "bypassPermissions",
            "--tools",
            "default",
            "--model",
            args.frontier_cli_model,
            "--max-budget-usd",
            str(args.frontier_max_budget_usd),
            "--output-format",
            "json",
            "--print",
            case.prompt,
        ]
        started = time.perf_counter()
        stdout = ""
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=_workspace_environment(environment, workspace),
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            if completed.returncode != 0:
                process_error = (
                    completed.stderr.strip()[-1_000:]
                    or f"{args.frontier_cli} exited {completed.returncode}"
                )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
        latency = time.perf_counter() - started
        try:
            decoded = json.loads(stdout) if stdout else {}
            payload = decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            payload = {}
            if process_error is None and not timed_out:
                process_error = "frontier CLI returned invalid JSON"
        if isinstance(payload, dict):
            final = str(payload.get("result") or "")
            if payload.get("is_error") is True and process_error is None:
                process_error = str(payload.get("subtype") or "frontier CLI error")
            usage = payload.get("usage")
            if isinstance(usage, dict):
                tokens = sum(
                    int(usage.get(key, 0) or 0)
                    for key in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                        "output_tokens",
                    )
                )
                server_tools = usage.get("server_tool_use")
                if isinstance(server_tools, dict):
                    tool_calls += sum(
                        int(value or 0)
                        for value in server_tools.values()
                        if isinstance(value, int | float)
                    )
            tool_calls += max(0, int(payload.get("num_turns", 1) or 1) - 1)
            denials = payload.get("permission_denials")
            if isinstance(denials, list):
                tool_errors = len(denials)
            cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
            model_usage = payload.get("modelUsage")
            if isinstance(model_usage, dict) and len(model_usage) == 1:
                inference_model_id = str(next(iter(model_usage)))
        if isinstance(case, (PatchCase, LongHorizonCase)):
            artifact_correct, artifact_failure = _grade_patch_workspace(
                case,
                workspace,
            )

    evaluator_outcome = frontier_result_outcome(payload, final)
    if timed_out:
        evaluator_outcome = failed_transport_outcome(
            "frontier_cli", status="transport_error", finish_reason="timeout"
        )
    elif process_error is not None:
        evaluator_outcome = failed_transport_outcome(
            "frontier_cli", status="transport_error", finish_reason="process_error"
        )
    delivered = is_exact_terminal_success(evaluator_outcome)
    graded_correct = (
        bool(artifact_correct)
        if isinstance(case, (PatchCase, LongHorizonCase))
        else _grade(case, final)
    )
    return RunResult(
        case_id=case.case_id,
        repeat=repeat,
        condition="frontier",
        expected=True if isinstance(case, (PatchCase, LongHorizonCase)) else case.expected,
        final_text=final[:20_000],
        delivered=delivered,
        correct=delivered and graded_correct,
        latency_seconds=round(latency, 4),
        tokens=tokens,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        timed_out=timed_out,
        process_error=process_error,
        expected_verdict="allow",
        failure_owner=(None if delivered else "candidate"),
        evaluator_outcome=evaluator_outcome,
        inference_model_id=inference_model_id,
        cost_usd=cost,
        task_type=_task_type(case),
        artifact_correct=artifact_correct,
        artifact_failure=artifact_failure,
    )
