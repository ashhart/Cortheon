"""Local harness job execution with infrastructure-death retry."""

from __future__ import annotations

import argparse
import contextlib
import os
from typing import Any

from cortheon.benchmark_core.candidate_diagnostics import (
    _candidate_correct,
    _causal_stage_reason,
)
from cortheon.benchmark_core.condition_execution import (
    condition_control_payload,
    condition_receipt_valid,
    condition_token,
    fetch_runtime_receipt,
    prepare_condition,
)
from cortheon.benchmark_core.execution_provenance import (
    ExecutionPolicy,
    execute_host_process,
    execution_facts,
)
from cortheon.benchmark_core.generic_mcp_runner import (
    GenericProcessSpec,
    generic_evaluation_profile,
    generic_receipt,
    generic_result_metadata,
    generic_runtime_facts,
    generic_transcript_valid,
    prepare_generic_process,
)
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
from cortheon.benchmark_core.opencode_receipt import (
    OpenCodeReceipt,
    capture_opencode_receipt,
)
from cortheon.benchmark_core.outcomes import is_authenticated_withhold, is_exact_terminal_success
from cortheon.benchmark_core.run_support import (
    _event_statistics,
    _opencode_step_budget_exhausted,
    _parse_events,
    _runtime_metric_delta,
    _runtime_metric_snapshot,
    _substrate_telemetry_valid,
    _task_type,
)
from cortheon.benchmark_core.runner_hosts import prepare_native_host
from cortheon.benchmark_core.transport_outcomes import (
    failed_transport_outcome,
    parse_transport_outcome,
)
from cortheon.benchmark_core.workspace import (
    _grade_patch_workspace,
    _prepare_patch_case,
    _prepare_semantic_case,
    _workspace_environment,
    isolated_repository,
)


def run_job(
    args: argparse.Namespace,
    case: BenchmarkCase,
    *,
    repeat: int,
    treatment: bool,
    condition: str | None = None,
    evaluation_profile: dict[str, Any] | None = None,
    control_token: str | None = None,
    evaluator_control_payload: bytes | None = None,
) -> RunResult:
    condition = condition or ("cortheon" if treatment else "baseline")
    if args.host == "generic_mcp":
        evaluation_profile = generic_evaluation_profile(
            evaluation_profile,
            treatment=treatment,
        )
    environment = os.environ.copy()
    runtime_token = condition_token() if control_token is None else control_token
    applied_condition = prepare_condition(
        environment,
        evaluation_profile,
        treatment=treatment,
    )
    if treatment:
        environment["CORTHEON_RUNTIME_URL"] = args.runtime_url
    else:
        environment.pop("CORTHEON_RUNTIME_URL", None)
        environment.pop("CORTHEON_PLUGIN_DEBUG", None)
        environment.pop("CORTHEON_EVALUATOR_MAX_STEPS", None)
        environment.pop("CORTHEON_EVALUATOR_PROFILE", None)

    timed_out = False
    process_error: str | None = None
    stdout = ""
    artifact_correct: bool | None = None
    artifact_failure: str | None = None
    runtime_before: dict[str, int] | None = None
    runtime_delta: dict[str, int] | None = None
    events: list[dict[str, Any]] = []
    opencode_receipt: OpenCodeReceipt | None = None
    profile_receipt: dict[str, Any] | None = None
    generic_spec: GenericProcessSpec | None = None
    transcript_valid: bool | None = None

    with contextlib.ExitStack() as stack:
        command = prepare_native_host(
            args,
            case,
            treatment=treatment,
            environment=environment,
            stack=stack,
        )

        workspace = stack.enter_context(
            isolated_repository(
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
            )
        )
        if isinstance(case, (PatchCase, LongHorizonCase)):
            _prepare_patch_case(case, workspace)
        elif isinstance(
            case,
            (SemanticCase, DiagnosticCase, PlanningCase, ReasoningCase),
        ):
            _prepare_semantic_case(case, workspace)
        job_environment = _workspace_environment(environment, workspace)
        if evaluation_profile is not None and args.host != "generic_mcp":
            runtime_before = _runtime_metric_snapshot(args.runtime_url)
        policy = ExecutionPolicy(
            max_steps=int(getattr(args, "max_steps", 4)),
            max_tool_calls=int(getattr(args, "max_tool_calls", getattr(args, "max_steps", 4))),
            timeout_seconds=float(args.timeout_seconds),
            context_tokens=int(args.context_tokens),
            output_tokens=int(args.output_tokens),
        )
        control_payload = (
            evaluator_control_payload
            if evaluator_control_payload is not None
            else condition_control_payload(
                applied_condition,
                token=runtime_token,
                host=args.host,
                treatment=treatment,
                max_steps=policy.max_steps,
                max_host_tool_calls=policy.max_tool_calls,
            )
        )
        stdin_payload: bytes | None = None
        if args.host == "generic_mcp":
            if applied_condition is None:
                raise RuntimeError("generic MCP requires an evaluator-applied condition")
            if policy.max_tool_calls < 1:
                raise ValueError("generic MCP requires at least one allowed tool call")
            generic_spec = prepare_generic_process(
                args,
                case,
                repeat=repeat,
                condition=condition,
                applied=applied_condition,
                workspace=workspace,
                environment=job_environment,
                token=runtime_token,
                max_steps=policy.max_steps,
                max_tool_calls=policy.max_tool_calls,
                control_payload=evaluator_control_payload,
            )
            command = generic_spec.command
            job_environment = generic_spec.environment
            control_payload = generic_spec.control_payload
            stdin_payload = generic_spec.stdin_payload
        capture = execute_host_process(
            command,
            cwd=workspace,
            env=job_environment,
            host=args.host,
            policy=policy,
            control_payload=control_payload,
            stdin_payload=stdin_payload,
        )
        stdout = capture.stdout
        events = _parse_events(stdout)
        if args.host == "opencode":
            opencode_receipt = capture_opencode_receipt(
                args.opencode,
                events,
                cwd=workspace,
                env=job_environment,
            )
        timed_out = capture.timed_out
        latency = capture.latency_seconds
        if capture.returncode not in {0, None} and not timed_out and capture.budget_reason is None:
            process_error = (
                capture.stderr.strip()[-1_000:] or f"{args.host} exited {capture.returncode}"
            )
        if args.host == "generic_mcp":
            assert generic_spec is not None and applied_condition is not None
            transcript_valid = generic_transcript_valid(
                events,
                applied=applied_condition,
                spec=generic_spec,
                args=args,
                require_web=isinstance(case, ResearchCase),
            )
            runtime_delta = generic_runtime_facts(events)
            profile_receipt = generic_receipt(events)
        elif evaluation_profile is not None:
            runtime_delta = _runtime_metric_delta(
                runtime_before,
                _runtime_metric_snapshot(args.runtime_url),
            )
            if treatment and applied_condition is not None:
                profile_receipt = fetch_runtime_receipt(
                    args.runtime_url,
                    applied_condition.nonce,
                    token=runtime_token,
                )
        if isinstance(case, (PatchCase, LongHorizonCase)):
            artifact_correct, artifact_failure = _grade_patch_workspace(
                case,
                workspace,
            )
    facts = execution_facts(
        events,
        host=args.host,
        opencode_receipt=opencode_receipt,
    )
    identity_valid = bool(
        facts.identity_valid
        and facts.provider_id == args.provider
        and facts.model_id == args.model_id
        and transcript_valid is not False
    )
    identity_reason = facts.identity_reason
    if facts.identity_valid and not identity_valid:
        identity_reason = "execution_identity_mismatch"
    parsed_transport = parse_transport_outcome(events, host=args.host)
    final = parsed_transport.final_text
    evaluator_outcome = parsed_transport.outcome
    model_text_chars = sum(
        len(str(event.get("part", {}).get("text", "")))
        for event in events
        if isinstance(event, dict)
        and isinstance(event.get("part"), dict)
        and event["part"].get("type") == "text"
    )
    if args.host == "generic_mcp":
        model_text_chars = sum(
            len(str(event.get("content", ""))) for event in events if event.get("type") == "message"
        )
    if not final and not timed_out and process_error is None:
        process_error = (
            f"{args.host} produced no output"
            if not stdout.strip()
            else f"{args.host} returned no assistant answer"
        )
    if not identity_valid and process_error is None and not timed_out:
        process_error = f"{args.host} execution identity invalid: {identity_reason}"
    budget_reason = capture.budget_reason
    step_budget_exhausted = bool(
        budget_reason == "max_steps"
        or (args.host == "opencode" and _opencode_step_budget_exhausted(final))
    )
    if timed_out and step_budget_exhausted:
        # OpenCode has already emitted its terminal model-budget outcome. Its
        # local runner can linger after that event, but this is an eligible
        # task failure rather than missing inference or corrupt infrastructure.
        timed_out = False
        process_error = None
    if budget_reason is not None:
        process_error = None
        timed_out = False
        if not is_authenticated_withhold(evaluator_outcome):
            evaluator_outcome = failed_transport_outcome(
                args.host, status="incomplete", finish_reason=budget_reason
            )
    elif step_budget_exhausted:
        evaluator_outcome = failed_transport_outcome(
            "opencode", status="incomplete", finish_reason="step_budget_exhausted"
        )
    elif timed_out:
        evaluator_outcome = failed_transport_outcome(
            args.host, status="transport_error", finish_reason="timeout"
        )
    elif process_error is not None:
        evaluator_outcome = failed_transport_outcome(
            args.host, status="transport_error", finish_reason="process_error"
        )
    delivered = is_exact_terminal_success(evaluator_outcome)
    graded_correct = (
        bool(artifact_correct)
        if isinstance(case, (PatchCase, LongHorizonCase))
        else _grade(case, final)
    )
    candidate_correct = _candidate_correct(
        case,
        events,
        host=args.host,
        treatment=treatment,
        final=final,
        evaluator_outcome=evaluator_outcome,
    )
    causal_stage_reason = _causal_stage_reason(
        events,
        host=args.host,
        treatment=treatment,
    )
    (
        _host_tokens,
        tool_calls,
        tool_errors,
        host_tool_executions,
        blocked_tool_calls,
        unavailable_tool_calls,
    ) = _event_statistics(
        events,
        host=args.host,
    )
    runtime_started = (runtime_delta or {}).get("sessions_started", 0)
    runtime_observations = (runtime_delta or {}).get("observations_accepted", 0)
    runtime_completed = (runtime_delta or {}).get("sessions_completed", 0)
    runtime_evidence_closed = (runtime_delta or {}).get(
        "sessions_evidence_closed",
        0,
    )
    runtime_withheld = (runtime_delta or {}).get("completion_withheld", 0)
    runtime_abandoned = (runtime_delta or {}).get("sessions_abandoned", 0)
    runtime_controller_decisions = (runtime_delta or {}).get(
        "controller_decisions",
        0,
    )
    runtime_controller_alternatives = (runtime_delta or {}).get(
        "controller_alternatives_considered",
        0,
    )
    substrate_telemetry_valid = _substrate_telemetry_valid(runtime_delta) if treatment else None
    expected_intercepts = (
        evaluation_profile.get("config", {}).get("intercepts_final") if evaluation_profile else None
    )
    profile_receipt_valid = (
        transcript_valid
        if args.host == "generic_mcp"
        else condition_receipt_valid(
            applied_condition,
            profile_receipt,
            runtime_delta,
            treatment=treatment,
            host=args.host,
        )
    )
    return RunResult(
        case_id=case.case_id,
        repeat=repeat,
        condition=condition,
        expected=True if isinstance(case, (PatchCase, LongHorizonCase)) else case.expected,
        final_text=final[:20_000],
        delivered=delivered,
        correct=delivered and graded_correct,
        latency_seconds=round(latency, 4),
        tokens=facts.tokens if transcript_valid is not False else None,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        host_tool_executions=host_tool_executions,
        blocked_tool_calls=blocked_tool_calls,
        unavailable_tool_calls=unavailable_tool_calls,
        timed_out=timed_out,
        process_error=process_error,
        expected_verdict="allow",
        failure_owner=(
            None if delivered or is_authenticated_withhold(evaluator_outcome) else "candidate"
        ),
        evaluator_outcome=evaluator_outcome,
        inference_provider_id=facts.provider_id,
        inference_model_id=facts.model_id,
        execution_identity_valid=identity_valid,
        execution_identity_provenance=facts.identity_provenance,
        execution_measurements_valid=(facts.measurements_valid and transcript_valid is not False),
        execution_identity_reason=identity_reason,
        execution_measurement_reason=(
            "invalid_generic_mcp_claim_transcript"
            if transcript_valid is False
            else facts.measurement_reason
        ),
        observed_steps=facts.steps,
        policy_timeout_seconds=policy.timeout_seconds,
        policy_max_steps=policy.max_steps,
        policy_max_tool_calls=policy.max_tool_calls,
        policy_context_tokens=policy.context_tokens,
        policy_output_tokens=policy.output_tokens,
        policy_budget_reason=budget_reason,
        step_budget_exhausted=step_budget_exhausted,
        cost_usd=facts.cost_usd if transcript_valid is not False else None,
        task_type=_task_type(case),
        artifact_correct=artifact_correct,
        artifact_failure=artifact_failure,
        candidate_correct=candidate_correct,
        causal_stage_reason=causal_stage_reason,
        substrate_telemetry_valid=substrate_telemetry_valid,
        model_text_chars=model_text_chars,
        deliverable_chars=len(final or ""),
        runtime_sessions_started=runtime_started,
        runtime_observations_accepted=runtime_observations,
        runtime_sessions_completed=runtime_completed,
        runtime_sessions_evidence_closed=runtime_evidence_closed,
        runtime_sessions_abandoned=runtime_abandoned,
        runtime_completion_withheld=runtime_withheld,
        runtime_controller_decisions=runtime_controller_decisions,
        runtime_controller_alternatives_considered=runtime_controller_alternatives,
        condition_registry_version=(
            evaluation_profile.get("schema_version") if evaluation_profile else None
        ),
        condition_config_sha256=(
            evaluation_profile.get("config_sha256") if evaluation_profile else None
        ),
        condition_implementation_sha256=(
            evaluation_profile.get("implementation_sha256") if evaluation_profile else None
        ),
        condition_requires_runtime_completion=(expected_intercepts),
        condition_profile_receipt_valid=profile_receipt_valid,
        condition_observed_config_sha256=(
            (evaluation_profile or {}).get("config_sha256")
            if args.host == "generic_mcp" and transcript_valid and runtime_started == 1
            else profile_receipt.get("config_sha256")
            if profile_receipt
            else None
        ),
        condition_observed_implementation_sha256=(
            (evaluation_profile or {}).get("implementation_sha256")
            if args.host == "generic_mcp" and transcript_valid and runtime_started == 1
            else profile_receipt.get("implementation_sha256")
            if profile_receipt
            else None
        ),
        condition_adapter_receipt_valid=(
            profile_receipt_valid if treatment and evaluation_profile else None
        ),
        condition_operator_counts=(
            dict(profile_receipt["operator_counts"])
            if profile_receipt_valid is True
            and isinstance(profile_receipt, dict)
            and isinstance(profile_receipt.get("operator_counts"), dict)
            else None
        ),
        **generic_result_metadata(generic_spec, events, transcript_valid),
    )
