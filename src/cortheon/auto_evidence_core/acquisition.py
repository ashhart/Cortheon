"""Bounded evidence-agent orchestration behind the compatibility facade."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def run(
    loop: Any,
    bindings: ModuleType,
    task: str,
    *,
    proposed_action: str | None,
    context: str | None,
    evidence: list[str] | None,
    limits: Any,
) -> Any:
    explicit_tags = bindings.unique_tags(evidence or [])
    initial = bindings.DecisionLayer().evaluate(
        task,
        proposed_action=proposed_action,
        evidence=explicit_tags,
        context=context,
    )
    tags = list(explicit_tags)
    runs: list[Any] = []

    if initial.verdict == "needs_evidence":
        for missing in initial.required_evidence:
            agent_run = loop.run_agent(
                missing,
                task=task,
                proposed_action=proposed_action,
                context=context,
                limits=limits or bindings.AutoEvidenceLimits(),
            )
            runs.append(agent_run)
            tags = bindings.unique_tags(tags + agent_run.produced_tags)

    final = bindings.DecisionLayer().evaluate(
        task,
        proposed_action=proposed_action,
        evidence=tags,
        context=context,
    )
    return bindings.AutoDecisionReport(
        task=task,
        proposed_action=proposed_action,
        initial_decision=initial,
        final_decision=final,
        evidence_tags=tags,
        agent_runs=runs,
        notes=bindings.auto_notes(initial, final, runs),
    )


def run_agent(
    loop: Any,
    bindings: ModuleType,
    missing: str,
    *,
    task: str,
    proposed_action: str | None,
    context: str | None,
    limits: Any,
) -> Any:
    if missing == "current_package_evidence":
        return loop.package_agent(task, proposed_action, context, limits)
    if missing == "api_evidence":
        return loop.api_agent(task, proposed_action, context)
    if missing == "research_report":
        return loop.research_agent(
            missing,
            bindings.research_topic(task, proposed_action, context),
            limits,
            technology_choice=False,
        )
    if missing == "architecture_evidence":
        return loop.research_agent(
            missing,
            "current architecture benchmark implementation evidence for: "
            f"{bindings.research_topic(task, proposed_action, context)}",
            limits,
            technology_choice=True,
        )
    if missing == "repo_context":
        return loop.repo_agent(missing)
    return bindings.EvidenceAgentRun(
        agent="unknown_evidence_agent",
        missing_evidence=missing,
        status="skipped",
        produced_tags=[],
        summary=f"No auto-evidence agent is registered for {missing}.",
    )


def repo_agent(loop: Any, bindings: ModuleType, missing: str) -> Any:
    if not loop.repo_path:
        return bindings.EvidenceAgentRun(
            agent="repo_context_agent",
            missing_evidence=missing,
            status="manual_required",
            produced_tags=[],
            summary=(
                "Repository context requires a repo path (--repo / repo_path); "
                "internet evidence cannot satisfy it alone."
            ),
        )
    report = loop.engine.scan_repo(loop.repo_path, write_report=False)
    if report.errors or report.python_file_count == 0:
        return bindings.EvidenceAgentRun(
            agent="repo_context_agent",
            missing_evidence=missing,
            status="failed",
            produced_tags=[],
            summary=f"Repository scan of {report.root} produced no usable context.",
            errors=report.errors,
        )
    return bindings.EvidenceAgentRun(
        agent="repo_context_agent",
        missing_evidence=missing,
        status="completed",
        produced_tags=["repo_context"],
        summary=(
            f"Scanned {report.root}: {len(report.declared_dependencies)} declared dependency(ies), "
            f"python {report.python_requirement or 'unspecified'}, "
            f"tests: {'; '.join(report.test_commands) or 'none detected'}."
        ),
        details={
            "root": report.root,
            "managers": report.dependency_managers,
            "python_requirement": report.python_requirement,
            "declared_count": len(report.declared_dependencies),
            "test_commands": report.test_commands,
            "undeclared_imports": report.undeclared_imports[:8],
        },
    )


def package_agent(
    loop: Any,
    bindings: ModuleType,
    task: str,
    proposed_action: str | None,
    context: str | None,
    limits: Any,
) -> Any:
    try:
        report = loop.engine.recommend(task)
    except Exception as exc:  # pragma: no cover - connector failures are environment dependent.
        return bindings.EvidenceAgentRun(
            agent="package_evidence_agent",
            missing_evidence="current_package_evidence",
            status="failed",
            produced_tags=[],
            summary=f"Package recommendation failed: {type(exc).__name__}: {exc}",
            errors=[str(exc)],
        )

    tags = bindings.package_evidence_tags(report, proposed_action)
    if tags:
        return bindings.EvidenceAgentRun(
            agent="package_evidence_agent",
            missing_evidence="current_package_evidence",
            status="satisfied",
            produced_tags=tags,
            summary=f"Live package recommendation selected {report.winner}.",
            details=bindings.recommendation_details(report),
        )

    if report.winner and proposed_action:
        proposed = sorted(bindings.named_technology_candidates(proposed_action))
        return bindings.EvidenceAgentRun(
            agent="package_evidence_agent",
            missing_evidence="current_package_evidence",
            status="partial",
            produced_tags=[],
            summary=(
                f"Live package recommendation selected {report.winner}, but the proposed action "
                f"targets {', '.join(proposed) or 'a different option'}."
            ),
            details=bindings.recommendation_details(report),
        )

    research_run = loop.research_agent(
        "current_package_evidence",
        "current best technical options for: "
        f"{bindings.research_topic(task, proposed_action, context)}",
        limits,
        technology_choice=True,
    )
    research_run.agent = "package_research_agent"
    return research_run


def api_agent(
    loop: Any,
    bindings: ModuleType,
    task: str,
    proposed_action: str | None,
    context: str | None,
) -> Any:
    target = bindings.extract_api_target(
        " ".join(part for part in [task, proposed_action or "", context or ""] if part)
    )
    if not target:
        return bindings.EvidenceAgentRun(
            agent="api_symbol_agent",
            missing_evidence="api_evidence",
            status="manual_required",
            produced_tags=[],
            summary="Could not infer a package and symbol query from the proposed action.",
        )
    package, query = target
    try:
        report = loop.engine.retrieve_api_evidence(package, query)
    except Exception as exc:  # pragma: no cover - connector failures are environment dependent.
        return bindings.EvidenceAgentRun(
            agent="api_symbol_agent",
            missing_evidence="api_evidence",
            status="failed",
            produced_tags=[],
            summary=f"API evidence lookup failed for {package}:{query}: {type(exc).__name__}: {exc}",
            errors=[str(exc)],
        )
    tags = bindings.api_evidence_tags(report)
    status = "satisfied" if tags else "partial"
    summary = (
        f"Source artifact lookup found {len(report.matches)} match(es) for {package}:{query}."
        if tags
        else f"Source artifact lookup found no matches for {package}:{query}."
    )
    return bindings.EvidenceAgentRun(
        agent="api_symbol_agent",
        missing_evidence="api_evidence",
        status=status,
        produced_tags=tags,
        summary=summary,
        details={
            "package": report.package,
            "version": report.version,
            "query": report.query,
            "matches": [item.qualname for item in report.matches[:10]],
            "sources": bindings.api_source_details(report),
        },
        errors=report.errors,
    )


def research_agent(
    loop: Any,
    bindings: ModuleType,
    missing: str,
    topic: str,
    limits: Any,
    *,
    technology_choice: bool,
) -> Any:
    research_engine = loop.research_engine or bindings.ResearchEngine(
        ledger=getattr(loop.engine, "ledger", bindings.EvidenceLedger()),
        source_planner_strategy=loop.source_planner_strategy,
    )
    try:
        report = research_engine.research(
            topic,
            max_search_results=limits.max_search_results,
            max_scholarly_results=limits.max_scholarly_results,
            max_github_results=limits.max_github_results,
            max_trial_results=limits.max_trial_results,
            max_follow_up_queries=limits.max_follow_up_queries,
            max_adaptive_queries=limits.max_adaptive_queries,
            max_artifact_inspections=limits.max_artifact_inspections,
            max_pages=limits.max_pages,
            max_depth=limits.max_depth,
        )
    except Exception as exc:  # pragma: no cover - connector failures are environment dependent.
        return bindings.EvidenceAgentRun(
            agent="research_agent",
            missing_evidence=missing,
            status="failed",
            produced_tags=[],
            summary=f"Live research failed: {type(exc).__name__}: {exc}",
            errors=[str(exc)],
        )

    tags = bindings.research_evidence_tags(report, technology_choice=technology_choice)
    status = "satisfied" if bindings.research_agent_satisfied(missing, tags) else "partial"
    return bindings.EvidenceAgentRun(
        agent="research_agent",
        missing_evidence=missing,
        status=status,
        produced_tags=tags,
        summary=bindings.research_summary(report, tags),
        details=bindings.research_details(report),
        errors=report.errors,
    )
