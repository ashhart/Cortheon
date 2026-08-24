"""Knowledge-pool orchestration and domain-aware research execution."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def run(
    pooler: Any,
    bindings: ModuleType,
    task: str,
    *,
    proposed_action: str | None,
    context: str | None,
    evidence: list[str] | None,
    limits: Any,
) -> Any:
    limits = limits or bindings.AutoEvidenceLimits()
    explicit_tags = bindings.unique_tags(evidence or [])
    first_gate = bindings.DecisionLayer().evaluate(
        task,
        proposed_action=proposed_action,
        context=context,
        evidence=explicit_tags,
    )
    if first_gate.verdict == "block":
        decision = bindings.EvidenceAcquisitionLoop(
            pooler.engine,
            research_engine=pooler.research_engine,
            source_planner_strategy=pooler.source_planner_strategy,
        ).run(
            task,
            proposed_action=proposed_action,
            context=context,
            evidence=explicit_tags,
            limits=limits,
        )
        return bindings.build_blocked_report(task, proposed_action, decision)

    research_report = pooler._research(task, proposed_action, context, limits)
    research_tags = bindings.research_evidence_tags(
        research_report,
        technology_choice=True,
    )
    decision = bindings.EvidenceAcquisitionLoop(
        pooler.engine,
        research_engine=pooler.research_engine,
        source_planner_strategy=pooler.source_planner_strategy,
    ).run(
        task,
        proposed_action=proposed_action,
        context=context,
        evidence=bindings.unique_tags(explicit_tags + research_tags),
        limits=limits,
    )
    return bindings.build_report(task, proposed_action, decision, research_report, research_tags)


def research(
    pooler: Any,
    bindings: ModuleType,
    task: str,
    proposed_action: str | None,
    context: str | None,
    limits: Any,
) -> Any:
    research_engine = pooler.research_engine or bindings.ResearchEngine(
        ledger=getattr(pooler.engine, "ledger", bindings.EvidenceLedger()),
        source_planner_strategy=pooler.source_planner_strategy,
    )
    plan = bindings.build_task_research_plan(task, proposed_action)
    effective_limits = bindings._merge_limits(limits, plan)
    return research_engine.research(
        bindings.pool_topic(task, proposed_action, context),
        max_search_results=effective_limits.max_search_results,
        max_scholarly_results=effective_limits.max_scholarly_results,
        max_github_results=effective_limits.max_github_results,
        max_trial_results=effective_limits.max_trial_results,
        max_follow_up_queries=limits.max_follow_up_queries,
        max_adaptive_queries=limits.max_adaptive_queries,
        max_artifact_inspections=limits.max_artifact_inspections,
        max_pages=effective_limits.max_pages,
        max_depth=limits.max_depth,
    )
