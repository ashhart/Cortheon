"""Final pooled-report construction."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def build_blocked_report(
    bindings: ModuleType,
    task: str,
    proposed_action: str | None,
    decision: Any,
) -> Any:
    reasons = [check.reason for check in decision.final_decision.checks if check.status == "block"]
    return bindings.KnowledgePoolReport(
        task=task,
        proposed_action=proposed_action,
        generated_at=bindings.utc_now(),
        answer_status="blocked",
        verdict=decision.final_decision.verdict,
        confidence=decision.final_decision.confidence,
        best_supported_approach="Do not proceed. " + " ".join(reasons),
        key_steps=[],
        evidence_tags=decision.evidence_tags,
        decision=decision,
        source_summaries=[],
        source_plan=[],
        discovery_counts={},
        synthesis_status=None,
        evidence_gaps=decision.final_decision.required_evidence,
        notes=[
            "Knowledge pooling did not run because the proposed action hit a block gate.",
            *decision.notes,
        ],
        errors=[],
    )


def build_report(
    bindings: ModuleType,
    task: str,
    proposed_action: str | None,
    decision: Any,
    research_report: Any,
    research_tags: list[str],
) -> Any:
    synthesis = research_report.synthesis
    source_summaries = bindings.dedupe_sources(
        bindings.top_sources(research_report) + bindings.agent_pooled_sources(decision)
    )[:8]
    gaps = bindings.evidence_gaps(decision, research_report)
    status = bindings.answer_status(decision, research_report, source_summaries)
    verdict, demotion_notes = bindings.couple_verdict_to_answer_status(
        decision.final_decision.verdict,
        status,
    )
    return bindings.KnowledgePoolReport(
        task=task,
        proposed_action=proposed_action,
        generated_at=bindings.utc_now(),
        answer_status=status,
        verdict=verdict,
        confidence=bindings.answer_confidence(decision, research_report),
        best_supported_approach=bindings.best_approach(decision, research_report),
        key_steps=bindings.key_steps(decision, research_report),
        evidence_tags=bindings.unique_tags(decision.evidence_tags + research_tags),
        decision=decision,
        source_summaries=source_summaries,
        source_plan=bindings.source_plan_summary(research_report),
        discovery_counts=bindings.discovery_counts(research_report),
        synthesis_status=synthesis.status if synthesis else None,
        evidence_gaps=gaps,
        notes=bindings.unique_notes([*demotion_notes, *research_report.notes, *decision.notes]),
        errors=bindings.unique_notes(
            [
                *research_report.errors,
                *[error for agent_run in decision.agent_runs for error in agent_run.errors],
            ]
        ),
    )
