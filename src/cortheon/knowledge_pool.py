"""Pool current evidence into a concise answer and an honest decision gate."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from types import ModuleType

from cortheon.auto_evidence import (
    AutoDecisionReport,
    AutoEvidenceLimits,
    EvidenceAcquisitionLoop,
    research_evidence_tags,
    unique_tags,
)
from cortheon.decision import DecisionLayer
from cortheon.engine import CortheonEngine
from cortheon.knowledge_pool_core import answering, pooler, reports
from cortheon.knowledge_pool_core import limits as limit_rules
from cortheon.knowledge_pool_core import sources as source_helpers
from cortheon.ledger import EvidenceLedger
from cortheon.models import ResearchReport, utc_now
from cortheon.research import ResearchEngine
from cortheon.sanitize import scan_text
from cortheon.task_research_bridge import (
    TaskResearchPlan,
    build_task_research_plan,
    classify_task,
    domain_research_limits,
)

_LATE_BOUND_DEPENDENCIES = (
    DecisionLayer,
    EvidenceAcquisitionLoop,
    EvidenceLedger,
    ResearchEngine,
    build_task_research_plan,
    classify_task,
    domain_research_limits,
    research_evidence_tags,
    scan_text,
    unique_tags,
    utc_now,
)


@dataclass(slots=True)
class PooledSource:
    title: str | None
    url: str
    source_type: str
    authority_score: float | None
    relevance_score: float | None
    summary: str | None
    derived_claims: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourcePlanSummary:
    name: str
    source_type: str
    selected: bool
    reason: str
    trust_tier: str
    budget: int | None
    observed_count: int | None = None


@dataclass(slots=True)
class KnowledgePoolReport:
    task: str
    proposed_action: str | None
    generated_at: datetime
    answer_status: str
    verdict: str
    confidence: float
    best_supported_approach: str
    key_steps: list[str]
    evidence_tags: list[str]
    decision: AutoDecisionReport
    source_summaries: list[PooledSource]
    source_plan: list[SourcePlanSummary]
    discovery_counts: dict[str, int]
    synthesis_status: str | None
    evidence_gaps: list[str]
    notes: list[str]
    errors: list[str]


def _bindings() -> ModuleType:
    return sys.modules[__name__]


class KnowledgePooler:
    """Pools current source evidence into a concise task answer and decision gate."""

    def __init__(
        self,
        engine: CortheonEngine,
        *,
        research_engine: ResearchEngine | None = None,
        source_planner_strategy: str | None = "auto",
    ) -> None:
        self.engine = engine
        self.research_engine = research_engine
        self.source_planner_strategy = source_planner_strategy

    def run(
        self,
        task: str,
        *,
        proposed_action: str | None = None,
        context: str | None = None,
        evidence: list[str] | None = None,
        limits: AutoEvidenceLimits | None = None,
    ) -> KnowledgePoolReport:
        return pooler.run(
            self,
            _bindings(),
            task,
            proposed_action=proposed_action,
            context=context,
            evidence=evidence,
            limits=limits,
        )

    def _research(
        self,
        task: str,
        proposed_action: str | None,
        context: str | None,
        limits: AutoEvidenceLimits,
    ) -> ResearchReport:
        return pooler.research(
            self,
            _bindings(),
            task,
            proposed_action,
            context,
            limits,
        )


def _merge_limits(limits: AutoEvidenceLimits, plan: TaskResearchPlan) -> AutoEvidenceLimits:
    """Merge domain-specific discovery budgets with caller-owned loop limits."""
    return limit_rules.merge_limits(_bindings(), limits, plan)


def build_blocked_report(
    task: str,
    proposed_action: str | None,
    decision: AutoDecisionReport,
) -> KnowledgePoolReport:
    return reports.build_blocked_report(_bindings(), task, proposed_action, decision)


def build_report(
    task: str,
    proposed_action: str | None,
    decision: AutoDecisionReport,
    research_report: ResearchReport,
    research_tags: list[str],
) -> KnowledgePoolReport:
    return reports.build_report(
        _bindings(),
        task,
        proposed_action,
        decision,
        research_report,
        research_tags,
    )


def pool_topic(task: str, proposed_action: str | None, context: str | None) -> str:
    return answering.pool_topic(task, proposed_action, context)


def couple_verdict_to_answer_status(
    verdict: str,
    status: str,
) -> tuple[str, list[str]]:
    """Demote nominal allows when the pooled answer lacks supporting evidence."""
    return answering.couple_verdict_to_answer_status(verdict, status)


def answer_status(
    decision: AutoDecisionReport,
    research_report: ResearchReport,
    sources: list[PooledSource],
) -> str:
    return answering.answer_status(decision, research_report, sources)


def answer_confidence(decision: AutoDecisionReport, research_report: ResearchReport) -> float:
    return answering.answer_confidence(decision, research_report)


def best_approach(decision: AutoDecisionReport, research_report: ResearchReport) -> str:
    return answering.best_approach(decision, research_report)


def key_steps(decision: AutoDecisionReport, research_report: ResearchReport) -> list[str]:
    return answering.key_steps(_bindings(), decision, research_report)


def evidence_gaps(
    decision: AutoDecisionReport,
    research_report: ResearchReport,
) -> list[str]:
    return answering.evidence_gaps(_bindings(), decision, research_report)


def synthesis_gaps(report: ResearchReport) -> list[str]:
    return answering.synthesis_gaps(report)


def top_sources(report: ResearchReport, limit: int = 6) -> list[PooledSource]:
    return source_helpers.top_sources(_bindings(), report, limit)


def agent_pooled_sources(decision: AutoDecisionReport) -> list[PooledSource]:
    return source_helpers.agent_pooled_sources(_bindings(), decision)


def agent_winner(decision: AutoDecisionReport) -> str | None:
    return source_helpers.agent_winner(decision)


def dedupe_sources(sources: list[PooledSource]) -> list[PooledSource]:
    return source_helpers.dedupe_sources(sources)


def claims_by_source(report: ResearchReport) -> dict[str, list[str]]:
    return source_helpers.claims_by_source(report)


def string_or_none(value: object) -> str | None:
    return source_helpers.string_or_none(value)


def float_or_none(value: object) -> float | None:
    return source_helpers.float_or_none(value)


def source_plan_summary(report: ResearchReport) -> list[SourcePlanSummary]:
    return source_helpers.source_plan_summary(_bindings(), report)


def discovery_counts(report: ResearchReport) -> dict[str, int]:
    return source_helpers.discovery_counts(report)


def compact_text(text: str, limit: int = 360) -> str | None:
    return source_helpers.compact_text(text, limit)


def unique_notes(items: list[str]) -> list[str]:
    return source_helpers.unique_notes(items)
