"""Acquire bounded evidence before re-evaluating repository decisions."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from cortheon.auto_evidence_core import acquisition, classification, details
from cortheon.decision import DecisionLayer
from cortheon.engine import CortheonEngine
from cortheon.ledger import EvidenceLedger
from cortheon.models import (
    ApiEvidenceReport,
    DecisionReport,
    RecommendationReport,
    ResearchReport,
)
from cortheon.research import ResearchEngine

_LATE_BOUND_DEPENDENCIES = (DecisionLayer, EvidenceLedger)


@dataclass(slots=True)
class EvidenceAgentRun:
    agent: str
    missing_evidence: str
    status: str
    produced_tags: list[str]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutoDecisionReport:
    task: str
    proposed_action: str | None
    initial_decision: DecisionReport
    final_decision: DecisionReport
    evidence_tags: list[str]
    agent_runs: list[EvidenceAgentRun]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class AutoEvidenceLimits:
    max_search_results: int = 5
    max_scholarly_results: int = 8
    max_github_results: int = 3
    max_trial_results: int = 5
    max_follow_up_queries: int = 2
    max_adaptive_queries: int = 1
    max_artifact_inspections: int = 2
    max_pages: int = 10
    max_depth: int = 1


def _bindings() -> ModuleType:
    """Expose late-bound facade dependencies to the implementation modules."""
    return sys.modules[__name__]


class EvidenceAcquisitionLoop:
    """Runs bounded specialist evidence agents before re-checking a decision."""

    def __init__(
        self,
        engine: CortheonEngine,
        *,
        research_engine: ResearchEngine | None = None,
        source_planner_strategy: str | None = "auto",
        repo_path: str | None = None,
    ) -> None:
        self.engine = engine
        self.research_engine = research_engine
        self.source_planner_strategy = source_planner_strategy
        self.repo_path = repo_path

    def run(
        self,
        task: str,
        *,
        proposed_action: str | None = None,
        context: str | None = None,
        evidence: list[str] | None = None,
        limits: AutoEvidenceLimits | None = None,
    ) -> AutoDecisionReport:
        return acquisition.run(
            self,
            _bindings(),
            task,
            proposed_action=proposed_action,
            context=context,
            evidence=evidence,
            limits=limits,
        )

    def run_agent(
        self,
        missing: str,
        *,
        task: str,
        proposed_action: str | None,
        context: str | None,
        limits: AutoEvidenceLimits,
    ) -> EvidenceAgentRun:
        return acquisition.run_agent(
            self,
            _bindings(),
            missing,
            task=task,
            proposed_action=proposed_action,
            context=context,
            limits=limits,
        )

    def repo_agent(self, missing: str) -> EvidenceAgentRun:
        return acquisition.repo_agent(self, _bindings(), missing)

    def package_agent(
        self,
        task: str,
        proposed_action: str | None,
        context: str | None,
        limits: AutoEvidenceLimits,
    ) -> EvidenceAgentRun:
        return acquisition.package_agent(
            self,
            _bindings(),
            task,
            proposed_action,
            context,
            limits,
        )

    def api_agent(
        self,
        task: str,
        proposed_action: str | None,
        context: str | None,
    ) -> EvidenceAgentRun:
        return acquisition.api_agent(
            self,
            _bindings(),
            task,
            proposed_action,
            context,
        )

    def research_agent(
        self,
        missing: str,
        topic: str,
        limits: AutoEvidenceLimits,
        *,
        technology_choice: bool,
    ) -> EvidenceAgentRun:
        return acquisition.research_agent(
            self,
            _bindings(),
            missing,
            topic,
            limits,
            technology_choice=technology_choice,
        )


def unique_tags(tags: list[str]) -> list[str]:
    return classification.unique_tags(tags)


def package_evidence_tags(report: RecommendationReport, proposed_action: str | None) -> list[str]:
    return classification.package_evidence_tags(
        report,
        proposed_action,
        named_technology_candidates=named_technology_candidates,
        normalize_name=normalize_name,
    )


def api_evidence_tags(report: ApiEvidenceReport) -> list[str]:
    return classification.api_evidence_tags(report)


def api_source_details(report: ApiEvidenceReport) -> list[dict[str, Any]]:
    return details.api_source_details(report)


def research_evidence_tags(report: ResearchReport, *, technology_choice: bool) -> list[str]:
    return classification.research_evidence_tags(
        report,
        technology_choice=technology_choice,
        research_report_has_substance=research_report_has_substance,
        grounded_claim_count=grounded_claim_count,
        technology_report_has_substance=technology_report_has_substance,
        architecture_report_has_substance=architecture_report_has_substance,
        unique_tags=unique_tags,
    )


def research_agent_satisfied(missing: str, tags: list[str]) -> bool:
    return classification.research_agent_satisfied(missing, tags)


def research_report_has_substance(report: ResearchReport) -> bool:
    return classification.research_report_has_substance(report)


def technology_report_has_substance(report: ResearchReport) -> bool:
    return classification.technology_report_has_substance(report)


def architecture_report_has_substance(report: ResearchReport) -> bool:
    return classification.architecture_report_has_substance(report)


def grounded_claim_count(report: ResearchReport) -> int:
    return classification.grounded_claim_count(report)


def extract_api_target(text: str) -> tuple[str, str] | None:
    return classification.extract_api_target(text, findall=re.findall)


def named_technology_candidates(text: str) -> set[str]:
    return classification.named_technology_candidates(
        text,
        extract_api_target=extract_api_target,
        normalize_name=normalize_name,
        findall=re.findall,
    )


def normalize_name(value: str) -> str:
    return classification.normalize_name(value, substitute=re.sub)


def research_topic(task: str, proposed_action: str | None, context: str | None) -> str:
    return classification.research_topic(task, proposed_action, context)


def recommendation_details(report: RecommendationReport) -> dict[str, Any]:
    return details.recommendation_details(
        report,
        package_source_details=package_source_details,
    )


def package_source_details(report: RecommendationReport) -> list[dict[str, Any]]:
    return details.package_source_details(report)


def research_details(report: ResearchReport) -> dict[str, Any]:
    return details.research_details(report, grounded_claim_count=grounded_claim_count)


def research_summary(report: ResearchReport, tags: list[str]) -> str:
    return details.research_summary(report, tags)


def auto_notes(
    initial: DecisionReport,
    final: DecisionReport,
    runs: list[EvidenceAgentRun],
) -> list[str]:
    return details.auto_notes(initial, final, runs)
