"""Answer status, confidence, approach, and step selection."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def pool_topic(task: str, proposed_action: str | None, context: str | None) -> str:
    parts = [task]
    if proposed_action:
        parts.append(f"Proposed action: {proposed_action}")
    if context:
        parts.append(f"Context: {context}")
    return " ".join(part.strip() for part in parts if part and part.strip())


def couple_verdict_to_answer_status(verdict: str, status: str) -> tuple[str, list[str]]:
    if verdict == "allow" and status in {"thin_evidence", "thin_sources"}:
        return "needs_evidence", [
            "Verdict demoted from allow to needs_evidence: no risk pattern "
            "was detected, but the pooled evidence is too thin to endorse "
            "the proposed action."
        ]
    return verdict, []


def answer_status(decision: Any, research_report: Any, sources: list[Any]) -> str:
    if decision.final_decision.verdict == "block":
        return "blocked"
    if decision.final_decision.verdict == "needs_evidence":
        return "needs_evidence"
    if sources and any(
        tag in decision.evidence_tags
        for tag in {
            "package_verified",
            "recommendation_report",
            "api_evidence",
            "source_symbol_evidence",
        }
    ):
        return "answered"
    synthesis = research_report.synthesis
    if synthesis and synthesis.status != "insufficient_evidence" and sources:
        return "answered"
    if synthesis and synthesis.status != "insufficient_evidence":
        return "thin_sources"
    return "thin_evidence"


def answer_confidence(decision: Any, research_report: Any) -> float:
    synthesis = research_report.synthesis
    if synthesis is None:
        return min(decision.final_decision.confidence, 0.45)
    if synthesis.status == "insufficient_evidence" and "package_verified" in decision.evidence_tags:
        return round(min(decision.final_decision.confidence, 0.72), 3)
    if (
        synthesis.status == "insufficient_evidence"
        and "recommendation_report" in decision.evidence_tags
    ):
        return round(min(decision.final_decision.confidence, 0.68), 3)
    if synthesis.status == "insufficient_evidence":
        return min(decision.final_decision.confidence, 0.5)
    return round(min(decision.final_decision.confidence, max(0.35, synthesis.confidence)), 3)


def best_approach(decision: Any, research_report: Any) -> str:
    if decision.final_decision.verdict == "block":
        return "Do not proceed until the block gate is resolved."
    synthesis = research_report.synthesis
    if (
        synthesis
        and synthesis.status != "insufficient_evidence"
        and synthesis.current_best_direction
    ):
        return synthesis.current_best_direction
    if decision.final_decision.verdict == "needs_evidence":
        missing = ", ".join(decision.final_decision.required_evidence)
        return (
            f"Not enough current evidence to recommend an implementation yet. Missing: {missing}."
        )
    for agent_run in decision.agent_runs:
        winner = agent_run.details.get("winner")
        if winner:
            return f"Use {winner} as the current best supported option, bounded by the gathered evidence."
    return "Proceed conservatively with the supported option, and keep the cited source limits visible."


def key_steps(bindings: ModuleType, decision: Any, research_report: Any) -> list[str]:
    if decision.final_decision.verdict == "block":
        return []
    if decision.final_decision.verdict == "needs_evidence":
        steps = [
            f"Gather missing evidence: {item}."
            for item in decision.final_decision.required_evidence
        ]
        steps.extend(
            f"Close evidence gap: {gap}" for gap in bindings.synthesis_gaps(research_report)[:3]
        )
        return bindings.unique_notes(steps)[:6]

    steps: list[str] = []
    synthesis = research_report.synthesis
    if (
        synthesis
        and synthesis.status != "insufficient_evidence"
        and synthesis.current_best_direction
    ):
        steps.append(f"Default to: {synthesis.current_best_direction}")
    if synthesis and synthesis.status != "insufficient_evidence":
        steps.extend(f"Account for: {finding}" for finding in synthesis.key_findings[:4])
        steps.extend(f"Watch contested point: {item}" for item in synthesis.contested_points[:2])
    winner = bindings.agent_winner(decision)
    if winner:
        steps.append(f"Use {winner} as the selected option backed by current package evidence.")
        steps.append(
            "Check repo constraints before implementation, especially existing framework and Python-version constraints."
        )
        steps.append("Require source-symbol evidence before committing package-specific API calls.")
    for gap in bindings.synthesis_gaps(research_report)[:2]:
        steps.append(f"Treat as open evidence gap: {gap}")
    if not steps:
        steps.append("Implement only the portion directly supported by the gathered sources.")
    return bindings.unique_notes(steps)[:7]


def evidence_gaps(bindings: ModuleType, decision: Any, research_report: Any) -> list[str]:
    gaps = list(decision.final_decision.required_evidence)
    gaps.extend(bindings.synthesis_gaps(research_report))
    gaps.extend(
        item.next_action
        for item in research_report.source_coverage
        if item.expected and item.status != "covered" and item.next_action
    )
    return bindings.unique_notes(gaps)


def synthesis_gaps(report: Any) -> list[str]:
    if report.synthesis is None:
        return []
    return list(report.synthesis.evidence_gaps)
