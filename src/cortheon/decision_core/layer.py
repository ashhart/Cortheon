from __future__ import annotations

from cortheon.decision_core._compat import facade
from cortheon.models import DecisionReport


class DecisionLayer:
    """Deterministic policy gate between an LLM and task decisions.

    Alias ``PolicyGate`` is the honest name; ``DecisionLayer`` is kept for
    callers that already use it.
    """

    def evaluate(
        self,
        task: str,
        *,
        proposed_action: str | None = None,
        evidence: list[str] | None = None,
        context: str | None = None,
    ) -> DecisionReport:
        api = facade()
        evidence_set = {item.strip().lower() for item in evidence or [] if item.strip()}
        text = " ".join(part for part in [task, proposed_action or "", context or ""] if part)
        checks = api.build_checks(text, evidence_set)
        required = api.missing_evidence(checks)
        tools = api.recommended_tools(checks)
        verdict = api.verdict_for(checks)
        notes = api.notes_for(verdict, checks)
        return api.DecisionReport(
            task=task,
            proposed_action=proposed_action,
            verdict=verdict,
            confidence=api.confidence_for(checks, verdict),
            checks=checks,
            required_evidence=required,
            recommended_tools=tools,
            notes=notes,
            cortheon=None,
        )
