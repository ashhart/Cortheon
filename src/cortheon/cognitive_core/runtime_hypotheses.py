"""HypothesisMixin for CognitiveRuntime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.models import (
    HYPOTHESIS_STATUSES,
    CognitiveRuntimeError,
    Hypothesis,
    Investigation,
    PublicClaim,
    Requirement,
)
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.semantic_graph import _keywords
from cortheon.cognitive_core.tasks import (
    _ABDUCTIVE_GOAL_RE,
    _abductive_proposition,
    _is_discriminating_test_design_goal,
    _observation_score,
)
from cortheon.cognitive_core.text import _normalized, _string_list, _text
from cortheon.cognitive_protocol import evaluation_operator


class HypothesisMixin(RuntimeState):
    """Hypotheses responsibilities of CognitiveRuntime."""

    def _add_hypotheses(
        self,
        session: Investigation,
        raw_hypotheses: Iterable[dict[str, Any]],
    ) -> None:
        raw_hypotheses = list(raw_hypotheses)
        if raw_hypotheses and not evaluation_operator(
            session.evaluation_profile,
            "hypothesis_framing",
        ):
            raise CognitiveRuntimeError("hypothesis framing is disabled by the evaluation profile")
        if (
            raw_hypotheses
            and session.hypotheses
            and all(
                item.origin == "substrate_abduction"
                and not item.supporting_evidence
                and not item.contradicting_evidence
                and not item.bearing_evidence
                for item in session.hypotheses.values()
            )
        ):
            originated_ids = set(session.hypotheses)
            for request in session.requests.values():
                if request.status == "pending" and request.hypothesis_id in originated_ids:
                    request.status = "waived"
            session.hypotheses.clear()
        for raw in raw_hypotheses:
            if not isinstance(raw, dict):
                raise ValueError("each hypothesis must be an object")
            statement = _text(raw.get("statement"), "hypothesis.statement", maximum=2_000)
            falsification = _text(
                raw.get("falsification_test"),
                "hypothesis.falsification_test",
                maximum=2_000,
            )
            normalized = _normalized(statement)
            if any(
                _normalized(item.statement) == normalized for item in session.hypotheses.values()
            ):
                continue
            if len(session.hypotheses) >= session.profile.max_hypotheses:
                raise CognitiveRuntimeError(
                    f"the {session.profile.name} effort profile allows at most "
                    f"{session.profile.max_hypotheses} hypotheses"
                )
            # Hypothesis ids are monotonic for the life of an investigation:
            # clearing provisional candidates never reuses h1/h2, so a pending
            # request can never alias a replacement hypothesis.
            session.hypothesis_sequence += 1
            hypothesis_id = f"h{session.hypothesis_sequence}"
            session.hypotheses[hypothesis_id] = Hypothesis(
                hypothesis_id=hypothesis_id,
                statement=statement,
                falsification_test=falsification,
            )
        if raw_hypotheses:
            self._record_evaluation_operator(session, "hypothesis_framing")

    def _originate_competing_hypotheses(self, session: Investigation) -> None:
        """Create bounded abductive candidates when the task explicitly needs them."""

        if not evaluation_operator(session.evaluation_profile, "hypothesis_framing"):
            return

        if (
            session.hypotheses
            or session.deliverable == "code_change"
            or _is_discriminating_test_design_goal(session.goal)
            or _ABDUCTIVE_GOAL_RE.search(session.goal) is None
        ):
            return
        if self._context_pack(session)["deterministic_derivations"]:
            return
        evidence = [
            item
            for item in session.observations.values()
            if item.status != "failed"
            and (item.status == "verified" or item.host_receipt is not None)
            and not item.quarantine_flags
            and (
                item.kind in {"analysis", "documentation", "user", "web"}
                or (item.host_receipt is not None and item.host_receipt.get("tool") == "read")
            )
        ]
        if not evidence:
            return
        ranked = sorted(
            evidence,
            key=lambda item: (
                _observation_score(item, _keywords(session.goal)),
                item.sequence,
            ),
            reverse=True,
        )
        primary = ranked[0]
        primary_proposition = _abductive_proposition(primary.content, session.goal)
        if not primary_proposition:
            return
        origin_ids = [primary.evidence_id]
        candidates = [
            (
                "The leading explanation is that the observed outcome depends on "
                f"this evidence-grounded condition: {primary_proposition}",
                "Find a counterexample where the outcome occurs while this condition "
                "is absent, or where the condition occurs without the outcome.",
            ),
            (
                "A competing explanation is that the apparent relationship is caused "
                "by an unstated scope, timing, measurement, caller, or authority "
                "boundary rather than the leading condition.",
                "Compare the outcome across the relevant boundary and look for an "
                "independent measurement or authority record that separates the two "
                "explanations.",
            ),
        ]
        if len(ranked) > 1:
            secondary = ranked[1]
            secondary_proposition = _abductive_proposition(
                secondary.content,
                session.goal,
            )
            if (
                secondary_proposition
                and secondary_proposition != primary_proposition
                and secondary.source != primary.source
            ):
                origin_ids.append(secondary.evidence_id)
                candidates.append(
                    (
                        "A third explanation is an interaction: neither clue is "
                        "sufficient alone, but the outcome emerges when "
                        f"'{primary_proposition}' combines with "
                        f"'{secondary_proposition}'.",
                        "Test each clue independently and together; the interaction "
                        "hypothesis fails if either clue alone predicts the outcome.",
                    )
                )
        limit = min(
            session.profile.max_hypotheses,
            max(2, session.profile.min_hypotheses),
        )
        for statement, falsification_test in candidates[:limit]:
            session.hypothesis_sequence += 1
            hypothesis_id = f"h{session.hypothesis_sequence}"
            session.hypotheses[hypothesis_id] = Hypothesis(
                hypothesis_id=hypothesis_id,
                statement=statement[:2_000],
                falsification_test=falsification_test,
                origin="substrate_abduction",
                origin_evidence_ids=list(origin_ids),
            )
            self._metrics["hypotheses_originated"] += 1
        if candidates[:limit]:
            self._record_evaluation_operator(session, "hypothesis_framing")

    def _update_hypotheses(
        self,
        session: Investigation,
        updates: Iterable[dict[str, Any]],
        *,
        as_revision: bool = True,
    ) -> None:
        updates = list(updates)
        if (
            updates
            and as_revision
            and not evaluation_operator(
                session.evaluation_profile,
                "contradiction_revision",
            )
        ):
            raise CognitiveRuntimeError("hypothesis revision is disabled by the evaluation profile")
        for raw in updates:
            if not isinstance(raw, dict):
                raise ValueError("each hypothesis update must be an object")
            hypothesis_id = _text(
                raw.get("hypothesis_id"),
                "hypothesis_update.hypothesis_id",
                maximum=64,
            )
            status = _text(raw.get("status"), "hypothesis_update.status", maximum=32)
            if status not in HYPOTHESIS_STATUSES:
                raise ValueError("hypothesis status must be open, supported, refuted, or uncertain")
            hypothesis = session.hypotheses.get(hypothesis_id)
            if hypothesis is None:
                raise ValueError(f"unknown hypothesis_id: {hypothesis_id}")
            evidence_ids = _string_list(
                raw.get("evidence_ids") or (),
                "hypothesis_update.evidence_ids",
                maximum_items=session.profile.max_observations,
                maximum_chars=2_000,
            )
            unknown = [item for item in evidence_ids if item not in session.observations]
            if unknown:
                raise ValueError(f"unknown evidence ids: {', '.join(unknown)}")
            if status == "supported" and not (evidence_ids or hypothesis.supporting_evidence):
                raise ValueError("a supported hypothesis requires evidence")
            if status == "refuted" and not (evidence_ids or hypothesis.contradicting_evidence):
                raise ValueError("a refuted hypothesis requires contradicting evidence")
            # An uncertain hypothesis may carry zero evidence ids: no record
            # genuinely bears on it. That is not a protocol error — normal
            # verification treats it as untested and withholds the completion
            # with an actionable gap instead of raising.
            hypothesis.status = status
            if status == "refuted":
                target = hypothesis.contradicting_evidence
            elif status == "uncertain":
                # Bearing evidence is stored neutrally: it neither supports nor
                # contradicts the hypothesis, it merely grounds the residual
                # uncertainty in accepted live evidence.
                target = hypothesis.bearing_evidence
            else:
                target = hypothesis.supporting_evidence
            for evidence_id in evidence_ids:
                if evidence_id not in target:
                    target.append(evidence_id)
        if updates and as_revision:
            self._record_evaluation_operator(session, "contradiction_revision")

    def _claims(
        self,
        session: Investigation,
        raw_claims: Iterable[dict[str, Any]],
    ) -> list[PublicClaim]:
        claims: list[PublicClaim] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                raise ValueError("each claim must be an object")
            if len(claims) >= 20:
                raise ValueError("claims may contain at most 20 items")
            claim = _text(raw.get("claim"), "claim.claim", maximum=2_000)
            evidence_ids = _string_list(
                raw.get("evidence_ids") or (),
                "claim.evidence_ids",
                maximum_items=session.profile.max_observations,
                maximum_chars=2_000,
            )
            claims.append(PublicClaim(claim=claim, evidence_ids=evidence_ids))
        if not claims:
            raise ValueError("at least one explicit claim is required")
        return claims

    def _maybe_reframe_research(self, session: Investigation) -> None:
        """Correct a costly misframe deterministically.

        When every purposed research request has resolved or been waived
        without a single clean web observation, the research contract is
        unsatisfiable for this host, so the session is reframed as a general
        evidence-backed answer and the downgrade is surfaced as a caveat.
        """

        if session.deliverable != "research_answer":
            return
        if any(
            item.kind == "web" and item.status != "failed" for item in session.observations.values()
        ):
            return
        purposed = [
            request for request in session.requests.values() if request.parameters.get("purpose")
        ]
        if not purposed or any(request.status == "pending" for request in purposed):
            return
        if not any(
            item.status != "failed" and not item.quarantine_flags
            for item in session.observations.values()
        ):
            return
        session.deliverable = "answer"
        session.task_kind = "general"
        session.requirements = [
            Requirement(
                requirement_id="r1",
                statement="Resolve the reframed answer from accepted live evidence",
                proof="completion",
            )
        ]
        self._waive(session, "research_reframe")
        self._metrics["sessions_reframed"] += 1
