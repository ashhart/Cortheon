"""VerificationMixin for CognitiveRuntime."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.aggregate_alignment import _evidence_alignment_check
from cortheon.cognitive_core.claim_verification import _claim_verification_profiles
from cortheon.cognitive_core.claims import _claim_profiles_from_checks, _join_reasons
from cortheon.cognitive_core.models import CognitiveRuntimeError, Investigation, PublicClaim
from cortheon.cognitive_core.receipts import _digest
from cortheon.cognitive_core.requirements import _requirement_coverage
from cortheon.cognitive_core.runtime_completion import _validate_revision_completion
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.tasks import (
    _AMBIGUITY_GOAL_RE,
    _is_discriminating_test_design_goal,
    _is_hypothesis_design_goal,
)
from cortheon.cognitive_core.text import _string_list, _text
from cortheon.cognitive_protocol import evaluation_operator


class VerificationMixin(RuntimeState):
    def verify(
        self,
        session_id: str,
        *,
        answer: str,
        claims: Iterable[dict[str, Any]],
        completion_evidence_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            if not evaluation_operator(session.evaluation_profile, "verification"):
                raise CognitiveRuntimeError("verification is disabled by the evaluation profile")
            self._take_turn(session)
            session.phase = "verifying"
            normalized_answer = _text(
                answer,
                "answer",
                maximum=session.profile.max_context_chars,
            )
            _validate_revision_completion(session, normalized_answer)
            normalized_claims = self._claims(session, claims)
            completion_ids = _string_list(
                completion_evidence_ids,
                "completion_evidence_ids",
                maximum_items=session.profile.max_observations,
                maximum_chars=4_000,
            )
            session.draft = normalized_answer
            session.claims = normalized_claims

            checks = self._verification_checks(
                session,
                normalized_claims,
                completion_ids,
            )
            checks.append(
                _evidence_alignment_check(
                    session,
                    normalized_answer,
                    normalized_claims,
                )
            )
            passed = all(check["passed"] for check in checks)
            gaps = [check["reason"] for check in checks if not check["passed"]]
            if passed:
                session.phase = "ready"
                session.verified_answer_digest = _digest(normalized_answer)
                next_action = {
                    "type": "finish",
                    "instruction": (
                        "The bounded completion contract passed. Call cortheon_finish with "
                        "this exact answer; any changed answer must be verified again."
                    ),
                    "submit_via": "cortheon_finish",
                }
                verdict = "ready"
            else:
                session.phase = "investigating"
                session.verified_answer_digest = None
                next_action = self._failed_verification_action(
                    session,
                    checks,
                    gaps,
                )
                verdict = "needs_evidence"

            report = {
                "verdict": verdict,
                "checks": checks,
                "gaps": gaps,
                "answer_digest": _digest(normalized_answer),
                "claim_verification": _claim_profiles_from_checks(checks),
            }
            session.last_verification = report
            payload = self._payload(
                session,
                next_action=next_action,
                guidance=(
                    "Passing means the answer satisfied Cortheon's observable evidence "
                    "contract; it is not a claim of mathematical or universal truth."
                ),
            )
            payload["verification"] = report
            self._commit(session)
            self._record_evaluation_operator(session, "verification")
            return payload

    def _attack_surface(
        self,
        session: Investigation,
        claims: list[PublicClaim],
    ) -> list[dict[str, Any]]:
        attacks: list[dict[str, Any]] = []
        for index, claim in enumerate(claims, start=1):
            if not claim.evidence_ids:
                attacks.append(
                    {
                        "severity": "high",
                        "target": f"claim_{index}",
                        "issue": "The claim has no cited live evidence.",
                        "required_resolution": "Cite supporting evidence or remove/qualify the claim.",
                    }
                )
                continue
            unknown = [
                evidence_id
                for evidence_id in claim.evidence_ids
                if evidence_id not in session.observations
            ]
            if unknown:
                attacks.append(
                    {
                        "severity": "high",
                        "target": f"claim_{index}",
                        "issue": f"The claim cites unknown evidence: {', '.join(unknown)}.",
                        "required_resolution": "Use only evidence ids from this investigation.",
                    }
                )
                continue
            failed = [
                evidence_id
                for evidence_id in claim.evidence_ids
                if session.observations[evidence_id].status == "failed"
            ]
            if failed:
                attacks.append(
                    {
                        "severity": "high",
                        "target": f"claim_{index}",
                        "issue": f"The claim relies on failed evidence: {', '.join(failed)}.",
                        "required_resolution": "Replace the failed evidence or qualify the claim.",
                    }
                )
                continue
            quarantined = [
                evidence_id
                for evidence_id in claim.evidence_ids
                if session.observations[evidence_id].quarantine_flags
            ]
            if quarantined and len(quarantined) == len(claim.evidence_ids):
                attacks.append(
                    {
                        "severity": "high",
                        "target": f"claim_{index}",
                        "issue": (
                            "The claim relies only on quarantined evidence: "
                            f"{', '.join(quarantined)}."
                        ),
                        "required_resolution": (
                            "Gather independent clean evidence or remove the claim."
                        ),
                    }
                )

        untested = [
            hypothesis.hypothesis_id
            for hypothesis in session.hypotheses.values()
            if not any(
                evidence_id in session.observations
                and session.observations[evidence_id].status != "failed"
                and not session.observations[evidence_id].quarantine_flags
                for evidence_id in (
                    hypothesis.supporting_evidence
                    + hypothesis.contradicting_evidence
                    + hypothesis.bearing_evidence
                )
            )
        ]
        if untested:
            attacks.append(
                {
                    "severity": "high",
                    "target": "hypothesis_competition",
                    "issue": f"Untested hypotheses remain: {', '.join(untested)}.",
                    "required_resolution": "Test or explicitly mark each alternative uncertain.",
                }
            )
        supported_without_counter = [
            hypothesis.hypothesis_id
            for hypothesis in session.hypotheses.values()
            if any(
                evidence_id in session.observations
                and session.observations[evidence_id].status != "failed"
                and not session.observations[evidence_id].quarantine_flags
                for evidence_id in hypothesis.supporting_evidence
            )
            and not any(
                evidence_id in session.observations
                and session.observations[evidence_id].status != "failed"
                and not session.observations[evidence_id].quarantine_flags
                for evidence_id in hypothesis.contradicting_evidence
            )
            and sum(
                1
                for evidence_id in hypothesis.supporting_evidence
                if evidence_id in session.observations
                and session.observations[evidence_id].status != "failed"
                and not session.observations[evidence_id].quarantine_flags
            )
            < 2
        ]
        if (
            supported_without_counter
            and session.profile.name != "quick"
            and not _is_hypothesis_design_goal(session.goal)
        ):
            attacks.append(
                {
                    "severity": "medium",
                    "target": "confirmation_bias",
                    "issue": (
                        "Supported hypotheses lack a recorded falsification attempt: "
                        f"{', '.join(supported_without_counter)}."
                    ),
                    "required_resolution": "Seek the strongest counterexample before completion.",
                }
            )
        attacks.extend(
            {
                "severity": "high",
                "target": "completion",
                "issue": gap,
                "required_resolution": "Gather live completion evidence through the harness.",
            }
            for gap in self._completion_gaps(session, None)
        )
        if not attacks:
            attacks.append(
                {
                    "severity": "low",
                    "target": "strongest_conclusion",
                    "issue": (
                        "No structural gap was found; test whether the conclusion is "
                        "stronger than its cited evidence."
                    ),
                    "required_resolution": (
                        "Keep the narrowest wording supported by the evidence and preserve "
                        "any residual uncertainty."
                    ),
                }
            )
        return attacks[:12]

    def _verification_checks(
        self,
        session: Investigation,
        claims: list[PublicClaim],
        completion_evidence_ids: list[str],
    ) -> list[dict[str, Any]]:
        self._maybe_reframe_research(session)
        known_ids = set(session.observations)
        cited_ids = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
        unknown = sorted(cited_ids - known_ids)
        failed = sorted(
            evidence_id
            for evidence_id in cited_ids & known_ids
            if session.observations[evidence_id].status == "failed"
        )
        uncited_claims = [
            str(index) for index, claim in enumerate(claims, 1) if not claim.evidence_ids
        ]
        grounding_passed = not unknown and not failed and not uncited_claims
        grounding_reason = (
            "Every explicit claim cites available, non-failed live evidence."
            if grounding_passed
            else _join_reasons(
                [
                    f"claims without evidence: {', '.join(uncited_claims)}"
                    if uncited_claims
                    else "",
                    f"unknown evidence ids: {', '.join(unknown)}" if unknown else "",
                    f"failed evidence ids: {', '.join(failed)}" if failed else "",
                ]
            )
        )
        quarantine_only_claims = [
            str(index)
            for index, claim in enumerate(claims, 1)
            if claim.evidence_ids
            and all(
                evidence_id in session.observations
                and session.observations[evidence_id].quarantine_flags
                for evidence_id in claim.evidence_ids
            )
        ]
        quarantine_passed = not quarantine_only_claims
        quarantine_reason = (
            "No explicit claim relies solely on quarantined evidence."
            if quarantine_passed
            else (
                "claims relying only on quarantined evidence: " + ", ".join(quarantine_only_claims)
            )
        )
        claim_profiles = _claim_verification_profiles(
            session,
            claims,
            require_host_receipts=self.require_host_receipts,
        )
        truth_passed = all(profile["passed"] for profile in claim_profiles)
        truth_reason = (
            "Every claim completed its claim-specific truth operation."
            if truth_passed
            else "; ".join(
                f"claim {profile['claim_index']}: {', '.join(profile['gaps'])}"
                for profile in claim_profiles
                if not profile["passed"]
            )
        )
        requirement_coverage = _requirement_coverage(
            session,
            claims,
            completion_evidence_ids,
            require_citations=True,
        )
        uncovered_requirements = [
            item for item in requirement_coverage if item["status"] != "covered"
        ]
        requirements_passed = not uncovered_requirements
        requirements_reason = (
            "Every material task requirement is bound to accepted completion evidence."
            if requirements_passed
            else "; ".join(
                f"{item['requirement_id']} ({item['statement']}): {item['reason']}"
                for item in uncovered_requirements
            )
        )

        enough_hypotheses = len(session.hypotheses) >= session.profile.min_hypotheses
        tested = [
            item
            for item in session.hypotheses.values()
            if any(
                evidence_id in session.observations
                and session.observations[evidence_id].status != "failed"
                and not session.observations[evidence_id].quarantine_flags
                for evidence_id in (
                    item.supporting_evidence + item.contradicting_evidence + item.bearing_evidence
                )
            )
        ]
        supported = [
            item
            for item in session.hypotheses.values()
            if any(
                evidence_id in session.observations
                and session.observations[evidence_id].status != "failed"
                and not session.observations[evidence_id].quarantine_flags
                for evidence_id in item.supporting_evidence
            )
        ]
        all_tested = len(tested) == len(session.hypotheses) and bool(session.hypotheses)
        # Ambiguity may end with every evidence-tested branch still uncertain;
        # requiring a supported branch here would force a guess.
        calibrated_non_decision = all_tested and (
            bool(_AMBIGUITY_GOAL_RE.search(session.goal))
            or _is_discriminating_test_design_goal(session.goal)
        )
        hypotheses_passed = (
            enough_hypotheses and all_tested and (bool(supported) or calibrated_non_decision)
        )
        hypotheses_applicable = evaluation_operator(
            session.evaluation_profile,
            "hypothesis_framing",
        )
        if not hypotheses_applicable:
            hypotheses_passed = True
        tested_ids = {item.hypothesis_id for item in tested}
        untested = sorted(
            hypothesis_id for hypothesis_id in session.hypotheses if hypothesis_id not in tested_ids
        )
        hypotheses_reason = (
            "Competing hypotheses were tested and at least one has supporting evidence."
            if hypotheses_passed and supported
            else "Every competing hypothesis is evidence-tested and calibrated open."
            if hypotheses_passed
            else (
                f"need {session.profile.min_hypotheses} hypotheses, evidence for every "
                "alternative, and support for at least one"
                + (
                    "; untested hypotheses with no bearing evidence: "
                    + ", ".join(untested)
                    + " — collect neutral bearing evidence or a counterexample "
                    "for each, or resubmit the hypothesis with a supported or "
                    "refuted status and valid evidence"
                    if untested
                    else ""
                )
            )
        )

        completion_gaps = self._completion_gaps(
            session,
            completion_evidence_ids,
        )
        pending = [
            request.request_id
            for request in session.requests.values()
            if request.status == "pending"
        ]
        return [
            {
                "name": "claim_grounding",
                "passed": grounding_passed,
                "reason": grounding_reason,
            },
            {
                "name": "evidence_quarantine",
                "passed": quarantine_passed,
                "reason": quarantine_reason,
            },
            {
                "name": "claim_truth_operations",
                "passed": truth_passed,
                "reason": truth_reason,
                "profiles": claim_profiles,
            },
            {
                "name": "requirement_coverage",
                "passed": requirements_passed,
                "reason": requirements_reason,
                "requirements": requirement_coverage,
            },
            {
                "name": "hypothesis_competition",
                "passed": hypotheses_passed,
                "applicable": hypotheses_applicable,
                "reason": (
                    hypotheses_reason
                    if hypotheses_applicable
                    else "Not applicable: hypothesis framing is disabled in this condition."
                ),
            },
            {
                "name": "adversarial_challenge",
                "passed": session.challenge_count > 0
                or not evaluation_operator(
                    session.evaluation_profile,
                    "contradiction_revision",
                ),
                "applicable": evaluation_operator(
                    session.evaluation_profile,
                    "contradiction_revision",
                ),
                "reason": (
                    "The draft completed an adversarial challenge."
                    if session.challenge_count > 0
                    else ("Not applicable: contradiction revision is disabled in this condition.")
                    if not evaluation_operator(
                        session.evaluation_profile,
                        "contradiction_revision",
                    )
                    else "cortheon_challenge must run before completion"
                ),
            },
            {
                "name": "completion_evidence",
                "passed": not completion_gaps,
                "reason": (
                    "The task-specific completion evidence is present."
                    if not completion_gaps
                    else "; ".join(completion_gaps)
                ),
            },
            {
                "name": "pending_requests",
                "passed": not pending,
                "reason": (
                    "No evidence request remains pending."
                    if not pending
                    else f"unresolved evidence requests: {', '.join(pending)}"
                ),
            },
        ]
