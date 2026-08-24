"""RecommendationMixin for CognitiveRuntime."""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_core.adaptive_stopping import next_catalog_action
from cortheon.cognitive_core.models import Investigation
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.semantic_join import _semantic_join_analysis
from cortheon.cognitive_core.tasks import (
    _is_adaptive_stopping_goal,
    _is_contradiction_revision_goal,
    _is_cross_source_derivation_goal,
    _is_discriminating_test_design_goal,
    _is_hypothesis_design_goal,
    _requested_hypothesis_count,
)
from cortheon.cognitive_protocol import evaluation_operator


class RecommendationMixin(RuntimeState):
    """Recommendation responsibilities of CognitiveRuntime."""

    def _recommend(self, session: Investigation) -> dict[str, Any]:
        self._maybe_reframe_research(session)
        pending = self._pending_request(session)
        if pending is not None:
            return self._payload(
                session,
                next_action=self._execute_action(pending),
                guidance="Satisfy the pending request with a host tool before reasoning further.",
            )
        verification_enabled = evaluation_operator(
            session.evaluation_profile,
            "verification",
        )
        if session.evaluation_profile is not None and not verification_enabled:
            session.phase = "evidence_ready"
            return self._payload(
                session,
                next_action={
                    "type": "disengage",
                    "instruction": "Use the attached evidence; Cortheon will not intercept the answer.",
                },
                guidance="This condition does not synthesize or verify a candidate.",
            )
        if session.turns >= session.profile.max_turns:
            session.phase = "inconclusive"
            return self._payload(
                session,
                next_action={
                    "type": "finish",
                    "instruction": (
                        "The turn budget is exhausted without a verified answer. Preserve "
                        "the uncertainty and abandon the investigation."
                    ),
                    "submit_via": "cortheon_finish",
                },
                guidance="Cortheon does not promote a budget-exhausted task to completion.",
            )
        retrieval_enabled = evaluation_operator(session.evaluation_profile, "retrieval")
        code_discovery = self._code_discovery_response(session) if retrieval_enabled else None
        if code_discovery is not None:
            return code_discovery
        document_discovery = (
            self._document_discovery_response(session) if retrieval_enabled else None
        )
        if document_discovery is not None:
            return document_discovery
        if session.deliverable == "research_answer" and retrieval_enabled:
            request = self._request_for_research_protocol(session)
            if request is not None:
                session.phase = "investigating"
                return self._payload(
                    session,
                    next_action=self._execute_action(request),
                    guidance=(
                        "Use the host's web tools and preserve URL, retrieval time, "
                        "publication date when available, and the request purpose."
                    ),
                )
        revision_enabled = evaluation_operator(
            session.evaluation_profile,
            "contradiction_revision",
        )
        semantic_conflict = self._semantic_conflict_response(session) if revision_enabled else None
        if semantic_conflict is not None:
            self._record_evaluation_operator(session, "contradiction_revision")
            return semantic_conflict
        framing_enabled = evaluation_operator(
            session.evaluation_profile,
            "hypothesis_framing",
        )
        if framing_enabled and _is_hypothesis_design_goal(session.goal):
            session.phase = "framing"
            hypothesis_count = _requested_hypothesis_count(
                session.goal,
                session.profile.min_hypotheses,
            )
            if len(session.hypotheses) < hypothesis_count:
                return self._payload(
                    session,
                    next_action={
                        "type": "reason",
                        "instruction": (
                            f"Frame exactly {hypothesis_count} evidence-grounded causal "
                            "hypotheses: the strongest explanation and genuinely distinct "
                            "rivals. Give each an observable falsification test before "
                            "constructing the final answer."
                        ),
                        "submit_via": "cortheon_step",
                        "required_fields": ["hypotheses"],
                    },
                    guidance=(
                        "Return compact public hypotheses and falsification tests, not "
                        "private chain-of-thought."
                    ),
                )
            self._record_evaluation_operator(session, "hypothesis_framing")
            return self._payload(
                session,
                next_action={
                    "type": "complete",
                    "instruction": (
                        "Frame exactly two evidence-grounded causal hypotheses: the "
                        "strongest leading explanation and one genuinely distinct rival. "
                        "Keep their requested outcome and scope aligned. Bind the proposed "
                        "intervention, observable result, and refutation target to the "
                        "leader, then submit both hypotheses and narrow evidence-linked claims."
                    ),
                    "submit_via": "cortheon_complete",
                    "required_fields": [
                        "hypotheses",
                        "claims",
                        "answer",
                        "completion_evidence_ids",
                    ],
                },
                guidance=(
                    "Return compact public hypotheses and observable falsification tests, "
                    "not private chain-of-thought."
                ),
            )
        if _is_contradiction_revision_goal(session.goal):
            session.phase = "revising" if revision_enabled else "synthesizing"
            if revision_enabled and not session.draft:
                return self._payload(
                    session,
                    next_action={
                        "type": "reason",
                        "instruction": (
                            "Compare the source that established the original hypothesis "
                            "with the decisive later source. Record whether the decisive "
                            "source supports or refutes the prior, then name the revised "
                            "hypothesis. Do not include private chain-of-thought."
                        ),
                        "submit_via": "cortheon_step",
                        "required_fields": ["draft"],
                    },
                    guidance=(
                        "Make the changed conclusion and the evidence that forced it explicit."
                    ),
                )
            if revision_enabled:
                self._record_evaluation_operator(session, "contradiction_revision")
            return self._payload(
                session,
                next_action={
                    "type": "complete",
                    "instruction": (
                        "Submit the source-bound revision record as the final answer with "
                        "narrow evidence-linked claims and completion evidence ids."
                    ),
                    "submit_via": "cortheon_complete",
                    "required_fields": [
                        "claims",
                        "answer",
                        "completion_evidence_ids",
                    ],
                },
                guidance="Do not restore the superseded hypothesis after revision.",
            )
        derivation_enabled = evaluation_operator(
            session.evaluation_profile,
            "cross_source_derivation",
        )
        if derivation_enabled and _is_cross_source_derivation_goal(session.goal):
            session.phase = "deriving"
            self._record_evaluation_operator(session, "cross_source_derivation")
            if not session.draft:
                return self._payload(
                    session,
                    next_action={
                        "type": "reason",
                        "instruction": (
                            "Join the ordered source-bound premises into one relational chain. "
                            "Preserve every premise and return the chain's first subject, "
                            "requested terminal relation, and final object."
                        ),
                        "submit_via": "cortheon_step",
                        "required_fields": ["draft"],
                    },
                    guidance="Do not replace the chain subject with an intermediate node.",
                )
            return self._payload(
                session,
                next_action={
                    "type": "complete",
                    "instruction": (
                        "Submit the source-bound derivation with its ordered premise path, "
                        "narrow evidence-linked claims, and completion evidence ids."
                    ),
                    "submit_via": "cortheon_complete",
                    "required_fields": ["claims", "answer", "completion_evidence_ids"],
                },
                guidance="Keep the accepted first subject and terminal object unchanged.",
            )
        adaptive_enabled = evaluation_operator(
            session.evaluation_profile,
            "adaptive_stopping",
        )
        if adaptive_enabled and _is_adaptive_stopping_goal(session.goal):
            action = next_catalog_action(session)
            if action is not None:
                self._record_evaluation_operator(session, "adaptive_stopping")
                request = self._create_request(
                    session,
                    capability="read",
                    query=f"Execute the next highest-value probe: {action['action_id']}.",
                    reason=(
                        "This probe covers the public sufficiency contract at bounded cost "
                        f"{action['cost']}."
                    ),
                    success_condition=(
                        "Return the exact observed result before deciding whether another "
                        "probe has positive information value."
                    ),
                    parameters={"path": f"actions/{action['action_id']}.txt"},
                )
                session.phase = "investigating"
                return self._payload(
                    session,
                    next_action=self._execute_action(request),
                    guidance="Stop once every action required by the sufficiency contract ran.",
                )
        discrimination_enabled = evaluation_operator(
            session.evaluation_profile,
            "discriminating_evidence",
        )
        if (
            discrimination_enabled
            and _is_discriminating_test_design_goal(session.goal)
            and not session.draft
        ):
            session.phase = "discriminating"
            self._record_evaluation_operator(session, "discriminating_evidence")
            return self._payload(
                session,
                next_action={
                    "type": "reason",
                    "instruction": (
                        "Compare the available probes against every named hypothesis. "
                        "Choose the one whose positive and negative outcomes reverse "
                        "which hypothesis they support, and record that compact test design."
                    ),
                    "submit_via": "cortheon_step",
                    "required_fields": ["draft"],
                },
                guidance="Prefer information gain over confirmation or low cost alone.",
            )
        if framing_enabled:
            self._originate_competing_hypotheses(session)
        hypothesis_count = _requested_hypothesis_count(
            session.goal,
            session.profile.min_hypotheses,
        )
        if framing_enabled and len(session.hypotheses) < hypothesis_count:
            session.phase = "framing"
            return self._payload(
                session,
                next_action={
                    "type": "reason",
                    "instruction": (
                        f"Propose {hypothesis_count} genuinely distinct, "
                        "falsifiable hypotheses. Include the obvious explanation and at "
                        "least one boundary, caller, dependency, or alternative explanation."
                    ),
                    "submit_via": "cortheon_step",
                    "required_fields": ["hypotheses"],
                },
                guidance=(
                    "Do not reveal private chain-of-thought. Return compact public "
                    "hypotheses and observable falsification tests."
                ),
            )

        request = (
            self._request_for_untested_hypothesis(session)
            if retrieval_enabled
            and evaluation_operator(
                session.evaluation_profile,
                "discriminating_evidence",
            )
            else None
        )
        if request is not None:
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance=(
                    "Use the host's tools. Prefer evidence that discriminates between "
                    "hypotheses over evidence that merely confirms the first idea."
                ),
            )

        completion_gaps = self._completion_gaps(session, None)
        if completion_gaps:
            request = (
                self._request_for_gaps(session, completion_gaps) if retrieval_enabled else None
            )
            if request is not None:
                session.phase = "investigating"
                return self._payload(
                    session,
                    next_action=self._execute_action(request),
                    guidance="Close the observable completion gap using a host-owned tool.",
                )

        reasoning_enabled = any(
            evaluation_operator(session.evaluation_profile, operator)
            for operator in (
                "hypothesis_framing",
                "discriminating_evidence",
                "contradiction_revision",
                "cross_source_derivation",
            )
        )
        if session.evaluation_profile is not None and not reasoning_enabled:
            session.phase = "verifying"
            return self._payload(
                session,
                next_action={
                    "type": "await_candidate",
                    "instruction": (
                        "Submit the host model's candidate and its model-owned receipts "
                        "directly for verification."
                    ),
                    "submit_via": "cortheon_verify",
                },
                guidance="Cortheon will not originate reasoning or evidence requests.",
            )
        if not session.draft:
            session.phase = "synthesizing"
            return self._payload(
                session,
                next_action={
                    "type": "reason",
                    "instruction": (
                        "Synthesize a compact draft from the context pack. State explicit "
                        "claims with evidence ids and preserve unresolved uncertainty."
                    ),
                    "submit_via": "cortheon_challenge",
                    "required_fields": ["draft", "claims"],
                },
                guidance="Every factual claim must point to live evidence in this session.",
            )
        if session.challenge_count == 0 and evaluation_operator(
            session.evaluation_profile,
            "contradiction_revision",
        ):
            session.phase = "challenging"
            return self._payload(
                session,
                next_action={
                    "type": "challenge",
                    "instruction": (
                        "Submit the current draft and its evidence-linked claims for an "
                        "adversarial completion check."
                    ),
                    "submit_via": "cortheon_challenge",
                },
                guidance="Challenge the strongest conclusion before asking to verify it.",
            )
        session.phase = "verifying"
        return self._payload(
            session,
            next_action={
                "type": "verify",
                "instruction": (
                    "Submit the revised answer, explicit evidence-linked claims, and the "
                    "ids of evidence that demonstrate task completion."
                ),
                "submit_via": "cortheon_verify",
            },
            guidance="Verification is structural and evidence-gated, never self-certification.",
        )

    def _semantic_conflict_response(
        self,
        session: Investigation,
    ) -> dict[str, Any] | None:
        semantic_request = next(
            (
                request
                for request in session.requests.values()
                if request.parameters.get("operation") == "semantic_join"
            ),
            None,
        )
        if semantic_request is None:
            return None
        paths = [
            str(item)
            for item in semantic_request.parameters.get("paths", ())
            if isinstance(item, str)
        ]
        derivation = _semantic_join_analysis(
            session.goal,
            paths,
            (
                item
                for item in session.observations.values()
                if item.status != "failed" and not item.quarantine_flags
            ),
            require_all_documents=not bool(semantic_request.parameters.get("discovered")),
        )
        if derivation is None or derivation.get("status") != "conflicted":
            return None

        conflicts = [item for item in derivation.get("conflicts", ()) if isinstance(item, dict)]
        description = "; ".join(
            (
                f"{item.get('entity')} {item.get('relation')} = "
                + " or ".join(str(value) for value in item.get("targets", ()))
            )
            for item in conflicts
        )[:1_000]
        rounds = sum(
            1
            for request in session.requests.values()
            if request.parameters.get("operation") == "semantic_disambiguation"
        )
        if rounds >= session.strictness.max_request_attempts:
            session.phase = "inconclusive"
            return self._payload(
                session,
                next_action={
                    "type": "finish",
                    "instruction": (
                        "The bounded disambiguation budget is exhausted and the live "
                        f"sources still conflict ({description}). Report the unresolved "
                        "alternatives and abandon; do not choose one by plausibility."
                    ),
                    "submit_via": "cortheon_finish",
                },
                guidance=(
                    "A bounded inconclusive result is safer and more useful than a "
                    "doom loop or an invented tie-break."
                ),
            )

        request = self._create_request(
            session,
            capability="search",
            query=(
                "Resolve this cross-source conflict with one focused host search or "
                f"read: {description}. Find an explicit current, effective, or "
                "superseding mapping from an authoritative project source."
            ),
            reason="The candidate synthesis branches disagree on a load-bearing relation.",
            success_condition=(
                "Return a focused host-receipted line that explicitly names the entity, "
                "the current/effective relation, and the selected target. A scoped null "
                "result is acceptable and must not be retried unchanged."
            ),
            parameters={
                "operation": "semantic_disambiguation",
                "conflicts": conflicts[:4],
                "tool_call_budget": 1,
            },
        )
        session.phase = "investigating"
        return self._payload(
            session,
            next_action=self._execute_action(request),
            guidance=(
                "Replan around the contradiction. Seek authority or effective-date "
                "evidence, not another repetition of either branch."
            ),
        )
