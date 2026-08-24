"""RequestMixin for CognitiveRuntime."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.alignment import _FALSIFICATION_DESIGN_RE
from cortheon.cognitive_core.frontier_policy import needs_frontier_grounding
from cortheon.cognitive_core.models import (
    CognitiveRuntimeError,
    EvidenceRequest,
    Hypothesis,
    Investigation,
)
from cortheon.cognitive_core.profiles import (
    _capability_for_falsification,
    _capability_for_kind,
    _evidence_action_cost,
    _evidence_action_reliability,
    _has_hint,
)
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.semantic_graph import _semantic_terms
from cortheon.cognitive_core.tasks import (
    _AMBIGUITY_GOAL_RE,
    _CROSS_SOURCE_HINTS,
    _goal_code_paths,
    _goal_code_symbols,
    _goal_document_paths,
    _infer_join_operation,
    _is_discriminating_test_design_goal,
)
from cortheon.cognitive_core.text import _lookup_phrase_target, _lookup_target_match
from cortheon.cognitive_graph import rank_information_gain
from cortheon.cognitive_protocol import evaluation_operator


class RequestMixin(RuntimeState):
    """Requests responsibilities of CognitiveRuntime."""

    def _initial_request(self, session: Investigation) -> EvidenceRequest:
        if not evaluation_operator(session.evaluation_profile, "retrieval"):
            raise CognitiveRuntimeError("retrieval is disabled by the evaluation profile")
        if needs_frontier_grounding(session.goal, session.task_kind):
            return self._environment_grounding_request(session)
        parameters: dict[str, Any] = {}
        if session.task_kind == "code":
            paths = _goal_code_paths(session.goal)
            if session.deliverable in {"code_change", "code_understanding"}:
                paths = list(dict.fromkeys([*paths, *_goal_document_paths(session.goal)]))
            symbols = _goal_code_symbols(session.goal)
            if session.deliverable == "code_understanding":
                lookup = _lookup_target_match(session.goal)
                phrase = _lookup_phrase_target(session.goal)
                scope = paths[0] if paths else None
                if len(paths) >= 2:
                    capability = "read_many"
                    parameters = {
                        "paths": paths[:6],
                        "symbols": (
                            []
                            if re.search(r"\bdiagnos(?:e|is|ing)\b", session.goal, re.I)
                            else symbols[:12]
                        ),
                    }
                    operation = _infer_join_operation(session.goal)
                    if operation is None and re.search(
                        r"\b(?:diagnos(?:e|is|ing)|debug|root\s+cause)\b",
                        session.goal,
                        flags=re.IGNORECASE,
                    ):
                        operation = "diagnostic_join"
                    if operation is not None:
                        parameters["operation"] = operation
                    query = (
                        "Read the named live files through the host and join only the "
                        f"evidence relevant to this question: {session.goal}"
                    )
                    success = (
                        "Return focused, separately sourced excerpts from every named "
                        "file so the answer can connect them without loading whole files."
                    )
                elif (lookup is not None or phrase is not None) and scope is not None:
                    capability = "grep"
                    target = (
                        phrase
                        if phrase is not None
                        else (lookup.group(1) if lookup is not None else "")
                    )
                    path = scope
                    parameters = {"pattern": target, "path": path}
                    query = (
                        f"Call the host grep tool with pattern '{target}' and path "
                        f"'{path}' to resolve: {session.goal}"
                    )
                    success = (
                        "Return the exact matching lines, or the host's explicit "
                        "zero-match result for that file."
                    )
                elif paths:
                    capability = "read_many"
                    parameters = {
                        "paths": paths[:1],
                        "symbols": symbols[:12],
                    }
                    query = (
                        "Read the named live file through the host and return only "
                        f"the evidence relevant to this question: {session.goal}"
                    )
                    success = (
                        "Return a focused host-read excerpt from the named file so "
                        "the answer does not rely on model memory."
                    )
                else:
                    capability = "search"
                    parameters = {
                        "operation": "code_discovery",
                        "max_candidates": 6,
                        "discovery_round": 1,
                        "prefer_tests": session.deliverable == "code_change",
                    }
                    query = (
                        "Search the live project for the smallest implementation, caller, "
                        "and test surface that controls this code question before reading "
                        f"broader context: {session.goal}"
                    )
                    success = (
                        "Return project-relative code paths with focused matching lines, "
                        "including a relevant test or observable boundary when one exists."
                    )
            elif paths:
                capability = "read_many"
                parameters = {
                    "paths": paths[:6],
                    "symbols": symbols[:12],
                }
                query = (
                    "Read the named implementation and test files through the host, "
                    f"then make the smallest verified change for: {session.goal}"
                )
                success = (
                    "Return focused live excerpts from each named file, centered on "
                    "the named symbols. Do not edit before this evidence is available."
                )
            else:
                capability = "search"
                parameters = {
                    "operation": "code_discovery",
                    "max_candidates": 6,
                    "discovery_round": 1,
                    "prefer_tests": True,
                }
                query = (
                    "Search the live project for the smallest implementation, caller, "
                    f"and test surface that controls this change: {session.goal}"
                )
                success = (
                    "Return project-relative implementation and test paths with focused "
                    "matching lines. Do not edit before this evidence is available."
                )
        elif session.task_kind == "research":
            capability = "search"
            revision_enabled = evaluation_operator(
                session.evaluation_profile,
                "contradiction_revision",
            )
            parameters = {
                "purpose": "contradiction_check" if revision_enabled else "corroboration",
                "minimum_independent_origins": 2,
                "require_primary_fetch": True,
                "require_contradiction_check": revision_enabled,
            }
            query = (
                "Search the current web for primary sources, independent "
                "corroboration, and the strongest credible disagreement, correction, "
                "or limitation. If no conflict is found, make that scoped result "
                f"explicit: {session.goal}"
                if revision_enabled
                else (
                    "Search the current web for primary sources and independent "
                    f"corroboration for: {session.goal}"
                )
            )
            success = (
                "Return dated, attributable results from distinct URL origins plus "
                "the strongest conflict found or an explicit scoped no-conflict result."
                if revision_enabled
                else "Return dated, attributable results from distinct URL origins."
            )
        elif session.task_kind == "documents":
            paths = _goal_document_paths(session.goal)
            if len(paths) >= 2:
                capability = "read_many"
                parameters = {
                    "paths": paths[:6],
                    "operation": "semantic_join",
                }
                query = (
                    "Read every named live document through the host, then connect the "
                    f"separately sourced facts needed to resolve: {session.goal}"
                )
                success = (
                    "Return focused passages from every named document. Preserve source "
                    "boundaries so Cortheon can test the cross-document bridge."
                )
            else:
                capability = "search"
                if _has_hint(session.goal, _CROSS_SOURCE_HINTS) or _AMBIGUITY_GOAL_RE.search(
                    session.goal
                ):
                    parameters = {
                        "operation": "document_discovery",
                        "extensions": ["md", "markdown", "rst", "txt"],
                        "max_candidates": 6,
                        "discovery_round": 1,
                    }
                query = (
                    "Search the live project for the smallest set of document paths and "
                    "matching lines that could form a cross-source evidence chain for: "
                    f"{session.goal}"
                )
                success = (
                    "Return project-relative document paths with focused matching lines. "
                    "Include bridge terms, not whole files or inferred conclusions."
                )
        elif session.task_kind == "decision":
            capability = "inspect"
            query = (
                "Gather the binding constraints, viable alternatives, and observable "
                f"success criteria for: {session.goal}"
            )
            success = "Return evidence for constraints and at least one real alternative."
        else:
            capability = "inspect"
            query = f"Gather the minimum live evidence needed to answer: {session.goal}"
            success = "Return a focused observation that can support or falsify an answer."
        return self._create_request(
            session,
            capability=capability,
            query=query,
            reason="Establish the live state before trusting the model's weights.",
            success_condition=success,
            parameters=parameters,
        )

    def _request_for_untested_hypothesis(
        self,
        session: Investigation,
    ) -> EvidenceRequest | None:
        if (
            (
                _FALSIFICATION_DESIGN_RE.search(session.goal)
                or _AMBIGUITY_GOAL_RE.search(session.goal)
                or _is_discriminating_test_design_goal(session.goal)
            )
            and session.hypotheses
            and all(
                hypothesis.origin == "substrate_abduction"
                for hypothesis in session.hypotheses.values()
            )
        ):
            return None
        untested = [
            hypothesis
            for hypothesis in session.hypotheses.values()
            if (
                not hypothesis.supporting_evidence
                and not hypothesis.contradicting_evidence
                and not hypothesis.bearing_evidence
            )
        ]
        selected = self._select_hypothesis_action(
            session,
            untested,
            challenge=False,
            mandatory=True,
        )
        if selected is not None:
            hypothesis, capability, query, controller = selected
            request = self._create_request(
                session,
                capability=capability,
                query=query,
                reason=(
                    f"Test {hypothesis.hypothesis_id}; the controller selected the "
                    "highest expected uncertainty reduction per unit cost."
                ),
                success_condition=(
                    "Return evidence that clearly supports or contradicts the "
                    "hypothesis, with a precise live source reference."
                ),
                hypothesis_id=hypothesis.hypothesis_id,
                parameters={"controller": controller},
            )
            self._record_evaluation_operator(session, "discriminating_evidence")
            return request
        challengeable = [
            hypothesis
            for hypothesis in session.hypotheses.values()
            if (
                hypothesis.status == "supported"
                and not hypothesis.contradicting_evidence
                and len(hypothesis.supporting_evidence) < 2
                and session.profile.name != "quick"
            )
        ]
        selected = self._select_hypothesis_action(
            session,
            challengeable,
            challenge=True,
            mandatory=False,
        )
        if selected is not None:
            hypothesis, capability, query, controller = selected
            request = self._create_request(
                session,
                capability=capability,
                query=query,
                reason=(
                    "Actively search for disconfirming evidence selected by expected "
                    "information gain."
                ),
                success_condition=(
                    "Return the strongest counterexample or a concrete failed "
                    "falsification attempt."
                ),
                hypothesis_id=hypothesis.hypothesis_id,
                parameters={"controller": controller},
            )
            self._record_evaluation_operator(session, "discriminating_evidence")
            return request
        return None

    def _select_hypothesis_action(
        self,
        session: Investigation,
        candidates: list[Hypothesis],
        *,
        challenge: bool,
        mandatory: bool,
    ) -> tuple[Hypothesis, str, str, dict[str, Any]] | None:
        if not candidates:
            return None
        weights = {
            hypothesis.hypothesis_id: {
                "open": 1.0,
                "uncertain": 1.0,
                "supported": 0.65,
                "refuted": 0.35,
            }[hypothesis.status]
            for hypothesis in session.hypotheses.values()
        }
        actions: list[dict[str, Any]] = []
        by_id: dict[str, tuple[Hypothesis, str, str]] = {}
        for hypothesis in candidates:
            query = (
                "Try to falsify this currently supported hypothesis: "
                f"{hypothesis.statement}. Test: {hypothesis.falsification_test}"
                if challenge
                else hypothesis.falsification_test
            )
            capability = _capability_for_falsification(
                session.task_kind,
                hypothesis.falsification_test,
            )
            query_terms = _semantic_terms(query)
            resolves = [
                other.hypothesis_id
                for other in session.hypotheses.values()
                if (
                    other.hypothesis_id == hypothesis.hypothesis_id
                    or len(
                        query_terms
                        & _semantic_terms(f"{other.statement} {other.falsification_test}")
                    )
                    >= 2
                )
            ]
            action_id = hypothesis.hypothesis_id
            actions.append(
                {
                    "action_id": action_id,
                    "resolves": resolves,
                    "cost": _evidence_action_cost(capability),
                    "reliability": _evidence_action_reliability(capability),
                }
            )
            by_id[action_id] = (hypothesis, capability, query)
        ranked = rank_information_gain(weights, actions)
        adaptive = evaluation_operator(session.evaluation_profile, "adaptive_stopping")
        if adaptive:
            self._record_evaluation_operator(session, "adaptive_stopping")
        if not ranked or (adaptive and not mandatory and ranked[0]["expected_utility"] <= 0):
            if not mandatory:
                self._metrics["controller_zero_gain_stops"] += 1
            return None
        selected = ranked[0]
        self._metrics["controller_decisions"] += 1
        self._metrics["controller_alternatives_considered"] += len(ranked)
        self._metrics["controller_information_gain_bits_total"] += float(
            selected["information_gain_bits"]
        )
        self._metrics["controller_expected_utility_total"] += float(selected["expected_utility"])
        hypothesis, capability, query = by_id[selected["action_id"]]
        controller = {
            "policy": "expected_information_gain_per_cost",
            "mandatory": mandatory,
            "selected": selected,
            "alternatives": ranked[1:4],
            "stop_when": (
                "mandatory evidence gate satisfied"
                if mandatory
                else "expected utility is zero or negative"
            ),
        }
        return hypothesis, capability, query, controller

    def _request_for_gaps(
        self,
        session: Investigation,
        gaps: Iterable[str],
    ) -> EvidenceRequest | None:
        gap_list = [item for item in gaps if item and "concise-change budget" not in item]
        if not gap_list:
            return None
        combined = " ".join(gap_list)
        lower = combined.lower()
        if "predates" in lower or "rerun" in lower:
            capability = "test"
            success = (
                "Rerun the relevant test against the final captured change and return "
                "the exact command, zero/nonzero outcome, and focused summary."
            )
        elif "diff" in lower or "what changed" in lower:
            capability = "diff"
            success = (
                "Return the focused live diff for the proposed change with enough "
                "location context to tie it to the answer."
            )
        elif "test" in lower or "execution" in lower:
            capability = "test"
            success = (
                "Return the exact command, outcome, and relevant failure or pass summary. "
                "The host must mark independently confirmed results as verified."
            )
        elif "local workspace" in lower or "local repository" in lower or "local project" in lower:
            capability = "inspect"
            success = (
                "Return a focused local code or document excerpt with a host read "
                "receipt and its project-relative location."
            )
        elif "source" in lower or "document" in lower or "corrobor" in lower:
            capability = "search"
            success = "Return a focused passage from a distinct, attributable live source."
        elif "code" in lower or "implementation" in lower:
            capability = "read"
            success = "Return only the relevant symbol or focused excerpt and its location."
        else:
            capability = _capability_for_kind(session.task_kind)
            success = "Return the minimum live observation that closes the named gap."
        return self._create_request(
            session,
            capability=capability,
            query=f"Close this completion gap for '{session.goal}': {gap_list[0]}",
            reason=gap_list[0],
            success_condition=success,
        )
