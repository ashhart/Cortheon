"""RequestFlowMixin for CognitiveRuntime."""

from __future__ import annotations

import copy
from typing import Any

from cortheon.cognitive_core.models import (
    _ASSIST_WAIVER_CAVEATS,
    _WAIVER_CAVEATS,
    CognitiveRuntimeError,
    EvidenceRequest,
    Investigation,
)
from cortheon.cognitive_core.profiles import _EXPLICIT_FRESHNESS_HINTS, _has_hint
from cortheon.cognitive_core.receipts import _observation_origin
from cortheon.cognitive_core.research_gaps import (
    _LOCAL_PROJECT_DOMAIN_RE,
    _effective_web_lineages,
    _is_local_project_evidence,
    _latest_release_goal,
)
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.text import _text
from cortheon.cognitive_protocol import evaluation_operator


class RequestFlowMixin(RuntimeState):
    """Request Flow responsibilities of CognitiveRuntime."""

    def _request_for_research_protocol(
        self,
        session: Investigation,
    ) -> EvidenceRequest | None:
        usable = [
            item
            for item in session.observations.values()
            if item.kind == "web" and item.status != "failed"
        ]
        origins = {_observation_origin(item) for item in usable}
        origins.discard(None)
        _, effective_lineages, syndicated = _effective_web_lineages(usable)
        purposes = {item.purpose for item in usable if item.purpose}
        waived = session.waivers
        revision_enabled = evaluation_operator(
            session.evaluation_profile,
            "contradiction_revision",
        )

        if not usable and revision_enabled and "contradiction_check" not in waived:
            if (
                self._purpose_rounds(session, "contradiction_check")
                >= session.strictness.max_request_attempts
            ):
                self._waive(session, "contradiction_check")
            else:
                return self._create_request(
                    session,
                    capability="search",
                    query=(
                        "Search the current web for primary sources that directly answer "
                        "the question, independent corroboration, and the strongest credible "
                        "disagreement, correction, or limitation. If no conflict is found, "
                        f"make that scoped result explicit: {session.goal}"
                    ),
                    reason=(
                        "Current research needs attributable live discovery plus an active "
                        "contradiction check."
                    ),
                    success_condition=(
                        "Return focused results from distinct URL origins with retrieval "
                        "time, publication or update date when available, and the strongest "
                        "conflict found or an explicit scoped no-conflict result."
                    ),
                    parameters={"purpose": "contradiction_check"},
                )
        if effective_lineages < 2 and "corroboration" not in waived:
            null_attested = any(
                item.kind == "web"
                and item.purpose == "corroboration"
                and (item.host_receipt or {}).get("outcome") == "no_match"
                for item in session.observations.values()
            )
            if (null_attested and not _latest_release_goal(session.goal)) or (
                self._purpose_rounds(session, "corroboration")
                >= session.strictness.corroboration_rounds
            ):
                self._waive(session, "corroboration")
            else:
                return self._create_request(
                    session,
                    capability="search",
                    query=(
                        "Find an independently worded source from a different publisher "
                        "or origin—not a mirror, press-release copy, or syndicated excerpt—"
                        f"that corroborates this question: {session.goal}"
                    ),
                    reason=(
                        "Distinct URLs do not establish independence when their evidence "
                        "is likely copied or syndicated."
                        if len(origins) >= 2 and syndicated
                        else "A single publisher cannot establish independent corroboration."
                    ),
                    success_condition=(
                        "Return a directly relevant result from a new URL origin with "
                        "retrieval and source-date metadata."
                    ),
                    parameters={"purpose": "corroboration"},
                )
        if usable and "primary_fetch" not in purposes and "primary_fetch" not in waived:
            if (
                self._purpose_rounds(session, "primary_fetch")
                >= session.strictness.max_request_attempts
            ):
                self._waive(session, "primary_fetch")
            else:
                return self._create_request(
                    session,
                    capability="fetch",
                    query=(
                        "Fetch the strongest primary-source URL already found for this "
                        f"question and return its directly relevant passage: {session.goal}"
                    ),
                    reason="Search snippets alone are not primary-source verification.",
                    success_condition=(
                        "Return a passage fetched from the source URL with retrieval time "
                        "and publication or update date when available."
                    ),
                    parameters={"purpose": "primary_fetch"},
                )
        if (
            revision_enabled
            and usable
            and "contradiction_check" not in purposes
            and "contradiction_check" not in waived
        ):
            if (
                self._purpose_rounds(session, "contradiction_check")
                >= session.strictness.max_request_attempts
            ):
                self._waive(session, "contradiction_check")
            else:
                return self._create_request(
                    session,
                    capability="search",
                    query=(
                        "Search specifically for the strongest credible disagreement, "
                        "correction, limitation, or counterevidence to the emerging answer "
                        f"for: {session.goal}"
                    ),
                    reason="Research must actively look for contradiction before completion.",
                    success_condition=(
                        "Return the strongest contradictory evidence found, or an explicit "
                        "scoped no-conflict result, with a source URL and retrieval time."
                    ),
                    parameters={"purpose": "contradiction_check"},
                )
        if (
            _LOCAL_PROJECT_DOMAIN_RE.search(session.goal)
            and "inspect" not in waived
            and not any(
                _is_local_project_evidence(item)
                for item in session.observations.values()
                if item.status != "failed" and not item.quarantine_flags
            )
        ):
            local_rounds = sum(
                1
                for item in session.requests.values()
                if item.capability == "inspect" and item.parameters.get("domain") == "local_project"
            )
            if local_rounds >= session.strictness.max_request_attempts:
                self._waive(session, "inspect")
            else:
                return self._create_request(
                    session,
                    capability="inspect",
                    query=(
                        "Inspect the local workspace or repository evidence explicitly "
                        f"required by this research goal: {session.goal}"
                    ),
                    reason=("The goal explicitly requires local workspace/repository grounding."),
                    success_condition=(
                        "Return a focused local code or document excerpt with a host read "
                        "receipt and its project-relative location."
                    ),
                    parameters={"domain": "local_project"},
                )
        if (
            _has_hint(session.goal, _EXPLICIT_FRESHNESS_HINTS)
            and "freshness_check" not in waived
            and not any(item.published_at for item in usable)
        ):
            if (
                self._purpose_rounds(session, "freshness_check")
                >= session.strictness.max_request_attempts
            ):
                self._waive(session, "freshness_check")
            else:
                return self._create_request(
                    session,
                    capability="search",
                    query=(
                        "Find a dated source that establishes the publication or update "
                        f"time for this freshness-sensitive question: {session.goal}"
                    ),
                    reason="The answer asks for time-sensitive information but lacks a source date.",
                    success_condition=(
                        "Return a relevant URL with an explicit publication or update date "
                        "and current retrieval time."
                    ),
                    parameters={"purpose": "freshness_check"},
                )
        return None

    def _create_request(
        self,
        session: Investigation,
        *,
        capability: str,
        query: str,
        reason: str,
        success_condition: str,
        hypothesis_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> EvidenceRequest:
        if not evaluation_operator(session.evaluation_profile, "retrieval"):
            raise CognitiveRuntimeError("retrieval is disabled by the evaluation profile")
        if len(session.requests) >= session.profile.max_observations:
            raise CognitiveRuntimeError("the evidence-request budget is exhausted")
        request_id = f"req{len(session.requests) + 1}"
        request = EvidenceRequest(
            request_id=request_id,
            capability=capability,
            query=_text(query, "request.query", maximum=3_000),
            reason=_text(reason, "request.reason", maximum=1_000),
            success_condition=_text(
                success_condition,
                "request.success_condition",
                maximum=1_500,
            ),
            parameters=copy.deepcopy(parameters or {}),
            hypothesis_id=hypothesis_id,
        )
        request.parameters.setdefault(
            "tool_call_budget",
            session.profile.max_calls_per_request,
        )
        session.requests[request_id] = request
        self._record_evaluation_operator(session, "retrieval")
        return request

    @staticmethod
    def _pending_request(session: Investigation) -> EvidenceRequest | None:
        return next(
            (item for item in session.requests.values() if item.status == "pending"),
            None,
        )

    def _waive(self, session: Investigation, key: str) -> None:
        """Record that an evidence requirement was downgraded, with its caveat."""

        if key in session.waivers:
            return
        if session.strictness.name == "assist" and key in _ASSIST_WAIVER_CAVEATS:
            session.waivers[key] = _ASSIST_WAIVER_CAVEATS[key]
        else:
            session.waivers[key] = _WAIVER_CAVEATS.get(
                key,
                f"The '{key}' evidence requirement was waived after repeated failed "
                "attempts; related claims carry reduced verification.",
            )
        self._metrics["requests_waived"] += 1

    def _register_request_attempt(
        self,
        session: Investigation,
        request: EvidenceRequest,
    ) -> bool:
        """Count a failed satisfaction attempt; waive the request when exhausted.

        The anti-doom-loop valve: after ``MAX_REQUEST_ATTEMPTS`` honest
        failures the request is downgraded with an explicit caveat that
        surfaces in every later payload. Returns True when now waived.
        """

        if request.status != "pending":
            return False
        request.attempts += 1
        if request.attempts < session.strictness.max_request_attempts:
            return False
        request.status = "waived"
        self._waive(
            session,
            str(request.parameters.get("purpose") or request.capability or "evidence"),
        )
        return True

    @staticmethod
    def _purpose_rounds(session: Investigation, purpose: str) -> int:
        return sum(
            1 for item in session.requests.values() if item.parameters.get("purpose") == purpose
        )

    @staticmethod
    def _execute_action(request: EvidenceRequest) -> dict[str, Any]:
        instruction = (
            "Choose and run the host capability that best satisfies this evidence "
            "request. Cortheon does not execute it."
        )
        budget = request.parameters.get("tool_call_budget")
        if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
            instruction += (
                f" Run at most {budget} host tool calls for this request, then "
                "submit the focused results with cortheon_observe before "
                "investigating further."
            )
        return {
            "type": "harness_tool",
            "instruction": instruction,
            "request": request.public(),
            "submit_via": "cortheon_observe",
        }
