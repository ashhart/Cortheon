"""Shared attribute and method contract for the CognitiveRuntime mixins.

The mixins reference each other through ``self``; this base class declares
that combined surface once so every mixin shares one typed contract.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from cortheon.cognitive_core.models import Investigation
from cortheon.cognitive_protocol import EVALUATION_OPERATORS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from cortheon.cognitive_core.models import (
        EvidenceRequest,
        Hypothesis,
        Investigation,
        PublicClaim,
    )


def initial_metrics() -> dict[str, float | int]:
    return {
        "sessions_started": 0,
        "sessions_completed": 0,
        "sessions_evidence_closed": 0,
        "sessions_abandoned": 0,
        "sessions_expired": 0,
        "completion_withheld": 0,
        "observations_accepted": 0,
        "requests_waived": 0,
        "requests_superseded": 0,
        "evidence_retracted": 0,
        "sessions_reframed": 0,
        "failed_submissions": 0,
        "controller_decisions": 0,
        "controller_alternatives_considered": 0,
        "controller_zero_gain_stops": 0,
        "controller_information_gain_bits_total": 0.0,
        "controller_expected_utility_total": 0.0,
        "hypotheses_originated": 0,
        "completion_latency_ms_total": 0.0,
        "completion_latency_ms_max": 0.0,
    }


def store_evaluation_receipt(
    receipts: OrderedDict[str, dict[str, Any]],
    profile: dict[str, Any],
    *,
    maximum: int,
) -> None:
    nonce = profile["nonce"]
    receipts[nonce] = {
        "schema_version": 1,
        "config_sha256": profile["config_sha256"],
        "implementation_sha256": profile["implementation_sha256"],
        "intercepts_final": profile["config"]["intercepts_final"],
        "cleanup_before_answer": profile["config"]["cleanup_before_answer"],
        "runtime_profile_received": True,
        "adapter_receipt": profile.get("adapter_receipt"),
        "operator_counts": dict.fromkeys(sorted(EVALUATION_OPERATORS), 0),
    }
    while len(receipts) > maximum:
        receipts.popitem(last=False)


def consume_evaluation_receipt(
    receipts: OrderedDict[str, dict[str, Any]],
    nonce: str,
) -> dict[str, Any]:
    if not isinstance(nonce, str):
        raise ValueError("nonce must be a string")
    receipt = receipts.pop(nonce, None)
    if receipt is None:
        raise ValueError("evaluation profile receipt not found")
    return receipt


def record_evaluation_operator(
    receipts: OrderedDict[str, dict[str, Any]],
    profile: dict[str, Any] | None,
    operator: str,
) -> None:
    if profile is None:
        return
    if operator not in EVALUATION_OPERATORS:
        raise ValueError("unknown evaluation operator")
    if profile["config"]["operators"][operator] is not True:
        raise ValueError("disabled evaluation operator executed")
    receipt = receipts.get(profile["nonce"])
    if receipt is None:
        raise ValueError("evaluation profile receipt not found")
    receipt["operator_counts"][operator] += 1


class RuntimeState:
    """Instance attributes owned by CognitiveRuntime.__init__."""

    max_sessions: int
    ttl_seconds: float
    require_host_receipts: bool
    _clock: Any
    _sessions: OrderedDict[str, Investigation]
    _evaluation_receipts: OrderedDict[str, dict[str, Any]]
    _lock: threading.RLock
    _metrics: dict[str, float | int]

    def _add_hypotheses(
        self, session: Investigation, raw_hypotheses: Iterable[dict[str, Any]]
    ) -> None:
        raise NotImplementedError()

    def _add_observation(self, session: Investigation, raw: dict[str, Any]) -> tuple[str, bool]:
        raise NotImplementedError()

    def _apply_completion_hypotheses(
        self, session: Investigation, raw_hypotheses: Iterable[dict[str, Any]]
    ) -> None:
        raise NotImplementedError()

    def _attack_surface(
        self, session: Investigation, claims: list[PublicClaim]
    ) -> list[dict[str, Any]]:
        raise NotImplementedError()

    def _claims(
        self, session: Investigation, raw_claims: Iterable[dict[str, Any]]
    ) -> list[PublicClaim]:
        raise NotImplementedError()

    def _code_discovery_response(self, session: Investigation) -> dict[str, Any] | None:
        raise NotImplementedError()

    @staticmethod
    def _cognition_brief(
        session: Investigation, *, next_action: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def _commit(self, session: Investigation) -> None:
        raise NotImplementedError()

    def _completion_gaps(
        self, session: Investigation, completion_evidence_ids: Iterable[str] | None
    ) -> list[str]:
        raise NotImplementedError()

    def _context_pack(self, session: Investigation) -> dict[str, Any]:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _document_discovery_response(self, session: Investigation) -> dict[str, Any] | None:
        raise NotImplementedError()

    @staticmethod
    def _execute_action(request: EvidenceRequest) -> dict[str, Any]:
        raise NotImplementedError()

    def _failed_verification_action(
        self, session: Investigation, checks: list[dict[str, Any]], gaps: list[str]
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def _environment_grounding_request(self, session: Investigation) -> EvidenceRequest:
        raise NotImplementedError()

    def _frontier_grounding_response(self, session: Investigation) -> dict[str, Any] | None:
        raise NotImplementedError()

    def _initial_request(self, session: Investigation) -> EvidenceRequest:
        raise NotImplementedError()

    def _next_research_source_request(self, session: Investigation) -> EvidenceRequest | None:
        raise NotImplementedError()

    def _maybe_reframe_research(self, session: Investigation) -> None:
        raise NotImplementedError()

    def _originate_competing_hypotheses(self, session: Investigation) -> None:
        raise NotImplementedError()

    def _payload(
        self, session: Investigation, *, next_action: dict[str, Any], guidance: str
    ) -> dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def _pending_request(session: Investigation) -> EvidenceRequest | None:
        raise NotImplementedError()

    def _purge_expired(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def _purpose_rounds(session: Investigation, purpose: str) -> int:
        raise NotImplementedError()

    def _recommend(self, session: Investigation) -> dict[str, Any]:
        raise NotImplementedError()

    def _record_completion(self, session: Investigation) -> None:
        raise NotImplementedError()

    def _record_evaluation_operator(self, session: Investigation, operator: str) -> None:
        raise NotImplementedError()

    def _register_request_attempt(self, session: Investigation, request: EvidenceRequest) -> bool:
        raise NotImplementedError()

    def _request_for_claim_truth(
        self, session: Investigation, profile: dict[str, Any]
    ) -> EvidenceRequest | None:
        raise NotImplementedError()

    def _request_for_gaps(
        self, session: Investigation, gaps: Iterable[str]
    ) -> EvidenceRequest | None:
        raise NotImplementedError()

    def _request_for_research_protocol(self, session: Investigation) -> EvidenceRequest | None:
        raise NotImplementedError()

    def _request_for_untested_hypothesis(self, session: Investigation) -> EvidenceRequest | None:
        raise NotImplementedError()

    @staticmethod
    def _scorecard(session: Investigation) -> dict[str, Any]:
        raise NotImplementedError()

    def _select_hypothesis_action(
        self,
        session: Investigation,
        candidates: list[Hypothesis],
        *,
        challenge: bool,
        mandatory: bool,
    ) -> tuple[Hypothesis, str, str, dict[str, Any]] | None:
        raise NotImplementedError()

    def _semantic_conflict_response(self, session: Investigation) -> dict[str, Any] | None:
        raise NotImplementedError()

    def _session(self, session_id: str) -> Investigation:
        raise NotImplementedError()

    def _take_turn(self, session: Investigation) -> None:
        raise NotImplementedError()

    def _update_hypotheses(
        self,
        session: Investigation,
        updates: Iterable[dict[str, Any]],
        *,
        as_revision: bool = True,
    ) -> None:
        raise NotImplementedError()

    def _verification_checks(
        self, session: Investigation, claims: list[PublicClaim], completion_evidence_ids: list[str]
    ) -> list[dict[str, Any]]:
        raise NotImplementedError()

    def _waive(self, session: Investigation, key: str) -> None:
        raise NotImplementedError()
