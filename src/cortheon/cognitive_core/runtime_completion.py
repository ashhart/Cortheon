"""CompletionMixin for CognitiveRuntime."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.adaptive_stopping import validate_adaptive_completion
from cortheon.cognitive_core.aggregate_alignment import _evidence_alignment_check
from cortheon.cognitive_core.claims import _claim_profiles_from_checks
from cortheon.cognitive_core.models import CognitiveRuntimeError, Hypothesis, Investigation
from cortheon.cognitive_core.receipts import _digest
from cortheon.cognitive_core.runtime_failed_verification import _completion_gaps
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.tasks import (
    _is_contradiction_revision_goal,
    _is_discriminating_test_design_goal,
)
from cortheon.cognitive_core.text import _normalized, _string_list, _text
from cortheon.cognitive_core.uncertainty_visibility import uncertainty_visibility_check
from cortheon.cognitive_protocol import (
    CORTHEON_CERTIFICATION_SCOPE,
    CORTHEON_PROTOCOL_VERSION,
    evaluation_operator,
)


class CompletionMixin(RuntimeState):
    """Completion responsibilities of CognitiveRuntime."""

    def complete(
        self,
        session_id: str,
        *,
        answer: str,
        claims: Iterable[dict[str, Any]],
        hypotheses: Iterable[dict[str, Any]],
        completion_evidence_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Challenge, verify, return, and discard an answer transactionally."""

        raw_hypotheses = list(hypotheses)
        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            if not evaluation_operator(session.evaluation_profile, "verification"):
                raise CognitiveRuntimeError("verification is disabled by the evaluation profile")
            framing_enabled = evaluation_operator(
                session.evaluation_profile,
                "hypothesis_framing",
            )
            revision_enabled = evaluation_operator(
                session.evaluation_profile,
                "contradiction_revision",
            )
            if framing_enabled and not raw_hypotheses:
                raise ValueError("hypotheses must contain at least one item")
            self._take_turn(session)
            if framing_enabled:
                self._supersede_provisional_hypothesis_requests(session)
                session.hypotheses.clear()
                self._apply_completion_hypotheses(session, raw_hypotheses)

            normalized_answer = _text(
                answer,
                "answer",
                maximum=session.profile.max_context_chars,
            )
            reasoning_binding = _validate_revision_completion(session, normalized_answer)
            normalized_claims = self._claims(session, claims)
            completion_ids = _string_list(
                completion_evidence_ids,
                "completion_evidence_ids",
                maximum_items=session.profile.max_observations,
                maximum_chars=4_000,
            )

            session.phase = "challenging" if revision_enabled else "verifying"
            session.draft = normalized_answer
            session.claims = normalized_claims
            session.challenge_count += int(revision_enabled)
            session.verified_answer_digest = None
            attacks = self._attack_surface(session, normalized_claims) if revision_enabled else []

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
            checks.append(
                {
                    "name": "uncertainty_visibility",
                    "passed": True,
                    "reason": (
                        "The test design preserves both named outcomes without deciding "
                        "which hypothesis is true."
                    ),
                }
                if _is_discriminating_test_design_goal(session.goal)
                else uncertainty_visibility_check(session, normalized_answer)
            )
            material_attacks = [item for item in attacks if item["severity"] in {"medium", "high"}]
            checks.append(
                {
                    "name": "adversarial_resolution",
                    "passed": not material_attacks,
                    "reason": (
                        "The transactional challenge found no unresolved material attack."
                        if not material_attacks
                        else "; ".join(str(item["issue"]) for item in material_attacks)
                    ),
                }
            )
            passed = all(check["passed"] for check in checks)
            gaps = [str(check["reason"]) for check in checks if not check["passed"]]
            report = {
                "verdict": "ready" if passed else "needs_evidence",
                "checks": checks,
                "gaps": gaps,
                "answer_digest": _digest(normalized_answer),
                "claim_verification": _claim_profiles_from_checks(checks),
            }
            session.last_verification = report
            self._record_evaluation_operator(session, "verification")
            if revision_enabled:
                self._record_evaluation_operator(session, "contradiction_revision")

            if passed:
                session.phase = "ready"
                session.verified_answer_digest = _digest(normalized_answer)
                result = {
                    "status": "complete",
                    "answer": normalized_answer,
                    "session_id": session.session_id,
                    "protocol_version": CORTHEON_PROTOCOL_VERSION,
                    "certification_scope": CORTHEON_CERTIFICATION_SCOPE,
                    "scorecard": self._scorecard(session),
                    "claim_verification": report["claim_verification"],
                    "attacks": attacks,
                    "discarded": True,
                    "retained_project_data": False,
                }
                if session.waivers:
                    result["caveats"] = sorted(session.waivers.values())
                if reasoning_binding is not None:
                    result["reasoning_binding"] = reasoning_binding
                self._record_completion(session)
                del self._sessions[session.session_id]
                return result

            self._metrics["completion_withheld"] += 1
            session.phase = "investigating"
            next_action = self._failed_verification_action(session, checks, gaps)
            payload = self._payload(
                session,
                next_action=next_action,
                guidance=(
                    "Completion was withheld. Follow the returned next action and retry "
                    "cortheon_complete; do not emit the unverified answer."
                ),
            )
            payload["attacks"] = attacks
            payload["verification"] = report
            self._commit(session)
            return payload

    def _supersede_provisional_hypothesis_requests(
        self,
        session: Investigation,
    ) -> None:
        """Retire pending requests tied to hypotheses this transaction discards.

        Only unevidenced provisional ``substrate_abduction`` requests are
        marked ``superseded`` (binding kept in
        ``parameters.superseded_hypothesis_id``, never counted satisfied);
        ordinary, ``host_model``, and evidenced requests still block.
        """

        for request in session.requests.values():
            if request.status != "pending" or request.hypothesis_id is None:
                continue
            hypothesis = session.hypotheses.get(request.hypothesis_id)
            if hypothesis is None or not self._is_provisional(hypothesis):
                continue
            request.parameters["superseded_hypothesis_id"] = request.hypothesis_id
            request.hypothesis_id = None
            request.status = "superseded"
            self._metrics["requests_superseded"] += 1

    @staticmethod
    def _is_provisional(hypothesis: Hypothesis) -> bool:
        return (
            hypothesis.origin == "substrate_abduction"
            and not hypothesis.supporting_evidence
            and not hypothesis.contradicting_evidence
            and not hypothesis.bearing_evidence
        )

    def _apply_completion_hypotheses(
        self,
        session: Investigation,
        raw_hypotheses: Iterable[dict[str, Any]],
    ) -> None:
        """Add and resolve public hypotheses supplied to the compact path."""

        for raw in raw_hypotheses:
            if not isinstance(raw, dict):
                raise ValueError("each completion hypothesis must be an object")
            statement = _text(
                raw.get("statement"),
                "completion_hypothesis.statement",
                maximum=2_000,
            )
            falsification_test = _text(
                raw.get("falsification_test"),
                "completion_hypothesis.falsification_test",
                maximum=2_000,
            )
            matching = next(
                (
                    item
                    for item in session.hypotheses.values()
                    if _normalized(item.statement) == _normalized(statement)
                ),
                None,
            )
            if matching is None:
                self._add_hypotheses(
                    session,
                    [
                        {
                            "statement": statement,
                            "falsification_test": falsification_test,
                        }
                    ],
                )
                matching = next(reversed(session.hypotheses.values()))
            self._update_hypotheses(
                session,
                [
                    {
                        "hypothesis_id": matching.hypothesis_id,
                        "status": raw.get("status"),
                        "evidence_ids": raw.get("evidence_ids") or (),
                    }
                ],
                as_revision=False,
            )

    def _completion_gaps(
        self,
        session: Investigation,
        completion_evidence_ids: Iterable[str] | None,
    ) -> list[str]:
        return _completion_gaps(
            session,
            completion_evidence_ids,
            require_host_receipts=self.require_host_receipts,
        )


_REVISION_RECORD_FIELDS = {
    "prior",
    "original_source",
    "decisive_source",
    "decisive_effect",
    "revised",
}
_REVISION_ANSWER_FIELDS = {"prior", "prior_status", "revised", "decisive_source"}


def _store_reasoning_draft(session: Investigation, draft: str) -> dict[str, str] | None:
    normalized = _text(
        draft,
        "draft",
        maximum=session.profile.max_context_chars,
        allow_empty=True,
    )
    binding = None
    if _is_revision_session(session):
        record = _closed_revision_object(normalized, _REVISION_RECORD_FIELDS, "record")
        contract = _revision_contract(session)
        _validate_revision_record(record, contract)
        session.revision_record = record
        session.revision_binding_digest = _digest(
            _canonical_revision({"record": record, "contract": contract})
        )
        binding = _revision_binding(session)
    session.draft = normalized
    session.verified_answer_digest = None
    return binding


def _validate_revision_completion(
    session: Investigation,
    answer: str,
) -> dict[str, str] | None:
    validate_adaptive_completion(session, answer)
    if not _is_revision_session(session):
        return None
    if session.revision_record is None:
        raise CognitiveRuntimeError("an accepted revision record is required before completion")
    record = dict(session.revision_record)
    contract = _revision_contract(session)
    expected = _digest(_canonical_revision({"record": record, "contract": contract}))
    if expected != session.revision_binding_digest:
        raise CognitiveRuntimeError("the accepted revision binding changed")
    _validate_revision_record(record, contract)
    final = _closed_revision_object(answer, _REVISION_ANSWER_FIELDS, "answer")
    if (
        any(final[field] != record[field] for field in ("prior", "revised", "decisive_source"))
        or final["prior_status"] != contract["status_map"][record["decisive_effect"]]
    ):
        raise CognitiveRuntimeError("final answer contradicts the public revision record")
    return _revision_binding(session)


def _revision_contract(session: Investigation) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for observation in session.observations.values():
        receipt = observation.host_receipt or {}
        raw_receipt_args = receipt.get("args")
        receipt_args = raw_receipt_args if isinstance(raw_receipt_args, dict) else {}
        receipt_path = receipt_args.get("path") or receipt_args.get("filePath")
        if (
            observation.status == "failed"
            or observation.quarantine_flags
            or observation.kind != "artifact"
            or not observation.source
            or receipt.get("tool") != "read"
            or receipt.get("outcome") != "result"
            or receipt_path != observation.source
        ):
            continue
        try:
            payload = json.loads(observation.content)
        except json.JSONDecodeError:
            continue
        contract = _parse_public_revision_contract(payload)
        if contract is not None:
            matches.append(
                {
                    "evidence_id": observation.evidence_id,
                    "evidence_digest": observation.digest,
                    **contract,
                }
            )
    if len(matches) != 1:
        raise CognitiveRuntimeError("exactly one public revision contract is required")
    return matches[0]


def _parse_public_revision_contract(payload: Any) -> dict[str, Any] | None:
    """Parse the one closed revision contract used by runtime and evaluator."""

    response = payload.get("response_schema") if isinstance(payload, dict) else None
    evidence = payload.get("evidence") if isinstance(payload, dict) else None
    fields = response.get("fields") if isinstance(response, dict) else None
    hypotheses = response.get("hypothesis_vocabulary") if isinstance(response, dict) else None
    statuses = response.get("status_vocabulary") if isinstance(response, dict) else None
    status_map = response.get("effect_status_map") if isinstance(response, dict) else None
    change_map = response.get("effect_changes_hypothesis") if isinstance(response, dict) else None
    sources: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            source = item.get("source_id") if isinstance(item, dict) else None
            if isinstance(source, str):
                sources.append(source)
    if (
        not isinstance(fields, list)
        or len(fields) != len(_REVISION_ANSWER_FIELDS)
        or set(fields) != _REVISION_ANSWER_FIELDS
        or not isinstance(hypotheses, list)
        or not 1 <= len(hypotheses) <= 32
        or len(set(hypotheses)) != len(hypotheses)
        or not all(isinstance(item, str) and 0 < len(item) <= 256 for item in hypotheses)
        or not isinstance(statuses, list)
        or not 1 <= len(statuses) <= 16
        or len(set(statuses)) != len(statuses)
        or not all(isinstance(item, str) and 0 < len(item) <= 64 for item in statuses)
        or not isinstance(status_map, dict)
        or not 2 <= len(status_map) <= 8
        or not all(
            isinstance(effect, str)
            and 0 < len(effect) <= 64
            and isinstance(status, str)
            and status in statuses
            for effect, status in status_map.items()
        )
        or set(status_map.values()) != set(statuses)
        or not isinstance(change_map, dict)
        or set(change_map) != set(status_map)
        or not all(type(value) is bool for value in change_map.values())
        or not isinstance(evidence, list)
        or len(sources) != len(evidence)
        or not 2 <= len(sources) <= 32
        or len(set(sources)) != len(sources)
        or not all(isinstance(source, str) and 0 < len(source) <= 256 for source in sources)
    ):
        return None
    return {
        "hypotheses": sorted(hypotheses),
        "sources": sorted(sources),
        "status_map": dict(status_map),
        "change_map": dict(change_map),
    }


def _validate_revision_record(record: dict[str, str], contract: dict[str, Any]) -> None:
    effect = record["decisive_effect"]
    if (
        record["prior"] not in contract["hypotheses"]
        or record["revised"] not in contract["hypotheses"]
        or record["original_source"] not in contract["sources"]
        or record["decisive_source"] not in contract["sources"]
        or record["original_source"] == record["decisive_source"]
        or effect not in contract["status_map"]
        or (record["prior"] != record["revised"]) is not contract["change_map"][effect]
    ):
        raise CognitiveRuntimeError("revision record is outside the public contract")


def _closed_revision_object(raw: str, fields: set[str], label: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CognitiveRuntimeError(f"revision {label} must be a JSON object") from exc
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not all(isinstance(item, str) and item for item in value.values())
    ):
        raise CognitiveRuntimeError(f"revision {label} has the wrong closed shape")
    return value


def _revision_binding(session: Investigation) -> dict[str, str]:
    if session.revision_binding_digest is None:
        raise CognitiveRuntimeError("revision binding is incomplete")
    return {
        "schema_version": "1",
        "reasoning_binding_sha256": session.revision_binding_digest,
    }


def _is_revision_session(session: Investigation) -> bool:
    return evaluation_operator(
        session.evaluation_profile,
        "contradiction_revision",
    ) and _is_contradiction_revision_goal(session.goal)


def _canonical_revision(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
