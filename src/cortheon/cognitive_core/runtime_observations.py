"""ObservationMixin for CognitiveRuntime."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.models import (
    OBSERVATION_KINDS,
    OBSERVATION_STATUSES,
    RESEARCH_PURPOSES,
    CognitiveRuntimeError,
    Investigation,
    Observation,
)
from cortheon.cognitive_core.receipts import (
    _digest,
    _host_evidence_receipt,
    _observation_digest,
    _validate_host_observation_batch,
)
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.text import (
    _optional_text,
    _optional_timestamp,
    _optional_url,
    _string_list,
    _text,
)
from cortheon.cognitive_protocol import evaluation_operator
from cortheon.sanitize import scan_text


class ObservationMixin(RuntimeState):
    """Observations responsibilities of CognitiveRuntime."""

    def observe(
        self,
        session_id: str,
        observations: Iterable[dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Accept bounded live evidence supplied by the host."""

        raw_observations = list(observations)
        if not raw_observations:
            raise ValueError("observations must contain at least one item")
        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            if request_id is not None:
                request = session.requests.get(request_id)
                if request is None:
                    raise ValueError(f"unknown request_id: {request_id}")
                if request.status != "pending":
                    raise ValueError(f"request is already resolved: {request_id}")
            else:
                request = None

            request_satisfied = True
            validation_error: str | None = None
            covered_before = set(request.covered_paths) if request is not None else set()
            if self.require_host_receipts:
                try:
                    request_satisfied = _validate_host_observation_batch(
                        request,
                        raw_observations,
                    )
                except CognitiveRuntimeError as exc:
                    if request is None or not self._register_request_attempt(session, request):
                        if request is not None:
                            self._commit(session)
                        raise
                    request_satisfied = False
                    validation_error = str(exc)[:500]

            accepted: list[str] = []
            duplicates: list[str] = []
            duplicate_ids: list[str] = []
            for raw in raw_observations:
                evidence_id, duplicate = self._add_observation(session, raw)
                if duplicate:
                    duplicates.append(_observation_digest(raw))
                    duplicate_ids.append(evidence_id)
                else:
                    accepted.append(evidence_id)

            if not accepted and not duplicates:
                raise ValueError("no observations were accepted")
            if request is not None:
                read_many_progressed = request.capability == "read_many" and bool(
                    set(request.covered_paths) - covered_before
                )
                if request_satisfied:
                    request.status = "completed"
                elif (
                    request.status == "pending"
                    and validation_error is None
                    and not read_many_progressed
                ):
                    self._register_request_attempt(session, request)
            session.verified_answer_digest = None
            if request is not None and request.hypothesis_id and duplicate_ids and not accepted:
                self._settle_duplicate_hypothesis_request(session, request, duplicate_ids)
            evidence_was_classified = bool(
                request is not None
                and request.hypothesis_id
                and any(
                    request.hypothesis_id
                    in {
                        *session.observations[evidence_id].supports,
                        *session.observations[evidence_id].contradicts,
                    }
                    for evidence_id in accepted
                )
            )
            cleanup_after_retrieval = bool(
                accepted
                and session.evaluation_profile is not None
                and session.evaluation_profile["config"]["cleanup_before_answer"] is True
            )
            if cleanup_after_retrieval:
                session.phase = "evidence_ready"
                response = self._payload(
                    session,
                    next_action={
                        "type": "disengage",
                        "instruction": "Use the attached evidence; Cortheon will not intercept the answer.",
                    },
                    guidance="Evidence acquisition is complete and ephemeral state is erased.",
                )
                response["status"] = "disengaged"
            elif (
                request is not None
                and request.hypothesis_id
                and accepted
                and not evidence_was_classified
                and evaluation_operator(
                    session.evaluation_profile,
                    "contradiction_revision",
                )
            ):
                session.phase = "connecting"
                response = self._payload(
                    session,
                    next_action={
                        "type": "reason",
                        "instruction": (
                            "Classify newly accepted evidence against the named "
                            "hypothesis. Submit supported, refuted, or uncertain with the "
                            "accepted evidence ids; do not gather the same evidence again."
                        ),
                        "submit_via": "cortheon_step",
                        "required_fields": ["hypothesis_updates"],
                    },
                    guidance=(
                        f"Update {request.hypothesis_id} using only accepted evidence ids: "
                        + ", ".join(accepted)
                    ),
                )
            else:
                response = self._recommend(session)
            response["accepted_evidence_ids"] = accepted
            response["duplicate_observations"] = len(duplicates)
            if validation_error is not None:
                response["validation_error"] = validation_error
            if cleanup_after_retrieval:
                self._sessions.pop(session.session_id, None)
                self._metrics["sessions_abandoned"] += 1
            else:
                self._commit(session)
            self._metrics["observations_accepted"] += len(accepted)
            return response

    def retract(
        self,
        session_id: str,
        evidence_ids: Iterable[str],
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        """Withdraw bad evidence without consuming a reasoning turn."""

        normalized_reason = _optional_text(reason or None, "reason", maximum=500) or ""
        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            ids = _string_list(
                evidence_ids,
                "evidence_ids",
                maximum_items=session.profile.max_observations,
                maximum_chars=4_000,
            )
            if not ids:
                raise ValueError("evidence_ids must contain at least one item")
            unknown = [item for item in ids if item not in session.observations]
            if unknown:
                raise ValueError(f"unknown evidence ids: {', '.join(unknown)}")
            for evidence_id in ids:
                observation = session.observations[evidence_id]
                observation.status = "failed"
                if "retracted" not in observation.quarantine_flags:
                    observation.quarantine_flags.append("retracted")
                session.observation_digests.discard(observation.digest)
                observation.supports = []
                observation.contradicts = []
                for hypothesis in session.hypotheses.values():
                    if evidence_id in hypothesis.supporting_evidence:
                        hypothesis.supporting_evidence.remove(evidence_id)
                        if hypothesis.status == "supported" and not hypothesis.supporting_evidence:
                            hypothesis.status = "open"
                    if evidence_id in hypothesis.contradicting_evidence:
                        hypothesis.contradicting_evidence.remove(evidence_id)
                        if hypothesis.status == "refuted" and not hypothesis.contradicting_evidence:
                            hypothesis.status = "open"
                    if evidence_id in hypothesis.bearing_evidence:
                        hypothesis.bearing_evidence.remove(evidence_id)
                        if hypothesis.status == "uncertain" and not hypothesis.bearing_evidence:
                            hypothesis.status = "open"
            session.verified_answer_digest = None
            response = self._recommend(session)
            response["retracted_evidence_ids"] = ids
            if normalized_reason:
                response["retraction_reason"] = normalized_reason
            self._commit(session)
            self._metrics["evidence_retracted"] += len(ids)
            return response

    def challenge(
        self,
        session_id: str,
        *,
        draft: str,
        claims: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        """Adversarially inspect a draft without pretending to prove its truth."""

        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            if not evaluation_operator(
                session.evaluation_profile,
                "contradiction_revision",
            ):
                raise CognitiveRuntimeError(
                    "contradiction revision is disabled by the evaluation profile"
                )
            self._take_turn(session)
            session.phase = "challenging"
            session.draft = _text(
                draft,
                "draft",
                maximum=session.profile.max_context_chars,
            )
            session.claims = self._claims(session, claims)
            session.challenge_count += 1
            session.verified_answer_digest = None

            attacks = self._attack_surface(session, session.claims)
            next_action: dict[str, Any]
            pending = self._pending_request(session)
            material_attacks = [item for item in attacks if item["severity"] in {"medium", "high"}]
            if pending is not None:
                next_action = self._execute_action(pending)
            elif material_attacks:
                next_action = {
                    "type": "reason",
                    "instruction": (
                        "Revise the draft against every listed attack. If an attack needs "
                        "new evidence, call cortheon_step with the unresolved question so "
                        "Cortheon can issue a targeted host-tool request."
                    ),
                    "submit_via": "cortheon_step",
                }
            else:
                next_action = {
                    "type": "verify",
                    "instruction": (
                        "The structural challenge found no open attack. Submit the revised "
                        "answer, explicit claims, and completion evidence to cortheon_verify."
                    ),
                    "submit_via": "cortheon_verify",
                }
            payload = self._payload(
                session,
                next_action=next_action,
                guidance=(
                    "These are public, evidence-linked attacks on the answer—not hidden "
                    "chain-of-thought. Resolve them or state the remaining uncertainty."
                ),
            )
            payload["attacks"] = attacks
            self._commit(session)
            self._record_evaluation_operator(session, "contradiction_revision")
            return payload

    def _add_observation(
        self,
        session: Investigation,
        raw: dict[str, Any],
    ) -> tuple[str, bool]:
        if not isinstance(raw, dict):
            raise ValueError("each observation must be an object")
        if len(session.observations) >= session.profile.max_observations:
            raise CognitiveRuntimeError("the observation-count budget is exhausted")
        kind = _text(raw.get("kind"), "observation.kind", maximum=32)
        if kind not in OBSERVATION_KINDS:
            raise ValueError(
                f"observation kind must be one of: {', '.join(sorted(OBSERVATION_KINDS))}"
            )
        status = _text(
            raw.get("status", "observed"),
            "observation.status",
            maximum=32,
        )
        if status not in OBSERVATION_STATUSES:
            raise ValueError("observation status must be observed, verified, or failed")
        content = _text(
            raw.get("content"),
            "observation.content",
            maximum=session.profile.max_observation_chars,
        )
        source = _optional_text(raw.get("source"), "observation.source", maximum=1_000)
        url = _optional_url(raw.get("url"), "observation.url")
        retrieved_at = _optional_timestamp(
            raw.get("retrieved_at"),
            "observation.retrieved_at",
        )
        published_at = _optional_timestamp(
            raw.get("published_at"),
            "observation.published_at",
            allow_date=True,
        )
        purpose = _optional_text(
            raw.get("purpose"),
            "observation.purpose",
            maximum=64,
        )
        if purpose is not None and purpose not in RESEARCH_PURPOSES:
            raise ValueError(
                "observation.purpose must be one of: " + ", ".join(sorted(RESEARCH_PURPOSES))
            )
        if kind != "web" and any((url, retrieved_at, published_at, purpose)):
            raise ValueError(
                "url, retrieved_at, published_at, and purpose are reserved for web evidence"
            )
        supports = _string_list(
            raw.get("supports") or (),
            "observation.supports",
            maximum_items=session.profile.max_hypotheses,
            maximum_chars=1_000,
        )
        contradicts = _string_list(
            raw.get("contradicts") or (),
            "observation.contradicts",
            maximum_items=session.profile.max_hypotheses,
            maximum_chars=1_000,
        )
        if contradicts and not evaluation_operator(
            session.evaluation_profile,
            "contradiction_revision",
        ):
            raise CognitiveRuntimeError(
                "contradiction revision is disabled by the evaluation profile"
            )
        unknown = [item for item in supports + contradicts if item not in session.hypotheses]
        if unknown:
            raise ValueError(f"unknown hypothesis ids: {', '.join(unknown)}")
        overlap = sorted(set(supports) & set(contradicts))
        if overlap:
            raise ValueError(
                "an observation cannot both support and contradict the same "
                f"hypothesis: {', '.join(overlap)}"
            )
        if status == "failed" and (supports or contradicts):
            raise ValueError("a failed observation cannot support or contradict a hypothesis")

        host_receipt = _host_evidence_receipt(content)
        evidence_content = content.partition("\n")[2] if host_receipt is not None else content
        scan = scan_text(evidence_content, preserve_layout=True)
        clean = scan.clean_text.strip()
        if not clean:
            clean = "[instruction-shaped content quarantined]"
            status = "failed"
            supports = []
            contradicts = []
        digest = _digest(
            "\x00".join(
                (
                    kind,
                    source or "",
                    status,
                    clean,
                    json.dumps(
                        host_receipt,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if host_receipt is not None
                    else "",
                    url or "",
                    retrieved_at or "",
                    published_at or "",
                    purpose or "",
                )
            )
        )
        if digest in session.observation_digests:
            existing = next(
                item.evidence_id for item in session.observations.values() if item.digest == digest
            )
            return existing, True
        current_chars = sum(len(item.content) for item in session.observations.values())
        if current_chars + len(clean) > session.profile.max_total_observation_chars:
            raise CognitiveRuntimeError("the total observation-character budget is exhausted")

        evidence_id = f"ev{len(session.observations) + 1}"
        observation = Observation(
            evidence_id=evidence_id,
            kind=kind,
            content=clean,
            source=source,
            status=status,
            supports=supports,
            contradicts=contradicts,
            quarantine_flags=["instruction_like_segment"] * len(scan.flags),
            sequence=len(session.observations) + 1,
            digest=digest,
            host_receipt=copy.deepcopy(host_receipt),
            url=url,
            retrieved_at=retrieved_at,
            published_at=published_at,
            purpose=purpose,
        )
        session.observations[evidence_id] = observation
        session.observation_digests.add(digest)
        for hypothesis_id in supports:
            hypothesis = session.hypotheses[hypothesis_id]
            if evidence_id not in hypothesis.supporting_evidence:
                hypothesis.supporting_evidence.append(evidence_id)
            if status == "verified":
                hypothesis.status = "supported"
        for hypothesis_id in contradicts:
            hypothesis = session.hypotheses[hypothesis_id]
            if evidence_id not in hypothesis.contradicting_evidence:
                hypothesis.contradicting_evidence.append(evidence_id)
            if status == "verified":
                hypothesis.status = "refuted"
        return evidence_id, False

    @staticmethod
    def _settle_duplicate_hypothesis_request(
        session: Investigation,
        request: Any,
        evidence_ids: list[str],
    ) -> None:
        """Treat seen evidence as neutral instead of requesting it forever."""

        selected = request.parameters.get("controller", {}).get("selected", {})
        resolved = selected.get("resolves") if isinstance(selected, dict) else None
        targets = resolved if isinstance(resolved, list) and resolved else [request.hypothesis_id]
        for hypothesis_id in targets:
            hypothesis = session.hypotheses.get(hypothesis_id)
            if (
                hypothesis is None
                or hypothesis.supporting_evidence
                or hypothesis.contradicting_evidence
            ):
                continue
            hypothesis.status = "uncertain"
            for evidence_id in evidence_ids:
                if evidence_id not in hypothesis.bearing_evidence:
                    hypothesis.bearing_evidence.append(evidence_id)
