"""LifecycleMixin for CognitiveRuntime."""

from __future__ import annotations

import copy
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from cortheon.cognitive_core.models import (
    CognitiveRuntimeError,
    EvidenceRequest,
    Investigation,
    InvestigationNotFound,
)
from cortheon.cognitive_core.profiles import EFFORT_PROFILES, STRICTNESS_PROFILES, TASK_KINDS
from cortheon.cognitive_core.receipts import _digest
from cortheon.cognitive_core.requirements import _extract_requirements
from cortheon.cognitive_core.runtime_completion import _store_reasoning_draft
from cortheon.cognitive_core.runtime_state import (
    RuntimeState,
    consume_evaluation_receipt,
    initial_metrics,
    record_evaluation_operator,
    store_evaluation_receipt,
)
from cortheon.cognitive_core.tasks import _infer_deliverable, _infer_task_kind
from cortheon.cognitive_core.text import _string_list, _text
from cortheon.cognitive_program import compile_program
from cortheon.cognitive_protocol import (
    CORTHEON_CERTIFICATION_SCOPE,
    CORTHEON_PROTOCOL_VERSION,
    CORTHEON_STORAGE_MODEL,
    evaluation_operator,
    normalize_evaluation_profile,
)


class LifecycleMixin(RuntimeState):
    """Lifecycle responsibilities of CognitiveRuntime."""

    def __init__(
        self,
        *,
        max_sessions: int = 32,
        ttl_seconds: float = 1_800.0,
        require_host_receipts: bool = False,
        clock: Any = time.monotonic,
    ) -> None:
        if not 1 <= max_sessions <= 1_024:
            raise ValueError("max_sessions must be between 1 and 1024")
        if not 30 <= ttl_seconds <= 86_400:
            raise ValueError("ttl_seconds must be between 30 and 86400")
        self.max_sessions = max_sessions
        self.ttl_seconds = float(ttl_seconds)
        self.require_host_receipts = bool(require_host_receipts)
        self._clock = clock
        self._sessions: OrderedDict[str, Investigation] = OrderedDict()
        self._evaluation_receipts: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = initial_metrics()

    @property
    def active_sessions(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._sessions)

    @property
    def metrics(self) -> dict[str, float | int | str]:
        """Return content-free process telemetry."""

        with self._lock:
            self._purge_expired()
            completed = int(self._metrics["sessions_completed"])
            total = float(self._metrics["completion_latency_ms_total"])
            return {
                "protocol_version": CORTHEON_PROTOCOL_VERSION,
                "storage": CORTHEON_STORAGE_MODEL,
                "active_sessions": len(self._sessions),
                **self._metrics,
                "completion_latency_ms_mean": (round(total / completed, 3) if completed else 0.0),
            }

    def start(
        self,
        goal: str,
        *,
        constraints: Iterable[str] = (),
        effort: str = "standard",
        task_kind: str = "auto",
        strictness: str = "standard",
        lease_seconds: float | None = None,
        evaluation_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start an investigation without reading or retaining a project file."""

        normalized_goal = _text(goal, "goal", maximum=8_000)
        normalized_constraints = _string_list(
            constraints,
            "constraints",
            maximum_items=12,
            maximum_chars=4_000,
        )
        if effort not in EFFORT_PROFILES:
            raise ValueError(f"effort must be one of: {', '.join(EFFORT_PROFILES)}")
        if task_kind not in TASK_KINDS:
            raise ValueError(f"task_kind must be one of: {', '.join(sorted(TASK_KINDS))}")
        if strictness not in STRICTNESS_PROFILES:
            raise ValueError(f"strictness must be one of: {', '.join(sorted(STRICTNESS_PROFILES))}")
        if lease_seconds is not None and not 10 <= lease_seconds <= 300:
            raise ValueError("lease_seconds must be between 10 and 300")
        normalized_evaluation_profile = normalize_evaluation_profile(evaluation_profile)

        with self._lock:
            self._purge_expired()
            if len(self._sessions) >= self.max_sessions:
                raise CognitiveRuntimeError(
                    "the in-memory session limit is full; finish or abandon an investigation"
                )
            now = self._clock()
            detected_kind = _infer_task_kind(normalized_goal)
            inferred_kind = detected_kind if task_kind == "auto" else task_kind
            deliverable = _infer_deliverable(normalized_goal, inferred_kind)
            profile = EFFORT_PROFILES[effort]
            requirements = _extract_requirements(
                normalized_goal,
                normalized_constraints,
                deliverable,
            )
            session = Investigation(
                session_id=f"vx_{secrets.token_urlsafe(18)}",
                goal=normalized_goal,
                constraints=normalized_constraints,
                requirements=requirements,
                task_kind=inferred_kind,
                deliverable=deliverable,
                profile=profile,
                program=compile_program(
                    goal=normalized_goal,
                    task_kind=inferred_kind,
                    deliverable=deliverable,
                    effort=effort,
                    requirements=(
                        (requirement.requirement_id, requirement.proof)
                        for requirement in requirements
                    ),
                    max_turns=profile.max_turns,
                    max_observations=profile.max_observations,
                    evaluation_profile=normalized_evaluation_profile,
                ),
                strictness=STRICTNESS_PROFILES[strictness],
                created_at=datetime.now(UTC).isoformat(),
                started_at=now,
                touched_at=now,
                expires_at=now + self.ttl_seconds,
                lease_seconds=float(lease_seconds) if lease_seconds is not None else None,
                lease_expires_at=(
                    now + float(lease_seconds) if lease_seconds is not None else None
                ),
                evaluation_profile=normalized_evaluation_profile,
            )
            self._sessions[session.session_id] = session
            self._metrics["sessions_started"] += 1
            if normalized_evaluation_profile is not None:
                store_evaluation_receipt(
                    self._evaluation_receipts,
                    normalized_evaluation_profile,
                    maximum=self.max_sessions * 2,
                )
            request = (
                self._initial_request(session)
                if evaluation_operator(session.evaluation_profile, "retrieval")
                else None
            )
            return self._payload(
                session,
                next_action=(
                    self._execute_action(request)
                    if request is not None
                    else {
                        "type": "await_candidate",
                        "instruction": (
                            "Use only model-owned host receipts, then submit the candidate "
                            "for verification. Cortheon will not originate evidence requests."
                        ),
                    }
                ),
                guidance=(
                    (
                        "Use the host's tools to satisfy this request from the live project. "
                        "Return only the smallest relevant excerpts or results through "
                        "cortheon_observe; do not send whole files."
                    )
                    if request is not None
                    else "Cortheon is verification-only and issued no evidence request."
                ),
            )

    def consume_evaluation_receipt(self, nonce: str) -> dict[str, Any]:
        """Consume one content-free receipt proving runtime profile application."""

        with self._lock:
            return consume_evaluation_receipt(self._evaluation_receipts, nonce)

    def _record_evaluation_operator(self, session: Investigation, operator: str) -> None:
        record_evaluation_operator(
            self._evaluation_receipts,
            session.evaluation_profile,
            operator,
        )

    def describe_sessions(self, *, limit: int = 3) -> dict[str, Any]:
        """Return bounded state needed to resume after host context loss."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise ValueError("limit must be an integer between 1 and 8")
        with self._lock:
            self._purge_expired()
            recent = list(self._sessions.values())[-limit:]
            summaries: list[dict[str, Any]] = []
            for index, session in enumerate(reversed(recent)):
                pending = self._pending_request(session)
                remembered_action = copy.deepcopy(session.last_next_action)
                remembered_request = (
                    remembered_action.get("request")
                    if isinstance(remembered_action, dict)
                    else None
                )
                if isinstance(remembered_request, dict):
                    remembered_request_id = remembered_request.get("request_id")
                    if pending is None or remembered_request_id != pending.request_id:
                        remembered_action = None
                summary: dict[str, Any] = {
                    "session_id": session.session_id,
                    "goal": session.goal[:500],
                    "phase": session.phase,
                    "task_kind": session.task_kind,
                    "deliverable": session.deliverable,
                    "program_id": session.program["program_id"],
                    "effort": session.profile.name,
                    "strictness": session.strictness.name,
                    "turns_remaining": max(0, session.profile.max_turns - session.turns),
                    "accepted_evidence_ids": list(session.observations),
                    "next_action": (
                        self._execute_action(pending)
                        if pending is not None
                        else remembered_action
                        if remembered_action is not None
                        else {
                            "type": "reason",
                            "instruction": (
                                "Continue this investigation: submit focused live "
                                "evidence with cortheon_observe or finish with "
                                "cortheon_complete."
                            ),
                            "submit_via": "cortheon_complete",
                        }
                    ),
                }
                if session.waivers:
                    summary["caveats"] = sorted(session.waivers.values())
                if index == 0:
                    summary["context"] = self._context_pack(session)
                summaries.append(summary)
            return {
                "protocol_version": CORTHEON_PROTOCOL_VERSION,
                "active_sessions": len(self._sessions),
                "storage": CORTHEON_STORAGE_MODEL,
                "sessions": summaries,
            }

    def note_failed_submission(
        self,
        session_id: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Count a failed evidence submission against its request budget."""

        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            request: EvidenceRequest | None = None
            if request_id is not None:
                candidate = session.requests.get(request_id)
                if candidate is not None and candidate.status == "pending":
                    request = candidate
            if request is None:
                request = self._pending_request(session)
            waived = False
            if request is not None:
                waived = self._register_request_attempt(session, request)
            self._metrics["failed_submissions"] += 1
            self._commit(session)
            response: dict[str, Any] = {
                "ok": True,
                "request_id": request.request_id if request is not None else None,
                "attempts": request.attempts if request is not None else 0,
                "waived": waived,
            }
            if session.waivers:
                response["caveats"] = sorted(session.waivers.values())
            return response

    def heartbeat(self, session_id: str) -> dict[str, Any]:
        """Renew a native-adapter lease without accepting task content."""

        with self._lock:
            session = self._session(session_id)
            return {
                "ok": True,
                "protocol_version": CORTHEON_PROTOCOL_VERSION,
                "session_id": session.session_id,
                "lease_seconds": session.lease_seconds,
                "storage": CORTHEON_STORAGE_MODEL,
            }

    def step(
        self,
        session_id: str,
        *,
        hypotheses: Iterable[dict[str, Any]] = (),
        hypothesis_updates: Iterable[dict[str, Any]] = (),
        open_questions: Iterable[str] = (),
        draft: str | None = None,
    ) -> dict[str, Any]:
        """Advance the task graph after the host model has reasoned."""

        with self._lock:
            session = copy.deepcopy(self._session(session_id))
            if session.evaluation_profile is not None and (
                hypotheses or hypothesis_updates or open_questions or draft is not None
            ):
                framing = evaluation_operator(
                    session.evaluation_profile,
                    "hypothesis_framing",
                )
                revision = evaluation_operator(
                    session.evaluation_profile,
                    "contradiction_revision",
                )
                if (hypotheses and not framing) or (
                    (hypothesis_updates or open_questions or draft is not None) and not revision
                ):
                    raise CognitiveRuntimeError(
                        "reasoning mutation is disabled by the evaluation profile"
                    )
            self._take_turn(session)
            self._add_hypotheses(session, hypotheses)
            self._update_hypotheses(session, hypothesis_updates)
            if open_questions:
                session.open_questions = _string_list(
                    open_questions,
                    "open_questions",
                    maximum_items=12,
                    maximum_chars=6_000,
                )
            if draft is not None:
                reasoning_binding = _store_reasoning_draft(session, draft)
            else:
                reasoning_binding = None
            response = self._recommend(session)
            if reasoning_binding is not None:
                response["reasoning_binding"] = reasoning_binding
            self._commit(session)
            return response

    def finish(
        self,
        session_id: str,
        *,
        mode: str = "complete",
        answer: str | None = None,
    ) -> dict[str, Any]:
        """Return a verified result or abandon the task, then erase all task state."""

        if mode not in {"complete", "abandon"}:
            raise ValueError("mode must be complete or abandon")
        with self._lock:
            session = self._session(session_id)
            if mode == "complete":
                if answer is None:
                    raise ValueError("answer is required when mode is complete")
                normalized_answer = _text(
                    answer,
                    "answer",
                    maximum=session.profile.max_context_chars,
                )
                if session.phase != "ready" or session.verified_answer_digest is None:
                    raise CognitiveRuntimeError(
                        "the investigation is not ready; cortheon_verify must pass first"
                    )
                if _digest(normalized_answer) != session.verified_answer_digest:
                    raise CognitiveRuntimeError(
                        "the answer changed after verification; verify the new answer first"
                    )
                result = {
                    "status": "complete",
                    "answer": normalized_answer,
                    "session_id": session.session_id,
                    "protocol_version": CORTHEON_PROTOCOL_VERSION,
                    "certification_scope": CORTHEON_CERTIFICATION_SCOPE,
                    "scorecard": self._scorecard(session),
                    "discarded": True,
                    "retained_project_data": False,
                }
                if session.waivers:
                    result["caveats"] = sorted(session.waivers.values())
                self._record_completion(session)
            else:
                result = {
                    "status": "abandoned",
                    "answer": None,
                    "session_id": session.session_id,
                    "protocol_version": CORTHEON_PROTOCOL_VERSION,
                    "certification_scope": CORTHEON_CERTIFICATION_SCOPE,
                    "scorecard": self._scorecard(session),
                    "discarded": True,
                    "retained_project_data": False,
                }
                self._metrics["sessions_abandoned"] += 1
            del self._sessions[session.session_id]
            return result

    def close_evidence(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            if session.deliverable != "document_synthesis":
                raise CognitiveRuntimeError("document synthesis required")
            if not session.observations:
                raise CognitiveRuntimeError("accepted evidence required")
            result = {
                "status": "evidence_closed",
                "answer_certified": False,
                "discarded": True,
                "retained_project_data": False,
            }
            self._metrics["sessions_evidence_closed"] += 1
            del self._sessions[session.session_id]
            return result

    def _session(self, session_id: str) -> Investigation:
        normalized = _text(session_id, "session_id", maximum=128)
        self._purge_expired()
        session = self._sessions.get(normalized)
        if session is None:
            raise InvestigationNotFound(
                "investigation not found; it may have expired or already been discarded"
            )
        now = self._clock()
        if now >= session.expires_at:
            del self._sessions[normalized]
            raise InvestigationNotFound("investigation expired and its state was discarded")
        session.touched_at = now
        session.expires_at = now + self.ttl_seconds
        if session.lease_seconds is not None:
            session.lease_expires_at = now + session.lease_seconds
        self._sessions.move_to_end(normalized)
        return session

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now >= session.expires_at
            or (session.lease_expires_at is not None and now >= session.lease_expires_at)
        ]
        for session_id in expired:
            del self._sessions[session_id]
        self._metrics["sessions_expired"] += len(expired)

    def _record_completion(self, session: Investigation) -> None:
        latency_ms = max(0.0, (self._clock() - session.started_at) * 1_000.0)
        self._metrics["sessions_completed"] += 1
        self._metrics["completion_latency_ms_total"] += latency_ms
        self._metrics["completion_latency_ms_max"] = max(
            float(self._metrics["completion_latency_ms_max"]),
            latency_ms,
        )

    def _commit(self, session: Investigation) -> None:
        self._sessions[session.session_id] = session
        self._sessions.move_to_end(session.session_id)

    def _take_turn(self, session: Investigation) -> None:
        if session.turns >= session.profile.max_turns:
            session.phase = "inconclusive"
            raise CognitiveRuntimeError(
                "the reasoning-turn budget is exhausted; abandon the investigation "
                "or finish a previously verified answer"
            )
        session.turns += 1
