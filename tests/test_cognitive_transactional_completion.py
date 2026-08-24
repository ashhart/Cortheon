"""Behavioral tests for transactional completion over pending hypothesis requests.

These drive the real CognitiveRuntime through the production causal-synthesis
state: a read_many request whose observation spawns provisional
substrate-abduction hypotheses and a pending counterexample request (req2)
tied to one of them, followed by cortheon_complete. They pin the round-14
correctness contracts: narrow supersession, monotonic hypothesis ids, honest
uncertain rivals, and missing-evidence paths that are never excused.
"""

from __future__ import annotations

import unittest
from typing import Any

from cortheon.cognitive_core.runtime import CognitiveRuntime

GOAL = (
    "Read facts/a.txt and facts/b.txt. Diagnose the causal explanation "
    "for the collision, disprove the rival, and give a discriminating test."
)
READ_A = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/a.txt"}}\n'
    "Northstar path A uses collision key amber."
)
READ_B = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/b.txt"}}\n'
    "Path B reuses key amber. Collision persists when compaction is disabled."
)
READ_B_NEUTRAL = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/b.txt"}}\n'
    "Path B reuses key amber. Compaction is scheduled nightly."
)
ANSWER = (
    "Cause: The collision occurs because both paths reuse the Northstar key "
    "amber.\nRival: Instead, cache compaction is the competing alternative; "
    "the accepted evidence does not settle it, so it remains uncertain.\n"
    "Test: Assign distinct keys; this distinguishing test would falsify the "
    "wrong mechanism: Cause predicts the collision disappears whereas Rival "
    "predicts the collision remains."
)
CLAIM = {
    "claim": (
        "The causal explanation for the collision is that both paths reuse "
        "the Northstar key amber; the rival cache-compaction alternative "
        "remains uncertain pending the falsification test."
    ),
    "evidence_ids": ["ev1", "ev2"],
}
CAUSE = {
    "statement": "The collision occurs because both paths reuse the Northstar key amber.",
    "falsification_test": "Assign distinct keys.",
    "status": "supported",
    "evidence_ids": ["ev1", "ev2"],
}
RIVAL_REFUTED = {
    "statement": "Instead, cache compaction is the competing alternative.",
    "falsification_test": "Assign distinct keys.",
    "status": "refuted",
    "evidence_ids": ["ev2"],
}
RIVAL_UNCERTAIN = {
    "statement": "Instead, cache compaction is the competing alternative.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}


def _causal_runtime(second_fact: str = READ_B) -> tuple[CognitiveRuntime, str, dict[str, Any]]:
    runtime = CognitiveRuntime()
    started = runtime.start(GOAL)
    session_id = started["session"]["session_id"]
    observed = runtime.observe(
        session_id,
        [
            {
                "kind": "documentation",
                "content": READ_A,
                "status": "verified",
                "source": "pi:read:facts/a.txt",
            },
            {
                "kind": "documentation",
                "content": second_fact,
                "status": "verified",
                "source": "pi:read:facts/b.txt",
            },
        ],
        request_id="req1",
    )
    return runtime, session_id, observed


def _complete(runtime: CognitiveRuntime, session_id: str, rival: dict[str, Any]):
    return runtime.complete(
        session_id,
        answer=ANSWER,
        claims=[CLAIM],
        hypotheses=[CAUSE, rival],
        completion_evidence_ids=["ev1", "ev2"],
    )


class TestTransactionalSupersession(unittest.TestCase):
    def test_provisional_req2_is_superseded_and_real_completion_succeeds(self):
        runtime, session_id, observed = _causal_runtime()
        request = observed["next_action"]["request"]
        assert request["request_id"] == "req2"
        assert request["hypothesis_id"] == "h1"
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result["status"] == "complete", result.get("verification", {}).get("gaps")
        # A successful completion irreversibly discards the in-memory session.
        assert runtime.active_sessions == 0

    def test_superseded_request_is_auditable_and_detached(self):
        runtime, session_id, _observed = _causal_runtime()
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result["status"] == "complete"
        assert runtime.metrics["requests_superseded"] == 1
        # The per-session scorecard records the supersession event itself,
        # distinct from satisfied and waived requests.
        scorecard = result["scorecard"]
        assert scorecard["superseded_requests"] == ["req2"]
        assert scorecard["satisfied_requests"] == ["req1"]
        assert scorecard["waived_requests"] == []

    def test_pending_ordinary_request_withholds(self):
        runtime, session_id, _observed = _causal_runtime()
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._create_request(
                session,
                capability="search",
                query="Find supporting context.",
                reason="Ordinary unresolved evidence need.",
                success_condition="A live source.",
            )
            runtime._commit(session)
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result.get("status") != "complete"
        pending = next(
            item for item in result["verification"]["checks"] if item["name"] == "pending_requests"
        )
        assert not pending["passed"]
        assert "req3" in pending["reason"]

    def test_pending_host_model_hypothesis_request_withholds(self):
        runtime, session_id, _observed = _causal_runtime()
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._add_hypotheses(
                session,
                [{"statement": "A host-model rival exists.", "falsification_test": "Check it."}],
            )
            host_id = next(reversed(session.hypotheses))
            runtime._create_request(
                session,
                capability="search",
                query="Test the host-model rival.",
                reason="Falsification evidence need.",
                success_condition="A live counterexample or confirmation.",
                hypothesis_id=host_id,
            )
            runtime._commit(session)
        result = runtime.complete(
            session_id,
            answer=ANSWER,
            claims=[CLAIM],
            hypotheses=[
                CAUSE,
                RIVAL_REFUTED,
                {
                    "statement": "A host-model rival exists.",
                    "falsification_test": "Check it.",
                    "status": "uncertain",
                    "evidence_ids": ["ev2"],
                },
            ],
            completion_evidence_ids=["ev1", "ev2"],
        )
        assert result.get("status") != "complete"
        with runtime._lock:
            session = runtime._session(session_id)
            blocking = [
                request
                for request in session.requests.values()
                if request.status == "pending" and request.hypothesis_id == host_id
            ]
        assert blocking, "the host_model hypothesis request must still block"

    def test_substrate_hypothesis_with_acquired_evidence_blocks(self):
        runtime, session_id, observed = _causal_runtime()
        request = observed["next_action"]["request"]
        assert request["hypothesis_id"] == "h1"
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._update_hypotheses(
                session,
                [{"hypothesis_id": "h1", "status": "supported", "evidence_ids": ["ev1"]}],
            )
            runtime._commit(session)
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result.get("status") != "complete"

    def test_withheld_completion_never_aliases_a_replacement_hypothesis(self):
        runtime, session_id, observed = _causal_runtime()
        assert observed["next_action"]["request"]["hypothesis_id"] == "h1"
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._create_request(
                session,
                capability="search",
                query="Hold this request open.",
                reason="Ordinary unresolved evidence need.",
                success_condition="A live source.",
            )
            runtime._commit(session)
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result.get("status") != "complete"
        with runtime._lock:
            session = runtime._session(session_id)
            requests = list(session.requests.values())
            hypothesis_ids = list(session.hypotheses)
        superseded = [item for item in requests if item.status == "superseded"]
        assert superseded
        for item in superseded:
            assert item.hypothesis_id is None
            assert item.parameters["superseded_hypothesis_id"] == "h1"
            # A superseded request never points at any live hypothesis id.
            assert item.parameters["superseded_hypothesis_id"] not in hypothesis_ids
            assert item.hypothesis_id not in hypothesis_ids
        # Monotonic ids: the replacement hypotheses never reuse h1/h2.
        assert "h1" not in hypothesis_ids and "h2" not in hypothesis_ids

    def test_hypothesis_ids_are_monotonic_across_clearing(self):
        runtime, session_id, _observed = _causal_runtime()
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._add_hypotheses(
                session,
                [
                    {
                        "statement": "A fresh replacement candidate.",
                        "falsification_test": "Check it.",
                    }
                ],
            )
            runtime._commit(session)
        with runtime._lock:
            session = runtime._session(session_id)
            ids = list(session.hypotheses)
        assert ids == ["h3"], ids

    def test_superseded_requests_do_not_count_as_satisfied(self):
        runtime, session_id, observed = _causal_runtime()
        assert observed["next_action"]["request"]["request_id"] == "req2"
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result["status"] == "complete"
        # The only reason no pending request blocked completion is that req2
        # was superseded (retired without satisfaction), never completed.
        assert runtime.metrics["requests_superseded"] == 1


class TestHonestUncertainty(unittest.TestCase):
    def test_supported_cause_plus_uncertain_rival_completes(self):
        runtime, session_id, _observed = _causal_runtime(READ_B_NEUTRAL)
        result = _complete(runtime, session_id, RIVAL_UNCERTAIN)
        assert result["status"] == "complete", result.get("verification", {}).get("gaps")
        # A rival held only by neutral bearing evidence is still tested.
        assert result["scorecard"]["tested_hypotheses"] == 2

    def test_overconfident_answer_over_neutral_evidence_is_withheld(self):
        # The rival is honestly uncertain over neutral evidence, but the
        # answer asserts it as disproven: hidden hypothesis metadata must
        # never certify a settled-sounding answer.
        runtime, session_id, _observed = _causal_runtime(READ_B_NEUTRAL)
        overconfident = ANSWER.replace(
            "the accepted evidence does not settle it, so it remains uncertain",
            "the accepted evidence disproves the rival compaction mechanism",
        )
        result = runtime.complete(
            session_id,
            answer=overconfident,
            claims=[CLAIM],
            hypotheses=[CAUSE, RIVAL_UNCERTAIN],
            completion_evidence_ids=["ev1", "ev2"],
        )
        assert result.get("status") != "complete"
        visibility = next(
            item
            for item in result["verification"]["checks"]
            if item["name"] == "uncertainty_visibility"
        )
        assert not visibility["passed"]

    def test_uncertain_rival_stores_bearing_evidence_neutrally(self):
        runtime, session_id, _observed = _causal_runtime(READ_B_NEUTRAL)
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._update_hypotheses(
                session,
                [{"hypothesis_id": "h2", "status": "uncertain", "evidence_ids": ["ev2"]}],
            )
            runtime._commit(session)
        with runtime._lock:
            session = runtime._session(session_id)
            rival = session.hypotheses["h2"]
        assert rival.status == "uncertain"
        assert rival.bearing_evidence == ["ev2"]
        assert rival.supporting_evidence == []
        assert rival.contradicting_evidence == []

    def test_uncertain_without_evidence_is_accepted_but_untested(self):
        """An uncertain hypothesis with zero evidence ids is a legal, honest
        empty binding: the update succeeds (no protocol error, no HTTP 400),
        and normal verification withholds the completion as untested with an
        actionable gap instead of certifying."""
        runtime, session_id, _observed = _causal_runtime(READ_B_NEUTRAL)
        with runtime._lock:
            session = runtime._session(session_id)
            runtime._update_hypotheses(
                session,
                [{"hypothesis_id": "h2", "status": "uncertain", "evidence_ids": []}],
            )
            runtime._commit(session)
        with runtime._lock:
            session = runtime._session(session_id)
            rival = session.hypotheses["h2"]
        assert rival.status == "uncertain"
        assert rival.bearing_evidence == []


class TestMissingEvidence(unittest.TestCase):
    def test_third_required_document_is_never_excused(self):
        runtime = CognitiveRuntime()
        goal = GOAL.replace(
            "Read facts/a.txt and facts/b.txt.",
            "Read facts/a.txt, facts/b.txt, and facts/c.txt.",
        )
        started = runtime.start(goal)
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": READ_A,
                    "status": "verified",
                    "source": "pi:read:facts/a.txt",
                },
                {
                    "kind": "documentation",
                    "content": READ_B,
                    "status": "verified",
                    "source": "pi:read:facts/b.txt",
                },
            ],
            request_id="req1",
        )
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result.get("status") != "complete"
        alignment = next(
            item
            for item in result["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        assert not alignment["passed"]
        assert "facts/c.txt" in alignment["reason"]

    def test_failed_third_read_is_still_missing_evidence(self):
        runtime = CognitiveRuntime()
        goal = GOAL.replace(
            "Read facts/a.txt and facts/b.txt.",
            "Read facts/a.txt, facts/b.txt, and facts/c.txt.",
        )
        started = runtime.start(goal)
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": READ_A,
                    "status": "verified",
                    "source": "pi:read:facts/a.txt",
                },
                {
                    "kind": "documentation",
                    "content": READ_B,
                    "status": "verified",
                    "source": "pi:read:facts/b.txt",
                },
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"error",'
                        '"args":{"filePath":"facts/c.txt"}}\n'
                        "unreadable"
                    ),
                    "status": "failed",
                    "source": "pi:read:facts/c.txt",
                },
            ],
            request_id="req1",
        )
        result = _complete(runtime, session_id, RIVAL_REFUTED)
        assert result.get("status") != "complete"
        alignment = next(
            item
            for item in result["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        assert not alignment["passed"]
        assert "facts/c.txt" in alignment["reason"]
