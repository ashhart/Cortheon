from __future__ import annotations

import unittest

from cognitive_adversarial_cases_common import hypothesis

from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveAtomicityTests(unittest.TestCase):
    def test_invalid_step_is_atomic_and_does_not_burn_a_turn(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Inspect behavior", effort="quick")
        session_id = started["session"]["session_id"]
        result = runtime.observe(
            session_id,
            [{"kind": "analysis", "content": "live observation"}],
            request_id="req1",
        )

        with self.assertRaises(ValueError):
            runtime.step(
                session_id,
                hypotheses=[
                    hypothesis("first"),
                    {"statement": "invalid", "falsification_test": ""},
                ],
            )

        result = runtime.step(session_id, hypotheses=[hypothesis("first")])
        self.assertEqual(result["session"]["turns_used"], 1)
        self.assertEqual(
            [item["hypothesis_id"] for item in result["context"]["hypotheses"]],
            ["h1"],
        )

    def test_invalid_observation_batch_is_atomic(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Inspect behavior")
        session_id = started["session"]["session_id"]

        with self.assertRaises(ValueError):
            runtime.observe(
                session_id,
                [
                    {"kind": "analysis", "content": "valid"},
                    {"kind": "not-a-kind", "content": "invalid"},
                ],
                request_id="req1",
            )

        result = runtime.observe(
            session_id,
            [{"kind": "analysis", "content": "valid"}],
            request_id="req1",
        )
        self.assertEqual(result["accepted_evidence_ids"], ["ev1"])
        self.assertEqual(result["session"]["observations_used"], 1)

    def test_invalid_challenge_and_verify_are_atomic(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Inspect behavior", effort="quick")
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [{"kind": "analysis", "content": "live evidence", "supports": []}],
            request_id="req1",
        )
        runtime.step(session_id, hypotheses=[hypothesis("the behavior is intentional")])
        runtime.observe(
            session_id,
            [
                {
                    "kind": "analysis",
                    "content": "the behavior is explicitly specified",
                    "supports": ["h1"],
                }
            ],
            request_id="req2",
        )

        with self.assertRaises(ValueError):
            runtime.challenge(
                session_id,
                draft="draft",
                claims=[{"claim": "bad", "evidence_ids": "ev2"}],
            )
        challenged = runtime.challenge(
            session_id,
            draft="draft",
            claims=[{"claim": "supported", "evidence_ids": ["ev2"]}],
        )
        self.assertEqual(challenged["session"]["turns_used"], 2)

        with self.assertRaises(ValueError):
            runtime.verify(
                session_id,
                answer="answer",
                claims=[{"claim": "", "evidence_ids": ["ev2"]}],
            )
        verified = runtime.verify(
            session_id,
            answer="answer",
            claims=[{"claim": "supported", "evidence_ids": ["ev2"]}],
            completion_evidence_ids=["ev2"],
        )
        self.assertEqual(verified["session"]["turns_used"], 3)

    def test_one_observation_cannot_both_support_and_contradict_hypothesis(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Inspect behavior", effort="quick")
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [{"kind": "analysis", "content": "initial"}],
            request_id="req1",
        )
        runtime.step(session_id, hypotheses=[hypothesis("one explanation")])

        with self.assertRaisesRegex(ValueError, "both support and contradict"):
            runtime.observe(
                session_id,
                [
                    {
                        "kind": "analysis",
                        "content": "ambiguous",
                        "supports": ["h1"],
                        "contradicts": ["h1"],
                    }
                ],
                request_id="req2",
            )
