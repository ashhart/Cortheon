from __future__ import annotations

import unittest
from datetime import UTC, datetime

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
    CognitiveRuntimeError,
)


class WaiverAndRetractionTests(unittest.TestCase):
    def test_single_origin_research_downgrades_with_caveats(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Research current guidance on widget safety from published sources.",
            task_kind="research",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        retrieved_at = datetime.now(UTC).isoformat()

        def web_observation(url: str, purpose: str, content: str) -> dict:
            return {
                "kind": "web",
                "content": content,
                "url": url,
                "retrieved_at": retrieved_at,
                "purpose": purpose,
            }

        first = runtime.observe(
            session_id,
            [
                web_observation(
                    "https://example.com/guidance",
                    "contradiction_check",
                    "The vendor guidance reports no known conflict for widget safety.",
                )
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )
        corroboration = first["next_action"]["request"]
        self.assertEqual(corroboration["parameters"]["purpose"], "corroboration")

        second = runtime.observe(
            session_id,
            [
                web_observation(
                    "https://example.com/guidance-archive",
                    "corroboration",
                    "The same vendor repeats the guidance on widget safety.",
                )
            ],
            request_id=corroboration["request_id"],
        )
        self.assertIn("single URL origin", " ".join(second.get("caveats", [])))
        fetch_request = second["next_action"]["request"]
        self.assertEqual(fetch_request["parameters"]["purpose"], "primary_fetch")

        third = runtime.observe(
            session_id,
            [
                web_observation(
                    "https://example.com/guidance",
                    "primary_fetch",
                    "The fetched primary guidance page confirms widget safety limits.",
                )
            ],
            request_id=fetch_request["request_id"],
        )
        self.assertNotEqual(third["next_action"].get("type"), "harness_tool")

        completed = runtime.complete(
            session_id,
            answer=(
                "Current vendor guidance at https://example.com/guidance sets the "
                "widget safety limits; no independent corroboration was available."
            ),
            claims=[
                {
                    "claim": "The vendor guidance sets widget safety limits.",
                    "evidence_ids": ["ev1", "ev2", "ev3"],
                }
            ],
            hypotheses=[
                {
                    "statement": "The vendor guidance sets the widget safety limits.",
                    "falsification_test": "Find a source that contradicts the guidance.",
                    "status": "supported",
                    "evidence_ids": ["ev1", "ev3"],
                }
            ],
            completion_evidence_ids=["ev1", "ev2", "ev3"],
        )
        self.assertEqual(completed["status"], "complete")
        self.assertTrue(any("single URL origin" in item for item in completed["caveats"]))
        self.assertIn("corroboration", completed["scorecard"]["waived_requirements"])

    def test_failed_receipt_attempts_waive_the_request(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start("Explain why the rollout stalled this afternoon.")
        session_id = started["session"]["session_id"]
        request_id = started["next_action"]["request"]["request_id"]
        self.assertEqual(started["next_action"]["request"]["capability"], "inspect")
        observation = {
            "kind": "analysis",
            "content": "The canary gate is still waiting.",
        }

        with self.assertRaises(CognitiveRuntimeError):
            runtime.observe(session_id, [observation], request_id=request_id)

        waived = runtime.observe(session_id, [observation], request_id=request_id)

        self.assertEqual(waived["accepted_evidence_ids"], ["ev1"])
        self.assertIn("validation_error", waived)
        self.assertTrue(waived.get("caveats"))
        self.assertNotEqual(waived["next_action"].get("type"), "harness_tool")
        self.assertEqual(runtime.metrics["requests_waived"], 1)

    def test_note_failed_submission_feeds_the_waiver_ladder(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start("Explain why the rollout stalled this afternoon.")
        session_id = started["session"]["session_id"]

        first = runtime.note_failed_submission(session_id)
        self.assertEqual(first["attempts"], 1)
        self.assertFalse(first["waived"])

        second = runtime.note_failed_submission(session_id)
        self.assertTrue(second["waived"])
        self.assertTrue(second.get("caveats"))

        described = runtime.describe_sessions()
        self.assertEqual(described["sessions"][0]["session_id"], session_id)
        self.assertNotEqual(
            described["sessions"][0]["next_action"]["type"],
            "harness_tool",
        )
        self.assertIn("context", described["sessions"][0])
        self.assertEqual(runtime.metrics["failed_submissions"], 2)

    def test_retract_unlinks_evidence_and_frees_resubmission(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Explain why the rollout stalled this afternoon.")
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [{"kind": "analysis", "content": "The canary gate is still waiting."}],
            request_id=started["next_action"]["request"]["request_id"],
        )
        runtime.step(
            session_id,
            hypotheses=[
                {
                    "statement": "The canary gate blocks the rollout.",
                    "falsification_test": "Check the gate status.",
                }
            ],
        )
        linked = runtime.observe(
            session_id,
            [
                {
                    "kind": "analysis",
                    "content": "A stale dashboard suggested the gate had cleared.",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
        )
        self.assertEqual(linked["accepted_evidence_ids"], ["ev2"])

        retracted = runtime.retract(
            session_id,
            ["ev2"],
            reason="The dashboard was from the wrong environment.",
        )

        self.assertEqual(retracted["retracted_evidence_ids"], ["ev2"])
        hypothesis = retracted["context"]["hypotheses"][0]
        self.assertEqual(hypothesis["status"], "open")
        self.assertEqual(hypothesis["supporting_evidence"], [])

        resubmitted = runtime.observe(
            session_id,
            [
                {
                    "kind": "analysis",
                    "content": "A stale dashboard suggested the gate had cleared.",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
        )
        self.assertEqual(resubmitted["accepted_evidence_ids"], ["ev3"])
        self.assertEqual(resubmitted["duplicate_observations"], 0)
        self.assertEqual(runtime.metrics["evidence_retracted"], 1)
