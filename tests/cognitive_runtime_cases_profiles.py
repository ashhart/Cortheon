from __future__ import annotations

import unittest
from datetime import UTC, datetime

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
    CognitiveRuntimeError,
)


class StrictnessProfileTests(unittest.TestCase):
    @staticmethod
    def _web_observation(url: str, purpose: str, content: str) -> dict:
        return {
            "kind": "web",
            "content": content,
            "url": url,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "purpose": purpose,
        }

    def test_assist_research_pre_arms_single_origin_corroboration(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Research current guidance on widget safety from published sources.",
            task_kind="research",
            effort="quick",
            strictness="assist",
        )
        self.assertEqual(started["session"]["strictness"], "assist")

        observed = runtime.observe(
            started["session"]["session_id"],
            [
                self._web_observation(
                    "https://example.com/guidance",
                    "contradiction_check",
                    "The vendor guidance reports no known conflict for widget safety.",
                )
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        self.assertEqual(
            observed["next_action"]["request"]["parameters"]["purpose"],
            "primary_fetch",
        )
        self.assertTrue(any("Assist strictness" in item for item in observed["caveats"]))

    def test_assist_waives_after_a_single_failed_receipt(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Explain why the rollout stalled this afternoon.",
            strictness="assist",
        )

        waived = runtime.observe(
            started["session"]["session_id"],
            [{"kind": "analysis", "content": "The canary gate is still waiting."}],
            request_id=started["next_action"]["request"]["request_id"],
        )

        self.assertEqual(waived["accepted_evidence_ids"], ["ev1"])
        self.assertTrue(waived.get("caveats"))

    def test_strict_requests_a_second_corroboration_round(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Research current guidance on widget safety from published sources.",
            task_kind="research",
            effort="quick",
            strictness="strict",
        )
        session_id = started["session"]["session_id"]
        first = runtime.observe(
            session_id,
            [
                self._web_observation(
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
                self._web_observation(
                    "https://example.com/guidance-archive",
                    "corroboration",
                    "The same vendor repeats the guidance on widget safety.",
                )
            ],
            request_id=corroboration["request_id"],
        )

        retry = second["next_action"]["request"]
        self.assertEqual(retry["parameters"]["purpose"], "corroboration")
        self.assertNotIn("caveats", second)

    def test_invalid_strictness_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictness"):
            CognitiveRuntime().start("Inspect code", strictness="ultra")

    def test_receipt_rejections_carry_a_copyable_example(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Does src/example.py import pathlib?",
            effort="quick",
        )

        with self.assertRaisesRegex(
            CognitiveRuntimeError,
            r"Correct example host_receipt: \{\"args\"",
        ):
            runtime.observe(
                started["session"]["session_id"],
                [
                    {
                        "kind": "code",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"bash",'
                            '"outcome":"result","args":{"command":"rg pathlib"}}\n'
                            "src/example.py:3:import pathlib"
                        ),
                    }
                ],
                request_id=started["next_action"]["request"]["request_id"],
            )


class ToolCallBudgetTests(unittest.TestCase):
    def test_requests_carry_an_effort_scaled_tool_call_budget(self) -> None:
        runtime = CognitiveRuntime()

        quick = runtime.start("Fix the parser bug and add tests", effort="quick")
        request = quick["next_action"]["request"]
        self.assertEqual(request["parameters"]["tool_call_budget"], 3)
        self.assertIn("at most 3 host tool calls", quick["next_action"]["instruction"])

        deep = runtime.start("Fix the tokenizer bug and add tests", effort="deep")
        self.assertEqual(
            deep["next_action"]["request"]["parameters"]["tool_call_budget"],
            8,
        )
        self.assertIn("at most 8 host tool calls", deep["next_action"]["instruction"])


class ResearchReframeTests(unittest.TestCase):
    def test_research_without_web_evidence_reframes_to_general_answer(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Research current guidance on widget safety from published sources.",
            task_kind="research",
            effort="quick",
            strictness="assist",
        )
        session_id = started["session"]["session_id"]

        observed = runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": "The internal handbook already documents the widget limits.",
                    "source": "internal-handbook.md",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        self.assertEqual(observed["session"]["deliverable"], "answer")
        self.assertTrue(any("reframed" in item for item in observed["caveats"]))

        completed = runtime.complete(
            session_id,
            answer="The internal handbook sets the widget safety limits.",
            claims=[
                {
                    "claim": "The handbook sets the widget safety limits.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "The handbook sets the widget safety limits.",
                    "falsification_test": "Check the handbook section on limits.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )

        self.assertEqual(completed["status"], "complete")
        self.assertTrue(any("reframed" in item for item in completed["caveats"]))
        self.assertEqual(runtime.metrics["sessions_reframed"], 1)

    def test_research_with_web_evidence_keeps_the_research_contract(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Research current guidance on widget safety from published sources.",
            task_kind="research",
            effort="quick",
        )

        observed = runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "web",
                    "content": "The vendor guidance documents the widget limits.",
                    "url": "https://example.com/guidance",
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "purpose": "contradiction_check",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        self.assertEqual(observed["session"]["deliverable"], "research_answer")
