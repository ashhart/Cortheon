from __future__ import annotations

import unittest

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
)


class ConciseChangeBudgetTests(unittest.TestCase):
    GOAL = "Apply a one-line fix to parser.py so the parser tests pass. Do not change the tests."

    @staticmethod
    def _session_with_change(goal: str, diff_content: str) -> tuple[CognitiveRuntime, str]:
        runtime = CognitiveRuntime()
        started = runtime.start(goal, effort="quick")
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "def parse(value): return value",
                    "source": "parser.py",
                }
            ],
            request_id="req1",
        )
        runtime.step(
            session_id,
            hypotheses=[
                {
                    "statement": "The parser mishandles empty input.",
                    "falsification_test": "Inspect the empty-input branch.",
                }
            ],
        )
        runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": diff_content,
                    "source": "git diff",
                    "supports": ["h1"],
                }
            ],
            request_id="req2",
        )
        runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "pytest: 12 passed",
                    "source": "pytest",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id="req3",
        )
        return runtime, session_id

    def _complete(self, runtime: CognitiveRuntime, session_id: str) -> dict:
        return runtime.complete(
            session_id,
            answer="Fixed parser.py with the required change.",
            claims=[
                {
                    "claim": "parser.py now parses empty input.",
                    "evidence_ids": ["ev2", "ev3"],
                }
            ],
            hypotheses=[
                {
                    "statement": "The parser mishandles empty input.",
                    "falsification_test": "Inspect the empty-input branch.",
                    "status": "supported",
                    "evidence_ids": ["ev2", "ev3"],
                }
            ],
            completion_evidence_ids=["ev2", "ev3"],
        )

    def test_oversized_patch_fails_the_concise_budget(self) -> None:
        big_diff = (
            "diff --git a/parser.py b/parser.py\n"
            "--- a/parser.py\n"
            "+++ b/parser.py\n"
            "@@\n" + "".join(f"-old line {index}\n+new line {index}\n" for index in range(6))
        )
        runtime, session_id = self._session_with_change(self.GOAL, big_diff)

        completed = self._complete(runtime, session_id)

        self.assertEqual(completed["verification"]["verdict"], "needs_evidence")
        self.assertTrue(
            any("concise-change budget" in gap for gap in completed["verification"]["gaps"])
        )
        self.assertEqual(completed["next_action"]["type"], "reason")
        self.assertIn("concise-change", completed["next_action"]["instruction"])

    def test_small_patch_passes_the_concise_budget(self) -> None:
        small_diff = (
            "diff --git a/parser.py b/parser.py\n"
            "--- a/parser.py\n"
            "+++ b/parser.py\n"
            "@@\n"
            "-    raise ParseError\n"
            "+    return empty_result\n"
        )
        runtime, session_id = self._session_with_change(self.GOAL, small_diff)

        completed = self._complete(runtime, session_id)

        self.assertEqual(completed["status"], "complete")
