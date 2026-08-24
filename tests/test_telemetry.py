import json
import tempfile
import unittest
from pathlib import Path

from cortheon.telemetry import (
    ProxyMetrics,
    agent_completion_outcome,
    enforcement_outcome,
    labeled_error_kind,
    patch_outcome,
)


class OutcomeContractTests(unittest.TestCase):
    def test_structural_allow_is_not_a_verified_completion(self) -> None:
        outcome = enforcement_outcome({"status": "clean_first_try", "verdict": "allow"})
        self.assertEqual(outcome["status"], "allowed")
        self.assertEqual(outcome["assurance"], "structural")
        self.assertFalse(outcome["verified_completion"])

    def test_behavioral_pass_is_verified_completion(self) -> None:
        outcome = enforcement_outcome(
            {
                "status": "clean_first_try",
                "verdict": "allow",
                "execution": {"verdict": "passed"},
            }
        )
        self.assertEqual(outcome["status"], "verified")
        self.assertEqual(outcome["assurance"], "behavioral")
        self.assertTrue(outcome["verified_completion"])

    def test_incomplete_upstream_is_inconclusive(self) -> None:
        outcome = enforcement_outcome(
            {"status": "incomplete_upstream", "verdict": "needs_evidence"}
        )
        self.assertEqual(outcome["status"], "inconclusive")
        self.assertEqual(outcome["verdict"], "needs_evidence")
        self.assertFalse(outcome["verified_completion"])

    def test_labeled_false_allow_and_false_block(self) -> None:
        self.assertEqual(labeled_error_kind({"verdict": "allow"}, "block"), "false_allow")
        self.assertEqual(labeled_error_kind({"verdict": "block"}, "allow"), "false_block")
        self.assertEqual(
            labeled_error_kind({"verdict": "needs_evidence"}, "allow"),
            "false_block",
        )
        self.assertIsNone(labeled_error_kind({"verdict": "allow"}, "allow"))

    def test_passing_repository_tests_are_a_verified_completion(self) -> None:
        outcome = patch_outcome("allow")
        self.assertEqual(outcome["assurance"], "repository_tests")
        self.assertTrue(outcome["verified_completion"])

    def test_grounded_agent_answer_is_a_verified_completion(self) -> None:
        outcome = agent_completion_outcome()
        self.assertEqual(outcome["assurance"], "agent_tools")
        self.assertEqual(outcome["verdict"], "allow")
        self.assertTrue(outcome["verified_completion"])


class ProxyMetricsTests(unittest.TestCase):
    def test_aggregates_and_persists_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            metrics = ProxyMetrics(path)
            metrics.observe(
                {
                    "outcome": {
                        "status": "verified",
                        "verdict": "allow",
                        "assurance": "behavioral",
                        "verified_completion": True,
                    },
                    "timing_ms": {"request_total": 12.5},
                },
                request_id="r1",
                expected_verdict="block",
                case_id="false-allow-case",
            )
            metrics.observe(
                {
                    "outcome": {
                        "status": "blocked",
                        "assurance": "structural",
                        "verified_completion": False,
                    },
                    "timing_ms": {"request_total": 7.5},
                },
                request_id="r2",
            )

            snapshot = metrics.snapshot()

            self.assertEqual(snapshot["requests"], 2)
            self.assertEqual(snapshot["verified_completions"], 1)
            self.assertEqual(snapshot["verified_completion_rate"], 0.5)
            self.assertEqual(snapshot["latency_ms"]["average"], 10.0)
            self.assertEqual(snapshot["evaluation"]["labeled_requests"], 1)
            self.assertEqual(snapshot["evaluation"]["false_allows"], 1)
            self.assertEqual(snapshot["evaluation"]["false_allow_rate"], 1.0)
            self.assertEqual(snapshot["evaluation"]["false_blocks"], 0)
            self.assertEqual(snapshot["evaluation"]["errors"], {"false_allow": 1})
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertFalse(events[0]["label_matches"])

    def test_agent_completion_and_tool_success_are_measured_separately(self) -> None:
        metrics = ProxyMetrics()
        metrics.observe(
            {
                "outcome": enforcement_outcome({"status": "no_code"}),
                "timing_ms": {"request_total": 12},
                "agent": {
                    "scorecard": {
                        "completed": True,
                        "tool_calls": 2,
                        "successful_tool_calls": 1,
                    }
                },
            }
        )

        snapshot = metrics.snapshot()
        self.assertEqual(
            snapshot["agent"],
            {
                "runs": 1,
                "completed": 1,
                "completion_rate": 1.0,
                "tool_calls": 2,
                "successful_tool_calls": 1,
                "tool_success_rate": 0.5,
            },
        )
        self.assertEqual(snapshot["verified_completions"], 0)


if __name__ == "__main__":
    unittest.main()
