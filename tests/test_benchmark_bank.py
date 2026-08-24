import json
import unittest
from pathlib import Path

from cortheon.decision import DecisionLayer

BANK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bank" / "tasks.json"


class BenchmarkBankShapeTests(unittest.TestCase):
    """Offline guards for the benchmark bank: schema integrity plus the gate
    verdicts that need no evidence to reach (so CI catches drift without network)."""

    def setUp(self) -> None:
        self.bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))

    def test_every_task_is_well_formed(self) -> None:
        seen = set()
        for task in self.bank["tasks"]:
            self.assertNotIn(task["id"], seen)
            seen.add(task["id"])
            self.assertIn(task["category"], self.bank["categories"])
            self.assertIn(task["expected_gate"], {"allow", "needs_evidence", "block"})
            # Every task needs a grading source: regex patterns, or a live
            # answer key fetched at grading time (post_cutoff tasks).
            self.assertTrue(task.get("good_patterns") or task.get("live_grading"))

    def test_no_evidence_gate_verdicts_match(self) -> None:
        # With zero evidence supplied, destructive tasks must block and
        # unsupported-action tasks must not shortcut to allow.
        for task in self.bank["tasks"]:
            report = DecisionLayer().evaluate(
                task["task"], proposed_action=task.get("proposed_action")
            )
            if task["expected_gate"] == "block":
                self.assertEqual(report.verdict, "block", task["id"])
            else:
                self.assertIn(report.verdict, {"needs_evidence", "allow"}, task["id"])

    def test_destructive_task_is_non_auto_satisfiable(self) -> None:
        task = next(
            t
            for t in self.bank["tasks"]
            if t["category"] == "security" and t["expected_gate"] == "block"
        )
        report = DecisionLayer().evaluate(
            task["task"],
            proposed_action=task.get("proposed_action"),
            evidence=["package_verified", "api_evidence", "repo_context", "tests_passed"],
        )
        self.assertEqual(report.verdict, "block")


if __name__ == "__main__":
    unittest.main()
