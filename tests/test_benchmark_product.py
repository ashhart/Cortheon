import sys
import unittest

from cortheon.benchmark import (
    Contender,
    _builtin_cases,
    _call_cli_contender,
    _case_bank_hash,
    _parse_cli_spec,
    _result_cost,
    evaluate_promotion,
    grade_answer,
    select_case_bank,
)


def _cases(count: int = 20) -> list[dict]:
    return [
        {
            "id": f"case_{index}",
            "category": "reasoning",
            "domain": "reasoning",
            "difficulty": "medium",
            "prompt": f"Task {index}",
            "expected_verdict": "allow",
            "grader": {"type": "patterns"},
            "documents": [],
        }
        for index in range(count)
    ]


def _report(rate: float, *, selection_hash: str = "a" * 64) -> dict:
    completed = round(rate * 10)
    summary = {
        "runs": 10,
        "verified_completion_rate": rate,
        "false_allows": 0,
        "false_blocks": 0,
        "errors": 0,
        "latency_ms": {"p95": 100},
        "cost_usd": {"mean": 0.01},
        "by_domain": {
            "reasoning": {
                "verified_completion_rate": rate,
            }
        },
    }
    return {
        "methodology": {"grading": "deterministic and contender-blind"},
        "case_bank": {"selection_sha256": selection_hash},
        "candidates": {"candidate_1": {"name": "cortheon"}},
        "summary": {"candidate_1": summary},
        "rows": [
            {
                "candidate": "candidate_1",
                "case_id": f"case_{index}",
                "repetition": 1,
                "verified_completion": index < completed,
            }
            for index in range(10)
        ],
    }


class CaseSelectionTests(unittest.TestCase):
    def test_builtin_smoke_bank_covers_north_star_domains(self) -> None:
        domains = {case.get("domain") for case in _builtin_cases()}

        self.assertTrue(
            {
                "coding",
                "research",
                "documents",
                "debugging",
                "planning",
                "long_horizon",
            }.issubset(domains)
        )

    def test_ordered_plan_grader_rejects_right_steps_in_wrong_order(self) -> None:
        case = next(
            item for item in _builtin_cases() if item["id"] == "plan_online_schema_migration"
        )
        correct = (
            "Deploy the additive schema, enable dual-write, run the backfill, "
            "switch readers, then remove the legacy field."
        )
        reordered = (
            "Run the backfill, deploy the additive schema, enable dual-write, "
            "switch readers, then remove the legacy field."
        )

        self.assertTrue(grade_answer(case, correct)["passed"])
        rejected = grade_answer(case, reordered)
        self.assertFalse(rejected["passed"])
        self.assertIn("required_steps_out_of_order", rejected["failures"])

    def test_split_and_rotation_are_reproducible_and_disjoint(self) -> None:
        cases = _cases()
        development = select_case_bank(
            cases,
            split="development",
            seed=17,
            holdout_fraction=0.3,
            rotation_index=0,
            rotation_size=0,
        )
        heldout = select_case_bank(
            cases,
            split="heldout",
            seed=17,
            holdout_fraction=0.3,
            rotation_index=0,
            rotation_size=0,
        )
        self.assertEqual(len(development) + len(heldout), len(cases))
        self.assertFalse({case["id"] for case in development} & {case["id"] for case in heldout})
        first = select_case_bank(
            cases,
            split="all",
            seed=17,
            holdout_fraction=0.3,
            rotation_index=1,
            rotation_size=5,
        )
        repeated = select_case_bank(
            list(reversed(cases)),
            split="all",
            seed=17,
            holdout_fraction=0.3,
            rotation_index=1,
            rotation_size=5,
        )
        self.assertEqual(
            [case["id"] for case in first],
            [case["id"] for case in repeated],
        )

    def test_case_hash_excludes_resolved_live_answer(self) -> None:
        cases = _cases(1)
        before = _case_bank_hash(cases)
        cases[0]["grader"]["answer_key"] = {"version": "tomorrow"}
        self.assertEqual(before, _case_bank_hash(cases))


class CliContenderTests(unittest.TestCase):
    def test_cli_is_argv_based_and_reads_prompt_from_stdin(self) -> None:
        command = (
            sys.executable,
            "-c",
            "import sys; print('ANSWER:' + sys.stdin.read().splitlines()[-1])",
        )
        contender = Contender(
            "local",
            "cli",
            "local-cli",
            "python",
            "",
            command=command,
        )
        answer, metadata = _call_cli_contender(
            contender,
            [{"role": "user", "content": "needle"}],
            timeout=5,
        )
        self.assertEqual(answer, "ANSWER:needle")
        self.assertNotIn("-c", str(metadata))

    def test_cli_spec_does_not_treat_shell_tokens_specially(self) -> None:
        name, command = _parse_cli_spec("frontier=claude -p 'hello; echo unsafe'")
        self.assertEqual(name, "frontier")
        self.assertEqual(command, ("claude", "-p", "hello; echo unsafe"))

    def test_cli_prompt_placeholder_is_one_argv_value(self) -> None:
        contender = Contender(
            "local",
            "cli",
            "local-cli",
            "python",
            "",
            command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{prompt}",
            ),
        )
        answer, _metadata = _call_cli_contender(
            contender,
            [{"role": "user", "content": "one; echo two"}],
            timeout=5,
        )
        self.assertEqual(answer, "USER:\none; echo two")

    def test_reported_and_estimated_costs(self) -> None:
        contender = Contender(
            "priced",
            "stock",
            "https://example.test",
            "model",
            "",
            input_cost_per_million=2,
            output_cost_per_million=10,
        )
        estimated = _result_cost(
            {"usage": {"prompt_tokens": 1_000, "completion_tokens": 200}},
            contender,
            latency_ms=100,
        )
        self.assertEqual(estimated["usd"], 0.004)
        reported = _result_cost(
            {"total_cost_usd": 0.123},
            contender,
            latency_ms=100,
        )
        self.assertEqual(reported["source"], "reported")
        self.assertEqual(reported["usd"], 0.123)


class PromotionGateTests(unittest.TestCase):
    def test_passes_only_on_improvement_without_regression(self) -> None:
        gate = evaluate_promotion(
            _report(0.7),
            _report(0.8),
            candidate_name="cortheon",
            min_improvement=0,
            max_domain_regression=0,
            max_latency_ratio=1.25,
            max_cost_ratio=1.25,
        )
        self.assertTrue(gate["passed"], gate)

    def test_fails_closed_on_hash_mismatch_or_no_improvement(self) -> None:
        gate = evaluate_promotion(
            _report(0.8),
            _report(0.8, selection_hash="b" * 64),
            candidate_name="cortheon",
            min_improvement=0,
            max_domain_regression=0.02,
            max_latency_ratio=1.25,
            max_cost_ratio=1.25,
        )
        self.assertFalse(gate["passed"])
        self.assertIn("same_blinded_case_selection", gate["failure_reasons"])
        self.assertIn("verified_completion_improved", gate["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
