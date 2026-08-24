import unittest

from cortheon.research_plan import plan_gap_follow_up_queries, plan_research_queries


class ResearchPlanTests(unittest.TestCase):
    def test_plan_includes_bounded_follow_up_queries(self) -> None:
        queries = plan_research_queries(
            "open-ended artificial life evolution benchmark",
            max_follow_up_queries=2,
        )

        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0].query, "open-ended evolution artificial life")
        self.assertEqual(queries[0].source, "user_topic")
        self.assertEqual(queries[1].query, "quality diversity novelty search artificial life")
        self.assertEqual(queries[2].query, "open-ended evolution benchmark")

    def test_biomedical_alife_plan_uses_source_friendly_queries(self) -> None:
        queries = plan_research_queries(
            (
                "We want an ALIFE-style cure engine that can discover senolytic therapies. "
                "Choose the strongest current architecture and tell the lab what to build first."
            ),
            max_follow_up_queries=2,
        )

        self.assertEqual(
            [item.query for item in queries],
            [
                "open-ended evolution artificial life architecture benchmark",
                "quality diversity novelty search artificial life benchmark",
                "senolytics cellular senescence clinical trial",
            ],
        )

    def test_plan_can_disable_follow_ups(self) -> None:
        queries = plan_research_queries("cure engine target discovery", max_follow_up_queries=0)

        self.assertEqual(
            [item.query for item in queries], ["therapeutic discovery clinical trial evidence"]
        )

    def test_gap_follow_up_targets_observed_synthesis_gap(self) -> None:
        existing = plan_research_queries(
            "open-ended artificial life evolution benchmark",
            max_follow_up_queries=1,
        )

        queries = plan_gap_follow_up_queries(
            "open-ended artificial life evolution benchmark",
            ["No clear benchmark or evaluation claim was extracted."],
            existing,
            max_adaptive_queries=1,
        )

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].source, "evidence_gap")
        self.assertEqual(
            queries[0].purpose, "close synthesis gap: find benchmark or evaluation evidence"
        )
        self.assertEqual(
            queries[0].target_gap, "No clear benchmark or evaluation claim was extracted."
        )
        self.assertIn("benchmark evaluation metrics dataset", queries[0].query)


if __name__ == "__main__":
    unittest.main()
