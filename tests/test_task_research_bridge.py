import unittest

from cortheon.task_research_bridge import (
    build_task_research_plan,
    classify_task,
    domain_research_limits,
    translate_task_to_queries,
)


class TaskResearchBridgeTests(unittest.TestCase):
    def test_classify_software_task(self) -> None:
        domain = classify_task("build a REST API for a Python service")
        self.assertEqual(domain.name, "software_engineering")
        self.assertTrue(domain.is_software)

    def test_classify_biomedical_task(self) -> None:
        domain = classify_task("cancer immunotherapy clinical trial evidence")
        self.assertEqual(domain.name, "biomedical")
        self.assertTrue(domain.is_biomedical)

    def test_classify_frontier_task(self) -> None:
        domain = classify_task("frontier artificial life open-ended evolution engines")
        self.assertEqual(domain.name, "frontier_research")
        self.assertTrue(domain.is_frontier)

    def test_classify_general_task(self) -> None:
        domain = classify_task("do something completely unrelated")
        self.assertEqual(domain.name, "general")

    def test_build_plan_software(self) -> None:
        plan = build_task_research_plan("build a REST API for a Python service", "Use FastAPI.")
        self.assertEqual(plan.domain, "software_engineering")
        self.assertTrue(plan.is_software)
        self.assertGreater(len(plan.research_queries), 0)
        self.assertIsNotNone(plan.domain_obj)

    def test_build_plan_biomedical(self) -> None:
        plan = build_task_research_plan("cancer immunotherapy clinical trial evidence")
        self.assertEqual(plan.domain, "biomedical")
        self.assertTrue(plan.is_biomedical)
        self.assertGreater(len(plan.research_queries), 0)

    def test_build_plan_frontier(self) -> None:
        plan = build_task_research_plan("frontier artificial life open-ended evolution engines")
        self.assertEqual(plan.domain, "frontier_research")
        self.assertTrue(plan.is_frontier)
        self.assertGreater(len(plan.research_queries), 0)

    def test_translate_queries_uses_task(self) -> None:
        domain = classify_task("build a REST API")
        queries = translate_task_to_queries("build a REST API", domain)
        self.assertGreater(len(queries), 0)
        # Each query should contain the task text.
        for query in queries:
            self.assertIn("build a rest api", query.lower())

    def test_domain_limits_software(self) -> None:
        domain = classify_task("build a REST API")
        limits = domain_research_limits(domain)
        self.assertEqual(limits["max_scholarly_results"], 0)
        self.assertGreater(limits["max_github_results"], 0)
        self.assertGreater(limits["max_pages"], 0)

    def test_domain_limits_biomedical(self) -> None:
        domain = classify_task("cancer immunotherapy clinical trial")
        limits = domain_research_limits(domain)
        self.assertGreater(limits["max_scholarly_results"], 0)
        self.assertEqual(limits["max_github_results"], 0)
        self.assertEqual(limits["max_pages"], 0)
        self.assertGreater(limits["max_trial_results"], 0)

    def test_domain_limits_frontier(self) -> None:
        domain = classify_task("frontier artificial life")
        limits = domain_research_limits(domain)
        self.assertGreater(limits["max_scholarly_results"], 0)
        self.assertGreater(limits["max_github_results"], 0)
        self.assertGreater(limits["max_pages"], 0)

    def test_plan_confidence_software(self) -> None:
        plan = build_task_research_plan("build a REST API for a Python service")
        self.assertGreater(plan.confidence, 0.0)

    def test_plan_confidence_general(self) -> None:
        plan = build_task_research_plan("do something completely unrelated")
        self.assertLess(plan.confidence, 0.5)

    def test_plan_includes_proposed_action_in_classification(self) -> None:
        plan = build_task_research_plan("build a service", "Use FastAPI.")
        self.assertEqual(plan.domain, "software_engineering")


if __name__ == "__main__":
    unittest.main()
