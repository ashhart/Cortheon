import unittest

from cortheon.models import CrawledPage, ResearchQuery, ScholarlyWork, utc_now
from cortheon.research import ResearchEngine, build_gap_closures
from cortheon.scholarly import ScholarlyDiscoveryResult


class ResearchEngineTests(unittest.TestCase):
    def test_research_report_includes_scholarly_works_and_claims(self) -> None:
        engine = ResearchEngine(
            search_provider=FakeSearchProvider(),
            scholarly_discovery=FakeScholarlyDiscovery(),
            github_discovery=FakeGitHubDiscovery(),
            crawler=FakeCrawler(),
        )

        report = engine.research(
            "open-ended artificial life",
            max_search_results=0,
            max_scholarly_results=1,
            max_pages=1,
            write_report=False,
        )

        self.assertEqual(len(report.scholarly_works), 1)
        self.assertEqual(len(report.discovery_passes), 4)
        self.assertEqual(report.discovery_passes[0].query, "open-ended evolution artificial life")
        self.assertTrue(report.discovery_passes[1].purpose)
        self.assertEqual(report.discovery_passes[-1].source, "evidence_gap")
        self.assertTrue(report.discovery_passes[-1].target_gap)
        self.assertEqual(len(report.crawled_pages), 1)
        self.assertTrue(report.claims)
        self.assertTrue(report.source_plan)
        self.assertIn("scholarly", {item.name for item in report.source_plan if item.selected})
        self.assertTrue(any(artifact.kind == "paper_pdf" for artifact in report.artifacts))
        self.assertIsNotNone(report.synthesis)
        self.assertTrue(report.synthesis.key_findings)
        self.assertTrue(any("scholarly work" in item.claim for item in report.evidence))
        self.assertTrue(
            any(item.source_type == "research_mission_plan" for item in report.evidence)
        )

    def test_gap_closure_records_closed_and_improved_status(self) -> None:
        closed = build_gap_closures(
            [
                ResearchQuery(
                    query="topic benchmark evaluation",
                    purpose="close synthesis gap",
                    source="evidence_gap",
                    target_gap="No clear benchmark or evaluation claim was extracted.",
                )
            ],
            ["No clear benchmark or evaluation claim was extracted."],
            [],
            before_claim_count=2,
            after_claim_count=5,
            before_source_count=2,
            after_source_count=4,
        )
        improved = build_gap_closures(
            [
                ResearchQuery(
                    query="topic survey review evidence",
                    purpose="close synthesis gap",
                    source="evidence_gap",
                    target_gap="Too few extracted claims for a strong synthesis.",
                )
            ],
            ["Too few extracted claims for a strong synthesis."],
            ["Too few extracted claims for a strong synthesis."],
            before_claim_count=1,
            after_claim_count=3,
            before_source_count=1,
            after_source_count=1,
        )

        self.assertEqual(closed[0].status, "closed")
        self.assertEqual(improved[0].status, "improved_but_open")


class FakeSearchProvider:
    name = "fake"

    def search(self, query, limit):
        return [], []


class FakeScholarlyDiscovery:
    def search(self, query, limit, connector_names=None):
        return ScholarlyDiscoveryResult(
            works=[
                ScholarlyWork(
                    title="Open-ended artificial life benchmark",
                    url="https://arxiv.org/abs/1",
                    abstract=(
                        "We introduce an open-ended artificial life benchmark that evaluates "
                        "ongoing novelty in evolving agents."
                    ),
                    authors=["A. Researcher"],
                    published_at=utc_now(),
                    source="arxiv",
                    venue="arXiv",
                    identifiers={"arxiv": "1"},
                    cited_by_count=None,
                    authority_score=0.9,
                )
            ],
            evidence=[],
            errors=[],
        )


class FakeGitHubDiscovery:
    def search(self, query, limit):
        return [], [], []

    def inspect_artifacts(self, artifacts, limit):
        return artifacts, [], []


class FakeCrawler:
    def crawl(self, seed_urls, allowed_domains=None, budget=None):
        return [
            CrawledPage(
                url="https://arxiv.org/abs/1",
                final_url="https://arxiv.org/abs/1",
                status=200,
                title="Open-ended artificial life benchmark",
                text="This paper proposes an artificial life benchmark for open-ended evolution.",
                links=[],
                source_type="paper",
                authority_score=0.9,
                fetched_at=utc_now(),
            )
        ], []


if __name__ == "__main__":
    unittest.main()
