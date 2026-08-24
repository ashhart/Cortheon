import unittest

from cortheon.artifacts import derive_research_artifacts
from cortheon.models import CrawledPage, ScholarlyWork, SearchResult, utc_now


class ResearchArtifactTests(unittest.TestCase):
    def test_derive_artifacts_from_papers_pages_and_search_results(self) -> None:
        now = utc_now()
        artifacts = derive_research_artifacts(
            "open-ended artificial life benchmark",
            [
                ScholarlyWork(
                    title="Open-ended artificial life benchmark",
                    url="https://arxiv.org/abs/2501.12345",
                    abstract="A benchmark for open-ended artificial life systems.",
                    authors=["A. Researcher"],
                    published_at=now,
                    source="arxiv",
                    venue="arXiv",
                    identifiers={"arxiv": "2501.12345", "doi": "10.1234/example"},
                    cited_by_count=None,
                    authority_score=0.9,
                    relevance_score=0.95,
                )
            ],
            [
                CrawledPage(
                    url="https://example.org/benchmark",
                    final_url="https://example.org/benchmark",
                    status=200,
                    title="ALIFE leaderboard",
                    text="The benchmark suite links code and benchmark dataset artifacts.",
                    links=[
                        "https://github.com/example/alife-benchmark",
                        "https://huggingface.co/datasets/example/alife",
                    ],
                    source_type="benchmark",
                    authority_score=0.7,
                    fetched_at=now,
                )
            ],
            search_results=[
                SearchResult(
                    title="ALIFE results on Papers with Code",
                    url="https://paperswithcode.com/sota/artificial-life-on-alife",
                    snippet="Leaderboard and benchmark results.",
                    provider="fixture",
                    rank=1,
                )
            ],
        )

        urls = {artifact.url for artifact in artifacts}
        kinds = {artifact.kind for artifact in artifacts}
        self.assertIn("https://arxiv.org/pdf/2501.12345", urls)
        self.assertIn("https://arxiv.org/e-print/2501.12345", urls)
        self.assertIn("https://doi.org/10.1234/example", urls)
        self.assertIn("https://github.com/example/alife-benchmark", urls)
        self.assertIn("https://huggingface.co/datasets/example/alife", urls)
        self.assertIn("paper_pdf", kinds)
        self.assertIn("paper_source", kinds)
        self.assertIn("code_repository", kinds)
        self.assertIn("dataset", kinds)
        self.assertIn("benchmark", kinds)


if __name__ == "__main__":
    unittest.main()
