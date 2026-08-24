import unittest

from cortheon.knowledge_pool import KnowledgePooler
from cortheon.models import (
    ApiEvidenceReport,
    ApiSymbol,
    CrawledPage,
    RecommendationReport,
    ResearchClaim,
    ResearchCoverageItem,
    ResearchReport,
    ResearchSourceDecision,
    ResearchSynthesis,
    SearchResult,
    SupportLevel,
    utc_now,
)


class KnowledgePoolTests(unittest.TestCase):
    def test_supported_task_returns_answer_and_allow_verdict(self) -> None:
        research = FakeResearchEngine(
            research_report(
                direction="Use FastAPI with an ASGI server and keep validation explicit.",
                gaps=[],
            )
        )

        report = KnowledgePooler(FakeEngine(), research_engine=research).run(
            "How should I build a REST API for a Python service now?",
            proposed_action="Use FastAPI.",
        )

        self.assertEqual(report.verdict, "allow")
        self.assertEqual(report.answer_status, "answered")
        self.assertIn("FastAPI", report.best_supported_approach)
        self.assertIn("technology_research_report", report.evidence_tags)
        self.assertEqual(report.source_summaries[0].url, "https://example.com/fastapi-guide")
        self.assertEqual(len(research.calls), 1)

    def test_blocked_action_does_not_pool_sources(self) -> None:
        research = FakeResearchEngine(research_report())

        report = KnowledgePooler(FakeEngine(), research_engine=research).run(
            "Release is blocked",
            proposed_action="Purge production auth variables from project settings.",
        )

        self.assertEqual(report.verdict, "block")
        self.assertEqual(report.answer_status, "blocked")
        self.assertEqual(report.source_summaries, [])
        self.assertIn("Do not proceed", report.best_supported_approach)
        self.assertEqual(research.calls, [])

    def test_fake_api_symbol_keeps_answer_evidence_limited(self) -> None:
        engine = FakeEngine(api_matches=[])
        report = KnowledgePooler(engine, research_engine=FakeResearchEngine(research_report())).run(
            "How should I stream bytes with httpx?",
            proposed_action="Use httpx.AsyncClient.fake_stream_now.",
        )

        self.assertEqual(report.verdict, "needs_evidence")
        self.assertEqual(report.answer_status, "needs_evidence")
        self.assertIn("api_evidence", report.evidence_gaps)
        self.assertNotIn("api_evidence", report.evidence_tags)
        self.assertEqual(engine.api_calls, [("httpx", "AsyncClient.fake_stream_now")])


class FakeEngine:
    def __init__(self, *, api_matches: list[str] | None = None) -> None:
        self.api_matches = ["httpx.AsyncClient.stream"] if api_matches is None else api_matches
        self.api_calls: list[tuple[str, str]] = []

    def recommend(self, task: str) -> RecommendationReport:
        return RecommendationReport(
            task=task,
            profile=None,
            generated_at=utc_now(),
            winner=None,
            candidates=[],
            evidence=[],
            notes=[],
        )

    def retrieve_api_evidence(self, package: str, query: str) -> ApiEvidenceReport:
        self.api_calls.append((package, query))
        return ApiEvidenceReport(
            package=package,
            version="1.0.0",
            query=query,
            artifact_filename=f"{package}-1.0.0.tar.gz",
            artifact_url=None,
            extracted_at=utc_now(),
            total_symbols=len(self.api_matches),
            matches=[
                ApiSymbol(
                    name=item.split(".")[-1],
                    kind="function",
                    module=package,
                    qualname=item,
                    signature=None,
                    file_path=f"{package}/__init__.py",
                    line=1,
                    docstring=None,
                )
                for item in self.api_matches
            ],
            evidence=[],
            errors=[],
        )


class FakeResearchEngine:
    def __init__(self, report: ResearchReport) -> None:
        self.report = report
        self.calls: list[tuple[str, dict[str, object]]] = []

    def research(self, topic: str, **kwargs: object) -> ResearchReport:
        self.calls.append((topic, kwargs))
        self.report.topic = topic
        if self.report.synthesis:
            self.report.synthesis.topic = topic
        return self.report


def research_report(
    *,
    direction: str = "Use the source-supported implementation path.",
    gaps: list[str] | None = None,
) -> ResearchReport:
    now = utc_now()
    claim = ResearchClaim(
        text="Current sources support the implementation direction.",
        source_url="https://example.com/fastapi-guide",
        source_title="Current FastAPI guide",
        source_type="web",
        support=SupportLevel.OBSERVED,
        confidence=0.8,
        stance="support",
        source_excerpt="Current sources support the implementation direction.",
        source_char_start=0,
        source_char_end=53,
    )
    return ResearchReport(
        topic="test topic",
        generated_at=now,
        search_provider="fake",
        seed_urls=[],
        search_results=[
            SearchResult(
                title="Current FastAPI guide",
                url="https://example.com/fastapi-guide",
                snippet="Current guidance for building a Python REST API.",
                provider="fake",
                rank=1,
            )
        ],
        scholarly_works=[],
        crawled_pages=[
            CrawledPage(
                url="https://example.com/fastapi-guide",
                final_url="https://example.com/fastapi-guide",
                status=200,
                title="Current FastAPI guide",
                text="Current guidance for building a Python REST API with explicit validation.",
                links=[],
                source_type="official_docs",
                authority_score=0.9,
                fetched_at=now,
            )
        ],
        artifacts=[],
        claims=[claim],
        source_lineage=[],
        synthesis=ResearchSynthesis(
            topic="test topic",
            generated_at=now,
            status="promising_but_incomplete",
            confidence=0.74,
            current_best_direction=direction,
            key_findings=["Use source-backed docs and verify package APIs before coding."],
            contested_points=[],
            evidence_gaps=gaps or [],
            clusters=[],
            contradictions=[],
        ),
        evidence=[],
        notes=[],
        errors=[],
        source_plan=[
            ResearchSourceDecision(
                name="web_search",
                source_type="web_search",
                selected=True,
                available=True,
                reason="Selected for current implementation guidance.",
                capabilities=["current_docs"],
                trust_tier="current_web",
                priority=0.8,
                budget=1,
                domains=["software"],
            )
        ],
        source_coverage=[
            ResearchCoverageItem(
                name="current_docs",
                status="covered",
                expected=True,
                observed_count=1,
                reason="Found current documentation-like source.",
                next_action="Use cited source boundaries.",
                source_names=["web_search"],
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
