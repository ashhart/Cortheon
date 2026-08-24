import unittest

from cortheon.auto_evidence import EvidenceAcquisitionLoop
from cortheon.models import (
    ApiEvidenceReport,
    ApiSymbol,
    Evidence,
    PackageMetadata,
    PackageReport,
    RecommendationReport,
    ResearchClaim,
    ResearchCoverageItem,
    ResearchReport,
    ResearchSynthesis,
    SupportLevel,
    utc_now,
)


class AutoEvidenceTests(unittest.TestCase):
    def test_matching_package_recommendation_satisfies_decision(self) -> None:
        loop = EvidenceAcquisitionLoop(FakeEngine(recommendation=recommendation_report("fastapi")))

        report = loop.run("Build a REST API", proposed_action="Use FastAPI.")

        self.assertEqual(report.initial_decision.verdict, "needs_evidence")
        self.assertEqual(report.final_decision.verdict, "allow")
        self.assertIn("recommendation_report", report.evidence_tags)
        self.assertEqual(report.agent_runs[0].agent, "package_evidence_agent")
        self.assertEqual(
            report.agent_runs[0].details["sources"][0]["source_type"], "pypi_package_metadata"
        )

    def test_mismatched_package_recommendation_does_not_rubber_stamp(self) -> None:
        loop = EvidenceAcquisitionLoop(FakeEngine(recommendation=recommendation_report("fastapi")))

        report = loop.run(
            "Build a REST API",
            proposed_action="Install the NexaAPI package for production.",
        )

        self.assertEqual(report.final_decision.verdict, "needs_evidence")
        self.assertEqual(report.agent_runs[0].status, "partial")
        self.assertNotIn("recommendation_report", report.evidence_tags)

    def test_api_symbol_agent_satisfies_real_source_symbol(self) -> None:
        loop = EvidenceAcquisitionLoop(
            FakeEngine(
                api_evidence_report=api_report(
                    "httpx", "AsyncClient.stream", matches=["httpx.AsyncClient.stream"]
                )
            )
        )

        report = loop.run(
            "Migrate downloader",
            proposed_action="Use httpx.AsyncClient.stream.",
            evidence=["recommendation_report"],
        )

        self.assertEqual(report.final_decision.verdict, "allow")
        self.assertIn("api_evidence", report.evidence_tags)

    def test_api_symbol_agent_keeps_fake_symbol_blocked_on_evidence(self) -> None:
        loop = EvidenceAcquisitionLoop(
            FakeEngine(api_evidence_report=api_report("httpx", "AsyncClient.fake", matches=[]))
        )

        report = loop.run(
            "Migrate downloader",
            proposed_action="Use httpx.AsyncClient.fake.",
            evidence=["recommendation_report"],
        )

        self.assertEqual(report.final_decision.verdict, "needs_evidence")
        self.assertNotIn("api_evidence", report.evidence_tags)
        self.assertEqual(
            report.agent_runs[0].details["sources"][0]["source_type"], "pypi_package_metadata"
        )

    def test_research_agent_satisfies_frontier_research_when_grounded(self) -> None:
        loop = EvidenceAcquisitionLoop(
            FakeEngine(),
            research_engine=FakeResearchEngine(
                research_report("cure engine", status="promising_but_incomplete")
            ),
        )

        report = loop.run(
            "Choose the best cure engine direction.",
            proposed_action="Build an ALIFE cure engine.",
        )

        self.assertEqual(report.final_decision.verdict, "allow")
        self.assertIn("research_report", report.evidence_tags)
        self.assertIn("grounded_claims", report.evidence_tags)

    def test_technology_choice_can_use_research_backed_evidence(self) -> None:
        loop = EvidenceAcquisitionLoop(
            FakeEngine(recommendation=empty_recommendation_report()),
            research_engine=FakeResearchEngine(
                research_report("vector database", status="promising_but_incomplete")
            ),
        )

        report = loop.run(
            "Pick the current best vector database for a coding-agent memory layer.",
            proposed_action="Select Qdrant as the production vector database.",
        )

        self.assertEqual(report.final_decision.verdict, "allow")
        self.assertIn("technology_research_report", report.evidence_tags)

    def test_architecture_commitment_needs_architecture_specific_research(self) -> None:
        loop = EvidenceAcquisitionLoop(
            FakeEngine(),
            research_engine=FakeResearchEngine(
                research_report(
                    "senolytic therapies",
                    status="promising_but_incomplete",
                    gaps=[
                        "Topic terms under-covered in extracted claims: alife, architecture, build."
                    ],
                )
            ),
        )

        report = loop.run(
            "Choose the strongest current architecture for an ALIFE cure engine.",
            proposed_action="Commit to the strongest cure-engine architecture and tell the lab to build it first.",
        )

        self.assertEqual(report.final_decision.verdict, "needs_evidence")
        self.assertIn("research_report", report.evidence_tags)
        self.assertNotIn("architecture_research_report", report.evidence_tags)
        self.assertIn("architecture_evidence", report.final_decision.required_evidence)
        self.assertEqual(report.agent_runs[-1].status, "partial")

    def test_destructive_action_does_not_run_auto_agents(self) -> None:
        loop = EvidenceAcquisitionLoop(FakeEngine())

        report = loop.run(
            "Release is blocked",
            proposed_action="Purge production auth variables from project settings.",
        )

        self.assertEqual(report.initial_decision.verdict, "block")
        self.assertEqual(report.final_decision.verdict, "block")
        self.assertEqual(report.agent_runs, [])


class FakeEngine:
    def __init__(
        self,
        *,
        recommendation: RecommendationReport | None = None,
        api_evidence_report: ApiEvidenceReport | None = None,
    ) -> None:
        self.recommendation = recommendation or empty_recommendation_report()
        self.api_report = api_evidence_report or api_report("httpx", "Client.stream", [])
        self.recommend_calls: list[str] = []
        self.api_calls: list[tuple[str, str]] = []

    def recommend(self, task: str) -> RecommendationReport:
        self.recommend_calls.append(task)
        return self.recommendation

    def retrieve_api_evidence(self, package: str, query: str) -> ApiEvidenceReport:
        self.api_calls.append((package, query))
        return self.api_report


class FakeResearchEngine:
    def __init__(self, report: ResearchReport) -> None:
        self.report = report
        self.calls: list[tuple[str, dict[str, object]]] = []

    def research(self, topic: str, **kwargs: object) -> ResearchReport:
        self.calls.append((topic, kwargs))
        self.report.topic = topic
        return self.report


def recommendation_report(winner: str) -> RecommendationReport:
    return RecommendationReport(
        task="test task",
        profile="test_profile",
        generated_at=utc_now(),
        winner=winner,
        candidates=[package_report(winner)],
        evidence=[Evidence(f"{winner} ranked highest", "unit_test", None)],
        notes=["test recommendation"],
    )


def empty_recommendation_report() -> RecommendationReport:
    return RecommendationReport(
        task="test task",
        profile=None,
        generated_at=utc_now(),
        winner=None,
        candidates=[],
        evidence=[],
        notes=["no profile"],
    )


def package_report(name: str) -> PackageReport:
    now = utc_now()
    return PackageReport(
        package=name,
        version="1.0.0",
        fetched_at=now,
        metadata=PackageMetadata(
            name=name,
            version="1.0.0",
            summary="A test package",
            requires_python=">=3.11",
            license="MIT",
            project_urls={},
            classifiers=[],
            requires_dist=[],
            release_upload_time=now,
            release_count=1,
            artifacts=[],
            source_url=f"https://pypi.org/pypi/{name}/json",
        ),
        vulnerabilities=None,
        github=None,
        documentation=None,
        verification=None,
        evidence=[],
        score=None,
        errors=[],
    )


def api_report(package: str, query: str, matches: list[str]) -> ApiEvidenceReport:
    return ApiEvidenceReport(
        package=package,
        version="1.0.0",
        query=query,
        artifact_filename=f"{package}.tar.gz",
        artifact_url=None,
        extracted_at=utc_now(),
        total_symbols=len(matches),
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
            for item in matches
        ],
        evidence=[],
        errors=[],
    )


def research_report(topic: str, *, status: str, gaps: list[str] | None = None) -> ResearchReport:
    now = utc_now()
    claim = ResearchClaim(
        text="Grounded evidence supports this technical direction.",
        source_url="https://example.com/source",
        source_title="Example source",
        source_type="web",
        support=SupportLevel.OBSERVED,
        confidence=0.8,
        source_excerpt="Grounded evidence supports this technical direction.",
        source_char_start=0,
        source_char_end=51,
    )
    return ResearchReport(
        topic=topic,
        generated_at=now,
        search_provider="fake",
        seed_urls=[],
        search_results=[],
        scholarly_works=[],
        crawled_pages=[],
        artifacts=[],
        claims=[claim],
        source_lineage=[],
        synthesis=ResearchSynthesis(
            topic=topic,
            generated_at=now,
            status=status,
            confidence=0.75,
            current_best_direction="Test direction",
            key_findings=["Grounded finding"],
            contested_points=[],
            evidence_gaps=gaps or [],
            clusters=[],
            contradictions=[],
        ),
        evidence=[],
        notes=[],
        source_coverage=[
            ResearchCoverageItem(
                name="grounded_claims",
                status="covered",
                expected=True,
                observed_count=1,
                reason="test",
                next_action="none",
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
