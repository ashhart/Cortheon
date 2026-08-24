import unittest

from cortheon.coverage import analyze_source_coverage
from cortheon.models import (
    ResearchArtifact,
    ResearchClaim,
    ResearchDiscoveryPass,
    ResearchSourceDecision,
    ScholarlyWork,
    SupportLevel,
)


class CoverageTests(unittest.TestCase):
    def test_biomedical_coverage_detects_trials_and_missing_claims(self) -> None:
        coverage = analyze_source_coverage(
            "cancer immunotherapy clinical trial evidence",
            source_plan=[
                source_decision("pubmed", "scholarly"),
                source_decision("clinicaltrials_gov", "trial_registry"),
            ],
            discovery_passes=[
                ResearchDiscoveryPass(
                    query="cancer immunotherapy clinical trial evidence",
                    purpose="primary",
                    source="user_topic",
                    scholarly_work_count=0,
                    search_result_count=0,
                    github_artifact_count=0,
                    seed_count=0,
                    registry_artifact_count=1,
                )
            ],
            scholarly_works=[],
            search_results=[],
            crawled_pages=[],
            artifacts=[
                ResearchArtifact(
                    kind="clinical_trial",
                    title="Cancer Immunotherapy Trial",
                    url="https://clinicaltrials.gov/study/NCT00000001",
                    source_url="https://clinicaltrials.gov/api/v2/studies",
                    provider="clinicaltrials_gov",
                    evidence="registered trial",
                    confidence=0.8,
                )
            ],
            claims=[],
        )
        by_name = {item.name: item for item in coverage}

        self.assertEqual(by_name["clinical_trial_registry"].status, "covered")
        self.assertEqual(by_name["clinical_trial_registry"].observed_count, 1)
        self.assertEqual(by_name["scholarly_literature"].status, "missing")
        self.assertEqual(by_name["grounded_claims"].status, "not_expected")

    def test_grounded_claim_coverage_when_scholarly_claims_exist(self) -> None:
        coverage = analyze_source_coverage(
            "open-ended artificial life benchmark evidence",
            source_plan=[source_decision("arxiv", "scholarly")],
            discovery_passes=[],
            scholarly_works=[
                ScholarlyWork(
                    title="Open-ended evolution benchmark",
                    url="https://arxiv.org/abs/1",
                    abstract="A benchmark.",
                    authors=[],
                    published_at=None,
                    source="arxiv",
                    venue="arXiv",
                    identifiers={"arxiv": "1"},
                    cited_by_count=None,
                    authority_score=0.9,
                )
            ],
            search_results=[],
            crawled_pages=[],
            artifacts=[],
            claims=[
                ResearchClaim(
                    text="A benchmark exists.",
                    source_url="https://arxiv.org/abs/1",
                    source_title="Open-ended evolution benchmark",
                    source_type="paper",
                    support=SupportLevel.OBSERVED,
                    confidence=0.8,
                    source_excerpt="A benchmark exists.",
                    source_char_start=0,
                    source_char_end=19,
                )
            ],
        )
        by_name = {item.name: item for item in coverage}

        self.assertEqual(by_name["scholarly_literature"].status, "covered")
        self.assertEqual(by_name["grounded_claims"].status, "covered")


def source_decision(name: str, source_type: str) -> ResearchSourceDecision:
    return ResearchSourceDecision(
        name=name,
        source_type=source_type,
        selected=True,
        available=True,
        reason="selected",
        capabilities=[],
        trust_tier="test",
        priority=0.8,
        budget=1,
    )


if __name__ == "__main__":
    unittest.main()
