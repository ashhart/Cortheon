import unittest
from datetime import timedelta

from cortheon.artifact_assessment import assess_artifacts
from cortheon.models import ResearchArtifact, utc_now


class ArtifactAssessmentTests(unittest.TestCase):
    def test_inspected_code_repository_can_rank_build_first(self) -> None:
        artifact = ResearchArtifact(
            kind="code_repository",
            title="fastapi/fastapi",
            url="https://github.com/fastapi/fastapi",
            source_url="https://api.github.com/search/repositories?q=fastapi",
            provider="github_search",
            evidence="GitHub search found repository fastapi/fastapi.",
            confidence=0.93,
            metadata={
                "repo": "fastapi/fastapi",
                "repository_health_score": "0.96",
                "primary_language": "Python",
                "primary_language_share": "0.98",
                "implementation_signals": (
                    "python_package,ci_config,tests,docs,license_file,install_docs,usage_docs"
                ),
                "license_spdx": "MIT",
                "archived": "false",
                "description": "FastAPI framework, high performance, ready for production.",
            },
        )

        assessment = assess_artifacts("fastapi python rest api framework", [artifact])[0]

        self.assertEqual(assessment.decision, "build_from_first")
        self.assertGreaterEqual(assessment.score, 0.82)
        self.assertFalse(any("not been inspected" in risk for risk in assessment.risks))
        self.assertTrue(any("Buildable project signals" in reason for reason in assessment.reasons))

    def test_recency_lifts_freshly_pushed_repository_over_stale_one(self) -> None:
        # Same repository health, differing only in push recency. Relative dates
        # keep the fresh/stale split true regardless of when the test runs.
        base = {
            "repo": "acme/engine",
            "repository_health_score": "0.9",
            "primary_language": "Python",
            "implementation_signals": "python_package,ci_config,tests,docs,usage_docs",
            "license_spdx": "MIT",
            "archived": "false",
        }
        fresh = code_artifact({**base, "pushed_at": (utc_now() - timedelta(days=5)).isoformat()})
        stale = code_artifact({**base, "pushed_at": (utc_now() - timedelta(days=1200)).isoformat()})

        fresh_assessment = assess_artifacts("acme engine", [fresh])[0]
        stale_assessment = assess_artifacts("acme engine", [stale])[0]

        self.assertGreater(fresh_assessment.score, stale_assessment.score)
        self.assertTrue(any("last pushed" in reason for reason in fresh_assessment.reasons))
        self.assertTrue(
            any("not been pushed to recently" in risk for risk in stale_assessment.risks)
        )
        self.assertFalse(
            any("not been pushed to recently" in risk for risk in fresh_assessment.risks)
        )

    def test_uninspected_code_repository_requires_more_inspection(self) -> None:
        artifact = ResearchArtifact(
            kind="code_repository",
            title="example/weak",
            url="https://github.com/example/weak",
            source_url=None,
            provider="github_search",
            evidence=None,
            confidence=0.62,
            metadata={"repo": "example/weak"},
        )

        assessment = assess_artifacts("open-ended artificial life", [artifact])[0]

        self.assertIn(assessment.decision, {"inspect_more", "background_reference"})
        self.assertTrue(any("not been inspected" in risk for risk in assessment.risks))

    def test_scholarly_artifact_is_read_first(self) -> None:
        artifact = ResearchArtifact(
            kind="paper_pdf",
            title="Open-ended artificial evolution",
            url="https://arxiv.org/pdf/1234.56789",
            source_url="https://arxiv.org/abs/1234.56789",
            provider="scholarly:arxiv",
            evidence="Derived arXiv PDF.",
            confidence=0.9,
            metadata={},
        )

        assessment = assess_artifacts("open-ended artificial life", [artifact])[0]

        self.assertEqual(assessment.decision, "read_first")


def code_artifact(metadata: dict) -> ResearchArtifact:
    return ResearchArtifact(
        kind="code_repository",
        title=metadata.get("repo", "acme/engine"),
        url=f"https://github.com/{metadata.get('repo', 'acme/engine')}",
        source_url="https://api.github.com/search/repositories",
        provider="github_search",
        evidence="GitHub search found repository.",
        confidence=0.9,
        metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
