import base64
import unittest

from cortheon.connectors.github import GitHubRepositorySearch, find_github_url, parse_owner_repo
from cortheon.models import ResearchArtifact


class GitHubParsingTests(unittest.TestCase):
    def test_parse_owner_repo(self) -> None:
        self.assertEqual(
            parse_owner_repo("https://github.com/encode/httpx"),
            ("encode", "httpx"),
        )

    def test_find_github_url_prefers_source(self) -> None:
        url = find_github_url(
            {
                "Homepage": "https://example.com",
                "Source": "https://github.com/fastapi/fastapi",
            }
        )

        self.assertEqual(url, "https://github.com/fastapi/fastapi")

    def test_repository_search_returns_code_artifacts(self) -> None:
        client = FakeSearchClient()
        artifacts, evidence, errors = GitHubRepositorySearch(client=client).search(
            "open-ended artificial life",
            limit=1,
        )

        self.assertFalse(errors)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].kind, "code_repository")
        self.assertEqual(artifacts[0].url, "https://github.com/example/alife")
        self.assertEqual(artifacts[0].metadata["stars"], "1234")
        self.assertEqual(evidence[0].source_type, "github_repository_search")
        self.assertIn("open-ended+artificial+life", client.url)

    def test_repository_artifact_inspection_enriches_metadata(self) -> None:
        artifact = ResearchArtifact(
            kind="code_repository",
            title="example/alife",
            url="https://github.com/example/alife",
            source_url="https://api.github.com/search/repositories?q=alife",
            provider="github_search",
            evidence="GitHub search found repository example/alife.",
            confidence=0.82,
            metadata={"repo": "example/alife"},
        )

        artifacts, evidence, errors = GitHubRepositorySearch(
            client=FakeInspectionClient()
        ).inspect_artifacts(
            [artifact],
            limit=1,
        )

        self.assertFalse(errors)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source_type, "github_artifact_inspection")
        self.assertEqual(artifacts[0].metadata["primary_language"], "Python")
        self.assertEqual(artifacts[0].metadata["license_spdx"], "MIT")
        self.assertIn("python_package", artifacts[0].metadata["implementation_signals"])
        self.assertIn("tests", artifacts[0].metadata["implementation_signals"])
        self.assertIn("usage_docs", artifacts[0].metadata["implementation_signals"])
        self.assertIn("repository_health_score", artifacts[0].metadata)
        self.assertGreaterEqual(artifacts[0].confidence, artifact.confidence)


class FakeSearchClient:
    def __init__(self) -> None:
        self.url = ""

    def get_json(self, url, headers=None):
        self.url = url
        return {
            "incomplete_results": False,
            "items": [
                {
                    "full_name": "example/alife",
                    "html_url": "https://github.com/example/alife",
                    "description": "Open-ended artificial life benchmark code",
                    "stargazers_count": 1234,
                    "forks_count": 56,
                    "archived": False,
                    "language": "Python",
                    "pushed_at": "2026-01-02T03:04:05Z",
                }
            ],
        }


class FakeInspectionClient:
    def get_json(self, url, headers=None):
        if url.endswith("/languages"):
            return {"Python": 900, "Shell": 100}
        if url.endswith("/contents"):
            return [
                {"name": "pyproject.toml", "type": "file"},
                {"name": "README.md", "type": "file"},
                {"name": "LICENSE", "type": "file"},
                {"name": "tests", "type": "dir"},
                {"name": ".github", "type": "dir"},
            ]
        if url.endswith("/readme"):
            content = base64.b64encode(
                b"# ALIFE\n\nInstallation: pip install alife\n\nUsage example and benchmark notes."
            ).decode("ascii")
            return {"content": content, "encoding": "base64"}
        return {
            "full_name": "example/alife",
            "description": "Open-ended artificial life benchmark code",
            "stargazers_count": 1234,
            "forks_count": 56,
            "open_issues_count": 4,
            "default_branch": "main",
            "pushed_at": "2026-01-02T03:04:05Z",
            "archived": False,
            "license": {"spdx_id": "MIT"},
            "topics": ["artificial-life", "benchmark"],
        }


if __name__ == "__main__":
    unittest.main()
