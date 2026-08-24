import unittest

from cortheon.option_ranker import (
    ECOSYSTEM_MAPPINGS,
    discover_candidates,
    keyword_extract_candidates,
)


class OptionRankerTests(unittest.TestCase):
    def test_discover_candidates_from_profile(self) -> None:
        candidates = discover_candidates("build a REST API for a Python service")
        packages = [pkg for pkg, _ in candidates]
        self.assertIn("fastapi", packages)
        self.assertIn("litestar", packages)
        self.assertIn("flask", packages)

    def test_discover_candidates_from_ecosystem_mapping(self) -> None:
        candidates = discover_candidates("I need a GraphQL server for my Python backend")
        packages = [pkg for pkg, _ in candidates]
        self.assertIn("strawberry", packages)
        self.assertIn("ariadne", packages)
        self.assertIn("graphene", packages)

    def test_discover_candidates_mobile(self) -> None:
        candidates = discover_candidates(
            "build a cross-platform mobile app with a reactive framework"
        )
        packages = [pkg for pkg, _ in candidates]
        self.assertIn("kivy", packages)
        self.assertIn("flutter", packages)

    def test_discover_candidates_no_match_returns_empty(self) -> None:
        candidates = discover_candidates("do something completely unrelated to software")
        self.assertEqual(candidates, [])

    def test_keyword_extract_candidates_quoted(self) -> None:
        candidates = keyword_extract_candidates('use "fastapi" for the API')
        packages = [pkg for pkg, _ in candidates]
        self.assertIn("fastapi", packages)

    def test_keyword_extract_candidates_use_pattern(self) -> None:
        candidates = keyword_extract_candidates("use httpx for HTTP requests")
        packages = [pkg for pkg, _ in candidates]
        self.assertIn("httpx", packages)

    def test_keyword_extract_candidates_no_false_positives(self) -> None:
        # "flutter" in "build a mobile app with flutter" should be extracted
        # because it follows the "with" pattern.
        candidates = keyword_extract_candidates("build a mobile app with flutter")
        packages = [pkg for pkg, _ in candidates]
        self.assertIn("flutter", packages)

    def test_keyword_extract_candidates_no_arbitrary_words(self) -> None:
        # Words not in explicit patterns should not be extracted.
        candidates = keyword_extract_candidates("build a mobile app for flutter development")
        packages = [pkg for pkg, _ in candidates]
        self.assertNotIn("flutter", packages)

    def test_ecosystem_mappings_cover_common_domains(self) -> None:
        # Ensure key domains are covered.
        keywords = set()
        for domain in ECOSYSTEM_MAPPINGS:
            keywords.update(ECOSYSTEM_MAPPINGS[domain])
        # At least the major frameworks should be mapped.
        all_packages = {
            pkg for domain in ECOSYSTEM_MAPPINGS for pkg, _ in ECOSYSTEM_MAPPINGS[domain]
        }
        self.assertIn("fastapi", all_packages)
        self.assertIn("httpx", all_packages)
        self.assertIn("pydantic", all_packages)

    def test_rank_options_returns_ranked_list(self) -> None:
        from cortheon.engine import CortheonEngine

        engine = CortheonEngine()
        report = engine.recommend(
            "I need a GraphQL server for my Python backend", write_report=False
        )
        self.assertIsNotNone(report.winner)
        self.assertGreater(len(report.candidates), 0)
        # Winner should be the top-scoring candidate.
        if report.candidates and report.candidates[0].score:
            self.assertEqual(report.winner, report.candidates[0].package)

    def test_rank_options_no_profile_uses_ranker(self) -> None:
        from cortheon.engine import CortheonEngine

        engine = CortheonEngine()
        # "GraphQL server" doesn't match a built-in profile but matches ecosystem mapping.
        report = engine.recommend(
            "I need a GraphQL server for my Python backend", write_report=False
        )
        self.assertIsNone(report.profile)
        self.assertIsNotNone(report.winner)
        self.assertGreater(len(report.candidates), 0)

    def test_rank_options_preserves_profile_path(self) -> None:
        from cortheon.engine import CortheonEngine

        engine = CortheonEngine()
        report = engine.recommend("build a REST API for a Python service", write_report=False)
        self.assertEqual(report.profile, "python_rest_api")
        self.assertEqual(report.winner, "fastapi")


if __name__ == "__main__":
    unittest.main()
