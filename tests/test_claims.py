import unittest

from cortheon.claims import extract_claims
from cortheon.models import ScholarlyWork, utc_now


class ClaimsTests(unittest.TestCase):
    def test_extracts_topic_relevant_claim_from_work(self) -> None:
        work = ScholarlyWork(
            title="Open-ended artificial life benchmark",
            url="https://arxiv.org/abs/1",
            abstract=(
                "We introduce an open-ended artificial life benchmark that evaluates "
                "whether evolving agents continue producing novel adaptive behavior."
            ),
            authors=["A. Researcher"],
            published_at=utc_now(),
            source="arxiv",
            venue="arXiv",
            identifiers={"arxiv": "1"},
            cited_by_count=None,
            authority_score=0.9,
        )

        claims = extract_claims("open-ended artificial life", [work], [])

        self.assertEqual(len(claims), 1)
        self.assertIn("open-ended artificial life benchmark", claims[0].text)
        self.assertGreater(claims[0].confidence, 0.6)
        self.assertIsNotNone(claims[0].source_excerpt)
        self.assertIsNotNone(claims[0].source_char_start)
        self.assertIsNotNone(claims[0].source_char_end)
        self.assertIn(claims[0].text, claims[0].source_excerpt)


if __name__ == "__main__":
    unittest.main()
