import unittest

from cortheon.lineage import build_source_lineage
from cortheon.models import ResearchClaim, ScholarlyWork, SupportLevel, utc_now


class LineageTests(unittest.TestCase):
    def test_build_source_lineage_maps_claims_to_sources(self) -> None:
        work = ScholarlyWork(
            title="Open-ended evolution",
            url="https://example.org/paper",
            abstract="A paper.",
            authors=[],
            published_at=utc_now(),
            source="fake",
            venue="Example",
            identifiers={},
            cited_by_count=None,
            authority_score=0.88,
            relevance_score=0.91,
        )
        claims = [
            ResearchClaim(
                text="Claim one.",
                source_url=work.url,
                source_title=work.title,
                source_type="scholarly:fake",
                support=SupportLevel.OBSERVED,
                confidence=0.7,
            ),
            ResearchClaim(
                text="Claim two.",
                source_url=work.url,
                source_title=work.title,
                source_type="scholarly:fake",
                support=SupportLevel.OBSERVED,
                confidence=0.8,
            ),
        ]

        lineage = build_source_lineage(claims, [work], [])

        self.assertEqual(len(lineage), 1)
        self.assertEqual(lineage[0].source_title, "Open-ended evolution")
        self.assertEqual(lineage[0].derived_claim_indexes, [0, 1])
        self.assertEqual(lineage[0].relevance_score, 0.91)


if __name__ == "__main__":
    unittest.main()
