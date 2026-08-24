import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from cortheon.models import ScholarlyWork
from cortheon.scholarly import (
    ArxivConnector,
    CompositeScholarlyDiscovery,
    PubMedConnector,
    ScholarlyDiscoveryResult,
    abstract_from_inverted_index,
    bounded_xml_root,
    scholarly_rank_key,
    score_work_recency,
    score_work_relevance,
    work_recency_score,
)


class ScholarlyTests(unittest.TestCase):
    def test_xml_parser_rejects_dtd_and_oversized_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "DTD"):
            bounded_xml_root(b'<!DOCTYPE root [<!ENTITY x "secret">]><root>&x;</root>')
        with self.assertRaisesRegex(ValueError, "exceeds"):
            bounded_xml_root(b"<root>" + b"x" * 5_000_000 + b"</root>")

    def test_openalex_abstract_from_inverted_index(self) -> None:
        abstract = abstract_from_inverted_index({"open": [0], "ended": [1], "evolution": [2]})

        self.assertEqual(abstract, "open ended evolution")

    def test_arxiv_connector_parses_atom_entries(self) -> None:
        connector = ArxivConnector(client=FakeAtomClient())

        result = connector.search("open ended evolution", 1)

        self.assertEqual(len(result.works), 1)
        self.assertEqual(result.works[0].title, "Open-Ended Evolution Test")
        self.assertEqual(result.works[0].authors, ["Ada Lovelace"])
        self.assertEqual(result.works[0].identifiers["arxiv"], "2601.00001v1")
        self.assertEqual(result.works[0].source, "arxiv")

    def test_pubmed_connector_parses_article_xml(self) -> None:
        connector = PubMedConnector(client=FakePubMedClient())

        result = connector.search("cancer immunotherapy trial", 1)

        self.assertEqual(len(result.works), 1)
        work = result.works[0]
        self.assertEqual(work.title, "Cancer Immunotherapy Trial")
        self.assertEqual(work.authors, ["Ada Lovelace", "Clinical Trial Group"])
        self.assertEqual(work.identifiers["pubmed"], "12345")
        self.assertEqual(work.identifiers["doi"], "10.1000/test")
        self.assertEqual(work.source, "pubmed")
        self.assertEqual(work.venue, "Journal of Test Medicine")
        self.assertIn("immunotherapy improved response", work.abstract or "")
        self.assertEqual(work.published_at.year, 2026)
        self.assertEqual(result.evidence[0].source_type, "pubmed_efetch")

    def test_relevance_beats_off_topic_authority(self) -> None:
        relevant = work(
            "Open-ended evolution in artificial life systems",
            "We introduce artificial life systems for open-ended evolution and novelty.",
            authority=0.72,
        )
        irrelevant = work(
            "The Multimodal Brain Tumor Image Segmentation Benchmark",
            "We report a benchmark for medical image segmentation.",
            authority=0.96,
        )

        discovery = CompositeScholarlyDiscovery(
            connectors=[FakeScholarlyConnector([irrelevant, relevant])]
        )
        result = discovery.search("open-ended artificial life evolution benchmark", 2)

        self.assertEqual(len(result.works), 1)
        self.assertEqual(result.works[0].title, relevant.title)
        self.assertGreater(result.works[0].relevance_score, 0.5)

    def test_recency_is_a_first_class_ranking_signal(self) -> None:
        # Equal relevance and a stale paper with *higher* authority: recency must
        # be what breaks the tie, so fresh-but-correct beats popular-but-stale.
        now = datetime(2026, 7, 1, tzinfo=UTC)
        stale = dated_work(
            "Seminal but stale", authority=0.95, published_at=datetime(2019, 1, 1, tzinfo=UTC)
        )
        fresh = dated_work(
            "Fresh state of the art", authority=0.80, published_at=datetime(2026, 5, 1, tzinfo=UTC)
        )
        stale.relevance_score = fresh.relevance_score = 0.6
        score_work_recency(stale, now)
        score_work_recency(fresh, now)

        self.assertEqual(fresh.recency_score, 0.98)
        self.assertEqual(stale.recency_score, 0.25)
        self.assertGreater(scholarly_rank_key(fresh), scholarly_rank_key(stale))

    def test_recency_score_decays_with_age_and_defaults_neutral(self) -> None:
        now = datetime(2026, 7, 1, tzinfo=UTC)
        self.assertEqual(work_recency_score(datetime(2026, 6, 1, tzinfo=UTC), now), 0.98)
        self.assertEqual(work_recency_score(datetime(2025, 1, 1, tzinfo=UTC), now), 0.75)
        self.assertEqual(work_recency_score(datetime(2010, 1, 1, tzinfo=UTC), now), 0.25)
        self.assertEqual(work_recency_score(None, now), 0.5)

    def test_pipeline_applies_recency_to_returned_works(self) -> None:
        dated = dated_work(
            "Open-ended artificial life evolution benchmark",
            authority=0.8,
            published_at=datetime(2026, 5, 1, tzinfo=UTC),
            abstract="A benchmark for open-ended artificial life evolution and novelty.",
        )
        discovery = CompositeScholarlyDiscovery(connectors=[FakeScholarlyConnector([dated])])

        result = discovery.search("open-ended artificial life evolution benchmark", 1)

        self.assertEqual(len(result.works), 1)
        self.assertGreater(result.works[0].recency_score, 0.0)

    def test_composite_caches_repeated_connector_variant_calls(self) -> None:
        connector = CountingScholarlyConnector(
            [
                work(
                    "Open-ended artificial life benchmark",
                    "A benchmark for open-ended artificial life evolution.",
                    authority=0.8,
                )
            ]
        )
        discovery = CompositeScholarlyDiscovery(connectors=[connector])

        discovery.search("open-ended artificial life evolution benchmark", 1)
        discovery.search("open-ended artificial life evolution benchmark", 1)

        self.assertEqual(connector.calls, 3)

    def test_composite_uses_selected_connectors_only(self) -> None:
        selected = CountingScholarlyConnector(
            [
                work(
                    "Open-ended artificial life benchmark",
                    "A benchmark for open-ended artificial life evolution.",
                    authority=0.8,
                )
            ],
            name="selected",
        )
        skipped = CountingScholarlyConnector(
            [
                work(
                    "Unrelated segmentation benchmark",
                    "A medical image segmentation benchmark.",
                    authority=0.9,
                )
            ],
            name="skipped",
        )
        discovery = CompositeScholarlyDiscovery(connectors=[selected, skipped])

        result = discovery.search(
            "open-ended artificial life evolution benchmark",
            2,
            connector_names=["selected"],
        )

        self.assertEqual(selected.calls, 3)
        self.assertEqual(skipped.calls, 0)
        self.assertEqual(result.evidence[-1].details["selected_connectors"], ["selected"])


@dataclass
class FakeResponse:
    body: bytes


class FakeAtomClient:
    def get(self, url, headers=None):
        return FakeResponse(
            b"""<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
              <entry>
                <id>https://arxiv.org/abs/2601.00001v1</id>
                <updated>2026-01-01T00:00:00Z</updated>
                <published>2026-01-01T00:00:00Z</published>
                <title>Open-Ended Evolution Test</title>
                <summary>We present a benchmark for open-ended artificial life systems.</summary>
                <author><name>Ada Lovelace</name></author>
                <arxiv:primary_category term="cs.AI" />
              </entry>
            </feed>"""
        )


class FakePubMedClient:
    def get_json(self, url, headers=None):
        return {"esearchresult": {"idlist": ["12345"]}}

    def get(self, url, headers=None):
        return FakeResponse(
            b"""<?xml version="1.0" encoding="UTF-8"?>
            <PubmedArticleSet>
              <PubmedArticle>
                <MedlineCitation>
                  <PMID>12345</PMID>
                  <Article>
                    <Journal>
                      <JournalIssue>
                        <PubDate>
                          <Year>2026</Year>
                          <Month>Jun</Month>
                          <Day>15</Day>
                        </PubDate>
                      </JournalIssue>
                      <Title>Journal of Test Medicine</Title>
                    </Journal>
                    <ArticleTitle>Cancer <i>Immunotherapy</i> Trial</ArticleTitle>
                    <Abstract>
                      <AbstractText Label="BACKGROUND">We studied cancer immunotherapy.</AbstractText>
                      <AbstractText Label="RESULTS">The immunotherapy improved response.</AbstractText>
                    </Abstract>
                    <AuthorList>
                      <Author>
                        <ForeName>Ada</ForeName>
                        <LastName>Lovelace</LastName>
                      </Author>
                      <Author>
                        <CollectiveName>Clinical Trial Group</CollectiveName>
                      </Author>
                    </AuthorList>
                  </Article>
                </MedlineCitation>
                <PubmedData>
                  <ArticleIdList>
                    <ArticleId IdType="doi">10.1000/test</ArticleId>
                  </ArticleIdList>
                </PubmedData>
              </PubmedArticle>
            </PubmedArticleSet>"""
        )


class FakeScholarlyConnector:
    name = "fake"

    def __init__(self, works, name="fake"):
        self.works = works
        self.name = name

    def search(self, query, limit):
        return ScholarlyDiscoveryResult(
            works=[score_work_relevance(item, query) for item in self.works],
            evidence=[],
            errors=[],
        )


class CountingScholarlyConnector(FakeScholarlyConnector):
    def __init__(self, works, name="fake"):
        super().__init__(works, name=name)
        self.calls = 0

    def search(self, query, limit):
        self.calls += 1
        return super().search(query, limit)


def work(title: str, abstract: str, authority: float) -> ScholarlyWork:
    return ScholarlyWork(
        title=title,
        url=f"https://example.org/{title}",
        abstract=abstract,
        authors=[],
        published_at=None,
        source="fake",
        venue=None,
        identifiers={},
        cited_by_count=None,
        authority_score=authority,
    )


def dated_work(title: str, authority: float, published_at, abstract: str = "") -> ScholarlyWork:
    return ScholarlyWork(
        title=title,
        url=f"https://example.org/{title}",
        abstract=abstract or None,
        authors=[],
        published_at=published_at,
        source="fake",
        venue=None,
        identifiers={},
        cited_by_count=None,
        authority_score=authority,
    )


if __name__ == "__main__":
    unittest.main()
