import unittest

from cortheon.models import ResearchClaim, SupportLevel
from cortheon.synthesis import classify_stance, synthesize_research


class SynthesisTests(unittest.TestCase):
    def test_synthesis_clusters_claims_and_selects_direction(self) -> None:
        claims = [
            claim(
                "We introduce an open-ended artificial life benchmark that evaluates novelty in evolving agents.",
                "https://example.org/a",
                0.82,
            ),
            claim(
                "The benchmark shows open-ended evolution can be studied through novelty search and diversity metrics.",
                "https://example.org/b",
                0.78,
            ),
            claim(
                "Open-ended artificial life remains difficult because objective functions often limit novelty.",
                "https://example.org/c",
                0.74,
            ),
        ]

        synthesis = synthesize_research("open-ended artificial life benchmark", claims)

        self.assertTrue(synthesis.clusters)
        self.assertIn(
            synthesis.status, {"contested", "emerging_consensus", "promising_but_incomplete"}
        )
        self.assertTrue(synthesis.key_findings)
        self.assertIn("open-ended artificial life", synthesis.current_best_direction)

    def test_stance_classifier_marks_challenges(self) -> None:
        self.assertEqual(classify_stance("This approach is limited and cannot scale."), "challenge")
        self.assertEqual(
            classify_stance("We demonstrate that this method improves results."), "support"
        )

    def test_synthesis_reports_explicit_contradictions(self) -> None:
        claims = [
            claim(
                "We demonstrate that novelty search supports open-ended evolution in artificial life.",
                "https://example.org/support",
                0.82,
                stance="support",
            ),
            claim(
                "Novelty search is limited and cannot produce sustained open-ended evolution in artificial life.",
                "https://example.org/challenge",
                0.8,
                stance="challenge",
            ),
        ]

        synthesis = synthesize_research("open-ended artificial life novelty search", claims)

        self.assertEqual(synthesis.status, "contested")
        self.assertTrue(synthesis.contradictions)
        self.assertEqual(synthesis.contradictions[0].support_claim_indexes, [0])
        self.assertEqual(synthesis.contradictions[0].challenge_claim_indexes, [1])

    def test_corroboration_rewards_independent_sources_over_claim_volume(self) -> None:
        # One cluster is echoed by three independent sources; the other repeats
        # from a single source. Equal per-claim confidence, so only corroboration
        # can separate them — the corroborated cluster must rank first and score higher.
        multi = [
            claim(
                "Novelty search sustains behavioral diversity across generations.",
                "https://src.example/a1",
                0.8,
            ),
            claim(
                "Behavioral diversity from novelty search improves exploration.",
                "https://src.example/a2",
                0.8,
            ),
            claim(
                "Novelty search increases diversity metrics in evolving populations.",
                "https://src.example/a3",
                0.8,
            ),
        ]
        single = [
            claim(
                "Genotype encoding choice shapes evolvability of representations.",
                "https://one.example/b",
                0.8,
            ),
            claim(
                "Representation encoding constrains evolvability in genotype space.",
                "https://one.example/b",
                0.8,
            ),
            claim(
                "Evolvability depends on genotype encoding and representation design.",
                "https://one.example/b",
                0.8,
            ),
        ]

        synthesis = synthesize_research("artificial life research directions", multi + single)

        corroborated = [
            cluster for cluster in synthesis.clusters if cluster.independent_source_count >= 3
        ]
        lone = [cluster for cluster in synthesis.clusters if cluster.independent_source_count == 1]
        self.assertTrue(corroborated, "expected a cluster backed by 3 independent sources")
        self.assertTrue(lone, "expected a single-source cluster")
        self.assertEqual(corroborated[0].corroboration, 0.667)
        self.assertEqual(lone[0].corroboration, 0.0)
        self.assertGreater(corroborated[0].support_score, lone[0].support_score)
        # Corroborated cluster wins the ranking despite identical per-claim confidence.
        self.assertGreaterEqual(synthesis.clusters[0].independent_source_count, 3)
        self.assertTrue(
            any(finding.startswith("Corroborated") for finding in synthesis.key_findings)
        )

    def test_single_source_claims_are_never_corroborated(self) -> None:
        claims = [
            claim(
                "Open-ended artificial life benchmarks measure sustained novelty.",
                "https://one.example/paper",
                0.85,
            ),
            claim(
                "Sustained novelty in open-ended artificial life needs diversity pressure.",
                "https://one.example/paper",
                0.85,
            ),
            claim(
                "Open-ended artificial life benchmarks track novelty over generations.",
                "https://one.example/paper",
                0.85,
            ),
        ]

        synthesis = synthesize_research("open-ended artificial life benchmark", claims)

        self.assertTrue(
            all(cluster.independent_source_count == 1 for cluster in synthesis.clusters)
        )
        self.assertTrue(all(cluster.corroboration == 0.0 for cluster in synthesis.clusters))
        self.assertNotEqual(synthesis.status, "emerging_consensus")
        self.assertFalse(
            any(finding.startswith("Corroborated") for finding in synthesis.key_findings)
        )

    def test_senolytic_cure_engine_direction_stays_evidence_first(self) -> None:
        claims = [
            claim(
                "Here, we report the discovery of three senolytics using cost-effective machine learning algorithms trained solely on published data.",
                "https://pubmed.ncbi.nlm.nih.gov/37301862/",
                0.8,
                stance="support",
            ),
            claim(
                "In the first clinical trial of senolytics, dasatinib and quercetin improved physical function in a senescence-associated disease.",
                "https://pubmed.ncbi.nlm.nih.gov/31542391/",
                0.8,
                stance="support",
            ),
            claim(
                "The first senolytic drugs dasatinib, quercetin, fisetin and navitoclax were discovered using a hypothesis-driven approach.",
                "https://pubmed.ncbi.nlm.nih.gov/32686219/",
                0.78,
            ),
        ]

        synthesis = synthesize_research(
            "ALIFE-style cure engine that can discover senolytic therapies architecture",
            claims,
        )

        self.assertIn("evidence-first discovery substrate", synthesis.current_best_direction)
        self.assertIn(
            "Do not claim the ALIFE/architecture layer is selected yet",
            synthesis.current_best_direction,
        )


def claim(text: str, url: str, confidence: float, stance: str = "neutral") -> ResearchClaim:
    return ResearchClaim(
        text=text,
        source_url=url,
        source_title="Test",
        source_type="paper",
        support=SupportLevel.OBSERVED,
        confidence=confidence,
        stance=stance,
    )


if __name__ == "__main__":
    unittest.main()
