"""Unit tests for the Pi adapter's pure evidence functions.

``buildEvidenceLedger`` and ``mapHypothesisEvidence`` are deterministic
TypeScript functions; Node's type-stripping loader executes the real reviewed
sources directly, so these tests exercise the shipped logic rather than a
Python reimplementation.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
LEDGER = ROOT / "src" / "cortheon" / "pi_core" / "evidence_ledger.ts"
MAPPING = ROOT / "src" / "cortheon" / "pi_core" / "evidence_mapping.ts"
MAX_HOST_EVIDENCE_CHARACTERS = 1_800

RUN_SCRIPT = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const module = await import(input.module);
const result =
  input.kind === "ledger"
    ? module.buildEvidenceLedger(input.payload.records)
    : module.mapHypothesisEvidence(input.payload.sections, input.payload.records);
console.log(JSON.stringify({ result: result === undefined ? null : result }));
process.exit(0);
"""


def _evaluate(module: Path, kind: str, payload: dict[str, object]) -> object:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", RUN_SCRIPT],
        input=json.dumps({"module": str(module), "kind": kind, "payload": payload}),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["result"]


def _ledger(records: list[dict[str, str]]) -> str | None:
    value = _evaluate(LEDGER, "ledger", {"records": records})
    assert value is None or isinstance(value, str)
    return value


def _entries(ledger: str) -> list[dict[str, str]]:
    parsed = json.loads(ledger)
    assert isinstance(parsed, list)
    return parsed


class BuildEvidenceLedgerTests(unittest.TestCase):
    def test_six_700_character_facts_fit_and_stay_bounded(self) -> None:
        records = [
            {"source": f"src/file-{index}.txt", "fact": f"fact {index} " + "x" * 690}
            for index in range(6)
        ]
        ledger = _ledger(records)
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertLessEqual(len(ledger), MAX_HOST_EVIDENCE_CHARACTERS)
        entries = _entries(ledger)
        self.assertEqual(len(entries), 6)
        for index, entry in enumerate(entries):
            self.assertEqual(entry["source"], f"src/file-{index}.txt")
            self.assertTrue(entry["fact"].startswith(f"fact {index} x"))

    def test_twelve_200_character_facts_preserve_every_source(self) -> None:
        records = [
            {"source": f"src/file-{index}.txt", "fact": f"fact {index} " + "x" * 190}
            for index in range(12)
        ]
        ledger = _ledger(records)
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertLessEqual(len(ledger), MAX_HOST_EVIDENCE_CHARACTERS)
        entries = _entries(ledger)
        self.assertEqual(len(entries), 12)
        by_source = {entry["source"]: entry["fact"] for entry in entries}
        for index in range(12):
            self.assertIn(f"src/file-{index}.txt", by_source)
            self.assertTrue(by_source[f"src/file-{index}.txt"].startswith(f"fact {index} x"))

    def test_reordered_input_produces_an_identical_ledger(self) -> None:
        records = [
            {"source": "src/b.txt", "fact": "bravo fact"},
            {"source": "src/a.txt", "fact": "alpha fact"},
            {"source": "src/c.txt", "fact": "charlie fact"},
        ]
        first = _ledger(records)
        second = _ledger(list(reversed(records)))
        self.assertEqual(first, second)
        assert first is not None
        sources = [entry["source"] for entry in _entries(first)]
        self.assertEqual(sources, sorted(sources))

    def test_empty_input_yields_an_empty_ledger(self) -> None:
        self.assertEqual(_ledger([]), "")

    def test_empty_trimmed_source_or_fact_fails_closed(self) -> None:
        self.assertIsNone(_ledger([{"source": "   ", "fact": "fact"}]))
        self.assertIsNone(_ledger([{"source": "src/a.txt", "fact": "   "}]))
        self.assertIsNone(
            _ledger(
                [
                    {"source": "src/a.txt", "fact": "real fact"},
                    {"source": "src/b.txt", "fact": "\t "},
                ]
            )
        )

    def test_fail_closed_when_labels_plus_one_code_point_cannot_fit(self) -> None:
        records = [
            {"source": "src/" + "L" * 400 + f"-{index}.txt", "fact": "fact"} for index in range(6)
        ]
        self.assertIsNone(_ledger(records))

    def test_preserves_every_source_when_labels_barely_fit(self) -> None:
        # JSON framing plus 8 medium labels leaves little beyond the
        # reserved first code point of every fact, so no source may be
        # dropped even though the excerpts are tiny.
        records = [
            {"source": "src/" + "L" * 84 + f"-{index}.txt", "fact": f"fact {index}"}
            for index in range(8)
        ]
        ledger = _ledger(records)
        self.assertIsNotNone(ledger)
        assert ledger is not None
        entries = _entries(ledger)
        self.assertEqual(len(entries), 8)
        for index, entry in enumerate(entries):
            self.assertEqual(entry["source"], "src/" + "L" * 84 + f"-{index}.txt")
            self.assertNotEqual(entry["fact"], "")

    def test_emoji_never_split_across_a_surrogate_pair(self) -> None:
        emoji_fact = "🚨" * 300
        ledger = _ledger(
            [
                {"source": "src/emoji.txt", "fact": emoji_fact},
                {"source": "src/plain.txt", "fact": "p" * 600},
            ]
        )
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertLessEqual(len(ledger), MAX_HOST_EVIDENCE_CHARACTERS)
        entries = {entry["source"]: entry["fact"] for entry in _entries(ledger)}
        excerpt = entries["src/emoji.txt"]
        self.assertNotEqual(excerpt, "")
        # A sliced excerpt must contain only whole surrogate pairs.
        for index, char in enumerate(excerpt):
            code = ord(char)
            if 0xD800 <= code <= 0xDBFF:
                self.assertNotEqual(
                    index + 1,
                    len(excerpt),
                    "excerpt ends inside a surrogate pair",
                )

    def test_longer_facts_receive_proportionally_larger_excerpts(self) -> None:
        records = [
            {"source": "src/short.txt", "fact": "s" * 60},
            {"source": "src/long.txt", "fact": "l" * 900},
        ]
        ledger = _ledger(records)
        assert ledger is not None
        entries = {entry["source"]: entry["fact"] for entry in _entries(ledger)}
        self.assertGreater(len(entries["src/long.txt"]), len(entries["src/short.txt"]) * 2)

    def test_labels_are_never_sliced(self) -> None:
        label = "src/" + "L" * 120 + ".txt"
        ledger = _ledger([{"source": label, "fact": "f" * 400}])
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(_entries(ledger)[0]["source"], label)

    def test_single_oversized_fact_is_excerpted_not_dropped(self) -> None:
        ledger = _ledger([{"source": "src/big.txt", "fact": "b" * 5_000}])
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertLessEqual(len(ledger), MAX_HOST_EVIDENCE_CHARACTERS)
        self.assertTrue(_entries(ledger)[0]["fact"].startswith("bbb"))

    def test_skewed_short_and_huge_sources_keep_the_short_excerpt(self) -> None:
        # A fair proportional quota can starve a short source to zero; the
        # code-point reservation must keep every accepted source nonempty.
        records = [
            {"source": "src/short.txt", "fact": "tiny fact"},
            {"source": "src/huge.txt", "fact": "h" * 4_000},
        ]
        ledger = _ledger(records)
        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertLessEqual(len(ledger), MAX_HOST_EVIDENCE_CHARACTERS)
        entries = {entry["source"]: entry["fact"] for entry in _entries(ledger)}
        self.assertEqual(entries["src/short.txt"], "tiny fact")
        self.assertTrue(entries["src/huge.txt"].startswith("hhh"))

    def test_barely_fitting_first_emoji_reserved_exactly(self) -> None:
        # The long label is sized so the leftover budget equals exactly the
        # escaped UTF-16 cost of one first code point per source: 2 units
        # for the emoji, 1 for ascii. The ledger must exist with one whole
        # code point per source; one label unit larger and even that
        # reservation cannot fit, so the adapter must withhold.
        emoji_record = {"source": "src/e.txt", "fact": "🚨" * 50}
        plain_record = {"source": "src/p.txt", "fact": "p" * 50}

        long_source = "src/" + "L" * 1_729 + ".txt"

        def records(extra_label_units: int) -> list[dict[str, str]]:
            stretched = "src/" + "L" * (1_729 + extra_label_units) + ".txt"
            return [
                {"source": stretched, "fact": emoji_record["fact"]},
                {"source": "src/p.txt", "fact": plain_record["fact"]},
            ]

        fitting = _ledger(records(0))
        self.assertIsNotNone(fitting)
        assert fitting is not None
        self.assertLessEqual(len(fitting), MAX_HOST_EVIDENCE_CHARACTERS)
        entries = {entry["source"]: entry["fact"] for entry in _entries(fitting)}
        self.assertEqual(entries[long_source], "🚨")
        self.assertEqual(entries["src/p.txt"], "p")
        self.assertIsNone(_ledger(records(1)))

    def test_unicode_facts_stay_bounded_and_non_empty(self) -> None:
        records = [
            {"source": "src/南京.txt", "fact": "配置键为「琥珀」。" * 40},
            {"source": "src/plain.txt", "fact": "plain ascii fact"},
        ]
        ledger = _ledger(records)
        assert ledger is not None
        self.assertLessEqual(len(ledger), MAX_HOST_EVIDENCE_CHARACTERS)
        entries = {entry["source"]: entry["fact"] for entry in _entries(ledger)}
        self.assertIn("src/南京.txt", entries)
        self.assertTrue(entries["src/南京.txt"].startswith("配置键为「琥珀」"))


SECTIONS = {
    "evidence": "",
    "cause": "The collision occurs because both paths reuse the Northstar key amber.",
    "rival": (
        "Instead, cache compaction is the competing alternative because the "
        "collision persists when compaction is disabled."
    ),
    "test": (
        "Assign distinct keys while holding compaction constant — this "
        "distinguishing test would falsify the wrong mechanism: Cause predicts "
        "the collision disappears whereas Rival predicts the collision remains."
    ),
}


def _mapping(
    records: list[dict[str, str]],
    sections: dict[str, str] | None = None,
) -> dict[str, object] | None:
    value = _evaluate(
        MAPPING,
        "mapping",
        {"sections": sections or SECTIONS, "records": records},
    )
    assert value is None or isinstance(value, dict)
    return value


class MapHypothesisEvidenceTests(unittest.TestCase):
    def test_cause_binds_to_supporting_records_and_rival_to_direct_contradiction(self):
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar path A uses collision key amber.",
            },
            {
                "id": "ev2",
                "source": "pi:read:facts/b.txt",
                "fact": (
                    "Path B reuses key amber. Collision persists when compaction is disabled."
                ),
            },
        ]
        mapping = _mapping(records)
        self.assertIsNotNone(mapping)
        assert mapping is not None
        self.assertEqual(mapping["causeStatus"], "supported")
        self.assertEqual(mapping["causeIds"], ["ev1", "ev2"])
        self.assertEqual(mapping["rivalStatus"], "refuted")
        self.assertEqual(mapping["rivalIds"], ["ev2"])
        self.assertEqual(mapping["cleanIds"], ["ev1", "ev2"])

    def test_counterexample_binds_only_that_evidence_id_to_refuted_rival(self):
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar path A uses collision key amber.",
            },
            {
                "id": "ev2",
                "source": "pi:read:facts/b.txt",
                "fact": "Path B reuses key amber and mentions compaction nightly.",
            },
            {
                "id": "ev3",
                "source": "pi:read:facts/c.txt",
                "fact": "The collision persists when compaction is disabled.",
            },
        ]
        mapping = _mapping(records)
        assert mapping is not None
        self.assertEqual(mapping["rivalStatus"], "refuted")
        self.assertEqual(mapping["rivalIds"], ["ev3"])

    def test_neutral_mention_becomes_uncertain_bearing_not_contradiction(self):
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar path A uses collision key amber.",
            },
            {
                "id": "ev2",
                "source": "pi:read:facts/b.txt",
                "fact": "Path B reuses key amber. Compaction is scheduled nightly.",
            },
        ]
        mapping = _mapping(records)
        self.assertIsNotNone(mapping)
        assert mapping is not None
        self.assertEqual(mapping["rivalStatus"], "uncertain")
        self.assertEqual(mapping["rivalIds"], ["ev2"])
        # ev2 shares only "amber" with the Cause, and ev1 alone covers just
        # half the Cause's distinctive anchors: one anchor is not support,
        # and one partial source is not credible aggregate coverage.
        self.assertEqual(mapping["causeIds"], [])

    def test_zero_match_grep_never_becomes_contradiction(self):
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar path A uses collision key amber.",
            },
            {
                "id": "ev2",
                "source": "pi:find:facts/",
                "fact": "No matches for pattern 'compaction disabled'.",
            },
        ]
        mapping = _mapping(records)
        assert mapping is not None
        self.assertEqual(mapping["rivalStatus"], "uncertain")

    def test_disabled_other_mechanism_never_refutes_the_rival(self) -> None:
        # Exact negative control: the persistence and disabled predicates are
        # both present, but they belong to logging, not the compaction rival;
        # the rival is merely called unrelated in a separate clause.
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar path A uses collision key amber.",
            },
            {
                "id": "ev2",
                "source": "pi:read:facts/b.txt",
                "fact": (
                    "The collision persists after logging was disabled; compaction is unrelated."
                ),
            },
        ]
        mapping = _mapping(records)
        assert mapping is not None
        self.assertEqual(mapping["rivalStatus"], "uncertain")
        self.assertEqual(mapping["rivalIds"], ["ev2"])

    def test_mechanism_disabled_without_outcome_stays_uncertain(self) -> None:
        # Positive paraphrase guard: the rival mechanism itself is disabled,
        # but the Cause's outcome is not observed to persist, so the record
        # bears on the rival without refuting it.
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar path A uses collision key amber.",
            },
            {
                "id": "ev2",
                "source": "pi:read:facts/b.txt",
                "fact": "Compaction is disabled on the staging node.",
            },
        ]
        mapping = _mapping(records)
        assert mapping is not None
        self.assertEqual(mapping["rivalStatus"], "uncertain")

    def test_records_without_runtime_ids_cannot_be_bound(self):
        records = [
            {"source": "pi:read:facts/a.txt", "fact": "Northstar key amber."},
        ]
        self.assertIsNone(_mapping(records))

    def test_no_cause_support_returns_uncertain_cause_with_no_ids(self):
        records = [
            {
                "id": "ev1",
                "source": "pi:read:facts/z.txt",
                "fact": "Completely unrelated weather report for the harbor.",
            },
        ]
        mapping = _mapping(records)
        self.assertIsNotNone(mapping)
        assert mapping is not None
        # The Cause stays uncertain with an honest empty binding — never a
        # fallback to unrelated clean ids — so /v1/complete is still called
        # (claims and completion keep cleanIds) and the runtime withholds
        # for a grounded, actionable reason.
        self.assertEqual(mapping["causeStatus"], "uncertain")
        self.assertEqual(mapping["causeIds"], [])
        self.assertEqual(mapping["rivalIds"], [])
        self.assertEqual(mapping["cleanIds"], ["ev1"])

    def test_cause_corroboration_requires_distinct_normalized_sources(self) -> None:
        # Duplicate records from one document — different ids, same source,
        # including case- and whitespace-varying labels of it — must not
        # masquerade as independent cross-source corroboration, and neither
        # record alone covers a strong majority of the Cause's anchors. The
        # identical facts from two genuinely distinct sources do support.
        sections = {
            **SECTIONS,
            "cause": (
                "The Northstar shard rotation leads to collision because writers reuse amber slots."
            ),
        }
        cases = [
            ("same source", "pi:read:facts/a.txt", "pi:read:facts/a.txt", False),
            (
                "case-insensitive prefix aliases",
                "PI:READ:facts/a.txt",
                "pi:read:facts/a.txt",
                False,
            ),
            (
                # Byte preservation: padded spellings are non-Pi sources and
                # stay byte-distinct identities, never collapsed.
                "padded spellings stay byte-distinct",
                "PI:READ:facts/a.txt",
                " pi:read:facts/a.txt ",
                True,
            ),
            ("dot-slash alias", "pi:read:./facts/a.txt", "pi:grep:facts/a.txt", False),
            (
                "parent-hop alias",
                "pi:grep:facts/../facts/a.txt",
                "pi:read:facts/a.txt",
                False,
            ),
            ("case-distinct paths", "pi:read:facts/A.txt", "pi:read:facts/a.txt", True),
            ("distinct sources", "pi:read:facts/a.txt", "pi:read:facts/b.txt", True),
        ]
        for label, source_a, source_b, supported in cases:
            with self.subTest(label):
                mapping = _mapping(
                    [
                        {
                            "id": "ev1",
                            "source": source_a,
                            "fact": "Northstar shard rotation config.",
                        },
                        {"id": "ev2", "source": source_b, "fact": "Writers reuse amber slots."},
                    ],
                    sections,
                )
                assert mapping is not None
                self.assertEqual(mapping["causeStatus"], "supported" if supported else "uncertain")
                self.assertEqual(mapping["causeIds"], ["ev1", "ev2"] if supported else [])
                self.assertEqual(mapping["cleanIds"], ["ev1", "ev2"])


if __name__ == "__main__":
    unittest.main()
