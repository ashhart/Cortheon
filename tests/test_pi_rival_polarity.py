"""Polarity-safe rival refutation probes for mapHypothesisEvidence.

The rival may be bound to refuted only by a grammatical, affirmative, local
relation between the rival mechanism and an absence predicate, tied to the
observed outcome the Cause names. Negated absences, enabled-not-disabled
forms, other disabled mechanisms, and project-name "outcomes" must all stay
uncertain. Node executes the real TypeScript directly.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
MAPPING = ROOT / "src" / "cortheon" / "pi_core" / "evidence_mapping.ts"

RUN_SCRIPT = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const module = await import(input.module);
const out = {};
for (const [name, call] of Object.entries(input.calls)) {
  out[name] = module.mapHypothesisEvidence(call.sections, call.records);
}
console.log(JSON.stringify(out));
process.exit(0);
"""

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
BASE_RECORD = {
    "id": "ev1",
    "source": "pi:read:facts/a.txt",
    "fact": "Northstar path A uses collision key amber.",
}

MUST_STAY_UNCERTAIN = [
    ("is not disabled + and", "Compaction is not disabled and the collision persists."),
    (
        "remains enabled, not disabled",
        "Compaction remains enabled, not disabled, so the collision persists.",
    ),
    ("was never disabled", "Compaction was never disabled and the collision remains."),
    ("cannot be disabled", "Compaction cannot be disabled, yet the collision persists."),
    ("semicolon", "Compaction is not disabled; the collision persists."),
    (
        "no evidence it was disabled",
        "There is no evidence compaction was disabled while the collision persists.",
    ),
    (
        "rather than disabled",
        "Compaction stayed on rather than disabled, and the collision remains.",
    ),
    (
        "other mechanism disabled, same sentence",
        "Compaction ran, but replication was disabled, and the collision persists.",
    ),
    (
        "other mechanism, comma-joined",
        "Compaction is nightly, replication is disabled, the collision persists.",
    ),
    (
        "absence-first ordering",
        "We disabled the nightly audit job before compaction; the collision persists.",
    ),
    (
        "absence-first, same sentence",
        "Replication was disabled while compaction ran and the collision persists.",
    ),
    (
        "outcome anchor is only the project name",
        "With compaction disabled, the Northstar rollout continued unchanged.",
    ),
    ("off inside prose", "Compaction handoff was clean and the collision remains."),
    ("zero-match grep", "No matches for pattern 'compaction disabled'."),
    (
        "intervening noun phrase breaks binding",
        "Compaction ran replication was disabled and the collision persists.",
    ),
    (
        "monitor is the disabled noun, not the mechanism",
        "Compaction monitor was disabled and the collision persists.",
    ),
    (
        "subsystem is the disabled noun, not the mechanism",
        "Compaction subsystem was disabled and the collision persists.",
    ),
    (
        "plural outcome head is not the derived head",
        "With compaction disabled the collisions persist.",
    ),
]

MUST_REFUTE = [
    "With compaction disabled the collision persists.",
    "Compaction was disabled and the collision remains.",
    "The collision persists when compaction is disabled.",
    "Collision persists when compaction is disabled.",
    "The collision persists with compaction disabled on the staging node.",
]


def _mapping_for(fact: str) -> dict[str, object]:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", RUN_SCRIPT],
        input=json.dumps(
            {
                "module": str(MAPPING),
                "calls": {
                    "probe": {
                        "sections": SECTIONS,
                        "records": [
                            BASE_RECORD,
                            {"id": "ev2", "source": "pi:read:facts/b.txt", "fact": fact},
                        ],
                    }
                },
            }
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])["probe"]
    assert isinstance(result, dict)
    return result


class RivalPolarityTests(unittest.TestCase):
    def test_negated_and_foreign_absences_stay_uncertain(self) -> None:
        for label, fact in MUST_STAY_UNCERTAIN:
            with self.subTest(label):
                mapping = _mapping_for(fact)
                self.assertEqual(mapping["rivalStatus"], "uncertain", fact)
                self.assertEqual(mapping["rivalIds"], ["ev2"], fact)

    def test_genuine_counterexamples_still_refute(self) -> None:
        for fact in MUST_REFUTE:
            with self.subTest(fact):
                mapping = _mapping_for(fact)
                self.assertEqual(mapping["rivalStatus"], "refuted", fact)
                self.assertEqual(mapping["rivalIds"], ["ev2"], fact)


if __name__ == "__main__":
    unittest.main()
