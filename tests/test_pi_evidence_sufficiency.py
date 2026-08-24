"""Direct state regressions for the causal evidence-sufficiency criterion.

Node's type-stripping loader executes the real reviewed TypeScript sources:
``merge.ts`` builds the authoritative clean evidence records and ``budget.ts``
judges sufficiency over them. The criterion is domain-neutral identity and
source independence — never response-batch or turn counting — so one batch
with two distinct clean sources can become sufficient while two batches
carrying one source, a repeated identity, or id-less/quarantined records
never can, and a pending runtime request always reopens discovery.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import cast

ROOT = Path(__file__).parents[1]
CORE = ROOT / "src" / "cortheon" / "pi_core"
SOURCE_DIR = ROOT / "src" / "cortheon"

CAUSAL_GOAL = (
    "Diagnose the causal explanation for the clash between the two ledgers, "
    "disprove the rival hypothesis, and give a discriminating test."
)
AMBIGUITY_GOAL = "The phrase in the spec is ambiguous; clarify it rather than guessing."

RUN_SCRIPT = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const merge = await import(input.mergeModule);
const budget = await import(input.budgetModule);
const state = await import(input.stateModule);
const results = [];
for (const payload of input.payloads || []) {
  const active = merge.mergePayload(payload);
  results.push({
    sufficient: budget.causalEvidenceSufficient(active),
    records: active?.evidenceRecords ?? null,
  });
}
for (const raw of input.actives || []) {
  results.push({ sufficient: budget.causalEvidenceSufficient(raw) });
}
console.log(JSON.stringify(results));
process.exit(0);
"""


def _run_node(
    core_dir: Path,
    payloads: list[dict[str, object]] | None = None,
    actives: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", RUN_SCRIPT],
        input=json.dumps(
            {
                "mergeModule": str(core_dir / "merge.ts"),
                "budgetModule": str(core_dir / "budget.ts"),
                "stateModule": str(core_dir / "state.ts"),
                "payloads": payloads or [],
                "actives": actives or [],
            }
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _sufficient(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    return _run_node(CORE, payloads=payloads)


def _payload(
    evidence: list[dict[str, object]],
    *,
    goal: str = CAUSAL_GOAL,
    request: dict[str, object] | None = None,
    session_id: str = "s1",
    deliverable: str = "document_synthesis",
) -> dict[str, object]:
    next_action = {"type": "harness_tool", "request": request} if request else {"type": "reason"}
    return {
        "session_id": session_id,
        "status": "observing",
        "session": {"deliverable": deliverable},
        "context": {"goal": goal, "evidence": evidence},
        "next_action": next_action,
    }


def _observation(
    evidence_id: str,
    source: str,
    content: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "source": source,
        "content": content,
        "status": "observed",
        **extra,
    }


TWO_SOURCES = [
    _observation("ev-1", "pi:read:facts/a.txt", "clean fact one"),
    _observation("ev-2", "pi:read:facts/b.txt", "clean fact two"),
]


class EvidenceSufficiencyTests(unittest.TestCase):
    def test_single_batch_two_distinct_sources_is_sufficient(self) -> None:
        results = _sufficient([_payload(TWO_SOURCES)])
        self.assertTrue(results[0]["sufficient"])
        self.assertEqual(len(cast("list[dict[str, object]]", results[0]["records"])), 2)

    def test_missing_source_plus_one_real_source_is_never_sufficient(
        self,
    ) -> None:
        """A missing source is not an independent source: merge may retain a
        conservative ``host`` display label for the blank source, but the
        two-source gate must count only normalized non-empty attributable
        sources — one unknown/blank source plus one real source never
        satisfies it."""
        batch = _payload(
            [
                _observation("ev-1", "", "clean fact one"),
                _observation("ev-2", "pi:read:facts/b.txt", "clean fact two"),
            ]
        )
        results = _sufficient([batch])
        self.assertFalse(results[0]["sufficient"])
        # The conservative host label is retained for display, yet the
        # direct real-source judgment over the merged records also rejects
        # it: the host-labeled record is not an attributable source.
        self.assertEqual(
            results[0]["records"],
            [
                {"id": "ev-1", "source": "host", "fact": "clean fact one"},
                {"id": "ev-2", "source": "pi:read:facts/b.txt", "fact": "clean fact two"},
            ],
        )
        self.assertFalse(
            _run_node(
                CORE,
                actives=[
                    {
                        **ONE_SOURCE_ACTIVE,
                        "evidenceRecords": [
                            {"id": "ev-1", "source": "host", "fact": "fact one"},
                            {
                                "id": "ev-2",
                                "source": "pi:read:facts/b.txt",
                                "fact": "fact two",
                            },
                        ],
                    }
                ],
            )[0]["sufficient"]
        )

    def test_whitespace_source_variants_are_one_source(self) -> None:
        """Two whitespace spellings of one source reach the gate as one
        source through merge's normalization, but on the active
        investigation they are byte-preserved non-Pi sources and therefore
        two identities: canonical identity never trims bytes."""
        batch = _payload(
            [
                _observation("ev-1", " pi:read:facts/a.txt\t", "clean fact one"),
                _observation("ev-2", "pi:read:facts/a.txt ", "clean fact two"),
            ]
        )
        results = _sufficient([batch])
        self.assertFalse(results[0]["sufficient"])
        self.assertEqual(
            {record["source"] for record in cast("list[dict[str, object]]", results[0]["records"])},
            {"pi:read:facts/a.txt"},
        )
        self.assertTrue(
            _run_node(
                CORE,
                actives=[
                    {
                        **ONE_SOURCE_ACTIVE,
                        "evidenceRecords": [
                            {
                                "id": "ev-1",
                                "source": " pi:read:facts/a.txt",
                                "fact": "fact one",
                            },
                            {
                                "id": "ev-2",
                                "source": "pi:read:facts/a.txt\t",
                                "fact": "fact two",
                            },
                        ],
                    }
                ],
            )[0]["sufficient"]
        )

    def test_whitespace_id_variants_collide_before_counting(self) -> None:
        """Ids are normalized before collision handling: two whitespace
        spellings of one id binding different facts collide and are excluded
        entirely — they can never pose as two independent identities."""
        batch = _payload(
            [
                _observation(" ev-1 ", "pi:read:facts/a.txt", "clean fact one"),
                _observation("ev-1", "pi:read:facts/b.txt", "conflicting fact"),
            ]
        )
        results = _sufficient([batch])
        self.assertFalse(results[0]["sufficient"])
        self.assertEqual(results[0]["records"], [])

    def test_two_real_distinct_sources_are_sufficient(self) -> None:
        """Two real, attributable, distinct sources satisfy the gate on the
        direct real-source judgment, with no blank or host label involved."""
        self.assertTrue(
            _run_node(
                CORE,
                actives=[
                    {
                        **ONE_SOURCE_ACTIVE,
                        "evidenceRecords": [
                            {
                                "id": "ev-1",
                                "source": "pi:read:facts/a.txt",
                                "fact": "fact one",
                            },
                            {
                                "id": "ev-2",
                                "source": "pi:read:facts/b.txt",
                                "fact": "fact two",
                            },
                        ],
                    }
                ],
            )[0]["sufficient"]
        )

    def test_two_batches_one_source_is_never_sufficient(self) -> None:
        batch_one = _payload([_observation("ev-1", "pi:read:facts/a.txt", "clean fact one")])
        batch_two = _payload(
            [
                _observation("ev-1", "pi:read:facts/a.txt", "clean fact one"),
                _observation("ev-2", "pi:read:facts/a.txt", "clean fact two"),
            ]
        )
        results = _sufficient([batch_one, batch_two])
        self.assertFalse(results[0]["sufficient"])
        self.assertFalse(results[1]["sufficient"])

    def test_repeated_identity_across_batches_is_never_sufficient(self) -> None:
        batch_one = _payload([_observation("ev-1", "pi:read:facts/a.txt", "clean fact one")])
        # The same id binding different source/fact pairs collides and is
        # excluded entirely; it never counts as a second identity.
        batch_two = _payload(
            [
                _observation("ev-1", "pi:read:facts/a.txt", "clean fact one"),
                _observation("ev-1", "pi:read:facts/b.txt", "conflicting fact"),
            ]
        )
        results = _sufficient([batch_one, batch_two])
        self.assertFalse(results[0]["sufficient"])
        self.assertFalse(results[1]["sufficient"])

    def test_idless_quarantined_and_failed_records_never_count(self) -> None:
        poisoned = _payload(
            [
                _observation("ev-1", "pi:read:facts/a.txt", "clean fact one"),
                _observation("", "pi:read:facts/b.txt", "no identity"),
                _observation(
                    "ev-3",
                    "pi:read:facts/c.txt",
                    "quarantined fact",
                    quarantine_flags=["instruction_like_segment"],
                ),
                _observation("ev-4", "pi:read:facts/d.txt", "failed fact", status="failed"),
            ]
        )
        results = _sufficient([poisoned])
        self.assertFalse(results[0]["sufficient"])
        self.assertEqual(
            results[0]["records"],
            [{"id": "ev-1", "source": "pi:read:facts/a.txt", "fact": "clean fact one"}],
        )

    def test_pending_request_prevents_sufficiency(self) -> None:
        pending = _payload(
            TWO_SOURCES,
            request={
                "request_id": "req-1",
                "capability": "reason",
                "query": "One more discriminating question.",
            },
        )
        results = _sufficient([pending])
        self.assertFalse(results[0]["sufficient"])

    def test_pending_request_reopens_after_sufficiency(self) -> None:
        reopen = _payload(
            TWO_SOURCES,
            request={
                "request_id": "req-2",
                "capability": "reason",
                "query": "A fresh runtime request reopens discovery.",
            },
        )
        results = _sufficient([_payload(TWO_SOURCES), reopen])
        self.assertTrue(results[0]["sufficient"])
        self.assertFalse(results[1]["sufficient"])

    def test_non_causal_shapes_are_never_sufficient(self) -> None:
        for payload in (
            _payload(TWO_SOURCES, goal=AMBIGUITY_GOAL),
            _payload(TWO_SOURCES, goal="Summarize the meeting notes."),
            _payload(TWO_SOURCES, deliverable="code_change"),
        ):
            self.assertFalse(_sufficient([payload])[0]["sufficient"], payload)

    def test_completed_investigation_is_never_sufficient(self) -> None:
        payload = _payload(TWO_SOURCES)
        payload["status"] = "complete"
        self.assertFalse(_sufficient([payload])[0]["sufficient"])


def _run_mutated(module_mutations: dict[str, tuple[str, str]], payloads=None, actives=None):
    # tempfile.TemporaryDirectory (not a leaked subprocess `mktemp -d`) so
    # the tree stays alive through the Node execution below and is cleaned
    # afterward.
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "cortheon"
        (root / "pi_core").mkdir(parents=True)
        for path in sorted((SOURCE_DIR / "pi_core").glob("*.ts")):
            text = path.read_text(encoding="utf-8")
            if path.stem in module_mutations:
                old, new = module_mutations[path.stem]
                assert text.count(old) == 1, (path.stem, old)
                text = text.replace(old, new)
            (root / "pi_core" / path.name).write_text(text, encoding="utf-8")
        return _run_node(root / "pi_core", payloads=payloads, actives=actives)


COLLIDED_ACTIVE = {
    "sessionId": "s1",
    "goal": CAUSAL_GOAL,
    "deliverable": "document_synthesis",
    "evidenceIds": [],
    "hypotheses": [],
    "completed": False,
    "admittedToolCalls": 0,
    "redundantDiscoveryCalls": 0,
    "automaticContinuations": 0,
    "needsContinuation": False,
    "pendingReadObservations": [],
    "evidenceRecords": [
        {"id": "ev-1", "source": "pi:read:facts/a.txt", "fact": "fact one"},
        {"id": "ev-1", "source": "pi:read:facts/b.txt", "fact": "fact two"},
        {"id": "ev-2", "source": "pi:read:facts/c.txt", "fact": "fact three"},
    ],
    "protectedTestPaths": [],
    "mutationTargets": [],
    "mutationEvidence": {},
    "initialFileHashes": {},
    "mutationInFlight": False,
}

ONE_SOURCE_ACTIVE = {
    **COLLIDED_ACTIVE,
    "evidenceRecords": [
        {"id": "ev-1", "source": "pi:read:facts/a.txt", "fact": "fact one"},
        {"id": "ev-2", "source": "pi:read:facts/a.txt", "fact": "fact two"},
    ],
}


class EvidenceSufficiencyMutationTests(unittest.TestCase):
    """Removing the identity or source half of the condition must reopen
    premature forcing on the same evidence the real guard rejects."""

    def test_identity_collision_never_suffices_until_guard_mutated(self) -> None:
        # Control: the reviewed guard rejects the collided identity state.
        self.assertFalse(_run_node(CORE, actives=[COLLIDED_ACTIVE])[0]["sufficient"])
        mutated = _run_mutated(
            {
                "budget": (
                    "if (identities.size !== records.length || identities.size < 2) return false;",
                    "if (identities.size < 2) return false;",
                )
            },
            actives=[COLLIDED_ACTIVE],
        )
        # Dropping the collision check lets one identity bound to two
        # records pose as two independent accepted sources: reopened.
        self.assertTrue(mutated[0]["sufficient"])

    def test_distinct_source_requirement_is_load_bearing(self) -> None:
        self.assertFalse(_run_node(CORE, actives=[ONE_SOURCE_ACTIVE])[0]["sufficient"])
        mutated = _run_mutated(
            {
                "budget": (
                    "return sources.size >= 2;",
                    "return true;",
                )
            },
            actives=[ONE_SOURCE_ACTIVE],
        )
        # Two unique ids from ONE source record now pass: reopened.
        self.assertTrue(mutated[0]["sufficient"])


if __name__ == "__main__":
    unittest.main()
