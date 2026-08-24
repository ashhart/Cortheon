"""Independent answer alignment for evidence-bound abduction.

The causal synthesis adapter prepends a host-built ``Evidence:`` ledger to
the model's ``Cause``/``Rival``/``Test`` lines.  That ledger quotes the
accepted records verbatim, so counting it as model output would let the
host's own evidence satisfy the checks that exist to prove the model
represented its sources and reached a causal conclusion.  The abductive
alignment check therefore reads the answer with exactly that leading
section removed, while the certified answer keeps it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any

from cortheon.cognitive_core.alignment import _model_authored_answer
from cortheon.cognitive_core.runtime import CognitiveRuntime

ALIGNMENT_SOURCE = Path(__file__).resolve().parents[1] / "src/cortheon/cognitive_core/alignment.py"

GOAL = (
    "Read facts/a.txt and facts/b.txt. Diagnose the causal explanation "
    "for the collision, disprove the rival, and give a discriminating test."
)
READ_A = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/a.txt"}}\n'
    "Northstar path A uses collision key amber because the amber lease "
    "routes through the shared index."
)
READ_B = (
    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
    '"args":{"filePath":"facts/b.txt"}}\n'
    "Path B reuses collision key amber, so the shared index explains the "
    "duplicate lease. A competing alternative would falsify the index theory."
)
LEDGER = (
    "Evidence: [pi:read:facts/a.txt] Northstar path A uses collision key amber "
    "because the amber lease routes through the shared index. "
    "[pi:read:facts/b.txt] Path B reuses collision key amber, so the shared "
    "index explains the duplicate lease. A competing alternative would falsify "
    "the index theory."
)
SYNTHESIZED_CAUSE = (
    "The collision occurs because both paths reuse collision key amber "
    "through the shared index lease."
)
RIVAL_LINE = "Rival: Instead, a competing alternative is nightly compaction."
TEST_LINE = "Test: A distinguishing test would falsify the wrong mechanism."
NONSENSE = "\n".join((LEDGER, "Cause: Blorp gribble wibble.", RIVAL_LINE, TEST_LINE))
SYNTHESIZED = "\n".join((LEDGER, f"Cause: {SYNTHESIZED_CAUSE}", RIVAL_LINE, TEST_LINE))
# The ledger carries "competing alternative" and "falsify"; the model's own
# lines carry neither.
BORROWED_MARKERS = "\n".join(
    (
        LEDGER,
        f"Cause: {SYNTHESIZED_CAUSE}",
        "Rival: Instead, nightly compaction explains the duplicate lease.",
        "Test: Assign distinct keys and re-run the import.",
    )
)
CAUSE_HYPOTHESIS = {
    "statement": SYNTHESIZED_CAUSE,
    "falsification_test": "Assign distinct keys.",
    "status": "supported",
    "evidence_ids": ["ev1", "ev2"],
}
RIVAL_HYPOTHESIS = {
    "statement": "Instead, nightly compaction is the competing alternative.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}


def _runtime() -> tuple[CognitiveRuntime, str]:
    runtime = CognitiveRuntime()
    session_id = runtime.start(GOAL)["session"]["session_id"]
    runtime.observe(
        session_id,
        [
            {
                "kind": "documentation",
                "content": READ_A,
                "status": "verified",
                "source": "pi:read:facts/a.txt",
            },
            {
                "kind": "documentation",
                "content": READ_B,
                "status": "verified",
                "source": "pi:read:facts/b.txt",
            },
        ],
        request_id="req1",
    )
    return runtime, session_id


def _complete(answer: str) -> dict[str, Any]:
    runtime, session_id = _runtime()
    return runtime.complete(
        session_id,
        answer=answer,
        claims=[{"claim": SYNTHESIZED_CAUSE, "evidence_ids": ["ev1", "ev2"]}],
        hypotheses=[CAUSE_HYPOTHESIS, RIVAL_HYPOTHESIS],
        completion_evidence_ids=["ev1", "ev2"],
    )


def _alignment(result: dict[str, Any]) -> dict[str, Any]:
    verification = result.get("verification") or result["scorecard"]["verification"]
    return next(item for item in verification["checks"] if item["name"] == "evidence_alignment")


class TestLedgerIsNotModelOutput(unittest.TestCase):
    def test_nonsense_cause_behind_an_evidence_paste_fails(self):
        result = _complete(NONSENSE)
        alignment = _alignment(result)
        self.assertFalse(alignment["passed"], alignment["reason"])
        self.assertIn("answer coverage across two source records", alignment["reason"])
        self.assertIn("an explicit causal bridge", alignment["reason"])
        self.assertIn("Evidence ledger is not model output", alignment["reason"])
        self.assertNotEqual(result.get("status"), "complete")

    def test_ledger_cannot_supply_the_alternative_or_falsification(self):
        alignment = _alignment(_complete(BORROWED_MARKERS))
        self.assertFalse(alignment["passed"], alignment["reason"])
        self.assertIn("an explicit alternative", alignment["reason"])
        self.assertIn("an observable falsification test", alignment["reason"])

    def test_synthesized_cause_completes_with_the_ledger_intact(self):
        result = _complete(SYNTHESIZED)
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))
        self.assertTrue(_alignment(result)["passed"])
        # The certified answer is returned byte for byte, ledger included.
        self.assertEqual(result["answer"], SYNTHESIZED)


class TestLeadingEvidenceSection(unittest.TestCase):
    def test_leading_ledger_is_removed_up_to_the_first_section(self):
        self.assertEqual(
            _model_authored_answer(SYNTHESIZED),
            "\n".join((f"Cause: {SYNTHESIZED_CAUSE}", RIVAL_LINE, TEST_LINE)),
        )

    def test_multiline_ledger_is_removed_up_to_the_first_section(self):
        answer = "Evidence: [a] one\n[b] two\nCause: c\nRival: r\nTest: t"
        self.assertEqual(_model_authored_answer(answer), "Cause: c\nRival: r\nTest: t")

    def test_answer_without_a_leading_ledger_is_untouched(self):
        answer = "Cause: c\nRival: r\nTest: t\nEvidence: [a] one"
        self.assertEqual(_model_authored_answer(answer), answer)

    def test_ledger_without_following_sections_drops_only_that_line(self):
        answer = "Evidence: [a] one\nThe collision came from the shared lease."
        self.assertEqual(
            _model_authored_answer(answer), "The collision came from the shared lease."
        )

    def test_prose_answer_is_untouched(self):
        answer = "The collision is explained by the shared index lease."
        self.assertEqual(_model_authored_answer(answer), answer)


class TestHelperScope(unittest.TestCase):
    def test_only_abductive_alignment_strips_the_preamble(self):
        tree = ast.parse(ALIGNMENT_SOURCE.read_text(encoding="utf-8"))
        callers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for inner in ast.walk(node)
            if isinstance(inner, ast.Name) and inner.id == "_model_authored_answer"
        }
        self.assertEqual(callers, {"_abductive_alignment_check"})


if __name__ == "__main__":
    unittest.main()
