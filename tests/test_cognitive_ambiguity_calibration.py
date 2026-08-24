"""Regression tests for calibrated ambiguity completion.

Synthetic scenario, invented for this suite: two unrelated departments
define the same operational phrase with incompatible clocks, and a third
document poses a request that names neither. The runtime must certify a
calibrated answer that keeps both readings open and asks the minimum
discriminating question, while one-sided guesses, omitted source coverage,
and quiet branch-settling stay fail-closed.
"""

from __future__ import annotations

import unittest

from cortheon.cognitive_runtime import CognitiveRuntime

_GOAL = (
    "Decide whether 'halve recovery time' admits a single defensible "
    "reading. Set out the incompatible readings and request the narrowest "
    "clarification that separates them before any roster change. "
    "Do not modify files."
)

_FILES = {
    "hangar_procedures.md": (
        "Hangar crew defines recovery time as minutes from bay door open to first tool checkout.\n"
    ),
    "rail_charter.md": (
        "Charter desk defines recovery time as hours from locomotive "
        "failure to replacement coupling.\n"
    ),
    "shift_brief.md": (
        "The brief says halve recovery time without naming a fleet, clock "
        "start, or reporting ledger.\n"
    ),
}

# A correct, honest non-decision: both evidence-backed readings stay open
# and the answer asks the one question that separates them.
_CALIBRATED_ANSWER = (
    "'Halve recovery time' is ambiguous: the two ledgers define recovery "
    "time differently, so the request is underspecified. Whether recovery "
    "time means minutes from bay door open to first tool checkout is "
    "unresolved, and whether it means hours from locomotive failure to "
    "replacement coupling is unresolved. Which reading governs the roster?"
)

_HOST_SUMMARY = "\n".join(f"[{path}] {content.strip()}" for path, content in _FILES.items())


class _AmbiguitySession:
    """Drive one ambiguity investigation through discovery and clean reads."""

    def __init__(self, test: unittest.TestCase) -> None:
        self.runtime = CognitiveRuntime()
        started = self.runtime.start(_GOAL, effort="quick")
        self.session_id = started["session"]["session_id"]
        discovery = started["next_action"]["request"]
        test.assertEqual(discovery["capability"], "search")
        observed = self.runtime.observe(
            self.session_id,
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"find","outcome":"result",'
                        '"args":{"command":"find . -name \\"*.md\\""}}\n' + path
                    ),
                    "source": f"pi:find:{path}",
                    "status": "observed",
                }
                for path in _FILES
            ],
            request_id=discovery["request_id"],
        )
        read_request = observed["next_action"]["request"]
        test.assertEqual(read_request["capability"], "read_many")
        reads = self.runtime.observe(
            self.session_id,
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{content}'
                    ),
                    "source": f"pi:read:{path}",
                    "status": "verified",
                }
                for path, content in _FILES.items()
            ],
            request_id=read_request["request_id"],
        )
        self.evidence_ids = reads["accepted_evidence_ids"]
        test.assertEqual(len(self.evidence_ids), 3)

    def complete(self, answer: str, hypotheses: list[dict]) -> dict:
        return self.runtime.complete(
            self.session_id,
            answer=answer,
            claims=[{"claim": _HOST_SUMMARY, "evidence_ids": self.evidence_ids}],
            hypotheses=hypotheses,
            completion_evidence_ids=self.evidence_ids,
        )


def _uncertain_hypotheses(evidence_ids: list[str]) -> list[dict]:
    return [
        {
            "statement": ("Recovery time means minutes from bay door open to first tool checkout."),
            "falsification_test": "Ask which ledger governs the roster.",
            "status": "uncertain",
            "evidence_ids": evidence_ids,
        },
        {
            "statement": (
                "Recovery time means hours from locomotive failure to replacement coupling."
            ),
            "falsification_test": "Ask which ledger governs the roster.",
            "status": "uncertain",
            "evidence_ids": evidence_ids,
        },
    ]


class AmbiguityCalibrationTest(unittest.TestCase):
    def test_calibrated_two_interpretation_answer_is_certified(self) -> None:
        session = _AmbiguitySession(self)
        completed = session.complete(
            _CALIBRATED_ANSWER,
            _uncertain_hypotheses(session.evidence_ids),
        )
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["answer"], _CALIBRATED_ANSWER)

    def test_guessing_one_interpretation_is_withheld(self) -> None:
        session = _AmbiguitySession(self)
        guess = (
            "The two ledgers conflict, but the roster drives the hangar "
            "floor, so 'halve recovery time' means minutes from bay door "
            "open to first tool checkout. Need the fleet rotation before "
            "any roster change."
        )
        result = session.complete(guess, _uncertain_hypotheses(session.evidence_ids))
        self.assertNotEqual(result.get("status"), "complete")
        failed = {
            check["name"] for check in result["verification"]["checks"] if not check["passed"]
        }
        self.assertIn("evidence_alignment", failed)

    def test_omitting_one_source_interpretation_is_withheld(self) -> None:
        session = _AmbiguitySession(self)
        one_sided = (
            "The request is underspecified: the hangar ledger defines "
            "recovery time as minutes from bay door open to first tool "
            "checkout, and nothing marks that clock as governing. Which "
            "reading applies?"
        )
        result = session.complete(one_sided, _uncertain_hypotheses(session.evidence_ids))
        self.assertNotEqual(result.get("status"), "complete")
        gaps = " ".join(result["verification"]["gaps"])
        self.assertIn("the distinctive content of every clean read", gaps)

    def test_methodological_treating_as_evidence_is_certified(self) -> None:
        session = _AmbiguitySession(self)
        methodological = (
            "Treating each source statement as evidence, the request "
            "remains ambiguous: whether recovery time means minutes from "
            "bay door open to first tool checkout is unresolved, and "
            "whether it means hours from locomotive failure to "
            "replacement coupling is unresolved. Which ledger governs "
            "the roster?"
        )
        result = session.complete(
            methodological,
            _uncertain_hypotheses(session.evidence_ids),
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], methodological)

    def test_quietly_settling_one_branch_is_withheld(self) -> None:
        session = _AmbiguitySession(self)
        settled = (
            "'Halve recovery time' is ambiguous: whether recovery time "
            "means minutes from bay door open to first tool checkout is "
            "unresolved, and whether it means hours from locomotive "
            "failure to replacement coupling is unresolved. Treating the "
            "hangar reading as the operative one for rostering, the crew "
            "clock is what the brief targets. Which ledger governs?"
        )
        result = session.complete(settled, _uncertain_hypotheses(session.evidence_ids))
        self.assertNotEqual(result.get("status"), "complete")
        failed = {
            check["name"] for check in result["verification"]["checks"] if not check["passed"]
        }
        self.assertIn("evidence_alignment", failed)


if __name__ == "__main__":
    unittest.main()
