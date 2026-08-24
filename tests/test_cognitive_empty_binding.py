"""Real-runtime contract for honest empty evidence bindings.

When no accepted record genuinely bears on a hypothesis, the adapter submits
it as ``uncertain`` with zero evidence ids. The runtime must accept that
submission structurally (no HTTP 400 / ValueError), treat the hypothesis as
untested, and withhold the completion with an actionable gap — never certify
an answer whose cause was never grounded in evidence.
"""

from __future__ import annotations

import unittest

from cortheon.cognitive_runtime import CognitiveRuntime


def _observed_evidence(runtime: CognitiveRuntime) -> tuple[str, list[str]]:
    started = runtime.start(
        "Read cohort_notes.md, routing_map.md, and capacity_limits.md. "
        "Infer why activation fell, compare competing hypotheses, and state one "
        "observation that would falsify the best explanation.",
        effort="quick",
    )
    facts = [
        (
            "cohort_notes.md",
            "Activation fell only for weekend migration accounts.",
        ),
        (
            "routing_map.md",
            "Weekend migrations use the legacy token broker.",
        ),
        (
            "capacity_limits.md",
            "The legacy broker rejects above 500; migration bursts reach 900.",
        ),
    ]
    observed = runtime.observe(
        started["session"]["session_id"],
        [
            {
                "kind": "documentation",
                "content": (
                    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                    f'"args":{{"filePath":"{path}"}}}}\n{fact}'
                ),
                "source": f"pi:read:{path}",
                "status": "verified",
            }
            for path, fact in facts
        ],
        request_id="req1",
    )
    return started["session"]["session_id"], observed["accepted_evidence_ids"]


class EmptyEvidenceBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = CognitiveRuntime()

    def test_uncertain_hypothesis_with_zero_ids_withholds_actionably(self) -> None:
        """An uncertain hypothesis with no bearing evidence is accepted by the
        completion contract, then withheld as untested with a gap naming the
        hypothesis and the next action — never a ValueError and never a
        certification."""
        session_id, evidence_ids = _observed_evidence(self.runtime)
        answer = (
            "The strongest explanation is legacy-broker overload because weekend "
            "migrations route through that broker and bursts of 900 exceed its 500 "
            "limit. A competing alternative is a weekend-only measurement "
            "artifact; whether it explains the fall remains uncertain. "
            "Falsify the leading explanation by finding a 900-request weekend "
            "burst through the legacy broker with normal activation."
        )
        result = self.runtime.complete(
            session_id,
            answer=answer,
            claims=[{"claim": answer, "evidence_ids": evidence_ids}],
            hypotheses=[
                {
                    "statement": "Legacy-broker overload explains the fall.",
                    "falsification_test": (
                        "Find a 900-request weekend burst with normal activation."
                    ),
                    # No record genuinely bears on the cause as framed here:
                    # zero ids must be a legal, honest binding.
                    "status": "uncertain",
                    "evidence_ids": [],
                },
                {
                    "statement": "A weekend-only measurement artifact explains the fall.",
                    "falsification_test": "Compare an independent activation measure.",
                    "status": "uncertain",
                    "evidence_ids": evidence_ids,
                },
            ],
            completion_evidence_ids=evidence_ids,
        )
        self.assertNotIn("answer", result)
        self.assertEqual(result["verification"]["verdict"], "needs_evidence")
        self.assertTrue(
            any("untested" in gap for gap in result["verification"]["gaps"]),
            result["verification"]["gaps"],
        )
        self.assertRegex(" ".join(result["verification"]["gaps"]), r"untested hypotheses[^;]*h\d")
        # The recovery instruction must be actionable and truthful: answer
        # prose alone cannot supply evidence or change hypothesis status.
        gaps_text = " ".join(result["verification"]["gaps"])
        self.assertNotIn("settle the question in the answer", gaps_text)
        self.assertRegex(
            gaps_text,
            r"bearing evidence or a counterexample.*supported or refuted status",
        )
        self.assertIn("next_action", result)

    def test_zero_id_binding_does_not_break_the_certified_path(self) -> None:
        """Control: the same session shape with a grounded supported cause and
        a bearing-evidence rival still certifies, so the empty-binding path is
        a withhold decision, not a blanket rejection."""
        session_id, evidence_ids = _observed_evidence(self.runtime)
        answer = (
            "The strongest explanation is legacy-broker overload because weekend "
            "migrations route through that broker and bursts of 900 exceed its 500 "
            "limit. A competing alternative is a weekend-only measurement "
            "artifact; whether it explains the fall remains uncertain. "
            "Falsify the leading explanation by finding a 900-request weekend "
            "burst through the legacy broker with normal activation."
        )
        result = self.runtime.complete(
            session_id,
            answer=answer,
            claims=[{"claim": answer, "evidence_ids": evidence_ids}],
            hypotheses=[
                {
                    "statement": "Legacy-broker overload explains the fall.",
                    "falsification_test": (
                        "Find a 900-request weekend burst with normal activation."
                    ),
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                },
                {
                    "statement": "A weekend-only measurement artifact explains the fall.",
                    "falsification_test": "Compare an independent activation measure.",
                    "status": "uncertain",
                    "evidence_ids": evidence_ids,
                },
            ],
            completion_evidence_ids=evidence_ids,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], answer)


if __name__ == "__main__":
    unittest.main()
