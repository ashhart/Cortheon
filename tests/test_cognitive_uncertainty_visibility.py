"""Real-runtime tests for the local uncertainty-visibility contract.

Each uncertain hypothesis submitted to ``cortheon_complete`` must be
visibly preserved: a local sentence or bounded clause of the answer must
contain both a distinctive content anchor from that hypothesis statement
and an explicit openness marker.  Unrelated uncertainty words, settled
rivals, and pronoun-only dismissals must be withheld.

The identifying rule that decides which anchors count -- shared tokens,
lone anchors, and rival framing -- is covered by
``test_cognitive_uncertainty_identity``.
"""

from __future__ import annotations

import unittest
from typing import Any

from cognitive_uncertainty_helpers import (
    ATLAS_RIVAL,
    CAUSE,
    CAUSE_AND_TEST,
    COMPACTION_RIVAL,
    complete_answer,
    rival_answer,
    start_session,
    visibility,
)

from cortheon.cognitive_core.uncertainty_visibility import _hypothesis_visibility, _segments

DEGENERATE_RIVAL = {
    "statement": "The rival is another possible explanation.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}


class TestWithheldAnswers(unittest.TestCase):
    def _assert_withheld(self, result: dict[str, Any]) -> None:
        self.assertNotEqual(result.get("status"), "complete")
        check = visibility(result)
        self.assertFalse(check["passed"], check["reason"])

    def test_settled_impossible_answer_is_withheld(self):
        runtime, session_id = start_session()
        result = complete_answer(
            runtime, session_id, rival_answer("Rival: Cache compaction is impossible.")
        )
        self._assert_withheld(result)

    def test_unrelated_uncertainty_word_is_withheld(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: The date is uncertain, but cache compaction cannot explain the collision."
        )
        self._assert_withheld(complete_answer(runtime, session_id, answer))

    def test_pronoun_only_dismissal_is_withheld(self):
        runtime, session_id = start_session()
        answer = rival_answer("Rival: The rival definitely does not explain it.")
        self._assert_withheld(complete_answer(runtime, session_id, answer))

    def test_overconfident_disproof_despite_marker_is_withheld(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: Cache compaction is the competing alternative; the accepted "
            "evidence disproves it."
        )
        self._assert_withheld(complete_answer(runtime, session_id, answer))

    def test_unmentioned_uncertain_hypothesis_is_withheld(self):
        runtime, session_id = start_session()
        result = complete_answer(runtime, session_id, CAUSE_AND_TEST, [CAUSE, ATLAS_RIVAL])
        self._assert_withheld(result)


class TestPreservedAnswers(unittest.TestCase):
    def test_group_uncertainty_can_introduce_two_named_interpretations(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Two alternatives remain uncertain: cache compaction and Atlas scheduling."
        )
        result = complete_answer(
            runtime, session_id, answer, [CAUSE, COMPACTION_RIVAL, ATLAS_RIVAL]
        )
        self.assertTrue(visibility(result)["passed"], result.get("gaps"))

    def test_group_uncertainty_does_not_cover_an_omitted_interpretation(self):
        runtime, session_id = start_session()
        answer = rival_answer("Two alternatives remain uncertain: cache compaction.")
        result = complete_answer(
            runtime, session_id, answer, [CAUSE, COMPACTION_RIVAL, ATLAS_RIVAL]
        )
        self.assertNotEqual(result.get("status"), "complete")
        self.assertFalse(visibility(result)["passed"])

    def test_remains_uncertain_paraphrase_completes(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: A competing alternative is cache compaction, which remains uncertain."
        )
        result = complete_answer(runtime, session_id, answer)
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))

    def test_competing_alternative_paraphrase_completes(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: A competing alternative is cache compaction; the accepted "
            "evidence does not settle it."
        )
        result = complete_answer(runtime, session_id, answer)
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))

    def test_ambiguous_anchor_paraphrase_completes(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: A competing alternative is Atlas scheduling, which is ambiguous."
        )
        result = complete_answer(runtime, session_id, answer, [CAUSE, ATLAS_RIVAL])
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))

    def test_two_uncertain_hypotheses_each_preserved_completes(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: A competing alternative is cache compaction, which remains "
            "uncertain; Atlas is ambiguous."
        )
        result = complete_answer(
            runtime, session_id, answer, [CAUSE, COMPACTION_RIVAL, ATLAS_RIVAL]
        )
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))

    def test_second_uncertain_hypothesis_unpreserved_is_withheld(self):
        runtime, session_id = start_session()
        answer = rival_answer(
            "Rival: Cache compaction remains uncertain; Atlas definitely does not contribute."
        )
        result = complete_answer(
            runtime, session_id, answer, [CAUSE, COMPACTION_RIVAL, ATLAS_RIVAL]
        )
        self.assertNotEqual(result.get("status"), "complete")
        self.assertFalse(visibility(result)["passed"])


class TestDegenerateGenericHypothesis(unittest.TestCase):
    def test_no_anchor_hypothesis_fails_closed_with_unanchorable(self):
        outcome = _hypothesis_visibility(
            DEGENERATE_RIVAL["statement"], _segments("The date is uncertain.")
        )
        self.assertEqual(outcome, (False, "unanchorable"))

    def test_no_anchor_hypothesis_fails_closed_when_omitted(self):
        outcome = _hypothesis_visibility(
            DEGENERATE_RIVAL["statement"], _segments("Cause: The collision is explained.")
        )
        self.assertEqual(outcome, (False, "unanchorable"))

    def test_unrelated_openness_runtime_completion_is_withheld(self):
        runtime, session_id = start_session()
        answer = rival_answer("Note: The date is uncertain.")
        result = complete_answer(runtime, session_id, answer, [CAUSE, DEGENERATE_RIVAL])
        self.assertNotEqual(result.get("status"), "complete")
        check = visibility(result)
        self.assertFalse(check["passed"], check["reason"])
        self.assertIn("resubmit", check["reason"])

    def test_omitted_degenerate_hypothesis_runtime_completion_is_withheld(self):
        runtime, session_id = start_session()
        result = complete_answer(runtime, session_id, CAUSE_AND_TEST, [CAUSE, DEGENERATE_RIVAL])
        self.assertNotEqual(result.get("status"), "complete")
        check = visibility(result)
        self.assertFalse(check["passed"], check["reason"])
        self.assertIn("resubmit", check["reason"])


if __name__ == "__main__":
    unittest.main()
