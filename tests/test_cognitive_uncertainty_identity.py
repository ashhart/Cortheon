"""Which anchors identify an uncertain hypothesis, and which never do.

An openness clause preserves a hypothesis only when it names *that*
mechanism.  Three ways it can fail to are pinned here: a token two open
hypotheses share identifies neither; a lone anchor identifies only when
its clause predicates openness of it and names nothing else; and rival
framing ("a competing alternative is ...") predicates nothing, so it
needs two matched anchors in its own clause.  Each rule is asserted both
directly and through the real ``cortheon_complete`` path.
"""

from __future__ import annotations

import unittest
from typing import Any

from cognitive_uncertainty_helpers import (
    ATLAS_RIVAL,
    CAUSE,
    COMPACTION_RIVAL,
    complete_answer,
    failed_checks,
    rival_answer,
    start_session,
    visibility,
)

from cortheon.cognitive_core.uncertainty_visibility import _hypothesis_visibility, _segments

# Two open hypotheses that share the tokens "cache" and "Northstar": one
# openness phrase naming only the first must never preserve the second.
COMPACTION_EVICTION = {
    "statement": "Cache compaction evicts the Northstar entries early.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}
REPLICA_LAG = {
    "statement": "Cache replica lag delays the Northstar writes.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}
# Two open hypotheses with identical content words: no clause can name one
# without the other, so the gate must fail closed rather than guess.
TWIN_A = {
    "statement": "Cache compaction is the competing alternative.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}
TWIN_B = {
    "statement": "Compaction of the cache is the competing alternative.",
    "falsification_test": "Assign distinct keys.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}
WEEKEND_RIVAL = {
    "statement": "A weekend-only measurement artifact skews the counts.",
    "falsification_test": "Re-run outside the weekend window.",
    "status": "uncertain",
    "evidence_ids": ["ev2"],
}
# An honest, fully aligned rival line that names exactly one open mechanism.
SHARED_RIVAL_LINE = "Rival: A competing alternative is cache compaction, which remains uncertain."
WEEKEND_LINE = "Rival: A competing alternative is a weekend-only measurement artifact."
# Framing split by a comma: the fragment before it carries only the topical
# word "cache" and names nothing else, while the real subject sits in the
# next clause.  It must not stand in for the replica-lag hypothesis.
SPLIT_FRAMING = "A competing alternative is cache, compaction of which remains uncertain."
SPLIT_FRAMING_LINE = "Rival: " + SPLIT_FRAMING
# The concise honest forms small models actually write, which must keep
# passing however the identifying rule tightens.
CONCISE_FORMS = (
    (COMPACTION_RIVAL, "Rival: Cache compaction remains uncertain."),
    (ATLAS_RIVAL, "Rival: Atlas is ambiguous."),
    (WEEKEND_RIVAL, WEEKEND_LINE),
    (COMPACTION_RIVAL, "Rival: A competing alternative is cache compaction."),
)


class TestSharedTokenLaundering(unittest.TestCase):
    """One openness phrase must not carry two hypotheses on a shared word."""

    def test_shared_token_preserves_only_the_named_hypothesis(self):
        segments = _segments(SHARED_RIVAL_LINE)
        named = COMPACTION_EVICTION["statement"]
        other = REPLICA_LAG["statement"]
        self.assertEqual(_hypothesis_visibility(named, segments, [other]), (True, "preserved"))
        self.assertEqual(_hypothesis_visibility(other, segments, [named]), (False, "shared_only"))

    def test_two_distinct_openness_clauses_preserve_both(self):
        segments = _segments(
            "Rival: A competing alternative is cache compaction, which remains "
            "uncertain; cache replica lag is also unresolved."
        )
        named = COMPACTION_EVICTION["statement"]
        other = REPLICA_LAG["statement"]
        self.assertEqual(_hypothesis_visibility(named, segments, [other]), (True, "preserved"))
        self.assertEqual(_hypothesis_visibility(other, segments, [named]), (True, "preserved"))

    def test_indistinguishable_hypotheses_fail_closed(self):
        segments = _segments(SHARED_RIVAL_LINE)
        first = TWIN_A["statement"]
        second = TWIN_B["statement"]
        self.assertEqual(
            _hypothesis_visibility(first, segments, [second]), (False, "indistinguishable")
        )
        self.assertEqual(
            _hypothesis_visibility(second, segments, [first]), (False, "indistinguishable")
        )

    def test_honest_forms_survive_a_second_open_hypothesis(self):
        # Small models write short honest clauses; each stays valid while a
        # second uncertain hypothesis is on the table.
        for statement, other, answer in (
            (
                ATLAS_RIVAL["statement"],
                COMPACTION_RIVAL["statement"],
                "Rival: Atlas is ambiguous.",
            ),
            (
                COMPACTION_RIVAL["statement"],
                ATLAS_RIVAL["statement"],
                "Rival: Cache compaction remains uncertain.",
            ),
            (WEEKEND_RIVAL["statement"], ATLAS_RIVAL["statement"], WEEKEND_LINE),
        ):
            with self.subTest(statement=statement):
                self.assertEqual(
                    _hypothesis_visibility(statement, _segments(answer), [other]),
                    (True, "preserved"),
                )


class TestSharedTokenRuntime(unittest.TestCase):
    """The same contract through the real runtime completion path."""

    def test_shared_token_answer_is_withheld(self):
        # Every other gate passes, so the withholding is attributable to the
        # second hypothesis riding on the shared word "cache".
        runtime, session_id = start_session()
        result = complete_answer(
            runtime,
            session_id,
            rival_answer(SHARED_RIVAL_LINE),
            [CAUSE, COMPACTION_EVICTION, REPLICA_LAG],
        )
        self.assertNotEqual(result.get("status"), "complete")
        self.assertEqual(failed_checks(result), {"uncertainty_visibility"})
        check = visibility(result)
        self.assertIn(REPLICA_LAG["statement"], check["reason"])
        self.assertNotIn(COMPACTION_EVICTION["statement"], check["reason"])

    def test_two_distinct_openness_clauses_complete(self):
        runtime, session_id = start_session()
        result = complete_answer(
            runtime,
            session_id,
            rival_answer(
                "Rival: A competing alternative is cache compaction, which remains "
                "uncertain; cache replica lag is also unresolved."
            ),
            [CAUSE, COMPACTION_EVICTION, REPLICA_LAG],
        )
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))

    def test_indistinguishable_hypotheses_ask_for_a_resubmission(self):
        runtime, session_id = start_session()
        result = complete_answer(
            runtime, session_id, rival_answer(SHARED_RIVAL_LINE), [CAUSE, TWIN_A, TWIN_B]
        )
        self.assertNotEqual(result.get("status"), "complete")
        self.assertEqual(failed_checks(result), {"uncertainty_visibility"})
        check = visibility(result)
        self.assertIn("resubmit", check["reason"])
        self.assertIn(TWIN_A["statement"], check["reason"])
        self.assertIn(TWIN_B["statement"], check["reason"])


class TestLoneAnchorIdentification(unittest.TestCase):
    """A lone anchor preserves only when its clause names nothing else.

    ``cache`` is a five-character content word of the replica-lag
    hypothesis, so a bare length floor let ``cache compaction remains
    uncertain`` stand in for it while naming another mechanism entirely.
    """

    def test_wrong_mechanism_lone_anchor_is_not_preservation(self):
        for answer in (
            "Rival: Cache compaction remains uncertain.",
            "Rival: Cache compaction is an open question.",
            SHARED_RIVAL_LINE,
        ):
            with self.subTest(answer=answer):
                self.assertEqual(
                    _hypothesis_visibility(REPLICA_LAG["statement"], _segments(answer)),
                    (False, "unidentified"),
                )

    def test_concise_forms_keep_their_lone_or_paired_anchors(self):
        for rival, line in CONCISE_FORMS:
            with self.subTest(line=line):
                self.assertEqual(
                    _hypothesis_visibility(rival["statement"], _segments(line)),
                    (True, "preserved"),
                )

    def test_mixed_set_preserves_only_the_named_mechanism(self):
        # Distinctiveness and the identifying rule compose: the lone Atlas
        # anchor still stands while replica lag is withheld.
        segments = _segments("Rival: Cache compaction remains uncertain; Atlas is ambiguous.")
        self.assertEqual(
            _hypothesis_visibility(ATLAS_RIVAL["statement"], segments, [REPLICA_LAG["statement"]]),
            (True, "preserved"),
        )
        self.assertEqual(
            _hypothesis_visibility(REPLICA_LAG["statement"], segments, [ATLAS_RIVAL["statement"]]),
            (False, "unidentified"),
        )


class TestFramingIdentification(unittest.TestCase):
    """Rival framing asserts nothing, so one matched anchor never suffices."""

    def test_split_framing_fragment_is_not_preservation(self):
        # The comma strands "a competing alternative is cache", which names
        # nothing else and so passed the purity rule for predicative
        # openness while its real subject, compaction, sat in the next
        # clause.  Framing now needs two anchors, so this is withheld.
        self.assertEqual(
            _hypothesis_visibility(REPLICA_LAG["statement"], _segments(SPLIT_FRAMING)),
            (False, "unidentified"),
        )

    def test_pure_framing_by_one_anchor_is_not_preservation(self):
        for rival, line in (
            (ATLAS_RIVAL, "Rival: A competing alternative is Atlas."),
            (REPLICA_LAG, "Rival: A competing alternative is cache."),
        ):
            with self.subTest(line=line):
                self.assertEqual(
                    _hypothesis_visibility(rival["statement"], _segments(line)),
                    (False, "unidentified"),
                )

    def test_framing_with_two_anchors_preserves(self):
        # The honest concise framing of the very hypothesis the split
        # fragment tried to launder still passes.
        for rival, line in (
            (REPLICA_LAG, "Rival: A competing alternative is cache replica lag."),
            (ATLAS_RIVAL, "Rival: A competing alternative is Atlas scheduling."),
            (WEEKEND_RIVAL, WEEKEND_LINE),
        ):
            with self.subTest(line=line):
                self.assertEqual(
                    _hypothesis_visibility(rival["statement"], _segments(line)),
                    (True, "preserved"),
                )

    def test_predicative_markers_keep_the_lone_anchor_allowance(self):
        for rival, line in (
            (ATLAS_RIVAL, "Rival: Atlas is ambiguous."),
            (COMPACTION_RIVAL, "Rival: Compaction remains uncertain."),
            (ATLAS_RIVAL, "Rival: Atlas is unresolved."),
        ):
            with self.subTest(line=line):
                self.assertEqual(
                    _hypothesis_visibility(rival["statement"], _segments(line)),
                    (True, "preserved"),
                )


class TestLoneAnchorRuntime(unittest.TestCase):
    """The same identifying rule through the real completion path."""

    def _completion(self, line: str, hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        runtime, session_id = start_session()
        return complete_answer(runtime, session_id, rival_answer(line), hypotheses)

    def test_wrong_mechanism_answer_is_withheld(self):
        # Every other gate passes on this wording, so the withholding is
        # attributable to the clause naming compaction instead of lag.
        result = self._completion(SHARED_RIVAL_LINE, [CAUSE, REPLICA_LAG])
        self.assertNotEqual(result.get("status"), "complete")
        self.assertEqual(failed_checks(result), {"uncertainty_visibility"})
        check = visibility(result)
        self.assertIn(REPLICA_LAG["statement"], check["reason"])
        self.assertIn("resubmit", check["reason"])

    def test_split_framing_answer_is_withheld(self):
        result = self._completion(SPLIT_FRAMING_LINE, [CAUSE, REPLICA_LAG])
        self.assertNotEqual(result.get("status"), "complete")
        self.assertEqual(failed_checks(result), {"uncertainty_visibility"})
        check = visibility(result)
        self.assertIn(REPLICA_LAG["statement"], check["reason"])
        self.assertIn("resubmit", check["reason"])

    def test_two_anchor_framing_completes(self):
        result = self._completion(
            "Rival: A competing alternative is cache replica lag.", [CAUSE, REPLICA_LAG]
        )
        self.assertEqual(result.get("status"), "complete", result.get("gaps"))

    def test_concise_forms_still_pass_the_visibility_gate(self):
        # These forms pass uncertainty_visibility itself; the abductive goal
        # separately demands explicit alternative framing, so completion may
        # still be withheld by evidence_alignment.
        for rival, line in CONCISE_FORMS:
            with self.subTest(line=line):
                result = self._completion(line, [CAUSE, rival])
                if result.get("status") == "complete":
                    continue
                check = visibility(result)
                self.assertTrue(check["passed"], check["reason"])

    def test_mixed_set_names_only_the_unidentified_hypothesis(self):
        result = self._completion(
            "Rival: Cache compaction remains uncertain; Atlas is ambiguous.",
            [CAUSE, REPLICA_LAG, ATLAS_RIVAL],
        )
        self.assertNotEqual(result.get("status"), "complete")
        check = visibility(result)
        self.assertFalse(check["passed"], check["reason"])
        self.assertIn(REPLICA_LAG["statement"], check["reason"])
        self.assertNotIn(ATLAS_RIVAL["statement"], check["reason"])


if __name__ == "__main__":
    unittest.main()
