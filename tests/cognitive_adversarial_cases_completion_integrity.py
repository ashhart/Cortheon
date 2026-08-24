from __future__ import annotations

from cognitive_adversarial_cases_common import CompletionCase


class CompletionIntegrityMixin(CompletionCase):
    def test_test_result_must_follow_the_final_diff(self) -> None:
        runtime, session_id = self._code_session()
        runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "12 passed before editing",
                    "source": "pytest",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id="req3",
        )
        result = runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": "- return value\n+ return normalize(value)",
                    "source": "git diff",
                    "supports": ["h1"],
                }
            ],
            request_id="req4",
        )

        self.assertEqual(result["next_action"]["request"]["capability"], "test")
        self.assertIn("predates", result["next_action"]["request"]["reason"])

    def test_diff_that_disables_assertions_cannot_certify(self) -> None:
        runtime, session_id = self._code_session()
        runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": "- assert parse('') == empty\n+ pass",
                    "source": "git diff",
                    "supports": ["h1"],
                }
            ],
            request_id="req3",
        )
        result = runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "12 passed",
                    "source": "pytest",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id="req4",
        )

        self.assertEqual(result["next_action"]["request"]["capability"], "diff")
        self.assertIn("weaken", result["next_action"]["request"]["reason"])

    def test_replacing_an_assertion_is_not_misclassified_as_test_weakening(self) -> None:
        runtime, session_id = self._code_session()
        result = runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": (
                        "- assert parse('') == old_empty\n+ assert parse('') == normalized_empty"
                    ),
                    "source": "git diff",
                    "supports": ["h1"],
                }
            ],
            request_id="req3",
        )
        runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "12 passed",
                    "source": "pytest",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id="req4",
        )
        claims = [
            {
                "claim": "The updated assertion passes.",
                "evidence_ids": ["ev3", "ev4"],
            }
        ]
        runtime.step(session_id, draft="Updated and tested.")
        runtime.challenge(
            session_id,
            draft="Updated and tested.",
            claims=claims,
        )

        result = runtime.verify(
            session_id,
            answer="Updated and tested.",
            claims=claims,
            completion_evidence_ids=["ev3", "ev4"],
        )

        self.assertEqual(result["verification"]["verdict"], "ready")
