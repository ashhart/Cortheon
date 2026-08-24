from __future__ import annotations

from cognitive_adversarial_cases_common import CompletionCase


class CompletionEvidenceMixin(CompletionCase):
    def test_code_change_cannot_pass_without_diff(self) -> None:
        runtime, session_id = self._code_session()
        pending = runtime.step(session_id, draft="Fixed and tested.")
        self.assertEqual(pending["next_action"]["request"]["capability"], "diff")

    def test_new_file_diff_receipt_advances_instead_of_requesting_diff_again(
        self,
    ) -> None:
        runtime, session_id = self._code_session()
        pending = runtime.step(session_id, draft="Added the new plugin.")
        self.assertEqual(pending["next_action"]["request"]["capability"], "diff")

        observed = runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"diff",'
                        '"outcome":"changed","args":{'
                        '"cmd":"git diff --no-index /dev/null '
                        'neural-memory/src/index.ts",'
                        '"path":"neural-memory/src/index.ts"}}\n'
                        "Focused new-file diff adds neural-memory/src/index.ts "
                        "with the PI extension entrypoint."
                    ),
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id=pending["next_action"]["request"]["request_id"],
        )

        self.assertEqual(observed["accepted_evidence_ids"], ["ev3"])
        self.assertEqual(observed["next_action"]["request"]["capability"], "test")

    def test_changed_diff_receipt_without_a_target_is_not_completion_evidence(
        self,
    ) -> None:
        runtime, session_id = self._code_session()
        pending = runtime.step(session_id, draft="Fixed.")

        observed = runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"diff",'
                        '"outcome":"changed","args":{"command":"git diff --stat"}}\n'
                        "Changes were reported."
                    ),
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id=pending["next_action"]["request"]["request_id"],
        )

        self.assertEqual(observed["next_action"]["request"]["capability"], "diff")

    def test_required_completion_evidence_must_be_cited_by_answer(self) -> None:
        runtime, session_id = self._code_session()
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
        runtime.step(session_id, draft="Fixed and tested.")
        runtime.challenge(
            session_id,
            draft="Fixed and tested.",
            claims=[{"claim": "The fix changes normalization.", "evidence_ids": ["ev3"]}],
        )

        result = runtime.verify(
            session_id,
            answer="Fixed and tested.",
            claims=[{"claim": "The fix changes normalization.", "evidence_ids": ["ev3"]}],
            completion_evidence_ids=["ev3", "ev4"],
        )

        self.assertEqual(result["verification"]["verdict"], "needs_evidence")
        completion = next(
            item
            for item in result["verification"]["checks"]
            if item["name"] == "completion_evidence"
        )
        self.assertTrue(completion["passed"])
        requirements = next(
            item
            for item in result["verification"]["checks"]
            if item["name"] == "requirement_coverage"
        )
        self.assertFalse(requirements["passed"])
        self.assertIn("no completion claim binds", requirements["reason"])

    def test_code_change_passes_with_cited_diff_and_verified_test(self) -> None:
        runtime, session_id = self._code_session()
        runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": "- return value\n+ return normalize(value)",
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
                "claim": "The normalization change passes the parser tests.",
                "evidence_ids": ["ev3", "ev4"],
            }
        ]
        runtime.step(session_id, draft="Fixed and tested.")
        runtime.challenge(session_id, draft="Fixed and tested.", claims=claims)

        result = runtime.verify(
            session_id,
            answer="Fixed and tested.",
            claims=claims,
            completion_evidence_ids=["ev3", "ev4"],
        )

        self.assertEqual(result["verification"]["verdict"], "ready")
