from __future__ import annotations

from cognitive_runtime_cases_common import RuntimeTestCase


class CompletionMixin(RuntimeTestCase):
    def test_compact_completion_challenges_verifies_and_discards(self) -> None:
        started = self.runtime.start(
            "Does src/parser.py normalize empty input?",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        observed = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                        '"outcome":"result","args":{"filePath":"src/parser.py"}}\n'
                        "def parse(value): return normalize(value)"
                    ),
                    "source": "src/parser.py:1",
                    "status": "verified",
                }
            ],
            request_id="req1",
        )
        self.assertEqual(observed["accepted_evidence_ids"], ["ev1"])

        result = self.runtime.complete(
            session_id,
            answer="Yes. parse passes its input to normalize.",
            claims=[
                {
                    "claim": "parse passes its input to normalize.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "parse normalizes its input.",
                    "falsification_test": "Inspect the parse implementation.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["scorecard"]["turns"], 1)
        self.assertEqual(result["scorecard"]["challenges"], 1)
        self.assertEqual(
            result["scorecard"]["verification"]["verdict"],
            "ready",
        )
        self.assertEqual(self.runtime.active_sessions, 0)

    def test_compact_completion_allows_auxiliary_completion_evidence(self) -> None:
        started = self.runtime.start(
            "Does src/parser.py normalize empty input?",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        observed = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                        '"outcome":"result","args":{"filePath":"src/parser.py"}}\n'
                        "def parse(value): return normalize(value)"
                    ),
                    "source": "src/parser.py:1",
                    "status": "verified",
                },
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                        '"outcome":"result","args":{"filePath":"src/settings.py"}}\n'
                        "MAX_RETRIES = 3"
                    ),
                    "source": "src/settings.py:1",
                    "status": "verified",
                },
            ],
            request_id="req1",
        )
        self.assertEqual(observed["accepted_evidence_ids"], ["ev1", "ev2"])

        result = self.runtime.complete(
            session_id,
            answer="Yes. parse passes its input to normalize.",
            claims=[
                {
                    "claim": "parse passes its input to normalize.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "parse normalizes its input.",
                    "falsification_test": "Inspect the parse implementation.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1", "ev2"],
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["scorecard"]["verification"]["verdict"], "ready")

    def test_requirement_gate_reopens_only_the_missing_deliverable(self) -> None:
        started = self.runtime.start(
            "Update src/client.py to add a timeout and update README.md with usage. "
            "Run python3 -m pytest -q after the edit.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        self.assertEqual(
            started["next_action"]["request"]["parameters"]["paths"],
            ["src/client.py", "README.md"],
        )
        self.assertEqual(
            [item["proof"] for item in started["context"]["requirements"]],
            ["mutation", "mutation", "verification"],
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "class Client: timeout = None",
                    "source": "src/client.py",
                },
                {
                    "kind": "documentation",
                    "content": "# Client usage",
                    "source": "README.md",
                },
            ],
            request_id="req1",
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": (
                        "--- a/src/client.py\n+++ b/src/client.py\n"
                        "-    timeout = None\n+    timeout = 30"
                    ),
                    "source": "git diff",
                }
            ],
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "python3 -m pytest -q: 12 passed",
                    "source": "pytest",
                    "status": "verified",
                }
            ],
        )
        claim = {
            "claim": "The client timeout is implemented and the tests pass.",
            "evidence_ids": ["ev1", "ev2", "ev3", "ev4"],
        }
        hypothesis = {
            "statement": "The requested timeout change is complete.",
            "falsification_test": "Inspect the diff and rerun pytest.",
            "status": "supported",
            "evidence_ids": ["ev1", "ev2", "ev3", "ev4"],
        }
        withheld = self.runtime.complete(
            session_id,
            answer="Added the client timeout; tests pass.",
            claims=[claim],
            hypotheses=[hypothesis],
            completion_evidence_ids=["ev1", "ev2", "ev3", "ev4"],
        )
        requirement_check = next(
            item
            for item in withheld["verification"]["checks"]
            if item["name"] == "requirement_coverage"
        )
        self.assertEqual(
            [
                item["requirement_id"]
                for item in requirement_check["requirements"]
                if item["status"] != "covered"
            ],
            ["r2"],
        )
        self.assertIn("README.md", withheld["next_action"]["instruction"])

        after_documentation = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": (
                        "--- a/README.md\n+++ b/README.md\n"
                        "-# Client usage\n+# Client usage\n"
                        "+Timeout defaults to 30 seconds."
                    ),
                    "source": "git diff",
                }
            ],
        )
        test_request = after_documentation["next_action"]["request"]
        self.assertEqual(test_request["capability"], "test")
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "python3 -m pytest -q: 12 passed",
                    "source": "pytest final",
                    "status": "verified",
                }
            ],
            request_id=test_request["request_id"],
        )
        evidence_ids = [f"ev{index}" for index in range(1, 7)]
        completed = self.runtime.complete(
            session_id,
            answer="Added the client timeout, documented its usage, and verified tests.",
            claims=[
                {
                    "claim": ("The client timeout and README usage are updated, and tests pass."),
                    "evidence_ids": evidence_ids,
                }
            ],
            hypotheses=[
                {
                    "statement": "Every requested deliverable is complete.",
                    "falsification_test": "Inspect both diffs and rerun pytest.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            completion_evidence_ids=evidence_ids,
        )
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["scorecard"]["requirements"], 3)
        self.assertEqual(completed["scorecard"]["covered_requirements"], 3)

    def test_retracting_verification_reopens_only_that_requirement(self) -> None:
        started = self.runtime.start(
            "Fix the defect in src/parser.py and verify with tests.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "def parse(value): return value",
                    "source": "src/parser.py",
                }
            ],
            request_id="req1",
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": (
                        "--- a/src/parser.py\n+++ b/src/parser.py\n"
                        "-    return value\n+    return normalize(value)"
                    ),
                    "source": "git diff",
                },
                {
                    "kind": "test",
                    "content": "12 passed",
                    "source": "pytest",
                    "status": "verified",
                },
            ],
        )
        reopened = self.runtime.retract(
            session_id,
            ["ev3"],
            reason="The test observation was mis-marked.",
        )
        statuses = {item["proof"]: item["status"] for item in reopened["context"]["requirements"]}
        self.assertEqual(statuses["mutation"], "covered")
        self.assertEqual(statuses["verification"], "contradicted")

    def test_failed_compact_completion_is_atomic_and_stays_open(self) -> None:
        started = self.runtime.start(
            "Does src/parser.py normalize empty input?",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        self.runtime.observe(
            session_id,
            [{"kind": "code", "content": "def parse(value): return value"}],
            request_id="req1",
        )

        with self.assertRaisesRegex(ValueError, "unknown evidence ids"):
            self.runtime.complete(
                session_id,
                answer="Yes.",
                claims=[{"claim": "It normalizes.", "evidence_ids": ["ev999"]}],
                hypotheses=[
                    {
                        "statement": "It normalizes.",
                        "falsification_test": "Inspect it.",
                        "status": "supported",
                        "evidence_ids": ["ev999"],
                    }
                ],
                completion_evidence_ids=["ev999"],
            )

        result = self.runtime.step(
            session_id,
            hypotheses=[
                {
                    "statement": "It does not normalize.",
                    "falsification_test": "Inspect it.",
                }
            ],
        )
        self.assertEqual(result["session"]["turns_used"], 1)
