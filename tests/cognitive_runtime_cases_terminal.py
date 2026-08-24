from __future__ import annotations

from cognitive_runtime_cases_common import FakeClock, RuntimeTestCase

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
    CognitiveRuntimeError,
    InvestigationNotFound,
)


class TerminalMixin(RuntimeTestCase):
    def test_compact_completion_rejects_claim_that_contradicts_exact_search(self) -> None:
        started = self.runtime.start(
            "Use Cortheon. Does src/example.py import pathlib? Name the import if present.",
            effort="quick",
        )
        self.assertEqual(started["next_action"]["request"]["capability"], "grep")
        self.assertIn("pattern 'pathlib'", started["next_action"]["request"]["query"])
        session_id = started["session"]["session_id"]
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"grep","outcome":"no_match",'
                        '"args":{"pattern":"pathlib","path":"src/example.py"}}\n'
                        "No files found"
                    ),
                    "source": "opencode:grep",
                    "status": "verified",
                }
            ],
            request_id="req1",
        )

        rejected = self.runtime.complete(
            session_id,
            answer="Yes.",
            claims=[
                {
                    "claim": "src/example.py imports pathlib.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "src/example.py imports pathlib.",
                    "falsification_test": "Search the file for pathlib.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )

        checks = {item["name"]: item for item in rejected["verification"]["checks"]}
        self.assertFalse(checks["evidence_alignment"]["passed"])
        self.assertEqual(rejected["next_action"]["submit_via"], "cortheon_complete")
        self.assertEqual(self.runtime.active_sessions, 1)

        accepted = self.runtime.complete(
            session_id,
            answer="No.",
            claims=[
                {
                    "claim": "src/example.py does not import pathlib.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "src/example.py has no pathlib import.",
                    "falsification_test": "Search the file for pathlib.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )

        self.assertEqual(accepted["status"], "complete")
        self.assertEqual(accepted["answer"], "No.")
        self.assertEqual(self.runtime.active_sessions, 0)

    def test_named_entrypoint_mapping_becomes_an_exact_scoped_lookup(self) -> None:
        started = self.runtime.start(
            "Read pyproject.toml and report which command maps to cortheon.cognitive_cli:main.",
            effort="quick",
        )

        request = started["next_action"]["request"]
        self.assertEqual(request["capability"], "grep")
        self.assertEqual(
            request["parameters"],
            {
                "pattern": "cortheon.cognitive_cli:main",
                "path": "pyproject.toml",
                "tool_call_budget": 3,
            },
        )

    def test_url_path_is_not_misclassified_as_a_local_grep_scope(self) -> None:
        started = self.runtime.start(
            "Set up the existing repository. Install the CLI with "
            "curl -fsSL https://api.mlx.fast/install.sh | sh, then inspect the "
            "repository and run its baseline before making optimizations.",
            effort="quick",
        )

        request = started["next_action"]["request"]
        self.assertNotEqual(request["capability"], "grep")
        self.assertNotEqual(
            request["parameters"].get("path"),
            "api.mlx.fast/install.sh",
        )

    def test_requested_live_evidence_must_appear_in_atomic_lookup_answer(self) -> None:
        started = self.runtime.start(
            "Does src/example.py import pathlib? Cite the live file evidence.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"grep","outcome":"no_match",'
                        '"args":{"pattern":"pathlib","path":"src/example.py"}}\n'
                        "No matches found."
                    ),
                    "status": "verified",
                }
            ],
            request_id="req1",
        )
        completion = {
            "session_id": session_id,
            "claims": [
                {
                    "claim": "src/example.py does not import pathlib.",
                    "evidence_ids": ["ev1"],
                }
            ],
            "hypotheses": [
                {
                    "statement": "src/example.py has no pathlib import.",
                    "falsification_test": "Search the file.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            "completion_evidence_ids": ["ev1"],
        }

        rejected = self.runtime.complete(answer="No.", **completion)
        alignment = next(
            item
            for item in rejected["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("does not name", alignment["reason"])

        accepted = self.runtime.complete(
            answer="No — src/example.py returned zero matches for pathlib.",
            **completion,
        )
        self.assertEqual(accepted["status"], "complete")

    def test_idle_sessions_expire_and_are_erased(self) -> None:
        clock = FakeClock()
        runtime = CognitiveRuntime(ttl_seconds=30, clock=clock)
        started = runtime.start("Investigate a task")
        session_id = started["session"]["session_id"]
        clock.advance(31)

        with self.assertRaises(InvestigationNotFound):
            runtime.step(session_id)
        self.assertEqual(runtime.active_sessions, 0)

    def test_native_adapter_lease_reaps_crashed_host_and_heartbeat_renews(self) -> None:
        clock = FakeClock()
        runtime = CognitiveRuntime(ttl_seconds=1_800, clock=clock)
        started = runtime.start(
            "Inspect live code",
            effort="quick",
            lease_seconds=10,
        )
        session_id = started["session"]["session_id"]

        clock.advance(9)
        heartbeat = runtime.heartbeat(session_id)
        self.assertEqual(heartbeat["lease_seconds"], 10.0)
        clock.advance(9)
        self.assertEqual(runtime.active_sessions, 1)
        clock.advance(2)

        self.assertEqual(runtime.active_sessions, 0)
        self.assertEqual(runtime.metrics["sessions_expired"], 1)

    def test_session_and_observation_limits_fail_closed(self) -> None:
        runtime = CognitiveRuntime(max_sessions=1)
        runtime.start("First task")
        with self.assertRaisesRegex(CognitiveRuntimeError, "session limit"):
            runtime.start("Second task")

        quick = CognitiveRuntime()
        started = quick.start("Inspect code", effort="quick", task_kind="code")
        with self.assertRaisesRegex(ValueError, "2000-character"):
            quick.observe(
                started["session"]["session_id"],
                [{"kind": "code", "content": "x" * 2_001}],
                request_id="req1",
            )
