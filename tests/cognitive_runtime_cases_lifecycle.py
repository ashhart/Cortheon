from __future__ import annotations

import builtins
from unittest.mock import patch

from cognitive_runtime_cases_common import ANSWER, CLAIMS, RuntimeTestCase

from cortheon.cognitive_runtime import (
    InvestigationNotFound,
)


class LifecycleMixin(RuntimeTestCase):
    def _start_code_task(self) -> tuple[str, dict]:
        result = self.runtime.start("Fix the parser bug in src/parser.py and verify it with tests")
        return result["session"]["session_id"], result

    def _ready_code_task(self) -> str:
        session_id, _ = self._start_code_task()
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
        self.runtime.step(
            session_id,
            hypotheses=[
                {
                    "statement": "The parser mishandles empty input.",
                    "falsification_test": "Inspect the empty-input branch.",
                },
                {
                    "statement": "A caller passes an invalid value.",
                    "falsification_test": "Inspect callers and their tests.",
                },
            ],
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "if not value: raise ParseError",
                    "source": "src/parser.py:20",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id="req2",
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "all callers normalize value before parse",
                    "source": "src/caller.py:10",
                    "status": "verified",
                    "contradicts": ["h2"],
                }
            ],
            request_id="req3",
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "The empty-input regression test fails before the fix.",
                    "source": "tests/test_parser.py",
                    "supports": ["h1"],
                }
            ],
            request_id="req4",
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": "- raise ParseError\n+ return empty_result",
                    "source": "git diff",
                    "supports": ["h1"],
                }
            ],
            request_id="req5",
        )
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "test",
                    "content": "pytest tests/test_parser.py: 12 passed",
                    "source": "pytest",
                    "status": "verified",
                    "supports": ["h1"],
                }
            ],
            request_id="req6",
        )
        self.runtime.step(session_id, draft=ANSWER)
        challenge = self.runtime.challenge(
            session_id,
            draft=ANSWER,
            claims=CLAIMS,
        )
        self.assertEqual(challenge["next_action"]["type"], "verify")
        verified = self.runtime.verify(
            session_id,
            answer=ANSWER,
            claims=CLAIMS,
            completion_evidence_ids=["ev5", "ev6"],
        )
        self.assertEqual(verified["verification"]["verdict"], "ready")
        return session_id

    def test_full_code_lifecycle_discards_everything(self) -> None:
        session_id = self._ready_code_task()

        result = self.runtime.finish(session_id, answer=ANSWER)

        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["discarded"])
        self.assertFalse(result["retained_project_data"])
        self.assertEqual(self.runtime.active_sessions, 0)
        with self.assertRaises(InvestigationNotFound):
            self.runtime.step(session_id)

    def test_runtime_never_reads_or_writes_a_file(self) -> None:
        original_open = builtins.open
        with patch("builtins.open", side_effect=AssertionError("file access")):
            session_id, result = self._start_code_task()
            self.assertEqual(result["session"]["storage"], "memory_only")
            abandoned = self.runtime.finish(session_id, mode="abandon")
        self.assertIs(builtins.open, original_open)
        self.assertTrue(abandoned["discarded"])

    def test_start_requests_live_host_evidence_without_running_a_tool(self) -> None:
        session_id, result = self._start_code_task()

        self.assertTrue(session_id.startswith("vx_"))
        self.assertEqual(result["next_action"]["type"], "harness_tool")
        self.assertEqual(result["next_action"]["request"]["capability"], "read_many")
        self.assertIn("Cortheon does not execute it", result["next_action"]["instruction"])

    def test_start_exposes_a_bounded_adaptive_orientation(self) -> None:
        _, result = self._start_code_task()

        cognition = result["cognition"]
        self.assertEqual(cognition["stage"], "orient")
        self.assertEqual(cognition["task_frame"]["deliverable"], "code_change")
        self.assertTrue(cognition["program"]["program_id"].startswith("cp_"))
        self.assertEqual(
            cognition["program"]["active_operator"]["operator_id"],
            "inspect_surface",
        )
        self.assertEqual(
            result["session"]["program_id"],
            cognition["program"]["program_id"],
        )
        self.assertEqual(
            cognition["evidence_target"]["capability"],
            result["next_action"]["request"]["capability"],
        )
        self.assertIn(
            "Acquire the requested live evidence before trusting model memory.",
            cognition["reasoning_moves"],
        )
        graph = result["context"]["cognitive_graph"]
        self.assertEqual(graph["node_count"], len(graph["nodes"]))
        self.assertTrue(graph["digest"].startswith("cg_"))
        self.assertEqual(
            cognition["evidence_target"]["selection"]["action_id"],
            result["next_action"]["request"]["request_id"],
        )
        self.assertIn(
            "information_gain_bits",
            cognition["evidence_target"]["selection"],
        )
        self.assertIn("Revise when", cognition["decision_rule"])

    def test_information_gain_controller_selects_discriminating_evidence(self) -> None:
        started = self.runtime.start(
            "Determine why tenant requests are returning the wrong result.",
            effort="standard",
            task_kind="general",
        )
        session_id = started["session"]["session_id"]
        self.runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": "Failures affect several tenants after a cache rollout.",
                    "source": "incident note",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        stepped = self.runtime.step(
            session_id,
            hypotheses=[
                {
                    "statement": "A database timeout returns the wrong fallback.",
                    "falsification_test": "Inspect database timeout handling.",
                },
                {
                    "statement": "Tenant cache key normalization causes collisions.",
                    "falsification_test": "Inspect tenant cache key normalization.",
                },
                {
                    "statement": "Cache normalization drops the tenant prefix.",
                    "falsification_test": "Inspect cache normalization and tenant prefix.",
                },
            ],
        )

        request = stepped["next_action"]["request"]
        controller = request["parameters"]["controller"]
        selection = stepped["cognition"]["evidence_target"]["selection"]
        alternatives = stepped["cognition"]["evidence_target"]["alternatives"]
        self.assertEqual(controller["policy"], "expected_information_gain_per_cost")
        self.assertEqual(request["hypothesis_id"], "h2")
        self.assertEqual(request["capability"], "read")
        self.assertEqual(selection, controller["selected"])
        self.assertEqual(selection["resolves"], ["h2", "h3"])
        self.assertGreater(
            selection["expected_utility"],
            alternatives[-1]["expected_utility"],
        )
        self.assertEqual(
            stepped["cognition"]["evidence_target"]["stop_when"],
            "mandatory evidence gate satisfied",
        )
        self.assertEqual(self.runtime.metrics["controller_decisions"], 1)
        self.assertEqual(
            self.runtime.metrics["controller_alternatives_considered"],
            3,
        )
        self.assertGreater(  # pyright: ignore[reportCallIssue, reportArgumentType]
            self.runtime.metrics["controller_information_gain_bits_total"],
            0,  # pyright: ignore[reportArgumentType]
        )

    def test_cross_file_question_requests_separate_focused_live_reads(self) -> None:
        result = self.runtime.start(
            "Read DEFAULT_PORT in src/cortheon/cognitive_http.py and "
            "MAX_MESSAGE_CHARS in src/cortheon/cognitive_mcp.py; what is their sum?",
            effort="quick",
        )

        request = result["next_action"]["request"]
        self.assertEqual(request["capability"], "read_many")
        self.assertEqual(
            request["parameters"]["paths"],
            [
                "src/cortheon/cognitive_http.py",
                "src/cortheon/cognitive_mcp.py",
            ],
        )
        self.assertEqual(
            request["parameters"]["symbols"],
            ["DEFAULT_PORT", "MAX_MESSAGE_CHARS"],
        )
        self.assertEqual(request["parameters"]["operation"], "sum")
        self.assertIn("separately sourced excerpts", request["success_condition"])

    def test_diagnostic_join_originates_expected_actual_mismatch(self) -> None:
        started = self.runtime.start(
            "Diagnose auth using settings.py, factory.py, and auth_trace.log. "
            "Identify the exact mismatched claim and rule out issuer.",
            effort="quick",
        )
        request = started["next_action"]["request"]
        self.assertEqual(request["parameters"]["operation"], "diagnostic_join")
        observations = []
        contents = {
            "settings.py": 'EXPECTED_AUDIENCE = "orders-api"\n',
            "factory.py": 'return {"aud": "order-api", "iss": "identity"}\n',
            "auth_trace.log": (
                "issuer check: passed\n"
                "audience check: expected=orders-api actual=order-api failed\n"
            ),
        }
        for path, content in contents.items():
            observations.append(
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{content}'
                    ),
                    "source": f"pi:read:{path}",
                }
            )

        observed = self.runtime.observe(
            started["session"]["session_id"],
            observations,
            request_id=request["request_id"],
        )

        diagnosis = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(diagnosis["operation"], "diagnostic_chain")
        self.assertIn("Expected orders-api", diagnosis["answer"])
        self.assertIn("order-api", diagnosis["answer"])
        self.assertIn("Every other recorded check passed", diagnosis["answer"])
        self.assertNotIn("issuer", diagnosis["answer"])

    def test_diagnostic_join_originates_boundary_failure(self) -> None:
        started = self.runtime.start(
            "Diagnose why collect in pager.py returns no rows using "
            "api_contract.md and pager_trace.log.",
            effort="quick",
        )
        request = started["next_action"]["request"]
        observations = []
        contents = {
            "pager.py": "def collect(fetch):\n    page = 0\n    return fetch(page)\n",
            "api_contract.md": (
                "The endpoint uses one-based pages. Page 1 is valid and page 0 "
                "returns an empty result.\n"
            ),
            "pager_trace.log": "request page=0 status=200 rows=0\n",
        }
        for path, content in contents.items():
            observations.append(
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{content}'
                    ),
                    "source": f"pi:read:{path}",
                }
            )

        observed = self.runtime.observe(
            started["session"]["session_id"],
            observations,
            request_id=request["request_id"],
        )

        diagnosis = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(diagnosis["nodes"], ["page = 0", "one-based", "empty result"])

    def test_named_code_change_reads_implementation_and_test_before_edit(self) -> None:
        result = self.runtime.start(
            "Fix calculator.add in calculator.py so test_calculator.py passes. "
            "Do not change the test.",
            effort="quick",
        )

        request = result["next_action"]["request"]
        self.assertEqual(request["capability"], "read_many")
        self.assertEqual(
            request["parameters"]["paths"],
            ["calculator.py", "test_calculator.py"],
        )
        self.assertEqual(request["parameters"]["symbols"], ["add"])
        self.assertIn("Do not edit before", request["success_condition"])

    def test_named_documents_request_a_semantic_join(self) -> None:
        result = self.runtime.start(
            "Read service_catalog.md, change_policy.md, and org_directory.md as "
            "separate documents. Who approves the Checkout rollback?",
            effort="quick",
        )

        request = result["next_action"]["request"]
        self.assertEqual(request["capability"], "read_many")
        self.assertEqual(
            request["parameters"]["paths"],
            [
                "service_catalog.md",
                "change_policy.md",
                "org_directory.md",
            ],
        )
        self.assertEqual(request["parameters"]["operation"], "semantic_join")
        self.assertIn("source boundaries", request["success_condition"])

    def test_named_documents_take_precedence_over_ambiguous_code_terms(self) -> None:
        result = self.runtime.start(
            "Read dependency_map.md, asset_register.md, and oncall_roster.md as "
            "separate documents. Which responder owns the blocker?",
            effort="quick",
        )

        self.assertEqual(result["session"]["task_kind"], "documents")
        request = result["next_action"]["request"]
        self.assertEqual(request["capability"], "read_many")
        self.assertEqual(request["parameters"]["operation"], "semantic_join")
