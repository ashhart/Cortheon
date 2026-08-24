"""Host-receipt certification, argument repair, and evidence retraction."""

import json
import unittest

from cognitive_mcp_helpers import request

from cortheon.cognitive_mcp import CortheonMcpServer
from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveMcpEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = CortheonMcpServer(CognitiveRuntime())

    def test_structured_host_receipt_certifies_generic_mcp_lookup(self) -> None:
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 31,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {
                        "goal": "Does src/example.py import pathlib?",
                        "effort": "quick",
                    },
                },
            },
        )["result"]["structuredContent"]
        session_id = started["session"]["session_id"]
        request_id = started["next_action"]["request"]["request_id"]

        observed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 32,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_observe",
                    "arguments": {
                        "session_id": session_id,
                        "request_id": request_id,
                        "observations": [
                            {
                                "kind": "code",
                                "content": "No matches found.",
                                "host_receipt": {
                                    "tool": "grep",
                                    "outcome": "no_match",
                                    "args": {
                                        "pattern": "pathlib",
                                        "path": "src/example.py",
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        )["result"]["structuredContent"]

        self.assertEqual(observed["accepted_evidence_ids"], ["ev1"])
        completed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 33,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_complete",
                    "arguments": {
                        "session_id": session_id,
                        "answer": "No.",
                        "claims": [
                            {
                                "claim": "src/example.py does not import pathlib.",
                                "evidence_ids": ["ev1"],
                            }
                        ],
                        "hypotheses": [
                            {
                                "statement": "src/example.py has no pathlib import.",
                                "falsification_test": "Search the file for pathlib.",
                                "status": "supported",
                                "evidence_ids": ["ev1"],
                            }
                        ],
                        "completion_evidence_ids": ["ev1"],
                    },
                },
            },
        )["result"]["structuredContent"]

        self.assertEqual(completed["status"], "complete")
        self.assertEqual(self.server.runtime.active_sessions, 0)

    def test_structured_search_receipt_advances_code_discovery(self) -> None:
        self.server = CortheonMcpServer(CognitiveRuntime(require_host_receipts=True))
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 37,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {
                        "goal": "Extend the paired benchmark and add a regression test.",
                        "effort": "quick",
                    },
                },
            },
        )["result"]["structuredContent"]

        observed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 38,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_observe",
                    "arguments": {
                        "session_id": started["session"]["session_id"],
                        "request_id": started["next_action"]["request"]["request_id"],
                        "observations": [
                            {
                                "kind": "code",
                                "content": (
                                    "src/cortheon/cognitive_benchmark.py:2458: "
                                    "host dispatch\n"
                                    "tests/test_cognitive_benchmark.py:610: benchmark CLI"
                                ),
                                "status": "verified",
                                "host_receipt": {
                                    "tool": "search",
                                    "executor": "shell",
                                    "outcome": "match",
                                    "args": {
                                        "command": "rg -n 'host|benchmark' src tests",
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        )["result"]["structuredContent"]

        follow_up = observed["next_action"]["request"]
        self.assertEqual(follow_up["capability"], "read_many")
        self.assertEqual(follow_up["parameters"]["operation"], "code_context")
        self.assertEqual(
            follow_up["parameters"]["paths"][:2],
            [
                "src/cortheon/cognitive_benchmark.py",
                "tests/test_cognitive_benchmark.py",
            ],
        )

    def test_compact_tools_repair_json_encoded_array_arguments(self) -> None:
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 34,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "Does src/example.py import pathlib?"},
                },
            },
        )["result"]["structuredContent"]
        session_id = started["session"]["session_id"]
        request_id = started["next_action"]["request"]["request_id"]

        observed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 35,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_observe",
                    "arguments": {
                        "session_id": session_id,
                        "request_id": request_id,
                        "observations": json.dumps(
                            [
                                {
                                    "kind": "code",
                                    "content": "No matches found.",
                                    "host_receipt": {
                                        "tool": "grep",
                                        "outcome": "no_match",
                                        "args": {
                                            "pattern": "pathlib",
                                            "path": "src/example.py",
                                        },
                                    },
                                }
                            ]
                        ),
                    },
                },
            },
        )["result"]["structuredContent"]
        self.assertEqual(observed["accepted_evidence_ids"], ["ev1"])

        completed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 36,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_complete",
                    "arguments": {
                        "session_id": session_id,
                        "answer": "No.",
                        "claims": json.dumps(
                            [
                                {
                                    "claim": "src/example.py does not import pathlib.",
                                    "evidence_ids": ["ev1"],
                                }
                            ]
                        ),
                        "hypotheses": json.dumps(
                            [
                                {
                                    "statement": "The file has no pathlib import.",
                                    "falsification_test": "Search the file for pathlib.",
                                    "status": "supported",
                                    "evidence_ids": ["ev1"],
                                }
                            ]
                        ),
                        "completion_evidence_ids": json.dumps(["ev1"]),
                    },
                },
            },
        )["result"]["structuredContent"]

        self.assertEqual(completed["status"], "complete")
        self.assertEqual(self.server.runtime.active_sessions, 0)

    def test_retract_withdraws_mis_marked_evidence(self) -> None:
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 41,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "Does src/example.py import pathlib?"},
                },
            },
        )["result"]["structuredContent"]
        session_id = started["session"]["session_id"]
        observed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_observe",
                    "arguments": {
                        "session_id": session_id,
                        "request_id": started["next_action"]["request"]["request_id"],
                        "observations": [{"kind": "code", "content": "import pathlib\n"}],
                    },
                },
            },
        )["result"]["structuredContent"]
        self.assertEqual(observed["accepted_evidence_ids"], ["ev1"])

        retracted = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 43,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_retract",
                    "arguments": {
                        "session_id": session_id,
                        "evidence_ids": ["ev1"],
                        "reason": "The excerpt came from the wrong file.",
                    },
                },
            },
        )["result"]["structuredContent"]

        self.assertEqual(retracted["retracted_evidence_ids"], ["ev1"])
        self.assertIn("next_action", retracted)

    def test_malformed_observe_returns_example_and_waives_the_request(self) -> None:
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 61,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "Does src/example.py import pathlib?"},
                },
            },
        )["result"]["structuredContent"]
        session_id = started["session"]["session_id"]
        request_id = started["next_action"]["request"]["request_id"]

        def observe_empty(call_id: int) -> dict:
            return request(
                self.server,
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "method": "tools/call",
                    "params": {
                        "name": "cortheon_observe",
                        "arguments": {
                            "session_id": session_id,
                            "request_id": request_id,
                            "observations": [],
                        },
                    },
                },
            )

        first = observe_empty(62)
        self.assertEqual(first["error"]["code"], -32602)
        self.assertIn("Correct example call", first["error"]["message"])
        self.assertIn(request_id, first["error"]["message"])

        second = observe_empty(63)
        self.assertIn("waived", second["error"]["message"])

        resumed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 64,
                "method": "tools/call",
                "params": {"name": "cortheon_resume", "arguments": {}},
            },
        )["result"]["structuredContent"]
        self.assertEqual(resumed["sessions"][0]["session_id"], session_id)
        self.assertNotEqual(
            resumed["sessions"][0]["next_action"]["type"],
            "harness_tool",
        )


if __name__ == "__main__":
    unittest.main()
