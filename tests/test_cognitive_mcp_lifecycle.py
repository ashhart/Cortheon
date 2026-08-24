"""Investigation lifecycle over tools/call: start, compaction, resume."""

import unittest

from cognitive_mcp_helpers import request

from cortheon.cognitive_mcp import CortheonMcpServer
from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveMcpLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = CortheonMcpServer(CognitiveRuntime())

    def test_start_call_returns_structured_host_request(self) -> None:
        response = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "Fix the parser bug"},
                },
            },
        )

        result = response["result"]
        self.assertFalse(result["isError"])
        content = result["structuredContent"]
        self.assertEqual(content["session"]["storage"], "memory_only")
        self.assertEqual(content["next_action"]["type"], "harness_tool")

    def test_compact_response_never_names_a_hidden_tool(self) -> None:
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "Does src/example.py import pathlib?"},
                },
            },
        )["result"]["structuredContent"]
        observed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_observe",
                    "arguments": {
                        "session_id": started["session"]["session_id"],
                        "request_id": started["next_action"]["request"]["request_id"],
                        "observations": [
                            {
                                "kind": "code",
                                "content": "import json\n",
                                "source": "host:read",
                            }
                        ],
                    },
                },
            },
        )["result"]["structuredContent"]

        self.assertEqual(observed["next_action"]["submit_via"], "cortheon_complete")
        self.assertNotIn("cortheon_step", observed["next_action"]["instruction"])

    def test_resume_recovers_goal_and_next_action_after_context_loss(self) -> None:
        started = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 71,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "Does src/example.py import pathlib?"},
                },
            },
        )["result"]["structuredContent"]

        resumed = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 72,
                "method": "tools/call",
                "params": {"name": "cortheon_resume", "arguments": {}},
            },
        )["result"]["structuredContent"]

        session = resumed["sessions"][0]
        self.assertIn("pathlib", session["goal"])
        self.assertEqual(
            session["next_action"]["request"]["request_id"],
            started["next_action"]["request"]["request_id"],
        )
        self.assertIn("context", session)


if __name__ == "__main__":
    unittest.main()
