"""JSON-RPC framing, initialize, and runtime error mapping over stdio."""

import io
import json
import unittest

from cognitive_mcp_helpers import request

from cortheon.cognitive_mcp import CortheonMcpServer, serve
from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveMcpProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = CortheonMcpServer(CognitiveRuntime())

    def test_initialize_describes_ephemeral_boundary(self) -> None:
        response = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
        )

        result = response["result"]
        self.assertEqual(result["serverInfo"]["name"], "cortheon")
        self.assertIn("stores no project files", result["instructions"])
        self.assertEqual(set(result["capabilities"]), {"tools"})

    def test_protocol_errors_are_bounded_jsonrpc_errors(self) -> None:
        response = request(
            self.server,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_finish",
                    "arguments": {"session_id": "missing"},
                },
            },
        )

        self.assertEqual(response["error"]["code"], -32602)
        self.assertIn("not found", response["error"]["message"])

    def test_stdio_server_handles_initialize_and_notification(self) -> None:
        incoming = io.StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "initialize",
                    "params": {},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            )
            + "\n"
        )
        outgoing = io.StringIO()

        serve(self.server, incoming, outgoing)

        messages = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["id"], 7)


if __name__ == "__main__":
    unittest.main()
