from __future__ import annotations

import io
import json
import random
import unittest

from cortheon.cognitive_mcp import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    MAX_MESSAGE_CHARS,
    CortheonMcpServer,
    serve,
)
from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveProtocolFuzzTests(unittest.TestCase):
    def test_client_controlled_messages_never_raise_internal_errors(self) -> None:
        randomizer = random.Random(20260725)
        server = CortheonMcpServer(CognitiveRuntime(max_sessions=128))
        methods = [
            "initialize",
            "ping",
            "tools/list",
            "tools/call",
            "unknown",
            "",
            None,
            7,
        ]
        arbitrary = [
            None,
            True,
            False,
            0,
            1.5,
            "",
            [],
            [1, 2],
            {},
            {"name": "missing"},
            {"name": "cortheon_start", "arguments": {"goal": ""}},
            {"name": "cortheon_finish", "arguments": {"session_id": "missing"}},
        ]
        for request_id in range(2_000):
            message = {
                "jsonrpc": randomizer.choice(["2.0", "1.0", None]),
                "id": request_id,
                "method": randomizer.choice(methods),
                "params": randomizer.choice(arbitrary),
            }
            response = server.handle(message)
            json.dumps(response)
            if response and "error" in response:
                self.assertNotEqual(
                    response["error"]["code"],
                    JSONRPC_INTERNAL_ERROR,
                    message,
                )

    def test_oversized_stdio_message_is_rejected_and_server_recovers(self) -> None:
        oversized = (
            '{"jsonrpc":"2.0","id":1,"method":"ping","padding":"'
            + ("x" * MAX_MESSAGE_CHARS)
            + '"}\n'
        )
        valid = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}) + "\n"
        output = io.StringIO()

        serve(CortheonMcpServer(), io.StringIO(oversized + valid), output)

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0]["error"]["code"], JSONRPC_INVALID_PARAMS)
        self.assertEqual(messages[1], {"jsonrpc": "2.0", "id": 2, "result": {}})

    def test_notifications_never_receive_responses(self) -> None:
        server = CortheonMcpServer()
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "cortheon_start",
                    "arguments": {"goal": "notification"},
                },
            }
        )
        self.assertIsNone(response)

    def test_malformed_params_are_invalid_not_internal(self) -> None:
        server = CortheonMcpServer()
        for params in ([], "bad", 7, True):
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": params,
                }
            )
            self.assertEqual(
                response["error"]["code"],  # pyright: ignore[reportOptionalSubscript]
                JSONRPC_INVALID_PARAMS,
            )
