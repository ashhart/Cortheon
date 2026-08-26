"""The tool catalogue and the JSON Schemas tools/list publishes."""

import unittest

from cortheon.cognitive_mcp import tool_definitions


class CognitiveMcpToolCatalogueTests(unittest.TestCase):
    def test_surface_contains_only_cognitive_operations(self) -> None:
        names = [item["name"] for item in tool_definitions()]

        self.assertEqual(
            names,
            [
                "cortheon_start",
                "cortheon_observe",
                "cortheon_complete",
                "cortheon_retract",
                "cortheon_abandon",
                "cortheon_resume",
            ],
        )
        descriptions = " ".join(item["description"] for item in tool_definitions())
        self.assertNotIn("execute shell", descriptions.lower())
        self.assertNotIn("read file", descriptions.lower())
        for tool in tool_definitions():
            self.assertEqual(
                tool["annotations"],
                {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": False,
                    "openWorldHint": False,
                },
            )

    def test_advanced_surface_is_explicitly_opt_in(self) -> None:
        names = [item["name"] for item in tool_definitions(advanced=True)]

        self.assertEqual(
            names,
            [
                "cortheon_start",
                "cortheon_observe",
                "cortheon_complete",
                "cortheon_retract",
                "cortheon_abandon",
                "cortheon_resume",
                "cortheon_step",
                "cortheon_challenge",
                "cortheon_verify",
                "cortheon_finish",
            ],
        )

    def test_start_schema_offers_strictness_profiles(self) -> None:
        start_tool = next(item for item in tool_definitions() if item["name"] == "cortheon_start")

        strictness = start_tool["inputSchema"]["properties"]["strictness"]
        self.assertEqual(strictness["enum"], ["strict", "standard", "assist"])
        self.assertEqual(strictness["default"], "standard")

    def test_observe_schema_requires_request_id(self) -> None:
        observation_tool = next(
            item for item in tool_definitions() if item["name"] == "cortheon_observe"
        )

        self.assertIn("request_id", observation_tool["inputSchema"]["required"])
        observation = observation_tool["inputSchema"]["properties"]["observations"]["items"]
        self.assertIn("host_receipt", observation["properties"])

    def test_observe_schema_can_submit_source_review_evidence(self) -> None:
        observation_tool = next(
            item for item in tool_definitions() if item["name"] == "cortheon_observe"
        )
        observation = observation_tool["inputSchema"]["properties"]["observations"]["items"]

        self.assertIn("source_record", observation["properties"])
        self.assertIn(
            "scholarly_validation",
            observation["properties"]["purpose"]["enum"],
        )
        self.assertIn(
            "implementation_reference",
            observation["properties"]["purpose"]["enum"],
        )


if __name__ == "__main__":
    unittest.main()
