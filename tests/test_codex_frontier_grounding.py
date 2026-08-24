"""Codex host coverage for current knowledge inside code tasks."""

from cortheon.cognitive_hooks import CognitiveHookTracker
from cortheon.cognitive_runtime import CognitiveRuntime


def test_codex_tracker_brings_current_web_evidence_into_a_code_task() -> None:
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    registered = tracker.register(
        "codex",
        "session-frontier",
        "turn-frontier",
        goal=(
            "Implement a production-ready HTTP client using the exact installed runtime "
            "versions and current official documentation."
        ),
        effort="deep",
    )
    assert registered["next_action"]["request"]["parameters"]["operation"] == (
        "environment_grounding"
    )
    tracker.pre_tool(
        "codex",
        "session-frontier",
        "turn-frontier",
        "read",
        tool_input={"filePath": "pyproject.toml"},
    )
    grounded = tracker.post_tool(
        "codex",
        "session-frontier",
        "turn-frontier",
        "read",
        succeeded=True,
        tool_output="requires-python = '>=3.13'\nhttpx = '0.28.1'",
    )
    request = grounded["next_action"]["request"]
    assert request["capability"] == "search_or_fetch"
    assert request["parameters"]["operation"] == "frontier_discovery"

    tracker.pre_tool(
        "codex",
        "session-frontier",
        "turn-frontier",
        "websearch",
        tool_input={"query": "current compatible HTTP client API"},
    )
    discovered = tracker.post_tool(
        "codex",
        "session-frontier",
        "turn-frontier",
        "websearch",
        succeeded=True,
        tool_output="host search envelope",
        tool_metadata={
            "results": [
                {
                    "url": "https://docs.example.org/http-client",
                    "snippet": "The current API supports Python 3.13.",
                    "publishedAt": "2026-08-23",
                }
            ]
        },
    )
    assert discovered["next_action"]["request"]["parameters"]["operation"] == (
        "primary_source_fetch"
    )
