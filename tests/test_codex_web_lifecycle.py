"""Doom-loop bounds for unattributable Codex web results."""

from __future__ import annotations

from cortheon.cognitive_hooks import CognitiveHookTracker
from cortheon.cognitive_runtime import CognitiveRuntime


def test_unattributable_web_attempt_replans_and_stop_releases_terminally() -> None:
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    registered = tracker.register(
        "codex",
        "session",
        "turn",
        goal="Research the current Cortheon release from fresh web sources and cite it.",
        effort="quick",
    )
    first = registered["next_action"]["request"]
    assert first["parameters"]["purpose"] == "contradiction_check"

    results = []
    for _attempt in range(2):
        tracker.pre_tool(
            "codex",
            "session",
            "turn",
            "exec",
            tool_input={"code": "await tools.web__run({search_query:[{q:'release'}]})"},
        )
        results.append(
            tracker.post_tool(
                "codex",
                "session",
                "turn",
                "exec",
                succeeded=True,
                tool_output="A flattened result mentions https://source.example/release.",
                tool_metadata={},
            )
        )

    assert results[0]["next_action"]["request"]["request_id"] == first["request_id"]
    successor = results[1]["next_action"]["request"]
    assert successor["request_id"] != first["request_id"]
    assert successor["parameters"]["purpose"] == "corroboration"
    assert runtime.metrics["requests_waived"] == 1

    stops = [
        tracker.stop("codex", "session", "turn", answer="Unverified answer.")
        for _attempt in range(3)
    ]
    assert [item["allow"] for item in stops] == [False, False, True]
    assert stops[-1]["terminal"] is True
    assert stops[-1]["uncertified"] is True
    assert tracker.active_turns == runtime.active_sessions == 0
