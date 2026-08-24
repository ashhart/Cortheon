"""Manual, model-driven hook enforcement: phases, budgets, and identifiers."""

import pytest

from cortheon.cognitive_hooks import CognitiveHookTracker, cortheon_tool_phase


def test_cortheon_tool_phase_accepts_plain_and_codex_mcp_names():
    assert cortheon_tool_phase("cortheon_start") == "start"
    assert cortheon_tool_phase("mcp__cortheon__cortheon_observe") == "observe"
    assert cortheon_tool_phase("mcp__cortheon__cortheon_complete") == "complete"
    assert cortheon_tool_phase("Bash") is None


def test_hook_tracker_enforces_start_observe_and_certified_completion():
    tracker = CognitiveHookTracker()
    identity = ("codex", "session-secret", "turn-secret")

    registered = tracker.register(*identity)
    assert registered == {
        "started": False,
        "observed": False,
        "certified": False,
        "automatic": False,
    }

    nudged = tracker.pre_tool(*identity, "Bash")
    assert nudged["allow"] is True
    assert "cortheon_start" in nudged["guidance"]

    assert (
        tracker.pre_tool(
            *identity,
            "mcp__cortheon__cortheon_start",
        )["allow"]
        is True
    )
    tracker.post_tool(
        *identity,
        "mcp__cortheon__cortheon_start",
        succeeded=True,
    )
    assert tracker.pre_tool(*identity, "Bash")["allow"] is True

    observed = tracker.post_tool(
        *identity,
        "mcp__cortheon__cortheon_observe",
        succeeded=True,
    )
    assert observed["observed"] is True

    tracker.post_tool(
        *identity,
        "mcp__cortheon__cortheon_complete",
        succeeded=True,
        certified=False,
    )
    assert tracker.stop(*identity)["allow"] is False

    tracker.post_tool(
        *identity,
        "mcp__cortheon__cortheon_complete",
        succeeded=True,
        certified=True,
    )
    assert tracker.stop(*identity) == {
        "tracked": True,
        "allow": True,
        "certified": True,
    }
    assert tracker.active_turns == 0
    assert tracker.metrics["hook_turns_certified"] == 1
    assert tracker.metrics["hook_tools_denied"] == 0


def test_failed_cortheon_calls_do_not_advance_hook_phase():
    tracker = CognitiveHookTracker()
    identity = ("codex", "session", "turn")
    tracker.register(*identity)

    tracker.post_tool(
        *identity,
        "mcp__cortheon__cortheon_start",
        succeeded=False,
    )
    state = tracker.pre_tool(*identity, "Bash")

    assert state["allow"] is True
    assert state["started"] is False
    assert tracker.stop(*identity)["allow"] is False


def test_hook_tracker_releases_uncertified_after_stop_budget():
    tracker = CognitiveHookTracker()
    identity = ("codex", "session", "turn")
    tracker.register(*identity)

    decisions = [tracker.pre_tool(*identity, "view_image") for _ in range(5)]
    assert all(decision["allow"] is True for decision in decisions)

    first = tracker.stop(*identity)
    second = tracker.stop(*identity)
    released = tracker.stop(*identity)
    assert first["allow"] is False and "terminal" not in first
    assert second["allow"] is False and "terminal" not in second
    assert released["allow"] is True
    assert released["terminal"] is True
    assert released["uncertified"] is True
    assert "certify" in released["caveat"]
    assert tracker.active_turns == 0
    assert tracker.metrics["hook_uncertified_releases"] == 1

    continued = ("codex", "session", "continued-turn")
    tracker.register(*continued)
    tracker.post_tool(
        *continued,
        "mcp__cortheon__cortheon_start",
        succeeded=True,
    )
    first = tracker.stop(*continued)
    second = tracker.stop(*continued)
    released = tracker.stop(*continued)
    assert first["allow"] is False and "terminal" not in first
    assert second["allow"] is False and "terminal" not in second
    assert released["allow"] is True
    assert released["uncertified"] is True


def test_hook_tracker_degrades_a_host_session_after_repeated_failures():
    tracker = CognitiveHookTracker()
    for index in range(3):
        identity = ("codex", "flaky-session", f"turn-{index}")
        tracker.register(*identity)
        for _ in range(3):
            tracker.stop(*identity)
    assert tracker.metrics["hook_uncertified_releases"] == 3

    degraded = tracker.register("codex", "flaky-session", "turn-3")
    assert degraded["tracked"] is False
    assert degraded["degraded"] is True
    assert tracker.metrics["hook_degraded_registrations"] == 1
    assert tracker.pre_tool("codex", "flaky-session", "turn-3", "Bash")["allow"] is True
    assert tracker.stop("codex", "flaky-session", "turn-3")["allow"] is True

    tracker.end_session("codex", "flaky-session")
    revived = tracker.register("codex", "flaky-session", "turn-4")
    assert "degraded" not in revived


def test_hook_tracker_hashes_identifiers_and_cleans_whole_host_session():
    tracker = CognitiveHookTracker()
    tracker.register("codex", "do-not-retain-this-id", "turn-one")
    tracker.register("codex", "do-not-retain-this-id", "turn-two")

    assert "do-not-retain-this-id" not in repr(tracker._turns)
    assert tracker.end_session("codex", "do-not-retain-this-id") == {
        "ok": True,
        "removed_turns": 2,
    }
    assert tracker.active_turns == 0


def test_hook_tracker_rejects_invalid_identifiers():
    tracker = CognitiveHookTracker()

    with pytest.raises(ValueError, match="host_session_id"):
        tracker.register("codex", "", "turn")
