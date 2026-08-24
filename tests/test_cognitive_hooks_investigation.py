"""Automatic investigation: goal continuity, harvesting, and convergence."""

from cortheon.cognitive_hooks import CognitiveHookTracker
from cortheon.cognitive_runtime import CognitiveRuntime


def test_automatic_state_is_discarded_when_a_new_goal_arrives():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    tracker.register(
        "codex",
        "session",
        "turn-one",
        goal=(
            "Fix calculator.add in calculator.py so test_calculator.py passes. "
            "Run python3 -m pytest -q test_calculator.py after the edit."
        ),
    )
    assert runtime.active_sessions == 1

    registered = tracker.register(
        "codex",
        "session",
        "turn-two",
        goal=(
            "Read pyproject.toml and answer which console command maps to "
            "cortheon.cognitive_cli:main."
        ),
    )

    assert registered["automatic"] is True
    assert registered["next_action"]["request"]["capability"] == "grep"
    assert tracker.active_turns == 1
    assert runtime.active_sessions == 1
    assert runtime.metrics["sessions_abandoned"] == 1


def test_automatic_state_migrates_across_turns_for_the_same_goal():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    goal = (
        "Read pyproject.toml and answer which console command maps to cortheon.cognitive_cli:main."
    )
    tracker.register("codex", "session", "turn-one", goal=goal)
    tracker.register("codex", "session", "turn-two", goal=goal)

    assert tracker.active_turns == 1
    assert runtime.metrics["sessions_started"] == 1


def test_automatic_hook_harvests_model_commands_for_inspect_requests():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "session", "harvest-turn")

    registered = tracker.register(
        *identity,
        goal="Explain why the rollout stalled yesterday afternoon.",
    )
    request = registered["next_action"]["request"]
    assert request["capability"] == "inspect"

    pre = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "cat notes/rollout.md"},
    )
    assert pre["allow"] is True
    assert "updated_input" not in pre

    observed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="The rollout waits on the canary gate before promotion.",
    )
    assert observed["observed"] is True

    stopped = tracker.stop(
        *identity,
        answer="The rollout stalled because the canary gate was still waiting.",
    )
    assert stopped["allow"] is True
    assert stopped["certified"] is True
    assert runtime.active_sessions == 0


def test_read_many_requests_converge_under_output_truncation():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "truncated-session", "truncated-turn")
    goal = (
        "Fix calculator.add in calculator.py so test_calculator.py passes. "
        "Do not change the test. Run python3 -m pytest -q test_calculator.py "
        "after the edit."
    )

    registered = tracker.register(*identity, goal=goal)
    request = registered["next_action"]["request"]
    assert request["capability"] == "read_many"
    first_read = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "cat calculator.py"},
    )
    assert "updated_input" not in first_read
    partial = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=("def add(left: int, right: int) -> int:\n    return left - right\n"),
    )
    follow_up = partial["next_action"]["request"]
    assert follow_up["request_id"] == request["request_id"]
    assert follow_up["covered_paths"] == ["calculator.py"]

    retry = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "cat test_calculator.py"},
    )
    assert "updated_input" not in retry

    completed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=(
            "from calculator import add\n\n"
            "def test_adds_two_numbers() -> None:\n"
            "    assert add(2, 3) == 5\n"
        ),
    )
    assert completed["next_action"]["request"]["capability"] == "edit"


def test_bad_inferred_paths_never_replace_model_investigation_commands():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "log-regression-session", "log-regression-turn")
    goal = (
        "Read rtc.html in this directory and build a Roller Coaster Tycoon "
        "game in HTML and Three.js."
    )

    registered = tracker.register(*identity, goal=goal)
    request = registered["next_action"]["request"]
    assert request["capability"] == "read_many"
    assert "rtc.html" in request["parameters"]["paths"]
    assert "Three.js" not in request["parameters"]["paths"]

    missing_read = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "cat rtc.html"},
    )
    assert missing_read["allow"] is True
    assert "updated_input" not in missing_read

    failed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=False,
        tool_output="cat: rtc.html: No such file or directory",
    )
    assert failed["next_action"]["request"]["request_id"] == request["request_id"]

    discovery = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "ls -la"},
    )
    assert discovery["allow"] is True
    assert "updated_input" not in discovery

    reframed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="-rw-r--r-- 1 developer staff 73261 rct.html",
    )
    assert reframed["observed"] is True
    assert reframed.get("observation_error") is None
    assert (
        reframed.get("next_action", {}).get("request", {}).get("request_id")
        != request["request_id"]
    )
    assert runtime.metrics["requests_waived"] == 1

    unrelated = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "find . -maxdepth 1 -type f"},
    )
    assert unrelated["allow"] is True
    assert "updated_input" not in unrelated


def test_automatic_hook_tracker_rewrites_observes_and_certifies_lookup():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "session", "turn")
    goal = (
        "Read pyproject.toml and answer which console command maps to cortheon.cognitive_cli:main."
    )

    registered = tracker.register(*identity, goal=goal)
    assert registered["automatic"] is True
    request = registered["next_action"]["request"]
    assert request["capability"] == "grep"

    command = (
        f"rg -n --fixed-strings -- {request['parameters']['pattern']} "
        f"{request['parameters']['path']}"
    )
    pre_tool = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": command},
    )
    assert pre_tool["allow"] is True
    assert "updated_input" not in pre_tool

    observed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output='25:cortheon = "cortheon.cognitive_cli:main"',
    )
    assert observed["observed"] is True

    stopped = tracker.stop(
        *identity,
        answer="cortheon maps to cortheon.cognitive_cli:main.",
    )
    assert stopped["allow"] is True
    assert stopped["certified"] is True
    assert runtime.active_sessions == 0
    assert tracker.active_turns == 0


def test_register_passes_task_kind_through_to_the_runtime():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)

    registered = tracker.register(
        "codex",
        "kind-session",
        "kind-turn",
        goal="Summarize the notes in docs/notes.md",
        task_kind="general",
    )

    assert registered["automatic"] is True
    assert registered["next_action"]["request"]["capability"] == "inspect"
