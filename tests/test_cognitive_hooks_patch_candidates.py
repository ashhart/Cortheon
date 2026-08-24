"""Best-of-N candidate advancement and the chained quality-check step."""

from cognitive_hooks_helpers import drive_repair_reads

from cortheon.cognitive_hooks import CognitiveHookTracker
from cortheon.cognitive_runtime import CognitiveRuntime


def test_patch_loop_tries_next_candidate_when_the_real_test_fails():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "candidates-session", "candidates-turn")
    goal = (
        "Fix calculator.add in calculator.py so test_calculator.py passes. "
        "Do not change the test. Run python3 -m pytest -q test_calculator.py "
        "after the edit."
    )

    tracker.register(*identity, goal=goal)
    observed = drive_repair_reads(
        tracker,
        identity,
        test_source=(
            "from calculator import add\n\n"
            "def test_add_two_and_two() -> None:\n"
            "    assert add(2, 2) == 4\n"
        ),
    )
    assert observed["next_action"]["request"]["capability"] == "edit"

    first_edit = tracker.pre_tool(*identity, "Bash", tool_input={"command": "apply_patch"})
    assert "return left * right" in first_edit["updated_input"]["command"]
    tracker.post_tool(*identity, "Bash", succeeded=True, tool_output="Done!")
    tracker.pre_tool(*identity, "Bash", tool_input={"command": "diff"})
    after_diff = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=(
            "diff --git a/calculator.py b/calculator.py\n"
            "--- a/calculator.py\n"
            "+++ b/calculator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(left: int, right: int) -> int:\n"
            "-    return left - right\n"
            "+    return left * right\n"
        ),
    )
    assert after_diff["next_action"]["request"]["capability"] == "test"

    tracker.pre_tool(*identity, "Bash", tool_input={"command": "pytest"})
    failed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=False,
        tool_output="assert add(2, 3) == 5\nE  assert 6 == 5\n1 failed in 0.01s",
    )
    assert failed["next_action"]["request"]["capability"] == "edit"
    assert tracker.metrics["hook_auto_repair_candidates_advanced"] == 1

    second_edit = tracker.pre_tool(*identity, "Bash", tool_input={"command": "apply_patch"})
    assert "-    return left * right" in second_edit["updated_input"]["command"]
    assert "+    return left + right" in second_edit["updated_input"]["command"]
    tracker.post_tool(*identity, "Bash", succeeded=True, tool_output="Done!")
    tracker.pre_tool(*identity, "Bash", tool_input={"command": "diff"})
    tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=(
            "diff --git a/calculator.py b/calculator.py\n"
            "--- a/calculator.py\n"
            "+++ b/calculator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(left: int, right: int) -> int:\n"
            "-    return left - right\n"
            "+    return left + right\n"
        ),
    )
    tracker.pre_tool(*identity, "Bash", tool_input={"command": "pytest"})
    passed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="2 passed in 0.01s",
    )
    assert passed["next_action"]["type"] == "finish"

    stopped = tracker.stop(
        *identity,
        answer=(
            "Updated calculator.py to add the operands. "
            "python3 -m pytest -q test_calculator.py passed."
        ),
    )
    assert stopped["allow"] is True
    assert stopped["certified"] is True
    assert tracker.active_turns == 0


def test_automatic_patch_loop_chains_requested_quality_check():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "check-session", "check-turn")
    goal = (
        "Fix calculator.add in calculator.py so test_calculator.py passes. "
        "Do not change the test. Run python3 -m pytest -q test_calculator.py "
        "after the edit. Run ruff check src after the tests."
    )

    tracker.register(*identity, goal=goal)
    drive_repair_reads(
        tracker,
        identity,
        test_source=(
            "from calculator import add\n\n"
            "def test_adds_two_numbers() -> None:\n"
            "    assert add(2, 3) == 5\n"
        ),
    )
    tracker.pre_tool(*identity, "Bash", tool_input={"command": "apply_patch"})
    tracker.post_tool(*identity, "Bash", succeeded=True, tool_output="Done!")
    tracker.pre_tool(*identity, "Bash", tool_input={"command": "diff"})
    after_diff = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=(
            "diff --git a/calculator.py b/calculator.py\n"
            "--- a/calculator.py\n"
            "+++ b/calculator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(left: int, right: int) -> int:\n"
            "-    return left - right\n"
            "+    return left + right\n"
        ),
    )
    assert after_diff["next_action"]["request"]["capability"] == "test"

    test_step = tracker.pre_tool(*identity, "Bash", tool_input={"command": "pytest"})
    assert test_step["updated_input"]["command"] == ("python3 -m pytest -q test_calculator.py")
    after_test = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="1 passed in 0.01s",
    )
    check_request = after_test["next_action"]["request"]
    assert check_request["capability"] == "test"
    assert check_request["parameters"]["command"] == ["ruff", "check", "src"]

    check_step = tracker.pre_tool(*identity, "Bash", tool_input={"command": "lint"})
    assert check_step["updated_input"]["command"] == "ruff check src"
    after_check = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="All checks passed!",
    )
    assert after_check["next_action"]["type"] == "finish"
    assert tracker.metrics["hook_auto_checks_scheduled"] == 1
    assert tracker.metrics["hook_auto_checks_passed"] == 1

    stopped = tracker.stop(
        *identity,
        answer=(
            "Updated calculator.py to add the operands. "
            "python3 -m pytest -q test_calculator.py passed and ruff check src is clean."
        ),
    )
    assert stopped["allow"] is True
    assert stopped["certified"] is True
    assert tracker.active_turns == 0
