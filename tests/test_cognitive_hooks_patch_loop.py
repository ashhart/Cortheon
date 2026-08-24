"""The end-to-end automatic patch loop: edit, diff, test, certify."""

from cognitive_hooks_helpers import drive_repair_reads

from cortheon.cognitive_hooks import CognitiveHookTracker
from cortheon.cognitive_runtime import CognitiveRuntime


def test_automatic_patch_loop_applies_diff_tests_and_certifies():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "patch-session", "patch-turn")
    goal = (
        "Fix calculator.add in calculator.py so test_calculator.py passes. "
        "Do not change the test. Run python3 -m pytest -q test_calculator.py "
        "after the edit and report the verified result."
    )

    registered = tracker.register(*identity, goal=goal)
    request = registered["next_action"]["request"]
    assert request["capability"] == "read_many"

    implementation_read = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "cat calculator.py"},
    )
    assert "updated_input" not in implementation_read
    partial = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=("def add(left: int, right: int) -> int:\n    return left - right\n"),
    )
    assert partial["next_action"]["request"]["request_id"] == request["request_id"]

    test_read = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "cat test_calculator.py"},
    )
    assert "updated_input" not in test_read
    observed = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output=(
            "from calculator import add\n\n"
            "def test_adds_two_numbers() -> None:\n"
            "    assert add(2, 3) == 5\n"
        ),
    )
    assert observed["next_action"]["request"]["capability"] == "edit"

    edit = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "apply_patch"},
    )
    assert "return left + right" in edit["updated_input"]["command"]
    after_edit = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="Done!",
    )
    assert after_edit["next_action"]["request"]["capability"] == "diff"

    diff = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "diff"},
    )
    assert diff["updated_input"]["command"] == "git diff -- calculator.py"
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

    test = tracker.pre_tool(
        *identity,
        "Bash",
        tool_input={"command": "pytest"},
    )
    assert test["updated_input"]["command"] == ("python3 -m pytest -q test_calculator.py")
    after_test = tracker.post_tool(
        *identity,
        "Bash",
        succeeded=True,
        tool_output="1 passed in 0.01s",
    )
    assert after_test["next_action"]["type"] == "finish"

    stopped = tracker.stop(
        *identity,
        answer=(
            "Updated calculator.py to add the operands. "
            "python3 -m pytest -q test_calculator.py passed."
        ),
    )
    assert stopped["allow"] is True
    assert stopped["certified"] is True
    assert tracker.metrics["hook_auto_repairs_derived"] == 1
    assert tracker.metrics["hook_auto_patches_applied"] == 1
    assert tracker.metrics["hook_auto_tests_passed"] == 1
    assert tracker.active_turns == 0
    assert runtime.active_sessions == 0


def test_automatic_patch_loop_blocks_protected_test_mutation():
    tracker = CognitiveHookTracker(runtime=CognitiveRuntime(require_host_receipts=True))
    identity = ("codex", "protected-session", "protected-turn")
    tracker.register(
        *identity,
        goal=(
            "Fix calculator.py so test_calculator.py passes. Do not change the "
            "test. Run python3 -m pytest -q test_calculator.py after the edit."
        ),
    )

    denied = tracker.pre_tool(
        *identity,
        "apply_patch",
        tool_input={
            "patch": (
                "*** Begin Patch\n"
                "*** Update File: test_calculator.py\n"
                "@@\n-assert False\n+assert True\n"
                "*** End Patch\n"
            )
        },
    )

    assert denied["allow"] is False
    assert "protected test" in denied["reason"]
    assert tracker.metrics["hook_protected_mutations_denied"] == 1


def test_patch_loop_reconciles_a_host_edit_without_post_tool_event():
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    identity = ("codex", "reconcile-session", "reconcile-turn")
    goal = (
        "Fix calculator.py so test_calculator.py passes. Do not change the test. "
        "Run python3 -m pytest -q test_calculator.py after the edit."
    )
    tracker.register(*identity, goal=goal)
    drive_repair_reads(
        tracker,
        identity,
        test_source=(
            "from calculator import add\ndef test_add() -> None:\n    assert add(2, 3) == 5\n"
        ),
    )
    edit = tracker.pre_tool(
        *identity,
        "apply_patch",
        tool_input={"patch": "model supplied patch"},
    )
    assert "return left + right" in edit["updated_input"]["patch"]

    reconciled = tracker.pre_tool(
        *identity,
        "update_plan",
        tool_input={"plan": []},
    )

    assert reconciled["allow"] is False
    assert reconciled["next_action"]["request"]["capability"] == "diff"
    assert tracker.metrics["hook_auto_edit_reconciliations"] == 1
