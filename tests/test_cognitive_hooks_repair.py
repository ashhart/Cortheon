"""The bounded repair contract the hook derives from live evidence."""

from cortheon.cognitive_repair import (
    changed_paths_from_diff,
    derive_simple_repair,
    protected_test_paths,
    requested_check_invocation,
    requested_test_invocation,
)


def test_bounded_repair_is_derived_only_from_live_literal_examples():
    plan = derive_simple_repair(
        [
            (
                "calculator.py",
                "def add(left: int, right: int) -> int:\n    return left - right\n",
            ),
            (
                "test_calculator.py",
                "from calculator import add\n\n"
                "def test_adds_two_numbers() -> None:\n"
                "    assert add(2, 3) == 5\n",
            ),
        ]
    )

    assert plan is not None
    assert plan.path == "calculator.py"
    assert plan.old_text == "    return left - right"
    assert plan.new_text == "    return left + right"
    assert "*** Update File: calculator.py" in plan.patch()


def test_repair_helpers_parse_test_contract_and_changed_paths():
    goal = (
        "Fix calculator.py so test_calculator.py passes. Do not change the test. "
        "Run python3 -m pytest -q test_calculator.py after the edit and report it."
    )

    invocation = requested_test_invocation(goal)
    assert invocation is not None
    assert invocation.shell_command() == "python3 -m pytest -q test_calculator.py"
    assert protected_test_paths(goal) == ("test_calculator.py",)
    assert protected_test_paths("Fix calculator.py without changing test_calculator.py.") == (
        "test_calculator.py",
    )
    assert requested_check_invocation(goal) is None
    assert changed_paths_from_diff(
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@\n"
        "-    return left - right\n"
        "+    return left + right\n"
    ) == {"calculator.py"}


def test_repair_candidates_cover_boundary_operand_and_off_by_one_classes():
    boundary = derive_simple_repair(
        [
            (
                "gate.py",
                "def allows(count: int, limit: int) -> bool:\n    return count < limit\n",
            ),
            (
                "test_gate.py",
                "from gate import allows\n\n"
                "def test_allows_at_limit() -> None:\n"
                "    assert allows(3, 3) == True\n"
                "    assert allows(4, 3) == False\n",
            ),
        ]
    )
    assert boundary is not None
    assert boundary.new_text == "    return count <= limit"

    swapped = derive_simple_repair(
        [
            ("delta.py", "def delta(a: int, b: int) -> int:\n    return a - b\n"),
            (
                "test_delta.py",
                "from delta import delta\n\n"
                "def test_delta() -> None:\n"
                "    assert delta(2, 5) == 3\n"
                "    assert delta(7, 2) == -5\n",
            ),
        ]
    )
    assert swapped is not None
    assert swapped.new_text == "    return b - a"

    off_by_one = derive_simple_repair(
        [
            ("seq.py", "def successor(value: int) -> int:\n    return value + 2\n"),
            (
                "test_seq.py",
                "from seq import successor\n\n"
                "def test_successor() -> None:\n"
                "    assert successor(1) == 2\n"
                "    assert successor(5) == 6\n",
            ),
        ]
    )
    assert off_by_one is not None
    assert off_by_one.new_text == "    return value + 1"


def test_repair_helpers_parse_quality_check_contract():
    goal = (
        "Fix calculator.py so test_calculator.py passes. Do not change the test. "
        "Run python3 -m pytest -q test_calculator.py after the edit. "
        "Run ruff check src after the tests."
    )

    check = requested_check_invocation(goal)
    assert check is not None
    assert check.shell_command() == "ruff check src"
    invocation = requested_test_invocation(goal)
    assert invocation is not None
    assert invocation.executable == "python3"
    assert requested_check_invocation("Run rm -rf build after the edit.") is None
    assert requested_check_invocation("Run mypy src after the edit.") is not None
