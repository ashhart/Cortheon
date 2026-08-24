"""Shared drivers for the automatic Cortheon hook patch-loop tests."""

from __future__ import annotations

from typing import Any

IMPLEMENTATION_SOURCE = "def add(left: int, right: int) -> int:\n    return left - right\n"


def drive_repair_reads(
    tracker: Any,
    identity: tuple[str, str, str],
    *,
    test_source: str,
    implementation: str = IMPLEMENTATION_SOURCE,
) -> dict[str, Any]:
    """Run the two model-owned reads that precede the bounded edit.

    Returns the final ``post_tool`` result so callers keep asserting on the
    step the hook schedules once both files are on the record.
    """

    tracker.pre_tool(*identity, "Bash", tool_input={"command": "cat calculator.py"})
    tracker.post_tool(*identity, "Bash", succeeded=True, tool_output=implementation)
    tracker.pre_tool(*identity, "Bash", tool_input={"command": "cat test_calculator.py"})
    return tracker.post_tool(*identity, "Bash", succeeded=True, tool_output=test_source)
