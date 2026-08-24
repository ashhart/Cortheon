"""Permanent mutation regressions for the Pi doom-loop gates.

Each mutation removes one load-bearing piece of the adapter's termination
machinery and proves the loop reopens (timeout or overrun). The
whole-operation stop is one of them: the mutation experiment that motivated
it produced 110 model requests in four seconds, so it stays pinned here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import (
    CEILING,
    MIXED_TURN,
    PROMPT,
    TOOL_TURN,
    never_finished_script,
    workspace,
)
from pi_recovery_helpers import (
    Servers,
    host_executions,
    parse_events,
    require_pi,
    run_pi,
)

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"


def _mutated_extension(tmp_path: Path, module: str, replacements: list[tuple[str, str]]) -> Path:
    root = tmp_path / "cortheon"
    (root / "pi_core").mkdir(parents=True)
    for path in sorted((SOURCE_DIR / "pi_core").glob("*.ts")):
        text = path.read_text(encoding="utf-8")
        if path.stem == module:
            for old, new in replacements:
                assert old in text, old
                text = text.replace(old, new)
        (root / "pi_core" / path.name).write_text(text, encoding="utf-8")
    facade = root / "pi_extension.ts"
    shutil.copy2(SOURCE_DIR / "pi_extension.ts", facade)
    return facade


def _run_doomed(extension: Path, tmp_path: Path, servers: Servers) -> Any:
    """Bounded mutation run: file-backed output, 4s timeout, delayed model."""
    return run_pi(
        extension,
        PROMPT,
        model_port=servers.model.server_port,
        runtime_port=servers.runtime.server_port,
        workspace=workspace(tmp_path),
        tmp_path=tmp_path,
        timeout=4,
        extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": CEILING},
        stdout_path=tmp_path / "stdout.jsonl",
        stderr_path=tmp_path / "stderr.log",
    )


def _assert_unbounded(requests_after_settle: int, completed: Any) -> None:
    executed = 0 if completed is None else len(host_executions(parse_events(completed.stdout)))
    assert executed > int(CEILING) or requests_after_settle > int(CEILING) // 2 + 3


@pytest.mark.parametrize(
    ("module", "replacements", "turns"),
    [
        ("tool_events", [("terminate: true,", "terminate: false,")], [TOOL_TURN]),
        ("budget", [("Boolean(active && toolBudgetExhausted(active))", "false")], [TOOL_TURN]),
        # A mixed batch needs the whole-operation stop: without it, the
        # unavailable tool's terminate-less result keeps the loop alive even
        # though every valid tool is blocked with terminate:true.
        ("tool_events", [("context.abort();", "void 0;")], [MIXED_TURN]),
    ],
    ids=["without-terminate", "without-hard-cap", "without-whole-operation-stop"],
)
def test_mutations_that_reopen_the_loop_fail_the_gate(
    tmp_path: Path, module: str, replacements: list[tuple[str, str]], turns: list[Any]
) -> None:
    """Removing terminate or the hard cap from the adapter lets the never
    finished runtime loop the model past every bound (timeout or overrun)."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    extension = _mutated_extension(tmp_path / "mutation", module, replacements)
    model_state: dict[str, Any] = {"requests": [], "turns": turns, "delay": 0.02}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finished_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        try:
            completed = _run_doomed(extension, tmp_path / "run", servers)
        except subprocess.TimeoutExpired:
            return
        _assert_unbounded(len(model_state["requests"]), completed)
