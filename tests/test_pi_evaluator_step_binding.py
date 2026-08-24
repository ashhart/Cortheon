"""Strict treatment-side binding of the evaluator's Pi step cap."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from test_pi_unified_budget import _stage_core

PROBE = r"""
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const state = await import(input.core + "/state.ts");
state.recordBoundStep();
const first = state.evaluatorBoundReached();
state.recordBoundStep();
const second = state.evaluatorBoundReached();
state.resetFinalization();
const staleAfterReset = state.evaluatorBoundReached();
process.env.CORTHEON_EVALUATOR_MAX_STEPS = "1";
state.recordBoundStep();
const dynamicAfterImport = state.evaluatorBoundReached();

console.log(JSON.stringify({
  first,
  second,
  staleAfterReset,
  dynamicAfterImport,
}));
"""


def test_pi_evaluator_step_cap_is_strict_bounded_and_task_scoped(tmp_path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    core = _stage_core(tmp_path / "pi_core")

    def probe(value: str | None) -> dict:
        environment = os.environ.copy()
        if value is None:
            environment.pop("CORTHEON_EVALUATOR_MAX_STEPS", None)
        else:
            environment["CORTHEON_EVALUATOR_MAX_STEPS"] = value
        completed = subprocess.run(
            [node, "--experimental-strip-types", "--input-type=module", "-e", PROBE],
            input=json.dumps({"core": str(core)}),
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        return json.loads(completed.stdout)

    invalid = [None, "", "0", "-1", "1.5", "01", "1e0", " 1", "1025", "NaN"]
    assert [probe(value)["first"] for value in invalid] == [False] * len(invalid)
    assert probe("2") == {
        "first": False,
        "second": True,
        "staleAfterReset": False,
        "dynamicAfterImport": False,
    }
    assert probe("1") == {
        "first": True,
        "second": True,
        "staleAfterReset": False,
        "dynamicAfterImport": True,
    }
