"""One unified automatic-follow-up budget: source-level and real-Pi proofs.

The contract: one initial agent operation plus at most ONE automatic
follow-up total per investigation — a repair continuation and an
answer-only continuation draw from the same budget and are never additive.
These tests pin the source invariants (no second counter, one shared gate),
prove the mixed-message terminal replacement retains toolCall pairing,
prove host-evidence source aliases collapse before the two-source check,
prove a user-authored ``[CORTHEON_CONTINUE]`` prefix is an ordinary new
turn, run the real lifecycle against the bundled artifact, and mutate the
shipped gate to show a reintroduced separate answer-only budget produces a
forbidden second follow-up.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import TOOL_TURN, workspace
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    require_pi,
    run_pi,
)
from pi_terminal_helpers import (
    AMBIGUITY_ANSWER,
    AMBIGUITY_PROMPT,
    EXTENSION,
    WITHHELD_MARKER,
    mutated_source,
    terminal_status_messages,
    withholding_ambiguity_script,
)

ROOT = Path(__file__).parents[1]
CORE = ROOT / "src" / "cortheon" / "pi_core"

CAUSAL_GOAL = (
    "Diagnose the causal explanation for the clash between the two "
    "ledgers, disprove the rival hypothesis, and give a discriminating test."
)


def _stage_core(target: Path) -> Path:
    """Copy pi_core beside node_modules satisfying its external imports."""
    target.mkdir(parents=True)
    for path in sorted(CORE.glob("*.ts")):
        (target / path.name).write_text(path.read_text(encoding="utf-8"))
    modules_root = target.parent / "node_modules"
    scoped = modules_root / "@earendil-works"
    scoped.mkdir(parents=True, exist_ok=True)
    pi_binary = Path(shutil.which("pi") or "").resolve()
    bundled = next(
        (
            parent / "node_modules"
            for parent in pi_binary.parents
            if (parent / "node_modules" / "@earendil-works" / "pi-ai").is_dir()
        ),
        None,
    )
    if bundled is not None:
        os.symlink(bundled / "@earendil-works" / "pi-ai", scoped / "pi-ai", True)
    else:
        vendor = scoped / "pi-ai"
        vendor.mkdir(parents=True, exist_ok=True)
        (vendor / "package.json").write_text(
            json.dumps(
                {
                    "name": "@earendil-works/pi-ai",
                    "exports": {".": "./index.js", "./compat": "./compat.js"},
                }
            ),
            encoding="utf-8",
        )
        (vendor / "index.js").write_text("export const uuidv7 = () => 'stub';\n", encoding="utf-8")
        (vendor / "compat.js").write_text(
            "export const complete = async () => { throw new Error('stub'); };\n",
            encoding="utf-8",
        )
    return target


MIXED_PROBE = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const state = await import(input.core + "/state.ts");
const terminal = await import(input.core + "/terminal.ts");
state.setEnabled(true);
state.markAnswerOnly();
state.setTerminalDisposition({ reason: "probe disposition", causal: false });
const entries = [];
const pi = {
  appendEntry(_type, data) { entries.push(data); },
  sendMessage() {},
};
const message = {
  content: [
    { type: "text", text: "uncertified draft text" },
    { type: "toolCall", id: "call-1", name: "read", arguments: { path: "a" } },
  ],
};
const result = await terminal.terminalDispositionResult(pi, message);
const later = await terminal.terminalDispositionResult(pi, {
  content: [{ type: "text", text: "later raw text" }],
});
const blocks = result?.message?.content ?? [];
console.log(JSON.stringify({
  textReplaced: Boolean(
    blocks.find((b) => b.type === "text")?.text?.startsWith("[Cortheon withheld:"),
  ),
  toolCallRetained: blocks.some((b) => b.type === "toolCall"),
  candidateCaptured: entries.some(
    (e) => e.stage === "completion" && e.candidate === "uncertified draft text",
  ),
  capturedOnce: entries.length === 1,
  laterReplaced: Boolean(
    later?.message?.content?.[0]?.text?.startsWith("[Cortheon withheld:"),
  ),
}));
process.exit(0);
"""


SOURCES_PROBE = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const merge = await import(input.core + "/merge.ts");
const state = await import(input.core + "/state.ts");
const budget = await import(input.core + "/budget.ts");
const results = {};
for (const [name, sources] of Object.entries(input.cases)) {
  state.setActive(undefined);
  merge.mergePayload({
    session_id: "s-" + name,
    status: "observing",
    session: { deliverable: "document_synthesis" },
    context: {
      goal: input.goal,
      evidence: sources.map((source, index) => ({
        evidence_id: "ev-" + index,
        source,
        content: "fact " + index,
      })),
    },
  });
  results[name] = budget.causalEvidenceSufficient(state.getActive());
}
console.log(JSON.stringify(results));
process.exit(0);
"""


def _node_probe(script: str, payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(
        subprocess.run(["mktemp", "-d"], capture_output=True, text=True, check=True).stdout.strip()
    )
    core = _stage_core(root / "pi_core")
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps({**payload, "core": str(core)}),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=root,
        env={**os.environ, "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


class UnifiedBudgetSourceTests(unittest.TestCase):
    def test_mixed_message_text_replaced_tool_call_retained(self) -> None:
        report = _node_probe(MIXED_PROBE, {})
        self.assertTrue(report["textReplaced"])
        self.assertTrue(report["toolCallRetained"])
        self.assertTrue(report["candidateCaptured"])
        self.assertTrue(report["capturedOnce"])
        self.assertTrue(report["laterReplaced"])

    def test_host_evidence_source_aliases_collapse(self) -> None:
        cases = {
            "read_grep_same_path": [
                "pi:read:facts/a.txt",
                "pi:grep:facts/a.txt",
            ],
            "prefix_case_same_path_bytes": [
                "PI:READ:facts/a.txt",
                "pi:Grep:facts/a.txt",
            ],
            "dot_slash_alias": ["pi:read:./facts/a.txt", "pi:grep:facts/a.txt"],
            "parent_hop_alias": [
                "pi:read:facts/w/../a.txt",
                "pi:grep:facts/a.txt",
            ],
            "bare_labels": ["pi:bash", "pi:ls"],
            "bare_plus_path": ["pi:bash", "pi:read:facts/a.txt"],
            "distinct_paths": ["pi:read:facts/a.txt", "pi:read:facts/b.txt"],
            "case_distinct_paths": ["pi:read:facts/A.txt", "pi:read:facts/a.txt"],
            "non_pi_urls": [
                "https://example.org/alpha",
                "https://example.org/beta",
            ],
            "non_pi_case_preserved": [
                "https://example.org/Alpha",
                "https://example.org/alpha",
            ],
            "pi_alias_vs_url": ["pi:read:facts/a.txt", "https://example.org/b"],
        }
        report = _node_probe(SOURCES_PROBE, {"cases": cases, "goal": CAUSAL_GOAL})
        self.assertFalse(report["read_grep_same_path"], report)
        self.assertFalse(report["prefix_case_same_path_bytes"], report)
        self.assertFalse(report["dot_slash_alias"], report)
        self.assertFalse(report["parent_hop_alias"], report)
        self.assertFalse(report["bare_labels"], report)
        self.assertFalse(report["bare_plus_path"], report)
        self.assertTrue(report["distinct_paths"], report)
        self.assertTrue(report["case_distinct_paths"], report)
        self.assertTrue(report["non_pi_urls"], report)
        self.assertTrue(report["non_pi_case_preserved"], report)
        self.assertTrue(report["pi_alias_vs_url"], report)


def _fresh_continuations(model_state: dict[str, Any], head: str = "[CORTHEON_CONTINUE]") -> int:
    count = 0
    for request in model_state["requests"]:
        messages = request.get("messages", [])
        last = messages[-1] if messages else None
        if not isinstance(last, dict):
            continue
        content = last.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [block.get("text", "") for block in content or [] if isinstance(block, dict)]
            if isinstance(content, list)
            else []
        )
        if any(text.startswith(head) for text in texts):
            count += 1
    return count


def _withhold_run(extension: Path, tmp_path: Path, turns: list[dict[str, Any]]):
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": withholding_ambiguity_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            extension,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    return completed, model_state, runtime_state


def test_mutation_reintroducing_separate_answer_only_budget_fails(tmp_path: Path) -> None:
    """Mutation proof of the unified budget: replacing the answer-only
    grant's shared-budget gate with an unconditional allowance (exactly a
    reintroduced separate answer-only budget) produces a forbidden second
    follow-up after the cap terminal — three agent operations — while the
    shipped adapter stops at one."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        TOOL_TURN,
        {"text": AMBIGUITY_ANSWER},
    ]
    gate = (
        "const mayAnswerFollowUp = Boolean(\n"
        "\t\t\t\tcurrent &&\n"
        "\t\t\t\t\t!current.completed &&\n"
        "\t\t\t\t\t!answerAlreadyDelivered() &&\n"
        "\t\t\t\t\tcurrent.automaticContinuations < MAX_AUTOMATIC_CONTINUATIONS,\n"
        "\t\t\t);\n"
        "\t\t\tif (mayAnswerFollowUp) {\n"
        "\t\t\t\tcurrent!.automaticContinuations += 1;",
        "const mayAnswerFollowUp = true;\n"
        "\t\t\tif (mayAnswerFollowUp) {\n"
        "\t\t\t\tif (current) current.automaticContinuations += 1;",
    )
    facade = mutated_source(tmp_path / "mutation", {"session_events": gate})
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": withholding_ambiguity_script(),
    }
    timed_out = False
    completed = None
    with Servers(model_state, runtime_state) as servers:
        try:
            completed = run_pi(
                facade,
                AMBIGUITY_PROMPT,
                model_port=servers.model.server_port,
                runtime_port=servers.runtime.server_port,
                workspace=workspace(tmp_path / "run"),
                tmp_path=tmp_path / "run",
                timeout=25,
            )
        except subprocess.TimeoutExpired:
            # The unbounded mutant may also simply never terminate.
            timed_out = True
    if completed is not None:
        assert completed.returncode == 0, completed.stderr
    # The mutated adapter scheduled a forbidden SECOND follow-up (the
    # answer-only grant ignored the shared budget the repair continuation
    # already spent): either repeated fresh follow-ups were requested after
    # the cap terminal or the run never terminated at all.
    assert timed_out or _fresh_continuations(model_state) >= 2, (
        timed_out,
        len(model_state["requests"]),
    )
    paths = [path for path, _body in runtime_state["records"]]
    assert timed_out or paths.count("/v1/complete") > 2, (timed_out, paths)
    # Control: the shipped adapter on the same scenario stops at one.
    shipped_completed, shipped_model, shipped_runtime = _withhold_run(
        EXTENSION, tmp_path / "shipped", turns
    )
    assert shipped_completed.returncode == 0, shipped_completed.stderr
    assert _fresh_continuations(shipped_model) == 1, shipped_model["requests"]
    shipped_paths = [path for path, _body in shipped_runtime["records"]]
    assert shipped_paths.count("/v1/complete") == 2, shipped_paths
    assert shipped_paths.count("/v1/abandon") == 1, shipped_paths


def test_bundled_artifact_runs_the_unified_terminal_scenario(tmp_path: Path) -> None:
    """The bundled Pi artifact is behaviorally exercised, not just parsed:
    bundle_pi_extension inlines the pinned sources into one file, and that
    file runs the repeated-withhold terminal scenario with the unified
    budget — one follow-up total, two completions, one abandon, one
    terminal, no third agent operation."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    sys.path.insert(0, str(ROOT))
    import build_support.pi_bundle as pi_bundle

    bundled = tmp_path / "bundled.ts"
    pi_bundle.bundle_pi_extension(
        ROOT / "src" / "cortheon" / "pi_extension.ts",
        ROOT / "src" / "cortheon" / "pi_core",
        bundled,
    )
    assert bundled.exists()
    turns = [
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
    ]
    completed, model_state, runtime_state = _withhold_run(bundled, tmp_path / "run", turns)
    assert completed.returncode == 0, completed.stderr
    assert _fresh_continuations(model_state) == 1, model_state["requests"]
    assert len(model_state["requests"]) == 2, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 2, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert len(terminal_status_messages(completed)) == 1
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert AMBIGUITY_ANSWER not in answers, answers


def test_user_authored_continuation_prefix_is_an_ordinary_new_turn(
    tmp_path: Path,
) -> None:
    """``[CORTHEON_CONTINUE]`` in-band is not authentication: only the
    exact extension-scheduled follow-up counts. A user prompt that begins
    with the prefix is an ordinary new turn — stale answer-only state and
    the held disposition are cleared, its answer is delivered verbatim, and
    no follow-up is scheduled for it."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    plain_answer = "An ordinary ungated answer."
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [
            {"text": AMBIGUITY_ANSWER},
            {"text": AMBIGUITY_ANSWER},
            {"text": plain_answer},
        ],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": withholding_ambiguity_script(),
    }
    forged = f"[CORTHEON_CONTINUE] {AMBIGUITY_PROMPT}"
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            [AMBIGUITY_PROMPT, forged],
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # Prompt 1 ran its own bounded lifecycle: withhold -> one repair
    # continuation -> withhold -> terminal (two model requests). Prompt 2
    # (the forged prefix) was an ordinary new turn with one request of its
    # own, answered verbatim.
    assert len(model_state["requests"]) == 3, len(model_state["requests"])
    repair_head = "[CORTHEON_CONTINUE] Completion was withheld."
    assert _fresh_continuations(model_state, repair_head) == 1, model_state["requests"]
    assert answers[-1] == plain_answer, answers
    assert len(terminal_status_messages(completed)) == 1
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
