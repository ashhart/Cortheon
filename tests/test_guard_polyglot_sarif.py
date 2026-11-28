"""Comprehensive offline tests for Cortheon Guard Polyglot Runners, SARIF 2.1.0, Git Hooks & Live Events."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from flash_forge.client import CallSpec, CompletionResult, ModelCapabilities
from flash_forge.guard import (
    GuardReport,
    ProvenBug,
    detect_runner,
    format_guard_report,
    format_guard_sarif,
    install_git_hook,
    run_guard,
    uninstall_git_hook,
)
from flash_forge.progress import LiveProgressReporter, format_event


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _init_git_repo(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "cortheon-test@example.com")
    _git(repo, "config", "user.name", "Cortheon Test")
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Initial clean commit")
    return repo


def _count_git_worktrees(repo: Path) -> int:
    proc = _git(repo, "worktree", "list", "--porcelain")
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("worktree "))


def _mk_result(
    content: str,
    *,
    tag: str = "",
    prompt_tok: int = 100,
    comp_tok: int = 50,
    raw_error: Optional[str] = None,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="error" if raw_error else "stop",
        prompt_tokens=prompt_tok,
        completion_tokens=comp_tok,
        reasoning_tokens=0,
        cached_tokens=0,
        latency_s=0.04,
        raw_error=raw_error,
        tag=tag,
    )


class _ScriptedClient:
    """Deterministic offline mock client returning scripted batch responses."""

    def __init__(self, batches: list[list[str] | Callable[[list[CallSpec]], list[CompletionResult]]]) -> None:
        self._batches = list(batches)
        self.recorded_specs: list[list[CallSpec]] = []

    async def probe(self) -> ModelCapabilities:
        return ModelCapabilities(model_id="mock-local-model")

    async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
        self.recorded_specs.append(specs)
        if not self._batches:
            raise AssertionError("Unexpected extra client.batch() call")
        nxt = self._batches.pop(0)
        if callable(nxt):
            return nxt(specs)
        assert len(nxt) == len(specs), f"Expected {len(specs)} scripted replies, got {len(nxt)}"
        return [_mk_result(text, tag=s.tag) for text, s in zip(nxt, specs)]

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. Polyglot Runner Auto-Detection Tests
# ---------------------------------------------------------------------------


def test_detect_runner_polyglot_extensions_and_overrides():
    # Pure language auto-detection
    assert detect_runner(["math_utils.py"]) == "python"
    assert detect_runner(["src/app.js", "README.md"]) == "node"
    assert detect_runner(["src/app.mjs"]) == "node"
    assert detect_runner(["src/app.cjs"]) == "node"
    assert detect_runner(["src/app.ts", "package.json"]) == "node"
    assert detect_runner(["src/app.mts"]) == "node"
    assert detect_runner(["pkg/math.go", "go.mod"]) == "go"
    assert detect_runner(["bin/clamp.sh"]) == "shell"

    # Rust / Zig / mixed / unknown fall back cleanly to Python driver
    assert detect_runner(["src/lib.rs"]) == "python"
    assert detect_runner(["src/main.zig"]) == "python"
    assert detect_runner(["a.py", "b.js"]) == "python"
    assert detect_runner(["README.md"]) == "python"
    assert detect_runner([]) == "python"

    # Explicit overrides & aliases
    assert detect_runner(["app.py"], runner="shell") == "shell"
    assert detect_runner(["app.py"], runner="sh") == "shell"
    assert detect_runner(["app.py"], runner="bash") == "shell"
    assert detect_runner(["app.py"], runner="node") == "node"
    assert detect_runner(["app.py"], runner="javascript") == "node"
    assert detect_runner(["app.py"], runner="ts") == "node"
    assert detect_runner(["app.js"], runner="python") == "python"
    assert detect_runner(["app.js"], runner="go") == "go"

    with pytest.raises(ValueError, match="Unsupported guard runner"):
        detect_runner(["app.py"], runner="cobol")


# ---------------------------------------------------------------------------
# 2. JavaScript / Node.js Polyglot Sandbox Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="Requires Node.js (`node`) on PATH")
def test_guard_node_js_real_git_repo_proves_regression_and_discards_false_alarms(tmp_path: Path):
    clean_js = (
        "function clamp(x, lo, hi) {\n"
        "  if (x < lo) return lo;\n"
        "  if (x > hi) return hi;\n"
        "  return x;\n"
        "}\n"
        "module.exports = { clamp };\n"
    )
    repo = _init_git_repo(tmp_path, "js_repo", {"clamp.js": clean_js})

    # Plant an off-by-one upper-bound regression in working tree
    buggy_js = (
        "function clamp(x, lo, hi) {\n"
        "  if (x < lo) return lo;\n"
        "  if (x > hi) return hi - 1;\n"
        "  return x;\n"
        "}\n"
        "module.exports = { clamp };\n"
    )
    (repo / "clamp.js").write_text(buggy_js, encoding="utf-8")

    # 5 adversaries exercising every gate condition:
    # (1) Prose-only code review slop -> discarded before execution
    adv_prose = "Consider adding JSDoc comments and checking upper-bound edge cases."
    # (2) Broken JS syntax -> discarded by `node --check` before execution
    adv_bad_syntax = "```javascript\nfunction broken( { return 1\n```"
    # (3) Passing fake JS test (exits 0) -> executed & discarded as non-failing
    adv_passing = (
        "```javascript\n"
        "// TITLE: Lower bound sanity check\n"
        "const assert = require('node:assert/strict');\n"
        "const { clamp } = require('./clamp.js');\n"
        "assert.strictEqual(clamp(-5, 0, 5), 0);\n"
        "assert.strictEqual(clamp(3, 0, 5), 3);\n"
        "```"
    )
    # (4) Invalid reproducer requiring nonexistent package -> discarded as invalid reproducer
    adv_bad_import = (
        "```javascript\n"
        "// TITLE: Bogus import\n"
        "const nope = require('definitely_not_installed_cortheon_xyz_pkg');\n"
        "console.log(nope);\n"
        "```"
    )
    # (5) Real failing JS reproducer proving the off-by-one regression!
    adv_real_bug = (
        "```javascript\n"
        "// TITLE: Off-by-one upper bound in JS clamp returns hi - 1\n"
        "const assert = require('node:assert/strict');\n"
        "const { clamp } = require('./clamp.js');\n"
        "assert.strictEqual(clamp(10, 0, 5), 5);\n"
        "```"
    )

    client = _ScriptedClient([[adv_prose, adv_bad_syntax, adv_passing, adv_bad_import, adv_real_bug]])
    status_before = _git(repo, "status", "--porcelain").stdout

    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=5,
            verify_baseline=True,
            runner="auto",
        )
    )

    assert report.runner == "node"
    assert report.clean is False
    assert report.diff_files == ["clamp.js"]
    assert report.adversaries_run == 5
    assert report.reproducers_executed == 3  # passing + bad_import + real_bug
    assert report.rejected_non_failing == 4  # prose + bad_syntax + passing + bad_import
    assert len(report.proven_bugs) == 1

    bug = report.proven_bugs[0]
    assert bug.id == "BUG-01"
    assert bug.language == "node"
    assert "Off-by-one upper bound in JS clamp" in bug.title
    assert bug.passed_on_base is True
    assert "AssertionError" in bug.failure_output
    assert "clamp.js" in bug.target_files

    # User's repo MUST remain completely untouched and all worktrees cleaned up
    assert (repo / "clamp.js").read_text(encoding="utf-8") == buggy_js
    assert _git(repo, "status", "--porcelain").stdout == status_before
    assert _count_git_worktrees(repo) == 1

    rendered = format_guard_report(report, use_color=False)
    assert "Runner: node" in rendered
    assert "```javascript" in rendered
    assert "REGRESSION (passed on HEAD)" in rendered

    sarif = format_guard_sarif(report)
    assert len(sarif["runs"][0]["results"]) == 1
    r0 = sarif["runs"][0]["results"][0]
    assert r0["level"] == "error"
    assert r0["properties"]["language"] == "node"
    assert r0["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "clamp.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Requires Node.js (`node`) on PATH")
def test_guard_node_esm_and_auto_fix_produces_verified_patch(tmp_path: Path):
    clean_mjs = (
        "export function clamp(x, lo, hi) {\n"
        "  if (x < lo) return lo;\n"
        "  if (x > hi) return hi;\n"
        "  return x;\n"
        "}\n"
    )
    repo = _init_git_repo(tmp_path, "mjs_repo", {"clamp.mjs": clean_mjs})
    buggy_mjs = clean_mjs.replace("return hi;", "return hi - 1;")
    (repo / "clamp.mjs").write_text(buggy_mjs, encoding="utf-8")

    adv_esm = (
        "```js\n"
        "// TITLE: ESM clamp upper bound returns hi - 1\n"
        "import assert from 'node:assert/strict';\n"
        "import { clamp } from './clamp.mjs';\n"
        "assert.strictEqual(clamp(10, 0, 5), 5);\n"
        "```"
    )
    bad_fix = "```javascript\n" + clean_mjs.replace("return hi;", "return hi - 2;") + "```"
    good_fix = "```javascript\n" + clean_mjs + "```"

    client = _ScriptedClient([[adv_esm], [bad_fix, good_fix]])
    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=1,
            verify_baseline=True,
            auto_fix=True,
            runner="auto",
        )
    )

    assert report.runner == "node"
    assert len(report.proven_bugs) == 1
    assert report.proven_bugs[0].passed_on_base is True
    assert report.patch_verified is True
    assert report.suggested_patch is not None
    assert "-  if (x > hi) return hi - 1;" in report.suggested_patch
    assert "+  if (x > hi) return hi;" in report.suggested_patch
    assert (repo / "clamp.mjs").read_text(encoding="utf-8") == buggy_mjs
    assert _count_git_worktrees(repo) == 1


def _node_major_version() -> int:
    node_bin = shutil.which("node")
    if not node_bin:
        return 0
    try:
        out = subprocess.run([node_bin, "--version"], capture_output=True, text=True, timeout=5.0).stdout.strip()
        return int(out.lstrip("v").split(".", 1)[0])
    except Exception:
        return 0


@pytest.mark.skipif(_node_major_version() < 23, reason="Requires Node.js >= 23 native TypeScript execution")
def test_guard_node_typescript_real_git_repo_proves_regression(tmp_path: Path):
    clean_ts = (
        "export function clamp(x: number, lo: number, hi: number): number {\n"
        "  if (x < lo) return lo;\n"
        "  if (x > hi) return hi;\n"
        "  return x;\n"
        "}\n"
    )
    repo = _init_git_repo(tmp_path, "ts_repo", {"clamp.ts": clean_ts})
    buggy_ts = clean_ts.replace("return hi;", "return hi - 1;")
    (repo / "clamp.ts").write_text(buggy_ts, encoding="utf-8")

    adv_ts = (
        "```typescript\n"
        "// TITLE: TypeScript clamp upper bound returns hi - 1\n"
        "import assert from 'node:assert/strict';\n"
        "import { clamp } from './clamp.ts';\n"
        "const got: number = clamp(10, 0, 5);\n"
        "assert.strictEqual(got, 5);\n"
        "```"
    )
    client = _ScriptedClient([[adv_ts]])
    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=1,
            verify_baseline=True,
            runner="auto",
        )
    )

    assert report.runner == "node"
    assert len(report.proven_bugs) == 1
    assert report.proven_bugs[0].language == "node"
    assert report.proven_bugs[0].passed_on_base is True
    assert "clamp.ts" in report.proven_bugs[0].target_files
    assert (repo / "clamp.ts").read_text(encoding="utf-8") == buggy_ts
    assert _count_git_worktrees(repo) == 1


@pytest.mark.skipif(
    shutil.which("node") is None or not (sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec")),
    reason="Requires Node.js + macOS Seatbelt (/usr/bin/sandbox-exec)",
)
def test_guard_node_sandbox_blocks_writes_to_user_repo(tmp_path: Path):
    clean_js = "module.exports = { f: (x) => x };\n"
    repo = _init_git_repo(tmp_path, "js_sb_repo", {"mod.js": clean_js})
    buggy_js = "module.exports = { f: (x) => x + 1 };\n"
    target = repo / "mod.js"
    target.write_text(buggy_js, encoding="utf-8")

    malicious_js = (
        "```javascript\n"
        "const fs = require('node:fs');\n"
        f"fs.writeFileSync({str(target.resolve())!r}, 'PWNED', 'utf8');\n"
        "```"
    )
    client = _ScriptedClient([[malicious_js]])
    asyncio.run(run_guard(client, repo, num_adversaries=1, verify_baseline=False, runner="auto"))  # type: ignore[arg-type]

    assert target.read_text(encoding="utf-8") == buggy_js
    assert _count_git_worktrees(repo) == 1


# ---------------------------------------------------------------------------
# 3. Shell Runner (`runner="shell"`) OS Sandbox Tests
# ---------------------------------------------------------------------------


def test_guard_shell_runner_real_git_repo_proves_bug_and_discards_exit_zero(tmp_path: Path):
    clean_sh = (
        "#!/usr/bin/env sh\n"
        "x=$1; lo=$2; hi=$3\n"
        'if [ "$x" -lt "$lo" ]; then echo "$lo"; exit 0; fi\n'
        'if [ "$x" -gt "$hi" ]; then echo "$hi"; exit 0; fi\n'
        'echo "$x"\n'
    )
    repo = _init_git_repo(tmp_path, "sh_repo", {"clamp.sh": clean_sh})

    # Plant off-by-one bug: upper bound returns hi - 1
    buggy_sh = (
        "#!/usr/bin/env sh\n"
        "x=$1; lo=$2; hi=$3\n"
        'if [ "$x" -lt "$lo" ]; then echo "$lo"; exit 0; fi\n'
        'if [ "$x" -gt "$hi" ]; then echo $((hi - 1)); exit 0; fi\n'
        'echo "$x"\n'
    )
    (repo / "clamp.sh").write_text(buggy_sh, encoding="utf-8")

    # (1) Prose slop -> discarded before execution
    adv_prose = "Looks good to me."
    # (2) Broken shell syntax (missing `fi`) -> discarded by `sh -n` before execution
    adv_bad_syntax = "```sh\nif [ 1 -eq 1 ]; then\n  echo hi\n```"
    # (3) Passing exit-0 false alarm -> executed & discarded
    adv_passing = (
        "```sh\n"
        "# TITLE: Lower bound sanity check\n"
        "set -e\n"
        'out=$(sh ./clamp.sh -5 0 5)\n'
        '[ "$out" = "0" ]\n'
        "```"
    )
    # (4) Invalid reproducer calling nonexistent external command -> discarded
    adv_bad_cmd = (
        "```sh\n"
        "set -e\n"
        "definitely_nonexistent_cortheon_tool_xyz 1 2 3\n"
        "```"
    )
    # (5) Real failing shell reproducer proving the off-by-one bug!
    adv_real_bug = (
        "```sh\n"
        "# TITLE: Shell clamp upper bound returns hi - 1\n"
        "set -e\n"
        'out=$(sh ./clamp.sh 10 0 5)\n'
        'if [ "$out" != "5" ]; then\n'
        '  echo "expected 5, got $out" >&2\n'
        "  exit 1\n"
        "fi\n"
        "```"
    )

    client = _ScriptedClient([[adv_prose, adv_bad_syntax, adv_passing, adv_bad_cmd, adv_real_bug]])
    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=5,
            verify_baseline=True,
            runner="shell",
        )
    )

    assert report.runner == "shell"
    assert report.clean is False
    assert report.reproducers_executed == 3  # passing + bad_cmd + real_bug
    assert report.rejected_non_failing == 4  # prose + bad_syntax + passing + bad_cmd
    assert len(report.proven_bugs) == 1

    bug = report.proven_bugs[0]
    assert bug.language == "shell"
    assert "Shell clamp upper bound returns hi - 1" in bug.title
    assert bug.passed_on_base is True
    assert "expected 5, got 4" in bug.failure_output
    assert "clamp.sh" in bug.target_files
    assert (repo / "clamp.sh").read_text(encoding="utf-8") == buggy_sh
    assert _count_git_worktrees(repo) == 1


@pytest.mark.skipif(
    not (sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec")),
    reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)",
)
def test_guard_shell_sandbox_blocks_writes_to_user_repo(tmp_path: Path):
    clean_sh = "#!/usr/bin/env sh\necho ok\n"
    repo = _init_git_repo(tmp_path, "sh_sb_repo", {"run.sh": clean_sh})
    buggy_sh = "#!/usr/bin/env sh\necho bad\n"
    target = repo / "run.sh"
    target.write_text(buggy_sh, encoding="utf-8")

    malicious_sh = (
        "```sh\n"
        "set -e\n"
        f"echo PWNED > '{str(target.resolve())}'\n"
        "```"
    )
    client = _ScriptedClient([[malicious_sh]])
    asyncio.run(run_guard(client, repo, num_adversaries=1, verify_baseline=False, runner="shell"))  # type: ignore[arg-type]

    assert target.read_text(encoding="utf-8") == buggy_sh
    assert _count_git_worktrees(repo) == 1


# ---------------------------------------------------------------------------
# 4. Go Polyglot Sandbox Execution Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("go") is None, reason="Requires Go toolchain (`go`) on PATH")
def test_guard_go_runner_real_git_repo_proves_regression(tmp_path: Path):
    go_mod = "module example.com/clamp\n\ngo 1.21\n"
    clean_go = (
        "package clamp\n\n"
        "func Clamp(x, lo, hi int) int {\n"
        "\tif x < lo {\n"
        "\t\treturn lo\n"
        "\t}\n"
        "\tif x > hi {\n"
        "\t\treturn hi\n"
        "\t}\n"
        "\treturn x\n"
        "}\n"
    )
    repo = _init_git_repo(tmp_path, "go_repo", {"go.mod": go_mod, "clamp.go": clean_go})

    buggy_go = clean_go.replace("\t\treturn hi\n", "\t\treturn hi - 1\n")
    (repo / "clamp.go").write_text(buggy_go, encoding="utf-8")

    adv_passing = (
        "```go\n"
        "// TITLE: Lower bound passes\n"
        "package clamp\n"
        'import "testing"\n'
        "func TestCortheonGuardPass(t *testing.T) {\n"
        "\tif got := Clamp(-5, 0, 5); got != 0 {\n"
        '\t\tt.Fatalf("expected 0, got %d", got)\n'
        "\t}\n"
        "}\n"
        "```"
    )
    adv_real_bug = (
        "```go\n"
        "// TITLE: Go Clamp upper bound off-by-one\n"
        "package clamp\n"
        'import "testing"\n'
        "func TestCortheonGuardUpper(t *testing.T) {\n"
        "\tif got := Clamp(10, 0, 5); got != 5 {\n"
        '\t\tt.Fatalf("expected 5, got %d", got)\n'
        "\t}\n"
        "}\n"
        "```"
    )

    client = _ScriptedClient([[adv_passing, adv_real_bug]])
    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=2,
            verify_baseline=True,
            runner="auto",
        )
    )

    assert report.runner == "go"
    assert report.clean is False
    assert report.reproducers_executed == 2
    assert report.rejected_non_failing == 1
    assert len(report.proven_bugs) == 1

    bug = report.proven_bugs[0]
    assert bug.language == "go"
    assert "Go Clamp upper bound off-by-one" in bug.title
    assert bug.passed_on_base is True
    assert "expected 5, got 4" in bug.failure_output
    assert "clamp.go" in bug.target_files
    assert (repo / "clamp.go").read_text(encoding="utf-8") == buggy_go
    assert _count_git_worktrees(repo) == 1


# ---------------------------------------------------------------------------
# 5. SARIF 2.1.0 Exporter Schema & Severity Tests
# ---------------------------------------------------------------------------


def test_format_guard_sarif_schema_with_proven_bugs_and_severity_mapping():
    report = GuardReport(
        repo_path="/tmp/example_repo",
        base_ref="HEAD",
        diff_files=["src/math_utils.py", "web/clamp.js", "scripts/tool.sh"],
        adversaries_run=4,
        reproducers_executed=3,
        rejected_non_failing=1,
        proven_bugs=[
            ProvenBug(
                id="BUG-01",
                role="regression_contract_breaker",
                title="Upper bound off-by-one in clamp",
                reproducer_code="from src.math_utils import clamp\nassert clamp(10, 0, 5) == 5",
                failure_output='Traceback (most recent call last):\n  File "src/math_utils.py", line 6, in clamp\nAssertionError: got 4',
                passed_on_base=True,
                target_files=["./src\\math_utils.py"],
                language="python",
            ),
            ProvenBug(
                id="BUG-02",
                role="edge_boundary_breaker",
                title="Pre-existing NaN handling bug",
                reproducer_code="assert.strictEqual(clamp(NaN, 0, 5), 0);",
                failure_output="AssertionError: NaN !== 0",
                passed_on_base=False,
                target_files=["/tmp/example_repo/web/clamp.js"],
                language="node",
            ),
            ProvenBug(
                id="BUG-03",
                role="custom_shell_breaker",
                title="New helper fails on empty arg",
                reproducer_code="sh ./scripts/tool.sh ''",
                failure_output="unexpected empty output",
                passed_on_base=None,
                target_files=["scripts/tool.sh"],
                language="shell",
            ),
        ],
        suggested_patch=None,
        patch_verified=False,
        wall_s=1.234,
        llm_calls=4,
        total_tokens=900,
        runner="python",
    )

    sarif = format_guard_sarif(report)

    # Strict JSON round-trip check
    round_tripped = json.loads(json.dumps(sarif))
    assert round_tripped == sarif

    # Top-level SARIF 2.1.0 schema
    assert sarif["version"] == "2.1.0"
    assert "sarif" in sarif["$schema"].lower() and "2.1.0" in sarif["$schema"]
    assert isinstance(sarif["runs"], list) and len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "Cortheon Guard"
    assert driver["version"] == "2.5.0"
    rule_ids = {r["id"] for r in driver["rules"]}
    assert "cortheon-guard/regression_contract_breaker" in rule_ids
    assert "cortheon-guard/edge_boundary_breaker" in rule_ids
    assert "cortheon-guard/custom_shell_breaker" in rule_ids

    results = run["results"]
    assert len(results) == 3

    # BUG-01: regression -> error, normalized relative URI, extracted line 6
    r1 = results[0]
    assert r1["ruleId"] == "cortheon-guard/regression_contract_breaker"
    assert r1["level"] == "error"
    assert "Upper bound off-by-one in clamp" in r1["message"]["text"]
    assert "AssertionError: got 4" in r1["message"]["text"]
    assert "assert clamp(10, 0, 5) == 5" in r1["message"]["text"]
    loc1 = r1["locations"][0]["physicalLocation"]
    assert loc1["artifactLocation"]["uri"] == "src/math_utils.py"
    assert loc1["region"]["startLine"] == 6
    assert r1["properties"]["role"] == "regression_contract_breaker"
    assert r1["properties"]["passed_on_base"] is True
    assert r1["properties"]["language"] == "python"

    # BUG-02: pre-existing -> warning, stripped repo_path prefix, default line 1
    r2 = results[1]
    assert r2["level"] == "warning"
    loc2 = r2["locations"][0]["physicalLocation"]
    assert loc2["artifactLocation"]["uri"] == "web/clamp.js"
    assert loc2["region"]["startLine"] == 1
    assert r2["properties"]["passed_on_base"] is False
    assert r2["properties"]["language"] == "node"

    # BUG-03: new code (passed_on_base=None) -> warning, custom rule registered
    r3 = results[2]
    assert r3["ruleId"] == "cortheon-guard/custom_shell_breaker"
    assert r3["level"] == "warning"
    assert r3["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "scripts/tool.sh"
    assert r3["locations"][0]["physicalLocation"]["region"]["startLine"] == 1
    assert r3["properties"]["passed_on_base"] is None
    assert r3["properties"]["language"] == "shell"


def test_format_guard_sarif_clean_zero_bug_report():
    clean_report = GuardReport(
        repo_path="/tmp/clean_repo",
        base_ref="HEAD",
        diff_files=[],
        adversaries_run=0,
        reproducers_executed=0,
        rejected_non_failing=0,
        proven_bugs=[],
        suggested_patch=None,
        patch_verified=False,
        wall_s=0.01,
        llm_calls=0,
        total_tokens=0,
    )

    sarif = format_guard_sarif(clean_report)
    assert json.loads(json.dumps(sarif)) == sarif
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["tool"]["driver"]["name"] == "Cortheon Guard"
    assert sarif["runs"][0]["tool"]["driver"]["version"] == "2.5.0"
    assert sarif["runs"][0]["results"] == []


# ---------------------------------------------------------------------------
# 6. Git Pre-Push Hook Installer / Uninstaller Tests
# ---------------------------------------------------------------------------


def test_git_hook_install_idempotent_refuse_clobber_force_uninstall_and_worktree(tmp_path: Path):
    repo = _init_git_repo(tmp_path, "hook_repo", {"README.md": "hello\n"})

    # 1. Fresh install creates executable pre-push hook
    hook_path = install_git_hook(repo, url="http://127.0.0.1:8080/v1")
    assert hook_path == (repo / ".git" / "hooks" / "pre-push").resolve()
    assert hook_path.is_file()
    mode = stat.S_IMODE(hook_path.stat().st_mode)
    assert mode == 0o755
    assert os.access(hook_path, os.X_OK)

    content = hook_path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env sh\n")
    assert "# CORTHEON-GUARD-HOOK" in content
    assert "cortheon guard --repo . --url http://127.0.0.1:8080/v1" in content
    assert "|| true" not in content

    # Verify generated hook passes POSIX `sh -n` syntax validation
    sh_check = subprocess.run(["sh", "-n", str(hook_path)], capture_output=True, text=True)
    assert sh_check.returncode == 0, f"Hook failed sh -n: {sh_check.stderr}"

    # 2. Idempotent update changes URL and non-blocking mode cleanly
    updated_path = install_git_hook(
        repo,
        url="http://127.0.0.1:18651/v1",
        block_on_bug=False,
    )
    assert updated_path == hook_path
    updated_content = hook_path.read_text(encoding="utf-8")
    assert "# CORTHEON-GUARD-HOOK" in updated_content
    assert "http://127.0.0.1:18651/v1" in updated_content
    assert "|| true" in updated_content
    assert subprocess.run(["sh", "-n", str(hook_path)], capture_output=True, text=True).returncode == 0

    # 3. Clean uninstall removes Cortheon hook and is idempotent when called again
    assert uninstall_git_hook(repo) is True
    assert not hook_path.exists()
    assert uninstall_git_hook(repo) is False

    # 4. Refuse to clobber an existing custom non-Cortheon hook unless force=True
    custom_script = "#!/usr/bin/env sh\necho 'custom user pre-push hook'\n"
    hook_path.write_text(custom_script, encoding="utf-8")
    os.chmod(hook_path, 0o755)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install_git_hook(repo, url="http://127.0.0.1:8080/v1", force=False)
    assert hook_path.read_text(encoding="utf-8") == custom_script

    # Also refuse to uninstall custom non-Cortheon hook unless force=True
    with pytest.raises(RuntimeError, match="Refusing to remove"):
        uninstall_git_hook(repo, force=False)
    assert hook_path.read_text(encoding="utf-8") == custom_script

    # force=True cleanly replaces custom hook
    install_git_hook(repo, url="http://127.0.0.1:9090/v1", force=True)
    assert "# CORTHEON-GUARD-HOOK" in hook_path.read_text(encoding="utf-8")
    assert "http://127.0.0.1:9090/v1" in hook_path.read_text(encoding="utf-8")
    assert uninstall_git_hook(repo) is True

    # 5. Linked git worktree resolves common .git/hooks cleanly via `git rev-parse --git-path hooks`
    wt = tmp_path / "linked_wt"
    _git(repo, "worktree", "add", "--detach", str(wt), "HEAD")
    try:
        wt_hook = install_git_hook(wt, url="http://127.0.0.1:8080/v1")
        assert wt_hook == hook_path
        assert wt_hook.is_file()
        assert uninstall_git_hook(wt) is True
        assert not wt_hook.exists()
    finally:
        _git(repo, "worktree", "remove", "--force", str(wt))


# ---------------------------------------------------------------------------
# 7. Live `on_event` Callback Sequence, Progress Compat & Fault Isolation
# ---------------------------------------------------------------------------


def test_guard_on_event_callback_sequence_progress_compat_and_fault_isolation(tmp_path: Path):
    clean_py = (
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        "    if x < lo: return lo\n"
        "    if x > hi: return hi\n"
        "    return x\n"
    )
    repo = _init_git_repo(tmp_path, "event_repo", {"math_utils.py": clean_py})

    # A. Clean empty diff emits start -> done cleanly
    empty_events: list[dict[str, Any]] = []
    empty_report = asyncio.run(
        run_guard(_ScriptedClient([]), repo, on_event=empty_events.append)  # type: ignore[arg-type]
    )
    assert empty_report.clean is True
    assert [e["event"] for e in empty_events] == ["start", "done"]
    assert all(e["engine"] == "guard" for e in empty_events)

    # B. Buggy diff emits start -> slot_done (per adversary) -> bug_proven -> done
    buggy_py = clean_py.replace("return hi", "return hi - 1")
    (repo / "math_utils.py").write_text(buggy_py, encoding="utf-8")

    adv1_prose = "Looks fine to me."
    adv2_pass = (
        "```python\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import clamp\n"
        "assert clamp(2, 0, 5) == 2\n"
        "```"
    )
    adv3_fail = (
        "```python\n"
        "# TITLE: clamp upper bound off by one\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import clamp\n"
        "assert clamp(10, 0, 5) == 5\n"
        "```"
    )

    events: list[dict[str, Any]] = []
    buf = io.StringIO()
    reporter = LiveProgressReporter(stream=buf, use_color=False)

    def _collect_and_render(ev: dict[str, Any]) -> None:
        events.append(ev)
        reporter.emit(ev)

    client = _ScriptedClient([[adv1_prose, adv2_pass, adv3_fail]])
    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=3,
            verify_baseline=True,
            on_event=_collect_and_render,
        )
    )

    assert len(report.proven_bugs) == 1
    seq = [e["event"] for e in events]
    assert seq == ["start", "slot_done", "slot_done", "slot_done", "bug_proven", "done"]
    assert all(e["engine"] == "guard" for e in events)
    assert events[0]["streams"] == 3
    assert events[4]["id"] == "BUG-01"
    assert events[4]["passed_on_base"] is True
    assert events[4]["language"] == "python"
    assert events[-1]["proven_bugs"] == 1
    assert events[-1]["clean"] is False

    # Verify LiveProgressReporter rendered every event cleanly with zero errors
    rendered_progress = buf.getvalue()
    assert "[cortheon] guard:" in rendered_progress
    assert "[BUG PROVEN]" in rendered_progress
    assert "REGRESSION" in rendered_progress
    assert "[cortheon] done" in rendered_progress
    for ev in events:
        line = format_event(ev, use_color=False)
        assert line.strip()

    # C. Fault isolation: a crashing `on_event` callback MUST NEVER crash `run_guard`
    def _exploding_callback(_ev: dict[str, Any]) -> None:
        raise RuntimeError("observer crashed!")

    client2 = _ScriptedClient([[adv3_fail]])
    report2 = asyncio.run(
        run_guard(
            client2,  # type: ignore[arg-type]
            repo,
            num_adversaries=1,
            verify_baseline=True,
            on_event=_exploding_callback,
        )
    )
    assert len(report2.proven_bugs) == 1
    assert report2.proven_bugs[0].passed_on_base is True
    assert _count_git_worktrees(repo) == 1
