"""End-to-end tests for Cortheon Guard (`flash_forge.guard`) and Stdio MCP Server (`flash_forge.mcp_server`)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from flash_forge.client import CallSpec, CompletionResult, ModelCapabilities
from flash_forge.engine import ForgeResult
from flash_forge.guard import GuardReport, ProvenBug, format_guard_report, run_guard
from flash_forge.mcp_server import (
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
    MCPServer,
    read_stdio_message,
    run_stdio_server,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc


def _init_math_repo(tmp_path: Path) -> Path:
    """Create a real git repository with clean `math_utils.py` committed at HEAD."""
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "cortheon-test@example.com")
    _git(repo, "config", "user.name", "Cortheon Test")

    clean_src = (
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Clamp x into the closed interval [lo, hi]."""\n'
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi\n"
        "    return x\n"
    )
    (repo / "math_utils.py").write_text(clean_src, encoding="utf-8")
    _git(repo, "add", "math_utils.py")
    _git(repo, "commit", "-m", "Initial clean clamp implementation")
    return repo


def _count_git_worktrees(repo: Path) -> int:
    proc = _git(repo, "worktree", "list", "--porcelain")
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("worktree "))


def _mk_result(content: str, *, tag: str = "", prompt_tok: int = 120, comp_tok: int = 60, raw_error: Optional[str] = None) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="error" if raw_error else "stop",
        prompt_tokens=prompt_tok,
        completion_tokens=comp_tok,
        reasoning_tokens=0,
        cached_tokens=0,
        latency_s=0.05,
        raw_error=raw_error,
        tag=tag,
    )


class _ScriptedClient:
    """Deterministic mock TensorFoldClient returning scripted batch responses."""

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
# Cortheon Guard tests
# ---------------------------------------------------------------------------


def test_guard_empty_diff_short_circuits_with_zero_llm_calls(tmp_path: Path):
    repo = _init_math_repo(tmp_path)
    client = _ScriptedClient([])  # Any LLM call would fail the test!

    report = asyncio.run(run_guard(client, repo))  # type: ignore[arg-type]

    assert report.clean is True
    assert report.diff_files == []
    assert report.adversaries_run == 0
    assert report.reproducers_executed == 0
    assert report.rejected_non_failing == 0
    assert report.proven_bugs == []
    assert report.llm_calls == 0
    assert report.total_tokens == 0
    assert _count_git_worktrees(repo) == 1

    rendered = format_guard_report(report, use_color=False)
    assert "[CLEAN] 0 proven bugs (0 reproducers executed, 0 broke the diff)" in rendered


def test_guard_end_to_end_real_git_repo_zero_fp_gate_and_baseline(tmp_path: Path):
    repo = _init_math_repo(tmp_path)

    # Introduce real off-by-one bug in the working tree
    buggy_src = (
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Clamp x into the closed interval [lo, hi]."""\n'
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi - 1\n"
        "    return x\n"
    )
    (repo / "math_utils.py").write_text(buggy_src, encoding="utf-8")

    # 3 adversaries:
    # (1) Prose-only code review slop (no executable Python code)
    adv1_prose = (
        "Looks mostly fine, though you might want to check edge cases around upper bounds "
        "and consider adding type hints or docstrings."
    )
    # (2) Valid Python test that PASSES (false alarm -> exits 0 -> must be discarded!)
    adv2_passing = (
        "```python\n"
        "# TITLE: Lower bound sanity check\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import clamp\n"
        "assert clamp(-5, 0, 5) == 0\n"
        "assert clamp(3, 0, 5) == 3\n"
        "```"
    )
    # (3) Real failing reproducer exposing the off-by-one upper-bound regression!
    adv3_real_bug = (
        "```python\n"
        "# TITLE: Off-by-one upper bound clamp returns hi - 1\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import clamp\n"
        "assert clamp(10, 0, 5) == 5, f'expected 5, got {clamp(10, 0, 5)}'\n"
        "```"
    )

    client = _ScriptedClient([[adv1_prose, adv2_passing, adv3_real_bug]])
    status_before = _git(repo, "status", "--porcelain").stdout

    report = asyncio.run(run_guard(client, repo, num_adversaries=3, verify_baseline=True))  # type: ignore[arg-type]

    # Mechanical proof gate assertions
    assert report.clean is False
    assert report.diff_files == ["math_utils.py"]
    assert report.adversaries_run == 3
    assert report.reproducers_executed == 2  # prose-only was rejected before execution
    assert report.rejected_non_failing == 2  # prose-only + passing false alarm
    assert report.llm_calls == 3
    assert report.total_tokens == 3 * 180
    assert len(report.proven_bugs) == 1

    bug = report.proven_bugs[0]
    assert bug.id == "BUG-01"
    assert bug.role == "regression_contract_breaker"
    assert "Off-by-one upper bound" in bug.title
    assert bug.passed_on_base is True  # Passed on HEAD, broke on buggy working tree!
    assert "AssertionError" in bug.failure_output
    assert "expected 5, got 4" in bug.failure_output
    assert "math_utils.py" in bug.target_files

    # User's repo MUST be untouched and ALL ephemeral worktrees MUST be removed
    assert (repo / "math_utils.py").read_text(encoding="utf-8") == buggy_src
    assert _git(repo, "status", "--porcelain").stdout == status_before
    assert _count_git_worktrees(repo) == 1

    # Report serialization and terminal formatting
    as_dict = report.to_dict()
    assert as_dict["clean"] is False
    assert len(as_dict["proven_bugs"]) == 1
    assert as_dict["proven_bugs"][0]["passed_on_base"] is True

    formatted = format_guard_report(report, use_color=False)
    assert "[PROVEN BUG] BUG-01" in formatted
    assert "REGRESSION (passed on HEAD)" in formatted
    assert "assert clamp(10, 0, 5) == 5" in formatted
    assert "AssertionError" in formatted


def test_guard_auto_fix_produces_and_verifies_unified_patch(tmp_path: Path):
    repo = _init_math_repo(tmp_path)

    buggy_src = (
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Clamp x into the closed interval [lo, hi]."""\n'
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi - 1\n"
        "    return x\n"
    )
    (repo / "math_utils.py").write_text(buggy_src, encoding="utf-8")

    adv_repro = (
        "```python\n"
        "# TITLE: Upper bound off-by-one in clamp\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import clamp\n"
        "assert clamp(10, 0, 5) == 5\n"
        "assert clamp(-2, 0, 5) == 0\n"
        "```"
    )
    # Auto-fix batch (B=2): candidate 1 still buggy, candidate 2 cleanly fixed!
    still_buggy_fix = (
        "```python\n"
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Clamp x into the closed interval [lo, hi]."""\n'
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi - 2\n"
        "    return x\n"
        "```"
    )
    fixed_file = (
        "```python\n"
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Clamp x into the closed interval [lo, hi]."""\n'
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi\n"
        "    return x\n"
        "```"
    )

    client = _ScriptedClient(
        [
            [adv_repro],
            [still_buggy_fix, fixed_file],
        ]
    )

    # Also exercise test_cmd verification alongside proven bug reproducers
    test_cmd = f'{sys.executable} -c "from math_utils import clamp; assert clamp(100, 0, 5) == 5 and clamp(2, 0, 5) == 2"'
    report = asyncio.run(
        run_guard(
            client,  # type: ignore[arg-type]
            repo,
            num_adversaries=1,
            verify_baseline=True,
            auto_fix=True,
            test_cmd=test_cmd,
        )
    )

    assert len(report.proven_bugs) == 1
    assert report.patch_verified is True
    assert report.suggested_patch is not None
    assert "--- a/math_utils.py" in report.suggested_patch
    assert "+++ b/math_utils.py" in report.suggested_patch
    assert "-        return hi - 1" in report.suggested_patch
    assert "+        return hi" in report.suggested_patch
    assert report.llm_calls == 3  # 1 adversary + 2 auto-fix candidates

    # User's working tree was NOT mutated by auto-fix
    assert (repo / "math_utils.py").read_text(encoding="utf-8") == buggy_src
    assert _count_git_worktrees(repo) == 1

    # Verify the generated unified patch applies cleanly to the buggy working tree!
    apply_proc = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", "-"],
        input=report.suggested_patch,
        capture_output=True,
        text=True,
    )
    assert apply_proc.returncode == 0, f"Suggested patch failed git apply --check: {apply_proc.stderr}"

    rendered = format_guard_report(report, use_color=False)
    assert "[VERIFIED PATCH]" in rendered


def test_guard_discards_invalid_reproducer_imports_and_handles_new_function_bugs(tmp_path: Path):
    repo = _init_math_repo(tmp_path)

    # Add a brand-new function `double_pos` with a bug (did NOT exist on HEAD!)
    updated_src = (
        (repo / "math_utils.py").read_text(encoding="utf-8")
        + "\n\ndef double_pos(n: int) -> int:\n"
        "    if n <= 0:\n"
        "        return 0\n"
        "    return n * 3  # bug: triples instead of doubles\n"
    )
    (repo / "math_utils.py").write_text(updated_src, encoding="utf-8")

    # (1) Invalid reproducer importing a non-existent module -> MUST be discarded!
    adv_bad_import = (
        "```python\n"
        "import definitely_not_installed_xyz_pkg\n"
        "assert False\n"
        "```"
    )
    # (2) Valid reproducer exercising newly added function -> fails on diff, ImportError on HEAD -> passed_on_base=None!
    adv_new_fn = (
        "```python\n"
        "# TITLE: New double_pos triples instead of doubling\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import double_pos\n"
        "assert double_pos(4) == 8, f'got {double_pos(4)}'\n"
        "```"
    )
    # (3) Duplicate reproducer with identical failure trace -> deduplicated!
    adv_new_fn_dup = (
        "```python\n"
        "# TITLE: Duplicate check on double_pos\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import double_pos\n"
        "assert double_pos(4) == 8, f'got {double_pos(4)}'\n"
        "```"
    )

    client = _ScriptedClient([[adv_bad_import, adv_new_fn, adv_new_fn_dup]])
    report = asyncio.run(run_guard(client, repo, num_adversaries=3, verify_baseline=True))  # type: ignore[arg-type]

    assert report.rejected_non_failing == 1  # adv_bad_import discarded
    assert len(report.proven_bugs) == 1  # adv_new_fn kept, dup deduplicated
    assert report.proven_bugs[0].passed_on_base is None  # new symbol didn't exist on HEAD, still reported!
    assert "got 12" in report.proven_bugs[0].failure_output
    assert _count_git_worktrees(repo) == 1


def test_guard_explicit_diff_text_and_target_syntax_error_is_caught(tmp_path: Path):
    repo = _init_math_repo(tmp_path)

    # Working tree stays completely clean; caller passes an explicit unified diff introducing a SyntaxError
    bad_diff = (
        "diff --git a/math_utils.py b/math_utils.py\n"
        "--- a/math_utils.py\n"
        "+++ b/math_utils.py\n"
        "@@ -4,4 +4,4 @@ def clamp(x: int, lo: int, hi: int) -> int:\n"
        "         return lo\n"
        "     if x > hi:\n"
        "-        return hi\n"
        "+        return hi (\n"
        "     return x\n"
    )
    repro = (
        "```python\n"
        "# TITLE: SyntaxError introduced in math_utils\n"
        "import sys; sys.path.insert(0, '.')\n"
        "import math_utils\n"
        "assert math_utils.clamp(10, 0, 5) == 5\n"
        "```"
    )
    client = _ScriptedClient([[repro]])
    report = asyncio.run(run_guard(client, repo, diff_text=bad_diff, num_adversaries=1, verify_baseline=True))  # type: ignore[arg-type]

    assert len(report.proven_bugs) == 1
    assert report.proven_bugs[0].passed_on_base is True
    assert "SyntaxError" in report.proven_bugs[0].failure_output
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _count_git_worktrees(repo) == 1


# ---------------------------------------------------------------------------
# Native Stdio MCP Server tests
# ---------------------------------------------------------------------------


def test_mcp_protocol_lifecycle_tools_list_and_error_handling():
    server = MCPServer(default_url="http://127.0.0.1:9123/v1")

    # 1. initialize
    init_resp = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "pytest-client", "version": "0.1"},
                    "capabilities": {},
                },
            }
        )
    )
    assert init_resp is not None
    assert init_resp["jsonrpc"] == "2.0"
    assert init_resp["id"] == 1
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"
    assert init_resp["result"]["serverInfo"]["name"] == "cortheon"
    assert "tools" in init_resp["result"]["capabilities"]

    # 2. notifications/initialized -> MUST return None
    notif_resp = asyncio.run(
        server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
    )
    assert notif_resp is None

    # 3. ping (including id=0 falsy edge case!)
    ping_resp = asyncio.run(server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "ping"}))
    assert ping_resp == {"jsonrpc": "2.0", "id": 0, "result": {}}

    # 4. tools/list exposes all 4 required tools
    list_resp = asyncio.run(server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    assert list_resp is not None
    tools = {t["name"]: t for t in list_resp["result"]["tools"]}
    assert set(tools.keys()) == {
        "cortheon_verify",
        "cortheon_guard",
        "cortheon_doctor",
        "cortheon_spec_verify",
    }
    assert "prompt" in tools["cortheon_verify"]["inputSchema"]["required"]
    assert "repo_path" in tools["cortheon_guard"]["inputSchema"]["required"]
    assert set(tools["cortheon_spec_verify"]["inputSchema"]["required"]) == {
        "prompt",
        "drafter_url",
        "verifier_url",
    }

    # 5. Unknown method -> -32601
    bad_method = asyncio.run(
        server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "nonexistent/method"})
    )
    assert bad_method is not None
    assert bad_method["error"]["code"] == ERR_METHOD_NOT_FOUND

    # 6. Unknown tool -> -32602
    bad_tool = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "cortheon_nonexistent", "arguments": {}},
            }
        )
    )
    assert bad_tool is not None
    assert bad_tool["error"]["code"] == ERR_INVALID_PARAMS
    assert "Unknown tool" in bad_tool["error"]["message"]

    # 7. Invalid params (missing required arg / bad types / bool-as-int trap) -> -32602
    invalid_cases = [
        {"name": "cortheon_verify", "arguments": {}},
        {"name": "cortheon_verify", "arguments": {"prompt": "hi", "mode": "bogus"}},
        {"name": "cortheon_verify", "arguments": {"prompt": "hi", "streams": True}},
        {"name": "cortheon_guard", "arguments": {"repo_path": "/tmp", "num_adversaries": 0}},
        {"name": "cortheon_guard", "arguments": {"repo_path": "/tmp", "auto_fix": "yes"}},
        {"name": "cortheon_doctor", "arguments": {"skip_live": "nope"}},
        {"name": "cortheon_spec_verify", "arguments": {"prompt": "hi", "drafter_url": "http://x"}},
    ]
    for idx, params in enumerate(invalid_cases, start=10):
        resp = asyncio.run(
            server.handle_request({"jsonrpc": "2.0", "id": idx, "method": "tools/call", "params": params})
        )
        assert resp is not None
        assert resp["error"]["code"] == ERR_INVALID_PARAMS, f"Case {params} should fail with -32602"


def test_mcp_tools_call_doctor_skip_live_and_mocked_verify(monkeypatch):
    # Ensure CORTHEON_URL env var is honored when no explicit default_url is given
    monkeypatch.setenv("CORTHEON_URL", "http://127.0.0.1:19999/v1")

    # 1. cortheon_doctor with skip_live=True (real local sandbox + solver + harness checks!)
    server = MCPServer()
    assert server.default_url == "http://127.0.0.1:19999/v1"

    doc_resp = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_doctor",
                    "arguments": {"skip_live": True},
                },
            }
        )
    )
    assert doc_resp is not None
    assert doc_resp["result"]["isError"] is False
    doc_payload = json.loads(doc_resp["result"]["content"][0]["text"])
    assert doc_payload["url"] == "http://127.0.0.1:19999/v1"
    checks_by_id = {c["id"]: c for c in doc_payload["checks"]}
    assert checks_by_id["sandbox"]["status"] in ("pass", "warn")
    assert checks_by_id["solver_stack"]["status"] == "pass"
    assert checks_by_id["live_probe"]["status"] == "skip"
    assert "Cortheon Doctor" in doc_payload["formatted"]

    # 2. cortheon_verify with mocked FlashForge.solve
    captured: dict[str, Any] = {}

    async def _fake_probe(self):
        return ModelCapabilities(model_id="mock-model")

    async def _fake_solve(self, prompt: str, *, force_mode=None, **kwargs):
        captured["prompt"] = prompt
        captured["force_mode"] = force_mode
        captured["base_url"] = self.config.base_url
        captured["num_streams"] = self.config.num_streams
        return ForgeResult(
            mode=force_mode or "reason",
            answer="42",
            code="",
            wall_s=0.12,
            token_stats={"calls": 2, "prompt_tokens": 100, "completion_tokens": 20},
            verified=True,
        )

    monkeypatch.setattr("flash_forge.client.TensorFoldClient.probe", _fake_probe)
    monkeypatch.setattr("flash_forge.engine.FlashForge.solve", _fake_solve)

    ver_resp = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 101,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_verify",
                    "arguments": {
                        "prompt": "Compute 6 * 7",
                        "mode": "reason",
                        "streams": 2,
                    },
                },
            }
        )
    )
    assert ver_resp is not None
    assert ver_resp["result"]["isError"] is False
    ver_payload = json.loads(ver_resp["result"]["content"][0]["text"])
    assert ver_payload["answer"] == "42"
    assert ver_payload["mode"] == "reason"
    assert ver_payload["verified"] is True
    assert captured == {
        "prompt": "Compute 6 * 7",
        "force_mode": "reason",
        "base_url": "http://127.0.0.1:19999/v1",
        "num_streams": 2,
    }


def test_mcp_tools_call_guard_and_spec_verify_and_tool_error_isolation(tmp_path: Path, monkeypatch):
    # 1. cortheon_guard on clean repo short-circuits cleanly without hitting network
    repo = _init_math_repo(tmp_path)
    server = MCPServer(default_url="http://127.0.0.1:19998/v1")

    g_resp = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 200,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_guard",
                    "arguments": {"repo_path": str(repo), "base_ref": "HEAD", "num_adversaries": 3},
                },
            }
        )
    )
    assert g_resp is not None
    assert g_resp["result"]["isError"] is False
    g_payload = json.loads(g_resp["result"]["content"][0]["text"])
    assert g_payload["clean"] is True
    assert g_payload["llm_calls"] == 0
    assert "[CLEAN]" in g_payload["formatted"]

    # 2. cortheon_spec_verify wired directly to flash_forge.spec_verify.solve_spec_verify
    async def _fake_complete(self, spec):
        code = "```python\nval = 19 * 70 + 7\nassert val == 1337\nprint(f'ANSWER: {val}')\n```"
        return _mk_result(code, tag=spec.tag)

    monkeypatch.setattr("flash_forge.client.TensorFoldClient.complete", _fake_complete)

    s_resp = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 201,
                "method": "tools/call",
                "params": {
                    "name": "cortheon_spec_verify",
                    "arguments": {
                        "prompt": "Compute 19 * 70 + 7 as an integer.",
                        "drafter_url": "http://127.0.0.1:8080/v1",
                        "verifier_url": "http://127.0.0.1:8081/v1",
                        "drafter_streams": 3,
                    },
                },
            }
        )
    )
    assert s_resp is not None
    assert s_resp["result"]["isError"] is False
    s_payload = json.loads(s_resp["result"]["content"][0]["text"])
    assert s_payload["verified"] is True
    assert s_payload["accepted_by_drafter"] is True
    assert s_payload["escalated"] is False
    assert s_payload["verifier_calls"] == 0
    assert s_payload["answer"] == "1337"

    # 3. Runtime failure inside a tool returns MCP isError=True without crashing server
    async def _boom(**kwargs):
        raise RuntimeError("backend unreachable")

    err_server = MCPServer(verify_fn=_boom)
    e_resp = asyncio.run(
        err_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 202,
                "method": "tools/call",
                "params": {"name": "cortheon_verify", "arguments": {"prompt": "test"}},
            }
        )
    )
    assert e_resp is not None
    assert e_resp["result"]["isError"] is True
    assert "backend unreachable" in e_resp["result"]["content"][0]["text"]


def test_mcp_stdio_dual_framing_ndjson_and_content_length():
    # A. Newline-delimited JSON-RPC (NDJSON)
    ndjson_in = io.BytesIO(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}\n'
        b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        b'{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
        b'not-valid-json\n'
    )
    ndjson_out = io.BytesIO()
    run_stdio_server(stdin=ndjson_in, stdout=ndjson_out)

    lines = [ln for ln in ndjson_out.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3  # notification produced no response!
    r1, r2, r3 = (json.loads(ln.decode("utf-8")) for ln in lines)
    assert r1["id"] == 1 and r1["result"]["serverInfo"]["name"] == "cortheon"
    assert r2 == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert r3["error"]["code"] == ERR_PARSE

    # B. Content-Length framed JSON-RPC (including multi-byte UTF-8)
    def _frame(obj: dict[str, Any]) -> bytes:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        return f"Content-Length: {len(body)}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n".encode("ascii") + body

    cl_in = io.BytesIO(
        _frame({"jsonrpc": "2.0", "id": "utf8—✅", "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
        + _frame({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + _frame({"jsonrpc": "2.0", "id": 99, "method": "tools/list"})
    )
    cl_out = io.BytesIO()
    run_stdio_server(stdin=cl_in, stdout=cl_out)

    cl_out.seek(0)
    raw1, mode1 = read_stdio_message(cl_out)
    raw2, mode2 = read_stdio_message(cl_out)
    raw3, mode3 = read_stdio_message(cl_out)

    assert mode1 == "content-length" and raw1 is not None
    assert mode2 == "content-length" and raw2 is not None
    assert mode3 == "eof" and raw3 is None

    msg1 = json.loads(raw1.decode("utf-8"))
    msg2 = json.loads(raw2.decode("utf-8"))
    assert msg1["id"] == "utf8—✅"
    assert msg1["result"]["protocolVersion"] == "2025-03-26"
    assert msg2["id"] == 99
    assert len(msg2["result"]["tools"]) == 4


def test_guard_untracked_file_preexisting_bug_and_lgtm_prose_trap(tmp_path: Path):
    repo = _init_math_repo(tmp_path)

    # Commit a file with a pre-existing bug at HEAD
    (repo / "legacy.py").write_text("def sub(a, b):\n    return a + b  # pre-existing bug\n", encoding="utf-8")
    _git(repo, "add", "legacy.py")
    _git(repo, "commit", "-m", "Add legacy module")

    # Touch legacy.py with a harmless comment AND create an untracked buggy file
    (repo / "legacy.py").write_text("# touched\ndef sub(a, b):\n    return a + b  # pre-existing bug\n", encoding="utf-8")
    (repo / "new_helper.py").write_text("def negate(x):\n    return x  # bug: forgot minus\n", encoding="utf-8")

    # (1) Single-word prose ("LGTM") parses as ast.Name — MUST still be discarded as non-reproducer!
    adv_lgtm = "LGTM"
    # (2) Pre-existing bug reproducer -> fails on both patched AND HEAD -> passed_on_base=False
    adv_preexisting = (
        "```python\n"
        "# TITLE: Pre-existing sub adds instead of subtracting\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from legacy import sub\n"
        "assert sub(5, 2) == 3\n"
        "```"
    )
    # (3) Untracked new file bug -> fails on patched, ModuleNotFoundError on HEAD -> passed_on_base=None
    adv_untracked = (
        "```python\n"
        "# TITLE: New negate fails to negate\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from new_helper import negate\n"
        "assert negate(5) == -5\n"
        "```"
    )

    client = _ScriptedClient([[adv_lgtm, adv_preexisting, adv_untracked]])
    report = asyncio.run(run_guard(client, repo, num_adversaries=3, verify_baseline=True))  # type: ignore[arg-type]

    assert report.rejected_non_failing == 1  # LGTM discarded
    assert report.reproducers_executed == 2
    assert len(report.proven_bugs) == 2
    by_role = {b.role: b for b in report.proven_bugs}
    assert by_role["state_concurrency_breaker"].passed_on_base is False
    assert by_role["regression_contract_breaker"].passed_on_base is None
    assert _count_git_worktrees(repo) == 1


def test_guard_auto_fix_accepts_unified_diff_output(tmp_path: Path):
    repo = _init_math_repo(tmp_path)
    buggy_src = (
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Clamp x into the closed interval [lo, hi]."""\n'
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi - 1\n"
        "    return x\n"
    )
    (repo / "math_utils.py").write_text(buggy_src, encoding="utf-8")

    adv_repro = (
        "```python\n"
        "# TITLE: clamp upper bound broken\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from math_utils import clamp\n"
        "assert clamp(10, 0, 5) == 5\n"
        "```"
    )
    unified_fix = (
        "```diff\n"
        "--- a/math_utils.py\n"
        "+++ b/math_utils.py\n"
        "@@ -4,4 +4,4 @@\n"
        "         return lo\n"
        "     if x > hi:\n"
        "-        return hi - 1\n"
        "+        return hi\n"
        "     return x\n"
        "```"
    )
    client = _ScriptedClient([[adv_repro], [unified_fix, "invalid no patch"]])
    report = asyncio.run(run_guard(client, repo, num_adversaries=1, auto_fix=True))  # type: ignore[arg-type]

    assert report.patch_verified is True
    assert report.suggested_patch is not None
    assert "+        return hi" in report.suggested_patch
    assert (repo / "math_utils.py").read_text(encoding="utf-8") == buggy_src
    assert _count_git_worktrees(repo) == 1


@pytest.mark.skipif(
    not (sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec")),
    reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)",
)
def test_guard_sandbox_prevents_reproducer_from_mutating_user_repo(tmp_path: Path):
    repo = _init_math_repo(tmp_path)
    buggy_src = (
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi - 1\n"
        "    return x\n"
    )
    target = repo / "math_utils.py"
    target.write_text(buggy_src, encoding="utf-8")

    malicious_repro = (
        "```python\n"
        "from pathlib import Path\n"
        f"Path({str(target.resolve())!r}).write_text('PWNED', encoding='utf-8')\n"
        "```"
    )
    client = _ScriptedClient([[malicious_repro]])
    asyncio.run(run_guard(client, repo, num_adversaries=1, verify_baseline=False))  # type: ignore[arg-type]

    # Seatbelt MUST block the write — user's repo file stays 100% intact
    assert target.read_text(encoding="utf-8") == buggy_src
    assert _count_git_worktrees(repo) == 1


def test_mcp_stdio_real_subprocess_handshake():
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.Popen(
        [sys.executable, "-m", "flash_forge.mcp_server", "--url", "http://127.0.0.1:18888/v1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        payload = (
            b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}\n'
            b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
            b'{"jsonrpc":"2.0","id":3,"method":"ping"}\n'
        )
        out_bytes, err_bytes = proc.communicate(input=payload, timeout=10.0)
        assert proc.returncode == 0, f"Server exited {proc.returncode}: {err_bytes.decode('utf-8', 'replace')}"
        lines = [ln for ln in out_bytes.splitlines() if ln.strip()]
        assert len(lines) == 3
        r_init, r_list, r_ping = (json.loads(ln.decode("utf-8")) for ln in lines)
        assert r_init["result"]["serverInfo"]["name"] == "cortheon"
        assert len(r_list["result"]["tools"]) == 4
        assert r_ping["result"] == {}
    finally:
        if proc.poll() is None:
            proc.kill()


def test_mcp_progress_notifications_from_meta_progress_token():
    """Supplying `_meta.progressToken` streams live `notifications/progress` before the final reply."""

    async def _fake_verify(*, prompt: str, mode=None, streams: int = 4, url: str = "", on_event=None):
        assert on_event is not None
        on_event({"engine": "reason", "event": "start", "mode": "reason", "streams": streams})
        on_event({"engine": "reason", "event": "slot_done", "slot": "pal_exact", "status": "ok", "latency_s": 1.2})
        on_event({"engine": "reason", "event": "early_abort", "consensus_count": 2, "cancelled": 2})
        on_event({"engine": "reason", "event": "done", "wall_s": 2.4})
        return {"mode": "reason", "answer": "42", "verified": True}

    server = MCPServer(verify_fn=_fake_verify)

    # 1. With _meta.progressToken -> progress notifications emitted in order before result
    req_with_tok = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "cortheon_verify",
            "arguments": {"prompt": "6 * 7", "streams": 4},
            "_meta": {"progressToken": "tok-verify-10"},
        },
    }
    in_buf = io.BytesIO((json.dumps(req_with_tok) + "\n").encode("utf-8"))
    out_buf = io.BytesIO()
    run_stdio_server(server=server, stdin=in_buf, stdout=out_buf)

    msgs = [json.loads(ln.decode("utf-8")) for ln in out_buf.getvalue().splitlines() if ln.strip()]
    progress_msgs = [m for m in msgs if m.get("method") == "notifications/progress"]
    result_msgs = [m for m in msgs if m.get("id") == 10]
    assert len(result_msgs) == 1
    assert msgs[-1]["id"] == 10  # final response arrives AFTER all progress notifications
    assert len(progress_msgs) >= 4
    steps = [m["params"]["progress"] for m in progress_msgs]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)  # strictly increasing
    assert all(m["params"]["progressToken"] == "tok-verify-10" for m in progress_msgs)
    joined_notes = " | ".join(m["params"].get("message", "") for m in progress_msgs)
    assert "pal_exact" in joined_notes
    assert "Early-abort" in joined_notes

    # 2. Without _meta.progressToken -> zero progress notifications emitted
    async def _fake_verify_plain(*, prompt: str, mode=None, streams: int = 4, url: str = "", on_event=None):
        assert on_event is None
        return {"mode": "reason", "answer": "42", "verified": True}

    plain_server = MCPServer(verify_fn=_fake_verify_plain)
    req_no_tok = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {"name": "cortheon_verify", "arguments": {"prompt": "6 * 7"}},
    }
    in_buf2 = io.BytesIO((json.dumps(req_no_tok) + "\n").encode("utf-8"))
    out_buf2 = io.BytesIO()
    run_stdio_server(server=plain_server, stdin=in_buf2, stdout=out_buf2)
    msgs2 = [json.loads(ln.decode("utf-8")) for ln in out_buf2.getvalue().splitlines() if ln.strip()]
    assert len(msgs2) == 1 and msgs2[0]["id"] == 11


def test_mcp_notifications_cancelled_kills_inflight_sandbox_child():
    """`notifications/cancelled` aborts an in-flight `tools/call`, kills its sandboxed child,
    suppresses any late response for that requestId, and keeps the server alive."""
    from flash_forge.mcp_server import serve_stdio
    from flash_forge.sandbox import run_python

    started = asyncio.Event()
    cancelled_seen = [False]
    sandbox_elapsed = [999.0]

    async def _slow_sandbox_verify(*, prompt: str, mode=None, streams: int = 4, url: str = "", on_event=None):
        started.set()
        t0 = time.perf_counter()
        try:
            # Spawn a real OS-sandboxed Python child that would sleep for 25s if not killed!
            await asyncio.to_thread(run_python, "import time\ntime.sleep(25)\nprint('DONE')", 25.0)
            return {"answer": "too_late"}
        except asyncio.CancelledError:
            cancelled_seen[0] = True
            sandbox_elapsed[0] = time.perf_counter() - t0
            raise

    async def _run_scenario():
        in_r_fd, in_w_fd = os.pipe()
        in_r = os.fdopen(in_r_fd, "rb", buffering=0)
        in_w = os.fdopen(in_w_fd, "wb", buffering=0)
        out_buf = io.BytesIO()

        server = MCPServer(verify_fn=_slow_sandbox_verify)
        srv_task = asyncio.create_task(serve_stdio(server, stdin=in_r, stdout=out_buf))
        try:
            in_w.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 42,
                            "method": "tools/call",
                            "params": {
                                "name": "cortheon_verify",
                                "arguments": {"prompt": "slow verification"},
                                "_meta": {"progressToken": "tok-42"},
                            },
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
            await asyncio.wait_for(started.wait(), timeout=5.0)
            await asyncio.sleep(0.25)  # give sandbox subprocess time to spawn

            t_cancel = time.perf_counter()
            in_w.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/cancelled",
                            "params": {"requestId": 42, "reason": "client timeout"},
                        }
                    )
                    + "\n"
                    + json.dumps({"jsonrpc": "2.0", "id": 43, "method": "ping"})
                    + "\n"
                ).encode("utf-8")
            )
            in_w.close()
            await asyncio.wait_for(srv_task, timeout=5.0)
            cancel_dt = time.perf_counter() - t_cancel
            return out_buf.getvalue(), cancel_dt
        finally:
            if not in_w.closed:
                in_w.close()
            if not srv_task.done():
                srv_task.cancel()
                await asyncio.gather(srv_task, return_exceptions=True)
            in_r.close()

    raw_out, cancel_dt = asyncio.run(_run_scenario())
    assert cancelled_seen[0] is True
    assert cancel_dt < 4.0, f"Cancellation took too long ({cancel_dt:.2f}s) — sandbox child may not have been killed!"
    msgs = [json.loads(ln.decode("utf-8")) for ln in raw_out.splitlines() if ln.strip()]
    # Cancelled request 42 MUST NOT emit a response; ping 43 MUST still succeed!
    assert not any(m.get("id") == 42 for m in msgs)
    assert any(m == {"jsonrpc": "2.0", "id": 43, "result": {}} for m in msgs)


def test_mcp_closed_stdout_pipe_cancels_inflight_call():
    """If the client disconnects and closes stdout while a tool call runs, stop burning compute."""
    from flash_forge.mcp_server import serve_stdio

    started = asyncio.Event()
    cancelled_seen = [False]

    async def _long_verify(*, prompt: str, mode=None, streams: int = 4, url: str = "", on_event=None):
        started.set()
        try:
            await asyncio.sleep(20.0)
            return {"answer": "should_not_finish"}
        except asyncio.CancelledError:
            cancelled_seen[0] = True
            raise

    async def _run_scenario():
        in_r_fd, in_w_fd = os.pipe()
        out_r_fd, out_w_fd = os.pipe()
        in_r = os.fdopen(in_r_fd, "rb", buffering=0)
        in_w = os.fdopen(in_w_fd, "wb", buffering=0)
        out_r = os.fdopen(out_r_fd, "rb", buffering=0)
        out_w = os.fdopen(out_w_fd, "wb", buffering=0)

        server = MCPServer(verify_fn=_long_verify)
        srv_task = asyncio.create_task(serve_stdio(server, stdin=in_r, stdout=out_w))
        try:
            in_w.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 77,
                            "method": "tools/call",
                            "params": {"name": "cortheon_verify", "arguments": {"prompt": "long"}},
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
            await asyncio.wait_for(started.wait(), timeout=5.0)
            # Simulate client abruptly closing both stdout reader and stdin writer
            out_r.close()
            in_w.close()
            await asyncio.wait_for(srv_task, timeout=4.0)
        finally:
            for f in (in_w, out_r, out_w, in_r):
                try:
                    if not f.closed:
                        f.close()
                except OSError:
                    pass
            if not srv_task.done():
                srv_task.cancel()
                await asyncio.gather(srv_task, return_exceptions=True)

    asyncio.run(_run_scenario())
    assert cancelled_seen[0] is True

