"""Unit tests + planted controls for FlashForge components."""

from __future__ import annotations

import subprocess
from pathlib import Path

from flash_forge.brain_state import BrainState
from flash_forge.code_forge import _extract_test_functions, _run_modular_suite
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import _answers_equivalent, extract_answer_from_stdout, normalize_answer
from flash_forge.referee import verify_citations_on_disk
from flash_forge.repo_index import apply_patch_ops, build_repo_map, parse_patch_ops
from flash_forge.sandbox import check_syntax, extract_python_code, run_in_ephemeral_worktree, run_python
from flash_forge.skill_loader import SkillPack
from flash_forge.tools import AgentToolbelt


def test_skill_pack_cards_exist_and_stay_compact():
    sp = SkillPack()
    required = [
        "contract", "oracle", "oracle_fuzzer", "race_minimal", "race_first_principles", "race_skeptic",
        "race_alt", "redteam_edge", "redteam_diff", "repair_surgical", "repair_rewrite",
        "pal_exact", "pal_alt_z3", "pal_skeptic", "reason_analytical",
        "falsifier_bruteforce", "falsifier_reread", "falsifier_z3_sympy",
    ]
    for name in required:
        text = sp.card(name)
        assert len(text) > 40
        # Local-first budget: every micro-card must stay short (<1100 chars ~ <250 tokens)
        assert len(text) < 1100, f"Card {name} too verbose ({len(text)} chars)"


def test_answer_normalization_and_equivalence():
    assert normalize_answer(r"\frac{14}{24}") == "7/12"
    assert normalize_answer("Fraction(7, 12)") == "7/12"
    assert normalize_answer("1,234,567") == "1234567"
    assert normalize_answer("42.0") == "42"
    assert _answers_equivalent("7/12", "14/24")
    assert _answers_equivalent("0.5833333333333334", "7/12")
    # Planted control: inequivalent answers must fail
    assert not _answers_equivalent("7/12", "8/12")


def test_sandbox_exec_and_timeout_control():
    res = run_python("from fractions import Fraction\nimport z3, sympy\nprint('ANSWER:', Fraction(2, 6))")
    assert res.ok
    assert extract_answer_from_stdout(res.stdout) == "1/3"

    # Planted control: infinite loop must time out cleanly
    t_res = run_python("while True: pass", timeout_s=0.4)
    assert not t_res.ok and t_res.timed_out


def test_modular_suite_and_planted_bug_detection():
    good_code = "def add(a, b):\n    return a + b\n"
    buggy_code = "def add(a, b):\n    return a + b + (1 if a == 0 else 0)\n"
    suite = (
        "def test_pos():\n    assert add(2, 3) == 5\n\n"
        "def test_zero():\n    assert add(0, 5) == 5\n"
    )
    assert set(_extract_test_functions(suite).keys()) == {"test_pos", "test_zero"}

    good_rows, _, err = _run_modular_suite(good_code, suite)
    assert err is None and all(r["ok"] for r in good_rows) and len(good_rows) == 2

    # Planted control: buggy_code passes test_pos and fails test_zero cleanly
    bad_rows, _, err2 = _run_modular_suite(buggy_code, suite)
    assert err2 is None
    by_name = {r["name"]: r["ok"] for r in bad_rows}
    assert by_name == {"test_pos": True, "test_zero": False}


def test_episodic_memory_fts5_roundtrip(tmp_path: Path):
    mem = EpisodicMemory(db_path=tmp_path / "test_lessons.db")
    lid = mem.add(
        domain="code",
        symptom="floating point equality drift in interval sweep line",
        dead_end="comparing float slopes directly",
        verified_rule="Use exact integer cross-products or fractions.Fraction for geometry comparisons.",
    )
    assert lid > 0
    hits = mem.retrieve("Write an interval sweep line geometry solver with exact slopes", domain="code")
    assert len(hits) == 1
    assert "cross-products" in hits[0].verified_rule
    mem.close()


def test_unclosed_fence_test_salvage():
    truncated = (
        "```python\n"
        "def test_ok1():\n"
        "    assert 1 + 1 == 2\n\n"
        "def test_ok2():\n"
        "    assert 2 * 3 == 6\n\n"
        "def test_cut_off():\n"
        "    assert (1 + 2"
    )
    salvaged = extract_python_code(truncated, salvage_truncated_tests=True)
    assert check_syntax(salvaged) is None
    assert set(_extract_test_functions(salvaged).keys()) == {"test_ok1", "test_ok2"}


def test_repo_index_and_multifile_patch_ops(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def foo(x):\n    return x + 1\n")
    (tmp_path / "pkg" / "b.py").write_text("from pkg.a import foo\ndef bar(x):\n    return foo(x) * 2\n")

    rmap = build_repo_map(tmp_path)
    assert "pkg/a.py" in rmap and "def foo(x)" in rmap and "def bar(x)" in rmap

    patch_reply = (
        "<<<EDIT: pkg/a.py>>>\n"
        "<<<SEARCH>>>\n"
        "def foo(x):\n"
        "    return x + 1\n"
        "<<<REPLACE>>>\n"
        "def foo(x):\n"
        "    return x + 10\n"
        "<<<END_EDIT>>>\n\n"
        "<<<FILE: pkg/c.py>>>\n"
        "def baz():\n"
        "    return 42\n"
        "<<<END_FILE>>>\n"
    )
    ops = parse_patch_ops(patch_reply)
    assert len(ops) == 2
    ok, err, touched = apply_patch_ops(tmp_path, ops)
    assert ok and not err and set(touched) == {"pkg/a.py", "pkg/c.py"}
    assert "x + 10" in (tmp_path / "pkg" / "a.py").read_text()
    assert "42" in (tmp_path / "pkg" / "c.py").read_text()

    # Planted control: broken syntax patch must be rejected
    bad_ops = parse_patch_ops("<<<FILE: pkg/c.py>>>\ndef broken(:\n<<<END_FILE>>>")
    ok2, err2, _ = apply_patch_ops(tmp_path, bad_ops)
    assert not ok2 and "SyntaxError" in err2


def test_brain_state_stuck_detector_and_referee_citations(tmp_path: Path):
    st = BrainState(goal="Test mission", workspace=str(tmp_path))
    st.turn = 1
    st.record_observation("read_file", "a.py:1-50", True, "content")
    assert not st.is_stuck()[0]
    st.turn = 2
    st.record_observation("read_file", "a.py:1-50", True, "content")
    assert st.is_stuck()[0]  # repeated identical action detected!

    # Referee citation check + planted control for hallucinated path/line!
    (tmp_path / "src.py").write_text("line1\nline2\nline3\n")
    n, errs = verify_citations_on_disk(tmp_path, "Verified in `src.py:2` and `src.py:1-3`.")
    assert n == 2 and not errs
    _, bad_errs = verify_citations_on_disk(tmp_path, "See `src.py:999` and `fake/file.zig:10`.")
    assert len(bad_errs) == 2


def test_agent_toolbelt_containment_and_syntax_guard(tmp_path: Path):
    tb = AgentToolbelt(tmp_path, read_only=False)
    assert tb.write_file("mod.py", "def sq(x):\n    return x * x\n").ok
    # Planted control: broken syntax refused before writing!
    assert not tb.write_file("bad.py", "def broken(:\n").ok
    assert not (tmp_path / "bad.py").exists()
    # Planted control: path escape refused!
    assert not tb.read_file("../../etc/passwd").ok
    # Surgical edit + python_probe
    assert tb.edit_file("mod.py", "return x * x", "return x * x + 1").ok
    probe = tb.python_probe("import mod\nassert mod.sq(3) == 10\nprint('OK')")
    assert probe.ok and "OK" in probe.output
    # Planted control: broken C/NEON syntax refused by clang -fsyntax-only before saving!
    assert tb.write_file("good.c", "int add(int a, int b) { return a + b; }\n").ok
    assert not tb.write_file("bad.c", "int bad(void) { int x = ; return x; }\n").ok
    assert not (tmp_path / "bad.c").exists()
    # Read-only mode blocks writes!
    ro = AgentToolbelt(tmp_path, read_only=True)
    assert not ro.write_file("x.py", "x = 1\n").ok


def test_ephemeral_git_worktree_isolation(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "calc.py").write_text("def double(x):\n    return x\n")
    subprocess.run(["git", "add", "calc.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    # Planted control: broken patch fails verify, original repo stays untouched
    bad = run_in_ephemeral_worktree(repo, {"calc.py": "def double(x):\n    return x * 3\n"}, "python3 -c 'import calc; assert calc.double(4) == 8'")
    assert not bad.ok
    good = run_in_ephemeral_worktree(repo, {"calc.py": "def double(x):\n    return x * 2\n"}, "python3 -c 'import calc; assert calc.double(4) == 8'")
    assert good.ok
    # Original repo must still have original buggy content (strict isolation!)
    assert "return x\n" in (repo / "calc.py").read_text()


def test_fatal_candidate_preserved_and_partial_timeout_salvaged():
    from flash_forge.code_forge import CodeCandidate, _apply_matrix_scores, _parse_arbiter_verdict

    # 1. Partial timeout salvage: test_fast passes before test_hang times out!
    code = "def f(x):\n    return x * 2\n"
    suite = (
        "import time\n"
        "def test_fast():\n    assert f(3) == 6\n\n"
        "def test_hang():\n    time.sleep(5)\n"
    )
    rows, _, fatal = _run_modular_suite(code, suite, timeout_s=0.6)
    assert fatal and "TIMEOUT" in fatal
    assert len(rows) == 1 and rows[0]["name"] == "test_fast" and rows[0]["ok"] is True

    # 2. Fatal candidate not erased by _apply_matrix_scores when absent from matrix
    c_fatal = CodeCandidate(
        name="crasher", wave=1, code="", raw_content="",
        fatal_error="TIMEOUT (8.0s)", tests_total=5, tests_passed=0, failed_tests=["__fatal__"],
    )
    _apply_matrix_scores([c_fatal], {}, set(), is_redteam=False)
    assert c_fatal.failed_tests == ["__fatal__"] and c_fatal.tests_total == 5 and not c_fatal.perfect

    # 3. Arbiter strict verdict parsing (INVALID must never parse as VALID!)
    assert _parse_arbiter_verdict("Looks wrong.\nVERDICT: INVALID") == "INVALID"
    assert _parse_arbiter_verdict("Matches spec.\nVERDICT: VALID") == "VALID"
    assert _parse_arbiter_verdict("Not sure about this") == "UNKNOWN"


def test_markdown_answer_extraction_ast_assert_and_decimal_guard():
    from flash_forge.reason_forge import _has_executable_assert, extract_answer_from_prose

    assert extract_answer_from_prose("Reasoning...\n**ANSWER: 20**") == "20"
    assert extract_answer_from_prose("Done.\n`ANSWER: 48`") == "48"
    assert extract_answer_from_prose("Therefore\nANSWER 27393") == "27393"
    assert extract_answer_from_stdout("working...\ndone\n") == ""
    assert extract_answer_from_stdout("solution:\n") == ""
    # Decimal precision guard: 10-digit truncated 0.3333333333 must NOT falsely snap to 1/3
    assert normalize_answer("0.3333333333") != "1/3"
    assert not _answers_equivalent("0.5", "0.5000000001")
    # AST assert check: comments & `assert True` do not count
    assert not _has_executable_assert("# assert x == 1\nprint('VERIFIED')\nassert True\n")
    assert _has_executable_assert("x = 42\nassert x == 42\n")


def test_sandbox_env_scrubbing_and_router_greeting_guard(monkeypatch):
    from flash_forge.router import _GREETING_ONLY_RE, _head_tail_excerpt

    monkeypatch.setenv("FF_SECRET_TOKEN_TEST", "super_secret_123")
    res = run_python("import os; print(os.environ.get('FF_SECRET_TOKEN_TEST', 'SCRUBBED'))")
    assert res.ok and res.stdout.strip() == "SCRUBBED"

    # Greeting regex must NOT match math prompts starting with "Hi,"
    assert _GREETING_ONLY_RE.fullmatch("Hi!") is not None
    assert _GREETING_ONLY_RE.fullmatch("Hi, calculate 17*23") is None
    # Head-tail excerpt preserves trailing ANSWER constraint
    long_txt = "A" * 2500 + "\n" + "B" * 2500 + "\nEnd with ANSWER: <int>"
    ex = _head_tail_excerpt(long_txt, max_chars=500)
    assert "End with ANSWER: <int>" in ex


def test_readonly_shell_lockdown_prefix_collision_and_test_cmd_spoof(tmp_path: Path):
    from flash_forge.tools import is_real_test_command

    ws = tmp_path / "repo"
    sibling = tmp_path / "repo_secret"
    ws.mkdir()
    sibling.mkdir()
    (ws / "ok.py").write_text("x = 1\n")
    (sibling / "creds.py").write_text("SECRET = 1\n")

    ro = AgentToolbelt(ws, read_only=True)
    # 1. Prefix-collision sibling directory escape MUST be blocked!
    assert not ro.read_file(str(sibling / "creds.py")).ok
    assert not ro.read_file("../repo_secret/creds.py").ok
    # 2. Shell chaining & arbitrary python execution MUST be blocked in read-only mode!
    assert not ro.run_command("ls; touch pwned.txt").ok
    assert not (ws / "pwned.txt").exists()
    assert not ro.run_command("python3 -c 'print(1)'").ok
    assert not ro.run_command("find . -exec echo {} +").ok
    assert not ro.run_command("git commit -m hack").ok
    assert not ro.python_probe("print('escape')").ok
    # Legitimate read-only argv command works cleanly
    assert ro.run_command("ls").ok

    # 3. Referee citation check MUST block prefix-collision paths!
    _, errs = verify_citations_on_disk(ws, "See `../repo_secret/creds.py:1`")
    assert len(errs) == 1

    # 4. Test command detector MUST reject spoofed commands (`echo pytest`, `cat pytest.ini`)!
    assert not is_real_test_command("echo pytest")
    assert not is_real_test_command("cat pytest.ini")
    assert not is_real_test_command("grep cargo test README.md")
    assert is_real_test_command("pytest -q")
    assert is_real_test_command("python3 -m pytest tests/")
    assert is_real_test_command("cc -O3 kernel.c bench.c -o bench_runner && ./bench_runner")
    assert is_real_test_command("zig build test")


def test_modular_suite_handles_future_imports_and_blocks_symbol_shadowing():
    # Monster Bug #1 + #2 regression tests (exposed by G27 & G40 in Gauntlet-50!)
    good_solution = "class Widget:\n    def val(self) -> int:\n        return 42\n"
    buggy_solution = "class Widget:\n    def val(self) -> int:\n        return -1\n"

    poisoned_test_suite = (
        "from __future__ import annotations\n"
        "from solution import Widget\n\n"
        "# Generated test accidentally pastes a broken copy of Widget (shadowing solution!)\n"
        "class Widget:\n"
        "    def val(self) -> int:\n"
        "        raise RuntimeError('SHADOWED COPY CALLED!')\n\n"
        "def test_uses_real_solution():\n"
        "    assert Widget().val() == 42\n"
    )
    rows_ok, _, fatal_ok = _run_modular_suite(good_solution, poisoned_test_suite)
    assert fatal_ok is None
    assert len(rows_ok) == 1 and rows_ok[0]["ok"] is True

    # Planted control: buggy solution MUST still fail (proving solution.Widget is truly tested!)
    rows_bad, _, fatal_bad = _run_modular_suite(buggy_solution, poisoned_test_suite)
    assert fatal_bad is None
    assert len(rows_bad) == 1 and rows_bad[0]["ok"] is False


def test_compile_time_syntax_guard_and_compact_traceback_headline():
    # Compile-time SyntaxError that ast.parse misses (`nonlocal` without outer binding!)
    bad_nonlocal = "def f():\n    nonlocal ghost\n    ghost = 1\n"
    err = check_syntax(bad_nonlocal)
    assert err is not None and "SyntaxError" in err

    # Compact traceback puts exact exception headline FIRST and strips temp dir paths!
    sol = "def boom():\n    raise ValueError('exact_sentinel_msg_99')\n"
    suite = "def test_boom():\n    boom()\n"
    rows, _, fatal = _run_modular_suite(sol, suite)
    assert fatal is None and len(rows) == 1 and not rows[0]["ok"]
    assert rows[0]["err"].startswith("ValueError: exact_sentinel_msg_99")
    assert "/ff_exec_" not in rows[0]["err"]


def test_helper_aware_test_context_and_strict_b2_consensus():
    import asyncio
    from dataclasses import dataclass
    from flash_forge.code_forge import _arbitrate_tests_strict, _build_test_context, _extract_module_helpers

    suite = (
        "import hashlib\n\n"
        "def _brute_ref(x):\n"
        "    return x + 999  # subtle helper bug\n\n"
        "def test_ok():\n"
        "    assert 1 == 1\n\n"
        "def test_calls_helper():\n"
        "    assert foo(1) == _brute_ref(1)\n"
    )
    helpers = _extract_module_helpers(suite)
    assert "def _brute_ref(x):" in helpers and "def test_ok" not in helpers
    ctx = _build_test_context(suite, "test_calls_helper")
    assert "def _brute_ref(x):" in ctx and "def test_calls_helper():" in ctx

    @dataclass
    class _FakeRes:
        ok: bool
        content: str

    class _FakeClient:
        def __init__(self, replies):
            self.replies = replies

        async def batch(self, specs):
            return [_FakeRes(True, r) for r in self.replies]

    # 2/2 VALID -> VALID; 1 VALID + 1 INVALID -> INVALID; 1 VALID + 1 UNKNOWN -> DISPUTED
    fake = _FakeClient([
        "ok\nVERDICT: VALID", "ok\nVERDICT: VALID",
        "ok\nVERDICT: VALID", "bug\nVERDICT: INVALID",
        "ok\nVERDICT: VALID", "unsure",
    ])
    verdicts = asyncio.run(
        _arbitrate_tests_strict(
            fake,
            "spec",
            [("t_both_valid", "c1", "e1"), ("t_split_invalid", "c2", "e2"), ("t_disputed", "c3", "e3")],
        )
    )
    assert verdicts == {"t_both_valid": "VALID", "t_split_invalid": "INVALID", "t_disputed": "DISPUTED"}


def test_cortheon_public_sdk_exports():
    import cortheon

    assert hasattr(cortheon, "Cortheon")
    assert hasattr(cortheon, "CortheonConfig")
    assert hasattr(cortheon, "CortheonResult")
    assert hasattr(cortheon, "ReplicaAgent")
    assert hasattr(cortheon, "create_app")


def test_client_soft_effort_mapping_and_nongrammar_choice():
    from flash_forge.client import CallSpec, ModelCapabilities, TensorFoldClient

    c = TensorFoldClient(base_url="http://127.0.0.1:9999/v1")
    c._is_tensorfold = True
    c._caps = ModelCapabilities(
        model_id="deepseek-v4.1-flash-1017",
        context_length=65536,
        structures=False,
        thinking_budget=False,
        thinking_switch=False,
        supported_efforts=["max", "high", "medium", "low", "minimal", "none"],
    )
    p_off = c._build_payload(CallSpec(messages=[{"role": "user", "content": "hi"}], enable_thinking=False, max_tokens=64))
    assert p_off["reasoning_effort"] == "none"
    assert p_off["chat_template_kwargs"] == {"enable_thinking": False}

    p_min = c._build_payload(CallSpec(messages=[{"role": "user", "content": "x"}], enable_thinking=True, thinking_budget=450, max_tokens=1500))
    assert p_min["reasoning_effort"] == "minimal"
    assert p_min["max_tokens"] >= 6144

    p_low = c._build_payload(CallSpec(messages=[{"role": "user", "content": "x"}], enable_thinking=True, thinking_budget=900, max_tokens=2000))
    assert p_low["reasoning_effort"] == "low"


def test_reason_format_gate_and_trailing_prose_strip():
    from flash_forge.reason_forge import _answer_satisfies_prompt_format

    assert normalize_answer("19. AIME answer 019 maybe final 19") == "19"
    p_int = "Find N. Every AIME answer is an integer from 000 to 999. End with `ANSWER: <integer>`."
    assert _answer_satisfies_prompt_format(p_int, "248") is True
    assert _answer_satisfies_prompt_format(p_int, "7*sqrt(7) + 13*sqrt(3) + 40") is False
    assert _answer_satisfies_prompt_format("Compute the exact rational probability.", "7/12") is True


def test_stdin_mode_cli_runner_and_planted_control():
    from flash_forge.code_forge import _is_stdin_problem

    prob = "Read N from standard input and print 2*N to standard output.\nSample Input 1:\n3\nSample Output 1:\n6\n"
    assert _is_stdin_problem(prob) is True
    assert _is_stdin_problem("Implement `def foo(x):` returning 2*x.") is False

    good_cli = "import sys\nn = int(sys.stdin.read().strip())\nprint(n * 2)\n"
    bad_cli = "import sys\nn = int(sys.stdin.read().strip())\nprint(n * 3)\n"
    suite = "def test_samples():\n    assert run_solution('3\\n') == '6'\n    assert run_solution('10\\n') == '20'\n"
    rows_ok, _, fat_ok = _run_modular_suite(good_cli, suite, stdin_mode=True)
    assert fat_ok is None and len(rows_ok) == 1 and rows_ok[0]["ok"] is True
    rows_bad, _, fat_bad = _run_modular_suite(bad_cli, suite, stdin_mode=True)
    assert fat_bad is None and len(rows_bad) == 1 and rows_bad[0]["ok"] is False

    from flash_forge.code_forge import _build_deterministic_sample_tests, _extract_prompt_io_samples

    atcoder_text = "### Sample Input 1\n```\n3\n```\n### Sample Output 1\n```\n6\n```\n### Sample Input 2\n```\n10\n```\n### Sample Output 2\n```\n20\n```\n"
    pairs = _extract_prompt_io_samples(atcoder_text)
    assert pairs == [("3", "6"), ("10", "20")]
    scode, snames = _build_deterministic_sample_tests(pairs)
    assert snames == {"test_prompt_sample_1", "test_prompt_sample_2"}
    s_rows, _, s_fat = _run_modular_suite(good_cli, scode, stdin_mode=True)
    assert s_fat is None and len(s_rows) == 2 and all(r["ok"] for r in s_rows)


def test_behavioral_differential_consensus_and_counterexample_adjudication():
    import asyncio
    from dataclasses import dataclass
    from flash_forge.code_forge import (
        CodeCandidate,
        _adjudicate_divergences,
        _analyze_behavioral_consensus,
        _rank_key,
        _run_modular_suite_with_diff,
    )

    # Simulate BigCodeBench_906: 2 candidates return relative path (correct), 1 returns abspath (buggy outlier!)
    rel_code = "import os\ndef make_path(d, f):\n    return os.path.join(d, f)\n"
    abs_code = "import os\ndef make_path(d, f):\n    return os.path.abspath(os.path.join(d, f))\n"
    suite = (
        "import tempfile, os\n"
        "def test_basic():\n"
        "    assert make_path('a', 'b.zip').endswith('b.zip')\n\n"
        "def diff_probe():\n"
        "    out = {}\n"
        "    out['rel'] = make_path('target', 'archive.zip')\n"
        "    with tempfile.TemporaryDirectory() as td:\n"
        "        out['tmp'] = make_path(td, 'x.zip')\n"
        "    return out\n"
    )
    r1, d1, _, f1 = _run_modular_suite_with_diff(rel_code, suite)
    r2, d2, _, f2 = _run_modular_suite_with_diff(rel_code + "# v2\n", suite)
    r3, d3, _, f3 = _run_modular_suite_with_diff(abs_code, suite)
    assert f1 is None and f2 is None and f3 is None
    assert "rel" in d1 and "tmp" in d1

    c1 = CodeCandidate(name="simple", wave=1, code=rel_code, raw_content="", tests_passed=1, tests_total=1, diff_outputs=d1)
    c2 = CodeCandidate(name="first", wave=1, code=rel_code + "# v2\n", raw_content="", tests_passed=1, tests_total=1, diff_outputs=d2)
    c3 = CodeCandidate(name="skeptic", wave=1, code=abs_code, raw_content="", tests_passed=1, tests_total=1, diff_outputs=d3)

    n_cases, divs = _analyze_behavioral_consensus([c1, c2, c3])
    assert n_cases == 2
    # Tempdir prefix normalized so 'tmp' AGREES across all 3, while 'rel' cleanly DIVERGES!
    assert [d[0] for d in divs] == ["rel"]
    assert c1.diff_majority_hits == 2 and c2.diff_majority_hits == 2 and c3.diff_majority_hits == 1
    assert _rank_key(c1) > _rank_key(c3)

    @dataclass
    class _FakeRes:
        ok: bool
        content: str

    class _FakeClient:
        async def batch(self, specs):
            return [_FakeRes(True, "Spec wants relative path.\nVERDICT: A") for _ in specs]

    notes = asyncio.run(_adjudicate_divergences(_FakeClient(), "Return os.path.join(d, f).", "def diff_probe(): ...", divs, [c1, c2, c3]))
    assert len(notes) == 1
    assert c1.diff_counterexample_wins == 1 and c3.diff_counterexample_losses == 1
    assert _rank_key(c1) > _rank_key(c3)


def test_starter_import_preservation_guard():
    from flash_forge.code_forge import (
        CodeCandidate,
        _analyze_behavioral_consensus,
        _ensure_starter_imports,
        _extract_required_exceptions,
        _extract_starter_imports,
        _extract_starter_symbols,
        _rank_key,
        _starter_api_coverage,
    )
    from flash_forge.think_forge import _should_run_executable_verifier

    prompt = "Complete:\n```python\nimport zipfile\nimport os\nimport shutil\ndef task_func(a, b):\n    pass\n```"
    imps = _extract_starter_imports(prompt)
    assert imps == ["import zipfile", "import os", "import shutil"]
    cand = "from __future__ import annotations\nimport os\ndef task_func(a, b):\n    return a\n"
    fixed = _ensure_starter_imports(cand, imps)
    assert fixed.splitlines()[0] == "from __future__ import annotations"
    assert "import shutil" in fixed and "import zipfile" in fixed
    assert fixed.count("import os") == 1

    # Unfenced starter snippet + AST API coverage scoring
    raw_starter = "import glob\nimport shutil\nfrom pathlib import Path\ndef f(x):\n    '''\n    Raises:\n        FileNotFoundError: missing\n        ValueError: bad\n    '''\n"
    raw_imps = _extract_starter_imports(raw_starter)
    syms = _extract_starter_symbols(raw_imps)
    assert syms == {"glob", "shutil", "Path"}
    assert _extract_required_exceptions(raw_starter) == {"FileNotFoundError", "ValueError"}

    full_api = "import glob, shutil\nfrom pathlib import Path\ndef f(x):\n    return [shutil.move(p, Path(x)) for p in glob.glob(x)]\n"
    partial_api = "import glob, shutil\nfrom pathlib import Path\ndef f(x):\n    return list(Path(x).iterdir())\n"
    assert _starter_api_coverage(full_api, syms) == 1.0
    assert _starter_api_coverage(partial_api, syms) < 0.5

    # Crasher filtering in behavioral consensus: 0-pass NameError crasher must NOT create spurious divergences!
    p1 = CodeCandidate(name="a", wave=1, code="x=1", raw_content="", tests_passed=4, tests_total=4, diff_outputs={"k1": "1", "k2": "2"})
    p2 = CodeCandidate(name="b", wave=1, code="x=1 # longer", raw_content="", tests_passed=4, tests_total=4, diff_outputs={"k1": "1", "k2": "2"})
    crash = CodeCandidate(name="c", wave=1, code="bad", raw_content="", tests_passed=0, tests_total=4, diff_outputs={"k1": "'EXC:NameError'", "k2": "'EXC:NameError'"})
    n_cases, divs = _analyze_behavioral_consensus([p1, p2, crash])
    assert n_cases == 2 and divs == []
    assert p1.diff_majority_hits == 2 and p2.diff_majority_hits == 2 and crash.diff_majority_hits == 0

    # Higher behavioral majority beats shorter buggy minority; starter_api_score breaks behavioral ties
    short_bug = CodeCandidate(name="short", wave=1, code="a=1", raw_content="", tests_passed=5, tests_total=5, diff_majority_hits=8, starter_api_score=1.0)
    long_maj = CodeCandidate(name="maj", wave=1, code="a=1\n" * 10, raw_content="", tests_passed=5, tests_total=5, diff_majority_hits=9, starter_api_score=0.5)
    assert _rank_key(long_maj) > _rank_key(short_bug)
    tie_high_api = CodeCandidate(name="hi_api", wave=1, code="a=1\n" * 8, raw_content="", tests_passed=5, tests_total=5, diff_majority_hits=9, starter_api_score=1.0)
    assert _rank_key(tie_high_api) > _rank_key(long_maj)

    # ThinkForge universal executable verifier triggers on both sizing & protocol/security prompts, not haikus
    assert _should_run_executable_verifier("Design a 50k TPS payment ledger with 35ms RTT and <40ms p99.") is True
    assert _should_run_executable_verifier("Audit our JWT refresh-token rotation protocol for replay attacks and race conditions.") is True
    assert _should_run_executable_verifier("Write a 3-line haiku about autumn rain.") is False


def test_reason_wave1_early_abort_cancels_stragglers(tmp_path: Path):
    import asyncio
    from flash_forge.client import CompletionResult
    from flash_forge.memory import EpisodicMemory
    from flash_forge.reason_forge import solve_reasoning
    from flash_forge.skill_loader import SkillPack

    cancelled_tags: list[str] = []
    seen_efforts: dict[str, str | None] = {}

    class _FakeClient:
        async def complete(self, spec):
            seen_efforts[spec.tag] = spec.reasoning_effort
            if spec.tag in ("w1_pal_exact", "w1_pal_alt_z3"):
                await asyncio.sleep(0.02)
                return CompletionResult(
                    content="```python\nassert 2 + 2 == 4\nprint('ANSWER: 208')\n```",
                    reasoning="",
                    finish_reason="stop",
                    prompt_tokens=100,
                    completion_tokens=40,
                    reasoning_tokens=10,
                    cached_tokens=0,
                    latency_s=0.02,
                    tag=spec.tag,
                )
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancelled_tags.append(spec.tag)
                raise
            return CompletionResult(
                content="ANSWER: 208",
                reasoning="",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=40,
                reasoning_tokens=10,
                cached_tokens=0,
                latency_s=10.0,
                tag=spec.tag,
            )

    mem = EpisodicMemory(db_path=tmp_path / "mem.db")
    skills = SkillPack()
    trace = asyncio.run(
        solve_reasoning(
            _FakeClient(),  # type: ignore[arg-type]
            skills,
            mem,
            "Find the number of positive integers <= 1000 divisible by 7 or 11 but not both.",
        )
    )
    assert trace.final_answer == "208"
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert set(cancelled_tags) == {"w1_pal_skeptic", "w1_analytical"}
    assert "w1_pal_exact" in seen_efforts and "w1_pal_alt_z3" in seen_efforts


def test_proxy_tool_turn_passthrough(monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from flash_forge.client import ModelCapabilities
    from flash_forge.proxy import create_app

    app = create_app(upstream_url="http://127.0.0.1:9999/v1")
    forwarded: list[dict] = []

    async def _fake_probe(self):
        return ModelCapabilities(model_id="glm-5.3-flash-tfq3")

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\\"path\\":\\"README.md\\"}"}}]},"finish_reason":"tool_calls"}]}'

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def post(self, url, json=None):
            forwarded.append({"url": url, "json": json})
            return _FakeResp()

    monkeypatch.setattr("flash_forge.client.TensorFoldClient.probe", _fake_probe)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    tc = TestClient(app)
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "cortheon",
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
            "messages": [{"role": "user", "content": "Read README.md"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert forwarded[0]["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert forwarded[0]["json"]["model"] == "glm-5.3-flash-tfq3"


def test_install_commands_all_harnesses(tmp_path: Path):
    from flash_forge.integrations import ALL_HARNESSES, install_commands

    home = tmp_path / "home"
    proj = tmp_path / "repo"
    home.mkdir()
    proj.mkdir()

    global_paths = install_commands(ALL_HARNESSES, home_dir=home, omp_profile="cortheon")
    assert len(global_paths) == len(ALL_HARNESSES)
    assert (home / ".omp/profiles/cortheon/agent/extensions/cortheon.ts").is_file()
    assert (home / ".pi/agent/extensions/cortheon.ts").is_file()
    assert (home / ".config/opencode/command/cortheon.md").is_file()
    assert (home / ".claude/commands/cortheon.md").is_file()
    assert (home / ".cursor/commands/cortheon.md").is_file()
    assert (home / ".agents/skills/cortheon/SKILL.md").is_file()

    # Project-local install works across all harnesses too
    proj_paths = install_commands(ALL_HARNESSES, home_dir=home, project_dir=proj)
    assert len(proj_paths) == len(ALL_HARNESSES)
    assert (proj / ".omp/extensions/cortheon.ts").is_file()
    assert (proj / ".pi/extensions/cortheon.ts").is_file()
    assert (proj / ".opencode/command/cortheon.md").is_file()
    assert (proj / ".claude/commands/cortheon.md").is_file()
    assert (proj / ".cursor/commands/cortheon.md").is_file()
    assert (proj / ".agents/skills/cortheon/SKILL.md").is_file()















