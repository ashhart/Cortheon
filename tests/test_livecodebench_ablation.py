"""Offline unit tests for LiveCodeBench v6 Hard 3-arm compute-matched runner.

Covers:
  - `best4_sample` selection hierarchy (public-pass count -> majority stdout tuple
    agreement among valid-syntax runs -> lowest temperature / earliest slot),
  - strict zero-leakage guarantee (never peeking at `private_cases`),
  - complete token & call logging across `vanilla`, `best4_sample`, and `forge`,
  - `--merge` across sharded problem ranges, disjoint arms, legacy `cortheon` files,
    and single-arm shard merges (preserving `bench.stats` compatibility),
  - exact Wilson 95% CIs and exact two-sided McNemar p-values,
  - path/URL/PII scrubbing, atomic checkpointing, CLI sharding, and `--resume`.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

import bench.run_livecodebench as lcb
from bench.run_livecodebench import (
    ALL_ARMS,
    BEST4_CONFIGS,
    build_lcb_payload,
    format_lcb_table,
    load_lcb_dataset,
    main as lcb_main,
    merge_lcb_files,
    run_one_best4_sample,
    run_one_forge,
    run_one_vanilla,
    scrub_detail_paths,
    select_best4_candidate,
)
from bench.stats import analyze_directory, analyze_pair, mcnemar_exact_p, wilson_ci
from flash_forge.client import CallSpec, CompletionResult, ModelCapabilities
from flash_forge.code_forge import CodeForgeTrace
from flash_forge.engine import ForgeResult


class _PoisonPrivateCases:
    """Sentinel container that fails the test immediately if ever read."""

    def _boom(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("best4_sample MUST NEVER read private_cases!")

    __iter__ = _boom
    __getitem__ = _boom
    __len__ = _boom
    __bool__ = _boom


def _mk_lcb_item(
    pid: str = "abc999_f",
    public_cases: list[dict[str, str]] | None = None,
    private_cases: Any = None,
) -> dict[str, Any]:
    pub = public_cases if public_cases is not None else [
        {"input": "2 3\n", "output": "5", "testtype": "stdin"},
        {"input": "10 20\n", "output": "30", "testtype": "stdin"},
    ]
    priv_for_cases = (
        [{"input": "7 8\n", "output": "15", "testtype": "stdin"}]
        if private_cases is None
        else (
            private_cases
            if isinstance(private_cases, list)
            else [{"input": "7 8\n", "output": "15", "testtype": "stdin"}]
        )
    )
    return {
        "pid": pid,
        "title": "Add Two Numbers",
        "platform": "atcoder",
        "prompt": "Read two integers a and b and print a + b.",
        "public_cases": pub,
        "private_cases": private_cases if private_cases is not None else priv_for_cases,
        "cases": pub + priv_for_cases,
        "n_pub": len(pub),
        "n_priv": len(priv_for_cases),
    }


def test_best4_selection_public_pass_ranking():
    public_cases = [
        {"input": "1\n", "output": "10"},
        {"input": "2\n", "output": "20"},
        {"input": "3\n", "output": "30"},
    ]
    c_pass1 = "n = int(input())\nprint(10 if n == 1 else 0)\n"
    c_pass3 = "n = int(input())\nprint(n * 10)\n"
    c_pass2 = "n = int(input())\nprint(n * 10 if n <= 2 else 999)\n"

    # Even though slot 0 has the lowest temp and slots 2 & 3 agree on stdout,
    # slot 1 passes 3/3 public cases and MUST win via rule (a).
    candidates = [
        (0.25, c_pass1),
        (0.45, c_pass3),
        (0.65, c_pass2),
        (0.85, c_pass2),
    ]
    selected, meta = select_best4_candidate(candidates, public_cases)
    assert selected == c_pass3
    assert meta["public_pass_counts"] == [1, 3, 2, 2]
    assert meta["selected_slot"] == 1
    assert meta["selected_temp"] == 0.45
    assert meta["selection_reason"] == "public_pass"


def test_best4_selection_majority_tiebreak_and_syntax_filtering():
    public_cases = [
        {"input": "1\n", "output": "10"},
        {"input": "2\n", "output": "20"},
    ]
    # 1. Three candidates tie at 1/2 public passes; slots 1 & 2 agree on ("10", "42"),
    #    slot 0 prints unique ("10", "999"), slot 3 has a syntax error.
    #    Majority agreement beats lower-temp slot 0, then lower temp picks slot 1 over slot 2.
    c_unique = "n = int(input())\nprint(10 if n == 1 else 999)\n"
    c_maj = "n = int(input())\nprint(10 if n == 1 else 42)\n"
    c_syn_err = "def broken(:\n    pass\n"

    selected, meta = select_best4_candidate(
        [(0.25, c_unique), (0.45, c_maj), (0.65, c_maj), (0.85, c_syn_err)],
        public_cases,
    )
    assert selected == c_maj
    assert meta["public_pass_counts"] == [1, 1, 1, 0]
    assert meta["selected_slot"] == 1
    assert meta["selected_agreement_count"] == 2
    assert meta["selection_reason"] == "majority_tiebreak"

    # 2. Two crashing candidates MUST NOT form a fake majority over a clean valid-syntax script.
    c_crash_a = "n = int(input())\nif n == 1: print(10)\nelse: 1 / 0\n"
    c_crash_b = "n = int(input())\nif n == 1: print(10)\nelse: raise ValueError('bad')\n"
    c_clean = "n = int(input())\nprint(10 if n == 1 else 77)\n"
    selected2, meta2 = select_best4_candidate(
        [(0.25, c_crash_a), (0.45, c_crash_b), (0.65, c_clean), (0.85, c_syn_err)],
        public_cases,
    )
    assert selected2 == c_clean
    assert meta2["public_pass_counts"] == [1, 1, 1, 0]
    assert meta2["selected_slot"] == 2

    # 3. Valid-syntax candidate beats syntax-error / empty candidates when tied at 0 passes.
    c_wrong_a = "input()\nprint('wrong_a')\n"
    c_wrong_b = "input()\nprint('wrong_b')\n"
    selected3, meta3 = select_best4_candidate(
        [(0.25, c_syn_err), (0.45, ""), (0.65, c_wrong_a), (0.85, c_wrong_b)],
        public_cases,
    )
    assert selected3 == c_wrong_a
    assert meta3["public_pass_counts"] == [0, 0, 0, 0]
    assert meta3["selected_slot"] == 2


def test_best4_selection_low_temp_tiebreak():
    public_cases = [
        {"input": "2 3\n", "output": "5"},
        {"input": "4 5\n", "output": "9"},
    ]
    # 1. All 4 pass 2/2 public cases -> identical stdout tuple -> lowest temp (slot 0) wins.
    c_slot0 = "a, b = map(int, input().split())\nprint(a + b)  # slot0\n"
    c_slot1 = "a, b = map(int, input().split())\nprint(b + a)  # slot1\n"
    c_slot2 = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n"
    c_slot3 = "x = list(map(int, input().split()))\nprint(sum(x))\n"

    selected, meta = select_best4_candidate(
        [(0.25, c_slot0), (0.45, c_slot1), (0.65, c_slot2), (0.85, c_slot3)],
        public_cases,
    )
    assert selected == c_slot0
    assert meta["public_pass_counts"] == [2, 2, 2, 2]
    assert meta["selected_slot"] == 0
    assert meta["selection_reason"] == "low_temp_tiebreak"

    # 2. Non-monotonic temperatures: lowest temperature wins even if not slot 0.
    selected_nm, meta_nm = select_best4_candidate(
        [(0.85, c_slot0), (0.25, c_slot1), (0.65, c_slot2), (0.45, c_slot3)],
        public_cases,
    )
    assert selected_nm == c_slot1
    assert meta_nm["selected_slot"] == 1
    assert meta_nm["selected_temp"] == 0.25

    # 3. Equal temperatures: earliest slot wins.
    selected_eq, meta_eq = select_best4_candidate(
        [(0.50, c_slot0), (0.50, c_slot1)],
        public_cases,
    )
    assert selected_eq == c_slot0
    assert meta_eq["selected_slot"] == 0


def test_best4_never_peeks_at_private_cases(monkeypatch: pytest.MonkeyPatch):
    """Candidate passing 2/2 public cases MUST be chosen even if another candidate passes private cases."""
    pub = [
        {"input": "2 3\n", "output": "5", "testtype": "stdin"},
        {"input": "10 20\n", "output": "30", "testtype": "stdin"},
    ]
    priv_real = [{"input": "7 8\n", "output": "15", "testtype": "stdin"}]

    # Put a poison sentinel in `private_cases` so any attempt to touch it explodes immediately!
    item = {
        "pid": "abc401_f",
        "title": "Leak Check",
        "platform": "atcoder",
        "prompt": "Read a b and print a + b.",
        "public_cases": pub,
        "private_cases": _PoisonPrivateCases(),
        "cases": pub + priv_real,
        "n_pub": len(pub),
        "n_priv": len(priv_real),
    }

    # Slot 0 overfits public cases (2/2 public, FAILS private!)
    overfit_pub = (
        "```python\n"
        "a, b = map(int, input().split())\n"
        "if (a, b) == (2, 3): print(5)\n"
        "elif (a, b) == (10, 20): print(30)\n"
        "else: print(-1)\n"
        "```"
    )
    # Slot 1 passes only 1/2 public cases, but passes private case (7, 8) -> 15
    pass_priv_1pub = (
        "```python\n"
        "a, b = map(int, input().split())\n"
        "print(999 if (a, b) == (10, 20) else a + b)\n"
        "```"
    )
    # Slot 2 passes 0/2 public cases, passes private case
    pass_priv_0pub = (
        "```python\n"
        "a, b = map(int, input().split())\n"
        "print(15 if (a, b) == (7, 8) else 0)\n"
        "```"
    )

    class _MockClient:
        base_url = "http://127.0.0.1:8080/v1"

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            contents = [overfit_pub, pass_priv_1pub, pass_priv_0pub, pass_priv_1pub]
            return [
                CompletionResult(
                    content=txt,
                    reasoning="r",
                    finish_reason="stop",
                    prompt_tokens=100,
                    completion_tokens=50,
                    reasoning_tokens=20,
                    cached_tokens=10,
                    latency_s=0.01,
                    tag=s.tag,
                )
                for s, txt in zip(specs, contents)
            ]

    grade_calls: list[int] = []
    real_grade = lcb.grade_livecodebench

    def _spy_grade(code: str, cases: list[dict[str, Any]], per_case_timeout_s: float = 4.0) -> tuple[bool, str]:
        grade_calls.append(len(cases))
        return real_grade(code, cases, per_case_timeout_s)

    monkeypatch.setattr(lcb, "grade_livecodebench", _spy_grade)

    row = asyncio.run(run_one_best4_sample(_MockClient(), item))  # type: ignore[arg-type]

    # Official grader called ONCE on ONLY the single selected candidate against all 3 cases
    assert grade_calls == [3]
    # Slot 0 won selection (2/2 public) despite failing private -> proves zero private-case peeking!
    assert row["extra"]["public_pass_counts"] == [2, 1, 0, 1]
    assert row["extra"]["selected_slot"] == 0
    assert row["passed"] is False
    assert "WA on case 3/3" in row["detail"]


def test_all_three_arms_token_logging_and_hyperparams(monkeypatch: pytest.MonkeyPatch):
    item = _mk_lcb_item(pid="abc387_f")
    good_code = "a, b = map(int, input().split())\nprint(a + b)"
    good_reply = f"```python\n{good_code}\n```"
    wrong_reply = "```python\nprint(0)\n```"

    class _MockClient:
        def __init__(self) -> None:
            self.base_url = "http://127.0.0.1:8080/v1"
            self.complete_specs: list[CallSpec] = []
            self.batch_specs: list[list[CallSpec]] = []

        async def complete(self, spec: CallSpec) -> CompletionResult:
            self.complete_specs.append(spec)
            return CompletionResult(
                content=good_reply,
                reasoning="thought",
                finish_reason="stop",
                prompt_tokens=320,
                completion_tokens=480,
                reasoning_tokens=210,
                cached_tokens=64,
                latency_s=0.05,
                tag=spec.tag,
            )

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            self.batch_specs.append(specs)
            replies = [wrong_reply, good_reply, wrong_reply, good_reply]
            out: list[CompletionResult] = []
            for idx, (s, txt) in enumerate(zip(specs, replies)):
                out.append(
                    CompletionResult(
                        content=txt,
                        reasoning="thought",
                        finish_reason="stop",
                        prompt_tokens=300 + idx * 10,
                        completion_tokens=400 + idx * 20,
                        reasoning_tokens=150 + idx * 5,
                        cached_tokens=50,
                        latency_s=0.05,
                        tag=s.tag,
                    )
                )
            return out

    # 1. Vanilla arm
    c_van = _MockClient()
    row_v = asyncio.run(run_one_vanilla(c_van, item))  # type: ignore[arg-type]
    assert len(c_van.complete_specs) == 1
    vs = c_van.complete_specs[0]
    assert vs.seed == 42 and vs.temperature == 0.6
    assert vs.thinking_budget == 1500 and vs.max_tokens == 8192 and vs.loop_guard is True
    assert row_v["arm"] == "vanilla"
    assert row_v["passed"] is True
    assert row_v["stats"] == {
        "calls": 1,
        "prompt_tokens": 320,
        "completion_tokens": 480,
        "reasoning_tokens": 210,
        "cached_tokens": 64,
        "total_tokens": 800,
    }

    # 2. best4_sample arm
    c_b4 = _MockClient()
    row_b4 = asyncio.run(run_one_best4_sample(c_b4, item))  # type: ignore[arg-type]
    assert len(c_b4.batch_specs) == 1
    b4s = c_b4.batch_specs[0]
    assert len(b4s) == 4
    assert [(s.seed, s.temperature) for s in b4s] == list(BEST4_CONFIGS)
    assert all(
        s.thinking_budget == 1000
        and s.max_tokens == 8192
        and s.loop_guard is True
        and s.messages == [{"role": "user", "content": item["prompt"]}]
        for s in b4s
    )
    assert row_b4["arm"] == "best4_sample"
    assert row_b4["passed"] is True
    # Sum across 4 slots: pt=300+310+320+330=1260, ct=400+420+440+460=1720, rt=150+155+160+165=630, cache=200
    assert row_b4["stats"] == {
        "calls": 4,
        "prompt_tokens": 1260,
        "completion_tokens": 1720,
        "reasoning_tokens": 630,
        "cached_tokens": 200,
        "total_tokens": 2980,
    }
    assert row_b4["extra"]["public_pass_counts"] == [0, 2, 0, 2]
    assert row_b4["extra"]["selected_slot"] == 1  # lower temp (0.45 < 0.85) among the two 2/2 passers

    # 3. Forge arm (verify CortheonConfig args & normalized total_tokens)
    captured_cfgs: list[Any] = []
    captured_solve_kwargs: list[dict[str, Any]] = []

    class _MockCortheon:
        def __init__(self, cfg: Any) -> None:
            captured_cfgs.append(cfg)

        async def __aenter__(self) -> "_MockCortheon":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def solve(self, prompt: str, **kwargs: Any) -> ForgeResult:
            captured_solve_kwargs.append(kwargs)
            trace = CodeForgeTrace(
                winner_name="w2_radically_simple",
                winner_code=good_code,
                wave2_triggered=True,
                redteam_valid_kills=["rt_1", "rt_2"],
                oracle_test_names=["t1", "t2", "t3"],
            )
            return ForgeResult(
                mode="code",
                answer=good_reply,
                code=good_code,
                raw_trace=trace,
                wall_s=12.345,
                # Intentionally omit `total_tokens` (just like FlashForge.solve) to verify normalization!
                token_stats={
                    "calls": 14,
                    "prompt_tokens": 15000,
                    "completion_tokens": 9000,
                    "reasoning_tokens": 3100,
                    "cached_tokens": 2500,
                },
            )

    monkeypatch.setattr(lcb, "Cortheon", _MockCortheon)
    row_f = asyncio.run(run_one_forge("http://127.0.0.1:8080/v1", item, streams=6))
    assert len(captured_cfgs) == 1
    assert captured_cfgs[0].base_url == "http://127.0.0.1:8080/v1"
    assert captured_cfgs[0].num_streams == 6
    assert captured_cfgs[0].deep is False
    assert captured_cfgs[0].enable_memory is False
    assert captured_solve_kwargs == [{"force_mode": None, "learn": False}]
    assert row_f["arm"] == "forge"
    assert row_f["passed"] is True
    assert row_f["wall_s"] == 12.35
    assert row_f["stats"] == {
        "calls": 14,
        "prompt_tokens": 15000,
        "completion_tokens": 9000,
        "reasoning_tokens": 3100,
        "cached_tokens": 2500,
        "total_tokens": 24000,
    }
    assert row_f["extra"] == {
        "routed_mode": "code",
        "winner": "w2_radically_simple",
        "wave2": True,
        "rt_kills": 2,
        "oracle_tests": 3,
    }


def test_merge_sharded_ranges_disjoint_arms_stats_and_mcnemar(tmp_path: Path):
    def _mk_row(
        arm: str,
        idx: int,
        passed: bool,
        wall: float,
        calls: int,
        pt: int,
        ct: int,
        rt: int,
        *,
        include_total: bool = True,
    ) -> dict[str, Any]:
        stats: dict[str, int] = {
            "calls": calls,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "reasoning_tokens": rt,
            "cached_tokens": 100,
        }
        if include_total:
            stats["total_tokens"] = pt + ct
        return {
            "arm": arm,
            "pid": f"abc{387 + idx:03d}_f",
            "title": f"Prob {idx}",
            "passed": passed,
            "detail": "AC 42/42 (500ms)" if passed else "WA on case 1/42",
            "wall_s": wall,
            "stats": stats,
        }

    # Craft 60 problems (0..59) with explicit non-nested discordant pairs:
    # vanilla:      0..23 passed (24/60 = 40.0%)
    # best4_sample: 0..21 + 24..30 + 55 passed (30/60 = 50.0%)
    #   vs vanilla:      n11=22, +fixed(n01)=8,  -broken(n10)=2, n00=28
    # forge:        0..22 + 24..39 + 50 passed (40/60 = 66.7%)
    #   vs vanilla:      n11=23, +fixed(n01)=17, -broken(n10)=1, n00=19
    #   vs best4_sample: n11=29, +fixed(n01)=11, -broken(n10)=1, n00=19
    s1_res: dict[str, list[dict[str, Any]]] = {a: [] for a in ALL_ARMS}
    s2_res: dict[str, list[dict[str, Any]]] = {a: [] for a in ALL_ARMS}

    for i in range(60):
        target = s1_res if i < 40 else s2_res
        v_ok = i < 24
        b4_ok = (i < 22) or (24 <= i < 31) or (i == 55)
        f_ok = (i < 23) or (24 <= i < 40) or (i == 50)
        target["vanilla"].append(_mk_row("vanilla", i, v_ok, 100.0, 1, 1000, 3000, 1200))
        target["best4_sample"].append(_mk_row("best4_sample", i, b4_ok, 140.0, 4, 4000, 12000, 3600))
        target["forge"].append(_mk_row("forge", i, f_ok, 220.0, 16, 20000, 24000, 8000))

    p_s1 = tmp_path / "lcb_0_40.json"
    p_s2 = tmp_path / "lcb_40_60.json"
    out_merged = tmp_path / "lcb_60_merged.json"

    p_s1.write_text(
        json.dumps(
            build_lcb_payload(
                model="glm-5.3-flash-tfq3",
                arms=list(ALL_ARMS),
                results_by_arm=s1_res,
                range_info={"start": 0, "end": 40},
            )
        ),
        encoding="utf-8",
    )
    p_s2.write_text(
        json.dumps(
            build_lcb_payload(
                model="glm-5.3-flash-tfq3",
                arms=list(ALL_ARMS),
                results_by_arm=s2_res,
                range_info={"start": 40, "end": 60},
            )
        ),
        encoding="utf-8",
    )

    # 1. Merge 2 multi-arm shards passed in reverse order via CLI --merge
    rc = lcb_main(["--merge", str(p_s2), str(p_s1), "--out", str(out_merged)])
    assert rc == 0
    merged = json.loads(out_merged.read_text(encoding="utf-8"))

    assert merged["model"] == "glm-5.3-flash-tfq3"
    assert merged["arms"] == ["vanilla", "best4_sample", "forge"]
    assert merged["range"] == {"start": 0, "end": 60, "shards": 2}
    assert merged["merged_from"] == ["lcb_40_60.json", "lcb_0_40.json"]

    expected_pids = [f"abc{387 + i:03d}_f" for i in range(60)]
    for arm in ALL_ARMS:
        assert [r["pid"] for r in merged["results"][arm]] == expected_pids

    # Verify exact per-arm summary metrics and exact Wilson 95% CIs
    for arm, exp_k, exp_wall, exp_calls, exp_pt, exp_ct, exp_rt, exp_tot in [
        ("vanilla", 24, 100.0, 1.0, 1000.0, 3000.0, 1200.0, 4000.0),
        ("best4_sample", 30, 140.0, 4.0, 4000.0, 12000.0, 3600.0, 16000.0),
        ("forge", 40, 220.0, 16.0, 20000.0, 24000.0, 8000.0, 44000.0),
    ]:
        s = merged["summary"][arm]
        lo, hi = wilson_ci(exp_k, 60)
        assert s["passed"] == exp_k
        assert s["total"] == 60
        assert math.isclose(s["accuracy_pct"], round(100.0 * exp_k / 60, 2), rel_tol=1e-9)
        assert s["wilson_ci_95_pct"] == [round(100.0 * lo, 2), round(100.0 * hi, 2)]
        assert s["avg_wall_s"] == exp_wall
        assert s["avg_llm_calls"] == exp_calls
        assert s["avg_prompt_tokens"] == exp_pt
        assert s["avg_completion_tokens"] == exp_ct
        assert s["avg_reasoning_tokens"] == exp_rt
        assert s["avg_total_tokens"] == exp_tot

    # Verify all 3 exact paired McNemar tests + fixed/broken counts
    pw_fv = merged["pairwise"]["forge_vs_vanilla"]
    assert (pw_fv["n"], pw_fv["n11"], pw_fv["n01"], pw_fv["n10"], pw_fv["n00"]) == (60, 23, 17, 1, 19)
    assert pw_fv["fixed"] == 17 and pw_fv["broken"] == 1
    assert math.isclose(pw_fv["mcnemar_exact_p"], mcnemar_exact_p(17, 1), rel_tol=1e-12)
    assert math.isclose(pw_fv["wall_ratio"], 2.2, rel_tol=1e-9)
    assert math.isclose(pw_fv["token_ratio"], 11.0, rel_tol=1e-9)

    pw_fb = merged["pairwise"]["forge_vs_best4_sample"]
    assert (pw_fb["n"], pw_fb["n11"], pw_fb["n01"], pw_fb["n10"], pw_fb["n00"]) == (60, 29, 11, 1, 19)
    assert pw_fb["fixed"] == 11 and pw_fb["broken"] == 1
    assert math.isclose(pw_fb["mcnemar_exact_p"], mcnemar_exact_p(11, 1), rel_tol=1e-12)
    assert math.isclose(pw_fb["token_ratio"], round(44000 / 16000, 3), rel_tol=1e-9)

    pw_bv = merged["pairwise"]["best4_sample_vs_vanilla"]
    assert (pw_bv["n"], pw_bv["n11"], pw_bv["n01"], pw_bv["n10"], pw_bv["n00"]) == (60, 22, 8, 2, 28)
    assert pw_bv["fixed"] == 8 and pw_bv["broken"] == 2
    assert math.isclose(pw_bv["mcnemar_exact_p"], mcnemar_exact_p(8, 2), rel_tol=1e-12)
    assert math.isclose(pw_bv["token_ratio"], 4.0, rel_tol=1e-9)

    # Markdown table includes all arms and all 3 comparisons
    md = format_lcb_table(merged)
    assert "`vanilla`" in md and "`best4_sample`" in md and "`forge`" in md
    assert "`forge` vs `vanilla`" in md
    assert "`forge` vs `best4_sample`" in md
    assert "`best4_sample` vs `vanilla`" in md
    assert "+17" in md and "-1" in md

    # 2. Disjoint-arm + sharded-range mixed merge (including legacy `"arm": "cortheon"` single-arm list!)
    v_0_40 = tmp_path / "v_0_40.json"
    v_40_60 = tmp_path / "v_40_60.json"
    b4_0_60 = tmp_path / "b4_0_60.json"
    legacy_cortheon = tmp_path / "legacy_cortheon_0_60.json"

    v_0_40.write_text(
        json.dumps(
            build_lcb_payload(
                model="glm-5.3-flash-tfq3",
                arms=["vanilla"],
                results_by_arm={"vanilla": s1_res["vanilla"]},
                range_info={"start": 0, "end": 40},
            )
        ),
        encoding="utf-8",
    )
    v_40_60.write_text(
        json.dumps(
            build_lcb_payload(
                model="glm-5.3-flash-tfq3",
                arms=["vanilla"],
                results_by_arm={"vanilla": s2_res["vanilla"]},
                range_info={"start": 40, "end": 60},
            )
        ),
        encoding="utf-8",
    )
    b4_0_60.write_text(
        json.dumps(
            build_lcb_payload(
                model="glm-5.3-flash-tfq3",
                arms=["best4_sample"],
                results_by_arm={"best4_sample": s1_res["best4_sample"] + s2_res["best4_sample"]},
                range_info={"start": 0, "end": 60},
            )
        ),
        encoding="utf-8",
    )
    # Legacy format: `"arm": "cortheon"`, list `"results"`, `stats` missing `total_tokens`
    legacy_rows = [
        _mk_row("cortheon", i, (i < 23) or (24 <= i < 40) or (i == 50), 220.0, 16, 20000, 24000, 8000, include_total=False)
        for i in range(60)
    ]
    legacy_cortheon.write_text(
        json.dumps({
            "dataset": "LiveCodeBench v6 Hard (AtCoder F/G)",
            "model": "glm-5.3-flash-tfq3",
            "arm": "cortheon",
            "passed": 40,
            "total": 60,
            "results": legacy_rows,
        }),
        encoding="utf-8",
    )

    mixed = merge_lcb_files([b4_0_60, v_40_60, legacy_cortheon, v_0_40])
    assert mixed["arms"] == ["vanilla", "best4_sample", "forge"]
    assert mixed["summary"] == merged["summary"]
    assert mixed["pairwise"] == merged["pairwise"]

    # 3. Single-arm shard merge preserves `bench/stats.py` list compatibility!
    stats_dir = tmp_path / "stats_compat"
    v_single = stats_dir / "official_lcb_v6_hard_glm53_vanilla.json"
    f_single = stats_dir / "official_lcb_v6_hard_glm53_forge.json"
    merge_lcb_files([v_0_40, v_40_60], out_path=v_single)
    merge_lcb_files([legacy_cortheon], out_path=f_single)

    v_data = json.loads(v_single.read_text(encoding="utf-8"))
    f_data = json.loads(f_single.read_text(encoding="utf-8"))
    assert isinstance(v_data["results"], list) and len(v_data["results"]) == 60
    assert isinstance(f_data["results"], list) and len(f_data["results"]) == 60
    assert v_data["arm"] == "vanilla" and f_data["arm"] == "forge"

    pair_stats = analyze_pair(v_data, f_data, suite_key="official_lcb_v6_hard_glm53")
    assert (pair_stats.n, pair_stats.n11, pair_stats.n01, pair_stats.n10, pair_stats.n00) == (60, 23, 17, 1, 19)
    assert math.isclose(pair_stats.mcnemar_p, mcnemar_exact_p(17, 1), rel_tol=1e-12)
    assert math.isclose(pair_stats.call_ratio or 0.0, 16.0, rel_tol=1e-9)
    assert math.isclose(pair_stats.token_ratio or 0.0, 11.0, rel_tol=1e-9)

    per_suite, _ = analyze_directory(stats_dir)
    assert len(per_suite) == 1
    assert per_suite[0].n == 60 and per_suite[0].cortheon_passed == 40


def test_pii_path_and_url_scrubbing(tmp_path: Path):
    # Direct unit tests on scrub_detail_paths
    assert (
        scrub_detail_paths('RE on case 1/10: File "/private/var/folders/4x/_q9b/T/ff_lcb_zz11/solution.py", line 2')
        == 'RE on case 1/10: File "<TMPDIR>/solution.py", line 2'
    )
    assert (
        scrub_detail_paths('File "/private/var/folders/4x/_q9b/T/ff_exec_12ab/_run.py", line 8')
        == 'File "<TMPDIR>/_run.py", line 8'
    )
    assert scrub_detail_paths('File "/private/var/folders/4x/_q9b') == 'File "<TMPDIR>'
    assert scrub_detail_paths('File "/tmp/ff_lcb_abc123/solution.py"') == 'File "<TMPDIR>/solution.py"'
    assert scrub_detail_paths("path=/Users/secretperson/Documents/repo/sol.py:42") == "path=<HOME>:42"
    assert scrub_detail_paths("path=/home/alice/work/file.py:42") == "path=<HOME>:42"
    assert scrub_detail_paths("see https://example.com/home/docs") == "see https://example.com/home/docs"
    assert (
        scrub_detail_paths(
            "EXC: ConnectError to http://gpu-box-01.corp.internal:8080/v1/chat/completions",
            redact_url="http://gpu-box-01.corp.internal:8080/v1",
        )
        == "EXC: ConnectError to <ENDPOINT>/chat/completions"
    )

    # End-to-end --merge PII check: directory path has a hostname, rows contain raw macOS temp & /Users paths
    host_dir = tmp_path / "gpu-host-secret.internal" / "runs"
    host_dir.mkdir(parents=True)
    s1 = host_dir / "shard_0_1.json"
    s2 = host_dir / "shard_1_2.json"
    out_path = tmp_path / "clean_merged.json"

    s1.write_text(
        json.dumps({
            "dataset": "livecodebench_v6_hard",
            "model": "test-model",
            "arm": "vanilla",
            "results": [
                {
                    "arm": "vanilla",
                    "pid": "abc387_f",
                    "title": "P1",
                    "passed": False,
                    "wall_s": 10.0,
                    "detail": (
                        'RE on case 1/5: File "/private/var/folders/aa/bb123/T/ff_lcb_9x9/solution.py", '
                        "ref=/Users/secretuser/Desktop/key.pem"
                    ),
                    "stats": {
                        "calls": 1,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "reasoning_tokens": 20,
                        "cached_tokens": 0,
                        "total_tokens": 150,
                    },
                }
            ],
        }),
        encoding="utf-8",
    )
    s2.write_text(
        json.dumps({
            "dataset": "livecodebench_v6_hard",
            "model": "test-model",
            "arm": "forge",
            "results": [
                {
                    "arm": "forge",
                    "pid": "abc387_f",
                    "title": "P1",
                    "passed": True,
                    "wall_s": 20.0,
                    "detail": "AC 5/5 (120ms)",
                    "stats": {
                        "calls": 4,
                        "prompt_tokens": 400,
                        "completion_tokens": 200,
                        "reasoning_tokens": 80,
                        "cached_tokens": 50,
                    },
                }
            ],
        }),
        encoding="utf-8",
    )

    rc = lcb_main(["--merge", str(s1), str(s2), "--out", str(out_path)])
    assert rc == 0
    raw = out_path.read_text(encoding="utf-8")
    merged = json.loads(raw)

    assert "/private/var/folders" not in raw
    assert "/var/folders" not in raw
    assert "/Users/" not in raw
    assert "secretuser" not in raw
    assert "ff_lcb_" not in raw
    assert "gpu-host-secret.internal" not in raw
    assert merged["merged_from"] == ["shard_0_1.json", "shard_1_2.json"]
    assert "<TMPDIR>/solution.py" in merged["results"]["vanilla"][0]["detail"]
    assert "<HOME>" in merged["results"]["vanilla"][0]["detail"]
    # Legacy forge stats missing total_tokens got normalized automatically
    assert merged["results"]["forge"][0]["stats"]["total_tokens"] == 600


def test_cli_sharding_backwards_compat_atomic_checkpoint_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Create a tiny 4-problem LiveCodeBench-format dataset on disk
    dataset_path = tmp_path / "mini_lcb.json"
    raw_items = [
        {
            "question_id": f"abc40{i}_f",
            "question_title": f"Sum {i}",
            "platform": "atcoder",
            "question_content": f"Read a b, print a + b + {i}.",
            "public_test_cases": json.dumps([
                {"input": "1 2\n", "output": str(3 + i), "testtype": "stdin"},
            ]),
            "private_test_cases": json.dumps([
                {"input": "10 20\n", "output": str(30 + i), "testtype": "stdin"},
            ]),
        }
        for i in range(4)
    ]
    dataset_path.write_text(json.dumps(raw_items), encoding="utf-8")

    # Verify load_lcb_dataset slicing for --limit, --start, --end
    assert [x["pid"] for x in load_lcb_dataset(dataset_path)] == ["abc400_f", "abc401_f", "abc402_f", "abc403_f"]
    assert [x["pid"] for x in load_lcb_dataset(dataset_path, limit=2)] == ["abc400_f", "abc401_f"]
    assert [x["pid"] for x in load_lcb_dataset(dataset_path, start=0, end=2)] == ["abc400_f", "abc401_f"]
    assert [x["pid"] for x in load_lcb_dataset(dataset_path, start=2, end=4)] == ["abc402_f", "abc403_f"]
    assert [x["pid"] for x in load_lcb_dataset(dataset_path, start=2, end=0)] == ["abc402_f", "abc403_f"]
    assert [x["pid"] for x in load_lcb_dataset(dataset_path, start=1, end=4, limit=2)] == ["abc401_f", "abc402_f"]

    call_log: list[str] = []
    forge_streams_seen: list[int] = []

    def _solution_for_prompt(prompt: str) -> str:
        # Extract offset from "print a + b + K."
        k = int(prompt.split("a + b + ")[1].split(".")[0])
        return f"a, b = map(int, input().split())\nprint(a + b + {k})"

    class _FakeClient:
        def __init__(self, base_url: str = "", **kwargs: Any) -> None:
            self.base_url = base_url
            self.caps = ModelCapabilities(model_id="mock-model-tfq3")

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def complete(self, spec: CallSpec) -> CompletionResult:
            call_log.append(spec.tag)
            code = _solution_for_prompt(spec.messages[-1]["content"])
            return CompletionResult(
                content=f"```python\n{code}\n```",
                reasoning="r",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=60,
                reasoning_tokens=20,
                cached_tokens=0,
                latency_s=0.01,
                tag=spec.tag,
            )

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            return [await self.complete(s) for s in specs]

    class _FakeCortheon:
        def __init__(self, cfg: Any) -> None:
            forge_streams_seen.append(cfg.num_streams)

        async def __aenter__(self) -> "_FakeCortheon":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def solve(self, prompt: str, **kwargs: Any) -> ForgeResult:
            call_log.append(f"forge:{prompt.split('a + b + ')[1].split('.')[0]}")
            code = _solution_for_prompt(prompt)
            return ForgeResult(
                mode="code",
                answer=f"```python\n{code}\n```",
                code=code,
                wall_s=1.5,
                token_stats={
                    "calls": 5,
                    "prompt_tokens": 500,
                    "completion_tokens": 300,
                    "reasoning_tokens": 100,
                    "cached_tokens": 50,
                },
            )

    monkeypatch.setattr(lcb, "TensorFoldClient", _FakeClient)
    monkeypatch.setattr(lcb, "Cortheon", _FakeCortheon)

    # 1. Backwards-compatible single-arm CLI: --arm vanilla --limit 2
    out_v = tmp_path / "lcb_vanilla.json"
    rc_v = lcb_main([
        "--dataset", str(dataset_path),
        "--url", "http://secret-gpu.internal:8080/v1",
        "--arm", "vanilla",
        "--limit", "2",
        "--out", str(out_v),
    ])
    assert rc_v == 0
    assert not out_v.with_suffix(".json.tmp").exists()
    raw_v = out_v.read_text(encoding="utf-8")
    assert "secret-gpu.internal" not in raw_v
    data_v = json.loads(raw_v)
    assert data_v["arm"] == "vanilla"
    assert isinstance(data_v["results"], list) and len(data_v["results"]) == 2
    assert data_v["passed"] == 2 and data_v["total"] == 2
    assert data_v["summary"]["vanilla"]["avg_total_tokens"] == 160.0

    # 2. Backwards-compatible single-arm CLI: --arm forge --limit 2 --streams 6
    out_f = tmp_path / "lcb_forge.json"
    rc_f = lcb_main([
        "--dataset", str(dataset_path),
        "--url", "http://127.0.0.1:8080/v1",
        "--arm", "forge",
        "--limit", "2",
        "--streams", "6",
        "--out", str(out_f),
    ])
    assert rc_f == 0
    assert forge_streams_seen == [6, 6]
    data_f = json.loads(out_f.read_text(encoding="utf-8"))
    assert data_f["arm"] == "forge"
    assert isinstance(data_f["results"], list) and len(data_f["results"]) == 2
    # Verify single-arm files work seamlessly with bench/stats.py!
    st = analyze_pair(data_v, data_f, suite_key="mini")
    assert st.n == 2 and st.vanilla_passed == 2 and st.cortheon_passed == 2
    assert math.isclose(st.token_ratio or 0.0, 5.0, rel_tol=1e-9)

    # 3. Multi-arm sharded CLI + --resume skips completed (arm, pid) pairs
    call_log.clear()
    out_s1 = tmp_path / "lcb_s1.json"
    rc_s1 = lcb_main([
        "--dataset", str(dataset_path),
        "--url", "http://127.0.0.1:8080/v1",
        "--arms", "vanilla,best4_sample,forge",
        "--start", "0",
        "--end", "2",
        "--out", str(out_s1),
    ])
    assert rc_s1 == 0
    data_s1 = json.loads(out_s1.read_text(encoding="utf-8"))
    assert data_s1["arms"] == ["vanilla", "best4_sample", "forge"]
    assert data_s1["range"] == {"start": 0, "end": 2}
    assert set(data_s1["pairwise"].keys()) == {"forge_vs_vanilla", "forge_vs_best4_sample", "best4_sample_vs_vanilla"}
    assert len(call_log) == 2 * (1 + 4 + 1)  # 2 problems * (1 vanilla + 4 best4 + 1 forge)

    # Resume with the exact same file -> 0 new LLM calls!
    call_log.clear()
    rc_res = lcb_main([
        "--dataset", str(dataset_path),
        "--url", "http://127.0.0.1:8080/v1",
        "--arms", "vanilla,best4_sample,forge",
        "--start", "0",
        "--end", "2",
        "--out", str(out_s1),
        "--resume",
    ])
    assert rc_res == 0
    assert call_log == []

    # Conflicting --arm and --arms raises SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        lcb_main([
            "--arm", "vanilla",
            "--arms", "forge",
            "--out", str(tmp_path / "bad.json"),
        ])
    assert exc_info.value.code == 2
