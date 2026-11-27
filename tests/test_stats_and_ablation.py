"""Unit tests for client raw-upstream safety, exact paired statistics, dataset fetcher, and 6-arm AIME ablation."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from bench.fetch_datasets import (
    BIGCODEBENCH_HARD_VERIFIED_IDS,
    fetch_aime2025,
    fetch_bigcodebench,
    fetch_livecodebench,
)
from bench.run_aime_ablation import (
    ALL_ARMS,
    MAJ4_CONFIGS,
    MAJ8_CONFIGS,
    build_ablation_payload,
    format_ablation_table,
    grade_aime,
    main as ablation_main,
    merge_ablation_files,
    run_maj8_prose,
    run_single_arm,
    scrub_detail_paths,
    select_maj4_answer,
)
from bench.stats import (
    WILSON_Z_95,
    analyze_directory,
    analyze_pair,
    mcnemar_exact_p,
    paired_contingency,
    wilson_ci,
)
from flash_forge.client import CallSpec, CompletionResult, ModelCapabilities, TensorFoldClient
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import solve_reasoning
from flash_forge.skill_loader import SkillPack


def test_client_cortheon_false_only_for_tensorfold():
    spec = CallSpec(messages=[{"role": "user", "content": "2+2?"}], max_tokens=64)

    # 1. TensorFold server -> MUST include `"cortheon": False` to prevent server-side double-nesting
    tf_client = TensorFoldClient(base_url="http://127.0.0.1:8080/v1")
    tf_client._caps = ModelCapabilities(model_id="glm-5.3-flash-tfq3", structures=True)
    tf_client._is_tensorfold = True
    tf_payload = tf_client._build_payload(spec)
    assert tf_payload.get("cortheon") is False

    # 2. Generic OpenAI-compatible server (Ollama / vLLM / llama.cpp) -> MUST NOT include `"cortheon"`
    oai_client = TensorFoldClient(base_url="http://127.0.0.1:11434/v1")
    oai_client._caps = ModelCapabilities(model_id="qwen3:32b", structures=True)
    oai_client._is_tensorfold = False
    oai_payload = oai_client._build_payload(spec)
    assert "cortheon" not in oai_payload

    # 3. Unprobed client without `_is_tensorfold` attribute -> defaults safe (omits `"cortheon"`)
    bare_client = TensorFoldClient(base_url="http://127.0.0.1:8000/v1")
    bare_client._caps = ModelCapabilities(model_id="llama-3")
    bare_payload = bare_client._build_payload(spec)
    assert "cortheon" not in bare_payload


def test_reason_forge_analytical_first_no_nameerror_and_disable_wave2(tmp_path: Path):
    """Regression test: analytical finishing before 2nd PAL must check `_answers_equivalent` cleanly."""
    seen_tags: list[str] = []

    class _FakeClient:
        async def complete(self, spec: CallSpec) -> CompletionResult:
            seen_tags.append(spec.tag)
            if spec.tag == "w1_analytical":
                # Finishes FIRST with a contradicting prose answer!
                await asyncio.sleep(0.005)
                return CompletionResult(
                    content="ANSWER: 999",
                    reasoning="",
                    finish_reason="stop",
                    prompt_tokens=100,
                    completion_tokens=40,
                    reasoning_tokens=10,
                    cached_tokens=0,
                    latency_s=0.005,
                    tag=spec.tag,
                )
            if spec.tag in ("w1_pal_exact", "w1_pal_alt_z3"):
                await asyncio.sleep(0.02)
                return CompletionResult(
                    content="```python\nassert 10 * 7 == 70\nprint('ANSWER: 70')\n```",
                    reasoning="",
                    finish_reason="stop",
                    prompt_tokens=100,
                    completion_tokens=40,
                    reasoning_tokens=10,
                    cached_tokens=0,
                    latency_s=0.02,
                    tag=spec.tag,
                )
            # 3rd PAL fails -> normally triggers Wave 2 because analytical contradicted 2 PALs
            await asyncio.sleep(0.04)
            return CompletionResult(
                content="```python\nraise ValueError('boom')\n```",
                reasoning="",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=40,
                reasoning_tokens=10,
                cached_tokens=0,
                latency_s=0.04,
                tag=spec.tag,
            )

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            raise AssertionError("Wave 2 batch must NOT run when enable_wave2=False")

    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    skills = SkillPack()
    trace = asyncio.run(
        solve_reasoning(
            _FakeClient(),  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 10 * 7. End with `ANSWER: <integer>`.",
            enable_wave2=False,
        )
    )
    assert trace.final_answer == "70"
    assert trace.wave2_triggered is False
    assert trace.consensus_type.startswith("wave1_only")
    assert all(not tag.startswith("w2_") for tag in seen_tags)


def test_wilson_ci_exact_values_and_edges():
    z2 = WILSON_Z_95**2

    assert wilson_ci(0, 0) == (0.0, 0.0)

    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0
    assert math.isclose(hi0, z2 / (10.0 + z2), rel_tol=1e-12)

    lo1, hi1 = wilson_ci(10, 10)
    assert math.isclose(lo1, 10.0 / (10.0 + z2), rel_tol=1e-12)
    assert hi1 == 1.0

    lo50, hi50 = wilson_ci(15, 30)
    assert math.isclose(lo50 + hi50, 1.0, rel_tol=1e-12)
    assert math.isclose(lo50, 0.33154, abs_tol=1e-4)
    assert math.isclose(hi50, 0.66846, abs_tol=1e-4)

    lo_perf, hi_perf = wilson_ci(50, 50)
    assert math.isclose(lo_perf, 50.0 / (50.0 + z2), rel_tol=1e-12)
    assert hi_perf == 1.0

    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)


def test_mcnemar_exact_p_values_and_large_m():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 3) == 1.0
    assert mcnemar_exact_p(1, 0) == 1.0
    assert math.isclose(mcnemar_exact_p(5, 0), 2.0 / 32.0, rel_tol=1e-15)
    assert math.isclose(mcnemar_exact_p(0, 5), 2.0 / 32.0, rel_tol=1e-15)
    assert math.isclose(mcnemar_exact_p(9, 2), 134.0 / 2048.0, rel_tol=1e-15)
    assert math.isclose(mcnemar_exact_p(10, 1), 24.0 / 2048.0, rel_tol=1e-15)
    assert math.isclose(mcnemar_exact_p(25, 0), 2.0 / (2**25), rel_tol=1e-15)
    assert math.isclose(mcnemar_exact_p(53, 0), 2.0 / (2**53), rel_tol=1e-15)

    # Large m > 1024 must NOT overflow IEEE-754 float conversion
    p_large = mcnemar_exact_p(1200, 800)
    assert 0.0 < p_large < 1e-18

    with pytest.raises(ValueError):
        mcnemar_exact_p(-1, 2)


def test_stats_paired_contingency_and_results_dir():
    ct = paired_contingency(
        {"p1": True, "p2": False, "p3": True, "p4": False, "p5": False},
        {"p1": True, "p2": True, "p3": False, "p4": False, "p5": True},
    )
    assert ct == {
        "n": 5,
        "n11": 1,
        "n01": 2,
        "n10": 1,
        "n00": 1,
        "baseline_passed": 2,
        "treatment_passed": 3,
    }

    # Verify token/call ratios when both arms log stats
    v_payload = {
        "dataset": "toy",
        "model": "m1",
        "results": [
            {"id": "P1", "correct": True, "wall_s": 10.0, "llm_calls": 1, "total_tokens": 1000},
            {"id": "P2", "correct": False, "wall_s": 20.0, "llm_calls": 1, "total_tokens": 2000},
        ],
    }
    c_payload = {
        "dataset": "toy",
        "model": "m1",
        "results": [
            {"id": "P1", "correct": True, "wall_s": 15.0, "llm_calls": 4, "total_tokens": 3000},
            {"id": "P2", "correct": True, "wall_s": 45.0, "llm_calls": 6, "total_tokens": 6000},
        ],
    }
    st = analyze_pair(v_payload, c_payload, suite_key="toy")
    assert st.n11 == 1 and st.n01 == 1 and st.n10 == 0 and st.n00 == 0
    assert math.isclose(st.wall_ratio or 0.0, 2.0, rel_tol=1e-9)
    assert math.isclose(st.call_ratio or 0.0, 5.0, rel_tol=1e-9)
    assert math.isclose(st.token_ratio or 0.0, 3.0, rel_tol=1e-9)

    # Verify real repository results/ ledger parses cleanly and matches exact pooled counts
    results_dir = Path(__file__).resolve().parent.parent / "results"
    per_suite, pooled = analyze_directory(results_dir)
    assert len(per_suite) == 11
    by_label = {p.suite_key: p for p in pooled}
    off = by_label["POOLED: All Official External"]
    assert off.n == 200
    assert off.vanilla_passed == 87 and off.cortheon_passed == 117
    assert (off.n11, off.n01, off.n10, off.n00) == (80, 37, 7, 76)
    assert off.mcnemar_p < 1e-5


def test_maj4_prose_clustering_normalization_and_tiebreak():
    # 1. Clear 2-vote majority beats lower-temperature 1-vote minority
    ans, meta = select_maj4_answer([
        (0.25, "We find ANSWER: 100"),
        (0.55, "Therefore \\boxed{200}"),
        (0.45, "**ANSWER: 200**"),
        (0.35, "ANSWER: 300"),
    ])
    assert ans == "200"
    assert meta["winner_count"] == 2
    assert meta["tie_broken_by_temp"] is False

    # 2. Exact 2-2 tie broken by lowest-temperature valid answer
    ans_t1, meta_t1 = select_maj4_answer([
        (0.25, "ANSWER: 70"),
        (0.55, "ANSWER: 42"),
        (0.45, "ANSWER: 70"),
        (0.35, "ANSWER: 42"),
    ])
    assert ans_t1 == "70"
    assert meta_t1["tie_broken_by_temp"] is True

    ans_t2, meta_t2 = select_maj4_answer([
        (0.25, "ANSWER: 42"),
        (0.55, "ANSWER: 70"),
        (0.45, "ANSWER: 70"),
        (0.35, "ANSWER: 42"),
    ])
    assert ans_t2 == "42"
    assert meta_t2["tie_broken_by_temp"] is True

    # 3. Leading zeros, float `.0`, boxed, and markdown wrappers cluster identically
    ans_norm, meta_norm = select_maj4_answer([
        (0.25, "Final result: \\boxed{070}"),
        (0.55, "`ANSWER: 70.0`"),
        (0.45, "ANSWER: 88"),
        (0.35, ""),
    ])
    assert ans_norm == "70"
    assert meta_norm["winner_count"] == 2

    # 4. Out-of-range (>999) and non-integer answers ignored when valid 0..999 integers exist;
    #    1-1 tie broken by lowest temperature among valid integers
    ans_filt, meta_filt = select_maj4_answer([
        (0.25, "ANSWER: 7/12"),
        (0.55, "ANSWER: 1024"),
        (0.45, "ANSWER: 042"),
        (0.35, "ANSWER: 99"),
    ])
    assert ans_filt == "99"
    assert meta_filt["valid_count"] == 2
    assert meta_filt["tie_broken_by_temp"] is True

    # 5. All empty returns ""
    ans_empty, meta_empty = select_maj4_answer([(0.25, ""), (0.55, ""), (0.45, ""), (0.35, "")])
    assert ans_empty == ""
    assert meta_empty["valid_count"] == 0

    # 6. AIME grader handles leading zeros & exact integer matching
    assert grade_aime("070", "70")[0] is True
    assert grade_aime("70.0", "70")[0] is True
    assert grade_aime("71", "70")[0] is False
    assert grade_aime("", "70")[0] is False


def test_ablation_merge_shards_and_summary(tmp_path: Path):
    def _row(idx: int, correct: bool, calls: int, tok: int, wall: float) -> dict[str, Any]:
        exp = str((idx * 17) % 1000)
        return {
            "id": f"AIME25_{idx:02d}",
            "expected": exp,
            "predicted": exp if correct else "999",
            "correct": correct,
            "wall_s": wall,
            "llm_calls": calls,
            "prompt_tokens": tok // 2,
            "completion_tokens": tok - tok // 2,
            "reasoning_tokens": tok // 4,
            "cached_tokens": 0,
            "total_tokens": tok,
            "detail": "ok" if correct else "wrong",
        }

    # Shard 1: problems 1..15; Shard 2: problems 16..30 (out-of-order input to test sorting!)
    shard1_res: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ALL_ARMS}
    shard2_res: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ALL_ARMS}

    for idx in range(1, 31):
        target = shard1_res if idx <= 15 else shard2_res
        target["vanilla_b1"].append(_row(idx, idx <= 15, 1, 2000, 30.0))      # 15/30
        target["maj4_prose"].append(_row(idx, idx <= 18, 4, 6000, 45.0))      # 18/30
        target["maj8_prose"].append(_row(idx, idx <= 19, 8, 12000, 90.0))     # 19/30
        target["pal_b1"].append(_row(idx, idx <= 20, 1, 1500, 20.0))          # 20/30
        target["cortheon_w1"].append(_row(idx, idx <= 23, 3, 4500, 35.0))     # 23/30
        target["cortheon_full"].append(_row(idx, idx <= 27, 5, 7000, 50.0))   # 27/30

    p1 = tmp_path / "shard_0_15.json"
    p2 = tmp_path / "shard_15_30.json"
    out_merged = tmp_path / "merged_30.json"

    p1.write_text(
        json.dumps(
            build_ablation_payload(
                model="test-model-tfq3",
                arms=list(ALL_ARMS),
                results_by_arm=shard1_res,
                range_info={"start": 0, "end": 15},
            )
        ),
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(
            build_ablation_payload(
                model="test-model-tfq3",
                arms=list(ALL_ARMS),
                results_by_arm=shard2_res,
                range_info={"start": 15, "end": 30},
            )
        ),
        encoding="utf-8",
    )

    # Run via CLI --merge passing shard 2 first to verify canonical sorting
    rc = ablation_main(["--merge", str(p2), str(p1), "--out", str(out_merged)])
    assert rc == 0
    merged = json.loads(out_merged.read_text(encoding="utf-8"))

    assert merged["model"] == "test-model-tfq3"
    assert merged["range"] == {"start": 0, "end": 30, "shards": 2}
    # Basenames only (zero host path PII!)
    assert merged["merged_from"] == ["shard_15_30.json", "shard_0_15.json"]

    for arm in ALL_ARMS:
        rows = merged["results"][arm]
        assert len(rows) == 30
        assert [r["id"] for r in rows] == [f"AIME25_{i:02d}" for i in range(1, 31)]

    assert merged["summary"]["vanilla_b1"]["correct"] == 15
    assert merged["summary"]["maj4_prose"]["correct"] == 18
    assert merged["summary"]["maj8_prose"]["correct"] == 19
    assert merged["summary"]["pal_b1"]["correct"] == 20
    assert merged["summary"]["cortheon_w1"]["correct"] == 23
    assert merged["summary"]["cortheon_full"]["correct"] == 27
    assert merged["summary"]["cortheon_full"]["accuracy_pct"] == 90.0

    # Check exact paired McNemar p-values: cortheon_full (27) vs vanilla_b1 (15) -> +12/-0 -> 2/2^12
    pw_v = merged["pairwise"]["cortheon_full_vs_vanilla_b1"]
    assert (pw_v["n11"], pw_v["n01"], pw_v["n10"], pw_v["n00"]) == (15, 12, 0, 3)
    assert math.isclose(pw_v["mcnemar_exact_p"], 2.0 / (2**12), rel_tol=1e-12)
    assert math.isclose(pw_v["token_ratio"], 3.5, rel_tol=1e-9)

    # cortheon_full (27) vs maj4_prose (18) -> +9/-0 -> 2/2^9
    pw_m = merged["pairwise"]["cortheon_full_vs_maj4_prose"]
    assert (pw_m["n11"], pw_m["n01"], pw_m["n10"], pw_m["n00"]) == (18, 9, 0, 3)
    assert math.isclose(pw_m["mcnemar_exact_p"], 2.0 / (2**9), rel_tol=1e-12)

    # cortheon_full (27) vs maj8_prose (19) -> +8/-0 -> 2/2^8
    pw_m8 = merged["pairwise"]["cortheon_full_vs_maj8_prose"]
    assert (pw_m8["n11"], pw_m8["n01"], pw_m8["n10"], pw_m8["n00"]) == (19, 8, 0, 3)
    assert math.isclose(pw_m8["mcnemar_exact_p"], 2.0 / (2**8), rel_tol=1e-12)

    # Also verify direct helper call
    direct = merge_ablation_files([p1, p2])
    assert direct["summary"]["cortheon_full"]["correct"] == 27


def test_all_six_ablation_arms_log_complete_schema():
    required_keys = {
        "id",
        "expected",
        "predicted",
        "correct",
        "wall_s",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
        "detail",
    }

    class _MockClient:
        def __init__(self) -> None:
            self.specs: list[CallSpec] = []
            self.reset_stats()

        def reset_stats(self) -> None:
            self.calls = 0
            self.pt = 0
            self.ct = 0
            self.rt = 0
            self.cache = 0

        def stats_dict(self) -> dict[str, int]:
            return {
                "calls": self.calls,
                "prompt_tokens": self.pt,
                "completion_tokens": self.ct,
                "reasoning_tokens": self.rt,
                "cached_tokens": self.cache,
            }

        async def complete(self, spec: CallSpec) -> CompletionResult:
            self.specs.append(spec)
            self.calls += 1
            self.pt += 200
            self.ct += 120
            self.rt += 80
            self.cache += 50

            tag = spec.tag
            if tag.startswith("abl_vanilla_b1"):
                content = "Step-by-step... ANSWER: 70"
            elif tag.startswith("abl_maj4_s0"):
                content = "ANSWER: 99"
            elif tag.startswith("abl_maj4_"):
                content = "ANSWER: 70"
            elif tag.startswith("abl_maj8_s0") or tag.startswith("abl_maj8_s1"):
                content = "ANSWER: 99"
            elif tag.startswith("abl_maj8_"):
                content = "ANSWER: 70"
            elif tag.startswith("abl_pal_b1"):
                content = "```python\nassert 7 * 10 == 70\nprint('ANSWER: 70')\n```"
            elif tag == "w1_pal_exact":
                content = "```python\nprint('ANSWER: 70')\n```"
            elif tag in ("w1_pal_alt_z3", "w1_pal_skeptic"):
                # Force Wave-1 disagreement so cortheon_w1 vs cortheon_full diverges!
                content = "```python\nraise RuntimeError('timeout')\n```"
            elif tag == "w1_analytical":
                content = "ANSWER: 99"
            elif tag.startswith("w2_"):
                content = "```python\nassert 7 * 10 == 70\nprint('ANSWER: 70')\n```"
            else:
                content = "A"
            return CompletionResult(
                content=content,
                reasoning="thinking...",
                finish_reason="stop",
                prompt_tokens=200,
                completion_tokens=120,
                reasoning_tokens=80,
                cached_tokens=50,
                latency_s=0.01,
                tag=tag,
            )

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            return [await self.complete(s) for s in specs]

        async def pairwise_vote(self, p_ab: str, p_ba: str, tag_prefix: str = "judge") -> str:
            self.calls += 2
            self.pt += 100
            self.ct += 4
            return "A"

    item = {
        "id": "AIME25_01",
        "problem_idx": 1,
        "prompt": "Compute 7 * 10. Note: Every AIME answer is an integer from 000 to 999.\nEnd your final response with `ANSWER: <integer>`.",
        "expected": "70",
    }
    skills = SkillPack()

    async def _run_all() -> dict[str, tuple[dict[str, Any], list[CallSpec]]]:
        out: dict[str, tuple[dict[str, Any], list[CallSpec]]] = {}
        for arm in ALL_ARMS:
            c = _MockClient()
            row = await run_single_arm(arm, c, skills, item)  # type: ignore[arg-type]
            out[arm] = (row, c.specs)
        return out

    ran = asyncio.run(_run_all())

    for arm in ALL_ARMS:
        row, _ = ran[arm]
        assert required_keys.issubset(row.keys()), f"Missing keys in {arm}: {required_keys - set(row.keys())}"
        assert row["id"] == "AIME25_01"
        assert row["expected"] == "70"
        assert row["total_tokens"] == row["prompt_tokens"] + row["completion_tokens"]
        assert row["llm_calls"] >= 1
        assert row["correct"] is True

    # Verify exact hyperparams of vanilla_b1
    _, v_specs = ran["vanilla_b1"]
    assert len(v_specs) == 1
    assert v_specs[0].seed == 42 and v_specs[0].temperature == 0.6
    assert v_specs[0].thinking_budget == 1500 and v_specs[0].max_tokens == 8192

    # Verify exact hyperparams of maj4_prose
    _, m_specs = ran["maj4_prose"]
    assert len(m_specs) == 4
    assert [(s.seed, s.temperature) for s in m_specs] == list(MAJ4_CONFIGS)
    assert all(s.thinking_budget == 1000 and s.max_tokens == 4096 for s in m_specs)

    # Verify exact hyperparams of maj8_prose
    _, m8_specs = ran["maj8_prose"]
    assert len(m8_specs) == 8
    assert [(s.seed, s.temperature) for s in m8_specs] == list(MAJ8_CONFIGS)
    assert all(s.thinking_budget == 1000 and s.max_tokens == 4096 for s in m8_specs)
    assert all(len(s.messages) == 1 and s.messages[0]["role"] == "user" for s in m8_specs)

    # Verify exact hyperparams of pal_b1
    _, p_specs = ran["pal_b1"]
    assert len(p_specs) == 1
    assert p_specs[0].seed == 101 and p_specs[0].temperature == 0.25
    assert p_specs[0].thinking_budget == 500 and p_specs[0].reasoning_effort == "low"

    # Verify cortheon_w1 did NOT trigger Wave 2 while cortheon_full DID trigger Wave 2
    w1_row, w1_specs = ran["cortheon_w1"]
    full_row, full_specs = ran["cortheon_full"]
    assert "wave2=False" in w1_row["detail"]
    assert not any(s.tag.startswith("w2_") for s in w1_specs)
    assert "wave2=True" in full_row["detail"]
    assert any(s.tag.startswith("w2_") for s in full_specs)


def test_maj8_prose_voting_disjoint_arm_merge_and_path_scrubbing(tmp_path: Path):
    """Verify maj8_prose (8 prose calls, no Python exec, voting/tiebreak), disjoint-arm --merge, and path scrubbing."""

    class _Maj8Client:
        def __init__(self, responses: list[str]) -> None:
            self.responses = responses
            self.batch_calls: list[list[CallSpec]] = []
            self.reset_stats()

        def reset_stats(self) -> None:
            self.calls = 0
            self.pt = 0
            self.ct = 0
            self.rt = 0
            self.cache = 0

        def stats_dict(self) -> dict[str, int]:
            return {
                "calls": self.calls,
                "prompt_tokens": self.pt,
                "completion_tokens": self.ct,
                "reasoning_tokens": self.rt,
                "cached_tokens": self.cache,
            }

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            self.batch_calls.append(specs)
            out: list[CompletionResult] = []
            for spec, text in zip(specs, self.responses):
                self.calls += 1
                self.pt += 250
                self.ct += 1500
                self.rt += 600
                self.cache += 100
                out.append(
                    CompletionResult(
                        content=text,
                        reasoning="prose reasoning",
                        finish_reason="stop",
                        prompt_tokens=250,
                        completion_tokens=1500,
                        reasoning_tokens=600,
                        cached_tokens=100,
                        latency_s=0.01,
                        tag=spec.tag,
                    )
                )
            return out

    item = {
        "id": "AIME25_07",
        "problem_idx": 7,
        "prompt": "Find the value. End your final response with `ANSWER: <integer>`.",
        "expected": "70",
    }

    # 1. Clear 4-vote majority across 8 parallel calls (including boxed/leading-zero/float normalization,
    #    and fenced python block that MUST NOT be executed!)
    responses_majority = [
        "ANSWER: 123",                                                       # s0 temp=0.20
        "```python\nraise SystemExit('must not execute')\n```\nANSWER: 070", # s1 temp=0.30
        "Therefore \\boxed{70}",                                             # s2 temp=0.40
        "ANSWER: 999",                                                       # s3 temp=0.50
        "`ANSWER: 70.0`",                                                    # s4 temp=0.35
        "ANSWER: 123",                                                       # s5 temp=0.45
        "ANSWER: 70",                                                        # s6 temp=0.55
        "ANSWER: 1024",                                                      # s7 temp=0.60 (out of 0..999 range)
    ]
    c_maj = _Maj8Client(responses_majority)
    row_maj = asyncio.run(run_maj8_prose(c_maj, item))  # type: ignore[arg-type]

    assert len(c_maj.batch_calls) == 1
    specs = c_maj.batch_calls[0]
    assert len(specs) == 8
    assert [s.seed for s in specs] == [101, 202, 303, 404, 505, 606, 707, 808]
    assert [s.temperature for s in specs] == [0.20, 0.30, 0.40, 0.50, 0.35, 0.45, 0.55, 0.60]
    assert all(s.thinking_budget == 1000 and s.max_tokens == 4096 for s in specs)
    assert row_maj["llm_calls"] == 8
    assert row_maj["predicted"] == "70"
    assert row_maj["correct"] is True
    assert "maj8_prose (4/8;" in row_maj["detail"]

    # 2. Exact 4-4 tie broken by lowest-temperature valid sample (0.20 < 0.30 -> 88 wins over 70)
    responses_tie = [
        "ANSWER: 88",  # s0 temp=0.20 (lowest temp!)
        "ANSWER: 70",  # s1 temp=0.30
        "ANSWER: 70",  # s2 temp=0.40
        "ANSWER: 70",  # s3 temp=0.50
        "ANSWER: 70",  # s4 temp=0.35
        "ANSWER: 88",  # s5 temp=0.45
        "ANSWER: 88",  # s6 temp=0.55
        "ANSWER: 88",  # s7 temp=0.60
    ]
    c_tie = _Maj8Client(responses_tie)
    row_tie = asyncio.run(run_maj8_prose(c_tie, item))  # type: ignore[arg-type]
    assert row_tie["predicted"] == "88"
    assert row_tie["correct"] is False
    assert "4/8 (tie->low_temp)" in row_tie["detail"]

    # 3. --merge merging disjoint arm files: 5-arm 30/30 file + maj8_prose 30/30 file -> 6-arm merged JSON
    def _mk_row(idx: int, correct: bool, calls: int, tok: int, wall: float, detail: str = "") -> dict[str, Any]:
        exp = str((idx * 19) % 1000)
        return {
            "id": f"AIME25_{idx:02d}",
            "expected": exp,
            "predicted": exp if correct else "999",
            "correct": correct,
            "wall_s": wall,
            "llm_calls": calls,
            "prompt_tokens": tok // 3,
            "completion_tokens": tok - tok // 3,
            "reasoning_tokens": tok // 5,
            "cached_tokens": 0,
            "total_tokens": tok,
            "detail": detail or ("OK" if correct else "WRONG"),
        }

    five_arms = ["vanilla_b1", "maj4_prose", "pal_b1", "cortheon_w1", "cortheon_full"]
    five_res: dict[str, list[dict[str, Any]]] = {a: [] for a in five_arms}
    maj8_rows: list[dict[str, Any]] = []

    for idx in range(1, 31):
        # Craft non-nested outcomes so discordant pairs (n01, n10) are both exercised cleanly:
        # vanilla_b1: 1..15 (15/30)
        # maj4_prose: 1..16 (16/30)
        # maj8_prose: 1..14 + 25..26 (16/30) -> vs cortheon_full (1..24): n11=14, n01=10, n10=2, n00=4
        # pal_b1:     1..8 (8/30)
        # cortheon_w1: 1..20 (20/30)
        # cortheon_full: 1..24 (24/30)
        five_res["vanilla_b1"].append(_mk_row(idx, idx <= 15, 1, 2800, 45.0))
        five_res["maj4_prose"].append(_mk_row(idx, idx <= 16, 4, 6500, 75.0))
        pal_detail = (
            'pal_b1 | exec_err=File "/private/var/folders/4x/_abc123def456gn/T/ff_exec_zz99/_run.py", line 2 '
            'and /Users/exampleuser/Documents/secret/proj.py | empty'
            if idx == 9
            else (
                'pal_b1 | exec_err=...(truncated)...\nabc123def456gn/T/ff_exec_a_o6mlwn/_run.py", line 4 | empty'
                if idx == 10
                else "pal_b1 | ok"
            )
        )
        five_res["pal_b1"].append(_mk_row(idx, idx <= 8, 1, 1400, 20.0, detail=pal_detail))
        five_res["cortheon_w1"].append(_mk_row(idx, idx <= 20, 4, 6600, 63.0))
        five_res["cortheon_full"].append(_mk_row(idx, idx <= 24, 6, 14400, 148.0))
        maj8_rows.append(_mk_row(idx, (idx <= 14 or idx in (25, 26)), 8, 13800, 140.0))

    five_path = tmp_path / "official_five_arm_30.json"
    maj8_path = tmp_path / "aime25_maj8_prose_30.json"
    merged_path = tmp_path / "official_six_arm_30.json"

    # Write five_path with raw unscrubbed paths in results to verify --merge scrubs pre-existing files too
    five_payload_raw = build_ablation_payload(
        model="glm-5.3-flash-tfq3",
        arms=five_arms,
        results_by_arm=five_res,
        range_info={"start": 0, "end": 30, "shards": 2},
    )
    five_payload_raw["results"]["pal_b1"][8]["detail"] = (
        'pal_b1 | exec_err=File "/private/var/folders/4x/_abc123def456gn/T/ff_exec_zz99/_run.py", line 2 '
        "and /Users/exampleuser/Documents/secret/proj.py | empty"
    )
    five_payload_raw["results"]["pal_b1"][9]["detail"] = (
        'pal_b1 | exec_err=...(truncated)...\nabc123def456gn/T/ff_exec_a_o6mlwn/_run.py", line 4 | empty'
    )
    five_path.write_text(json.dumps(five_payload_raw, indent=2), encoding="utf-8")

    maj8_payload = build_ablation_payload(
        model="glm-5.3-flash-tfq3",
        arms=["maj8_prose"],
        results_by_arm={"maj8_prose": maj8_rows},
        range_info={"start": 0, "end": 30},
    )
    maj8_path.write_text(json.dumps(maj8_payload, indent=2), encoding="utf-8")

    rc = ablation_main(["--merge", str(five_path), str(maj8_path), "--out", str(merged_path)])
    assert rc == 0

    raw_text = merged_path.read_text(encoding="utf-8")
    merged = json.loads(raw_text)

    # Canonical 6-arm order preserved regardless of input file order
    assert merged["arms"] == list(ALL_ARMS)
    assert merged["model"] == "glm-5.3-flash-tfq3"
    assert merged["range"] == {"start": 0, "end": 30, "shards": 2}
    assert merged["merged_from"] == ["official_five_arm_30.json", "aime25_maj8_prose_30.json"]

    for arm in ALL_ARMS:
        assert len(merged["results"][arm]) == 30
        assert [r["id"] for r in merged["results"][arm]] == [f"AIME25_{i:02d}" for i in range(1, 31)]

    assert merged["summary"]["vanilla_b1"]["correct"] == 15
    assert merged["summary"]["maj4_prose"]["correct"] == 16
    assert merged["summary"]["maj8_prose"]["correct"] == 16
    assert merged["summary"]["pal_b1"]["correct"] == 8
    assert merged["summary"]["cortheon_w1"]["correct"] == 20
    assert merged["summary"]["cortheon_full"]["correct"] == 24

    # Exact paired McNemar stats against vanilla_b1, maj4_prose, AND maj8_prose!
    pw_v = merged["pairwise"]["cortheon_full_vs_vanilla_b1"]
    assert (pw_v["n11"], pw_v["n01"], pw_v["n10"], pw_v["n00"]) == (15, 9, 0, 6)
    assert math.isclose(pw_v["mcnemar_exact_p"], mcnemar_exact_p(9, 0), rel_tol=1e-12)

    pw_m4 = merged["pairwise"]["cortheon_full_vs_maj4_prose"]
    assert (pw_m4["n11"], pw_m4["n01"], pw_m4["n10"], pw_m4["n00"]) == (16, 8, 0, 6)
    assert math.isclose(pw_m4["mcnemar_exact_p"], mcnemar_exact_p(8, 0), rel_tol=1e-12)

    pw_m8 = merged["pairwise"]["cortheon_full_vs_maj8_prose"]
    assert (pw_m8["n11"], pw_m8["n01"], pw_m8["n10"], pw_m8["n00"]) == (14, 10, 2, 4)
    assert math.isclose(pw_m8["mcnemar_exact_p"], mcnemar_exact_p(10, 2), rel_tol=1e-12)
    assert math.isclose(pw_m8["token_ratio"], round(14400 / 13800, 3), rel_tol=1e-9)

    pw_m8_v = merged["pairwise"]["maj8_prose_vs_vanilla_b1"]
    assert (pw_m8_v["n11"], pw_m8_v["n01"], pw_m8_v["n10"], pw_m8_v["n00"]) == (14, 2, 1, 13)
    assert math.isclose(pw_m8_v["mcnemar_exact_p"], mcnemar_exact_p(2, 1), rel_tol=1e-12)

    # Also verify reverse-order and mixed (5-arm full + 2 maj8 problem shards) merge produce identical stats
    maj8_s1 = tmp_path / "maj8_0_15.json"
    maj8_s2 = tmp_path / "maj8_15_30.json"
    maj8_s1.write_text(
        json.dumps(
            build_ablation_payload(
                model="glm-5.3-flash-tfq3",
                arms=["maj8_prose"],
                results_by_arm={"maj8_prose": maj8_rows[:15]},
                range_info={"start": 0, "end": 15},
            )
        ),
        encoding="utf-8",
    )
    maj8_s2.write_text(
        json.dumps(
            build_ablation_payload(
                model="glm-5.3-flash-tfq3",
                arms=["maj8_prose"],
                results_by_arm={"maj8_prose": maj8_rows[15:]},
                range_info={"start": 15, "end": 30},
            )
        ),
        encoding="utf-8",
    )
    mixed = merge_ablation_files([maj8_s2, five_path, maj8_s1])
    assert mixed["arms"] == list(ALL_ARMS)
    assert mixed["pairwise"]["cortheon_full_vs_maj8_prose"] == pw_m8
    assert "McNemar p vs `maj8_prose`" in format_ablation_table(mixed)

    # 4. Verify ephemeral temp paths & /Users/... paths were scrubbed cleanly from written JSON
    assert "/private/var/folders" not in raw_text
    assert "/var/folders" not in raw_text
    assert "/Users/" not in raw_text
    assert "exampleuser" not in raw_text
    assert "ff_exec_" not in raw_text
    assert "<TMPDIR>/_run.py" in merged["results"]["pal_b1"][8]["detail"]
    assert "<HOME>" in merged["results"]["pal_b1"][8]["detail"]
    assert "<TMPDIR>/_run.py" in merged["results"]["pal_b1"][9]["detail"]

    # Direct unit checks on scrub_detail_paths edge cases
    assert scrub_detail_paths('File "/private/var/folders/4x/_q9b/T/ff_exec_12ab/_ru') == 'File "<TMPDIR>/_ru'
    assert scrub_detail_paths('File "/private/var/folders/4x/_q9b') == 'File "<TMPDIR>'
    assert scrub_detail_paths('File "/tmp/ff_exec_abc123/_run.py"') == 'File "<TMPDIR>/_run.py"'
    assert scrub_detail_paths("path=/home/alice/work/file.py:42") == "path=<HOME>:42"
    assert scrub_detail_paths("see https://example.com/home/docs") == "see https://example.com/home/docs"


def test_fetch_datasets_mocked_pagination_and_filtering(tmp_path: Path):
    # Synthetic responses for AIME (30 rows unsorted), BCB (2 pages with 30 pinned IDs + extras), LCB (JSONL)
    aime_rows = [
        {"row": {"problem_idx": idx, "problem": f"Problem {idx}", "answer": (idx * 13) % 1000, "problem_type": ["Algebra"]}}
        for idx in reversed(range(1, 31))
    ]
    bcb_all = [
        {"row": {"task_id": tid, "complete_prompt": "def f(): pass", "instruct_prompt": "Write f", "test": "assert True"}}
        for tid in BIGCODEBENCH_HARD_VERIFIED_IDS
    ] + [
        {"row": {"task_id": f"BigCodeBench/{9000 + i}", "complete_prompt": "", "instruct_prompt": "", "test": ""}}
        for i in range(90)
    ]
    lcb_lines = [
        json.dumps({"question_id": "abc387_a", "question_title": "Easy", "platform": "atcoder", "difficulty": "easy", "question_content": "..."}),
        json.dumps({"question_id": "lc_hard", "question_title": "LC", "platform": "leetcode", "difficulty": "hard", "question_content": "..."}),
        json.dumps({"question_id": "abc387_f", "question_title": "Hard1", "platform": "atcoder", "difficulty": "hard", "question_content": "Solve 1"}),
        json.dumps({"question_id": "abc388_g", "question_title": "Hard2", "platform": "atcoder", "difficulty": "hard", "question_content": "Solve 2"}),
    ]

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "datasets-server.huggingface.co":
            ds = request.url.params.get("dataset")
            offset = int(request.url.params.get("offset", "0"))
            length = int(request.url.params.get("length", "100"))
            if ds == "MathArena/aime_2025":
                return httpx.Response(200, json={"num_rows_total": 30, "rows": aime_rows[offset : offset + length]})
            if ds == "bigcode/bigcodebench-hard":
                return httpx.Response(200, json={"num_rows_total": len(bcb_all), "rows": bcb_all[offset : offset + length]})
        if "test6.jsonl" in str(request.url):
            return httpx.Response(200, text="\n".join(lcb_lines) + "\n")
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        aime_path = fetch_aime2025(tmp_path, client=client, force=True)
        bcb_path = fetch_bigcodebench(tmp_path, client=client, force=True)
        lcb_path = fetch_livecodebench(tmp_path, client=client, force=True)

    aime_data = json.loads(aime_path.read_text(encoding="utf-8"))
    assert len(aime_data) == 30
    assert [r["problem_idx"] for r in aime_data] == list(range(1, 31))

    bcb_data = json.loads(bcb_path.read_text(encoding="utf-8"))
    assert [r["task_id"] for r in bcb_data] == BIGCODEBENCH_HARD_VERIFIED_IDS

    lcb_data = json.loads(lcb_path.read_text(encoding="utf-8"))
    assert [r["question_id"] for r in lcb_data] == ["abc387_f", "abc388_g"]
