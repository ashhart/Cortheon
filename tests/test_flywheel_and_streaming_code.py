"""Offline unit tests for the Self-Play Verification Flywheel and CodeForge v6.1 speedups."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flash_forge.client import CallSpec, CompletionResult
from flash_forge.code_forge import (
    CodeCandidate,
    _has_identical_behavioral_fingerprint,
    solve_code,
)
from flash_forge.flywheel import (
    FlywheelPair,
    export_dpo_jsonl,
    export_sft_jsonl,
    format_export_summary,
    harvest_from_results_dir,
    record_live_trace,
    scrub_pii,
)
from flash_forge.memory import EpisodicMemory
from flash_forge.skill_loader import SkillPack


# ===========================================================================
# Part 1: Self-Play Verification Flywheel (`flash_forge/flywheel.py`)
# ===========================================================================


def test_pii_scrubbing_and_record_live_trace_atomic_append(tmp_path: Path):
    # Direct scrubber checks including non-PII preservation controls
    raw = (
        "Loaded /Users/alice/Projects/secret/app.py and /home/bob/work/run.sh; "
        "sandbox /private/var/folders/ab/cdef1234/T/ff_exec_99/solution.py "
        "called http://192.168.1.5:8080/v1, 10.0.0.42, 172.16.8.9 with sk-abcdefghijklmnopqrstuvwxyz123456; "
        "safe tokens: 10.5, v10.0.1, 127.0.0.1:8080"
    )
    cleaned = scrub_pii(raw)
    assert "alice" not in cleaned
    assert "/Users/" not in cleaned
    assert "/home/bob" not in cleaned
    assert "/private/var/folders/" not in cleaned
    assert "192.168.1.5" not in cleaned
    assert "10.0.0.42" not in cleaned
    assert "172.16.8.9" not in cleaned
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in cleaned
    assert "<HOME>/Projects/secret/app.py" in cleaned
    assert "<TMPDIR>" in cleaned
    assert "<PRIVATE_IP>" in cleaned
    assert "<REDACTED_SECRET>" in cleaned
    # Controls: normal floats, 3-part versions, and loopback stay intact
    assert "10.5" in cleaned
    assert "v10.0.1" in cleaned
    assert "127.0.0.1:8080" in cleaned

    log_path = tmp_path / "nested" / "flywheel.jsonl"
    out1 = record_live_trace(
        log_path,
        prompt="Fix bug in /Users/alice/repo/calc.py hitting 192.168.1.5",
        chosen="def fix():\n    return 42\n",
        rejected="def fix():\n    raise RuntimeError('/private/var/folders/x/y/T/tmp1')\n",
        domain="code",
        model="local-qwen",
        metadata={"host": "10.1.2.3", "workdir": "/Users/alice/repo"},
    )
    assert out1 == log_path
    assert log_path.is_file()

    # Concurrent atomic appends must produce clean, uncorrupted JSON lines
    def _append(i: int) -> None:
        record_live_trace(
            log_path,
            prompt=f"Compute sum up to {i} on 10.0.0.{i}",
            chosen=f"ANSWER: {i * (i + 1) // 2}",
            rejected=f"ANSWER: {i}",
            domain="math",
            model="local-glm",
            metadata={"idx": i, "path": f"/Users/alice/run_{i}.txt"},
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_append, range(1, 11)))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 11
    for line in lines:
        rec = json.loads(line)
        assert "timestamp" in rec
        dumped = json.dumps(rec)
        assert "alice" not in dumped
        assert "/Users/" not in dumped
        assert "/private/var/folders/" not in dumped
        assert "192.168.1.5" not in dumped
        assert "10.1.2.3" not in dumped
        assert "10.0.0." not in dumped


def test_harvest_from_results_dir_paired_ablation_live_and_datasets(tmp_path: Path):
    results_dir = tmp_path / "results"
    datasets_dir = tmp_path / "datasets"
    results_dir.mkdir()
    datasets_dir.mkdir()

    # Synthetic datasets for prompt joining
    (datasets_dir / "aime_2025.json").write_text(
        json.dumps(
            [
                {"problem_idx": 1, "problem": "Find 1 + 1.", "answer": 2},
                {"problem_idx": 2, "problem": "Find the area of heptagon AFNBCEM.", "answer": 588},
                {"problem_idx": 3, "problem": "Find remainder of N mod 1000.", "answer": 16},
            ]
        ),
        encoding="utf-8",
    )
    (datasets_dir / "livecodebench_v6_hard.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "abc387_f",
                    "question_title": "Count Arrays",
                    "question_content": "Given N, M and A, count valid arrays modulo 998244353.",
                }
            ]
        ),
        encoding="utf-8",
    )

    # 1. Paired *_vanilla.json + *_cortheon.json (AIME math)
    (results_dir / "aime_glm_vanilla.json").write_text(
        json.dumps(
            {
                "dataset": "MathArena/aime_2025",
                "model": "glm-5.3-flash",
                "arm": "vanilla",
                "results": [
                    {"pid": "AIME25_01", "passed": True, "detail": "OK (2)"},
                    {"pid": "AIME25_02", "passed": False, "detail": "WRONG: got='720' expected='588'"},
                    {"pid": "AIME25_03", "passed": True, "detail": "OK (16)"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "aime_glm_cortheon.json").write_text(
        json.dumps(
            {
                "dataset": "MathArena/aime_2025",
                "model": "glm-5.3-flash",
                "arm": "cortheon",
                "results": [
                    {"pid": "AIME25_01", "passed": True, "detail": "OK (2)"},
                    {"pid": "AIME25_02", "passed": True, "detail": "OK (588)"},
                    {"pid": "AIME25_03", "passed": False, "detail": "WRONG: got='99' expected='16'"},
                ],
            }
        ),
        encoding="utf-8",
    )

    # 2. Paired code JSONs with full code traces
    (results_dir / "lcb_glm_vanilla.json").write_text(
        json.dumps(
            {
                "dataset": "LiveCodeBench v6 Hard",
                "model": "glm-5.3-flash",
                "arm": "vanilla",
                "results": [
                    {
                        "pid": "abc387_f",
                        "passed": False,
                        "title": "Count Arrays",
                        "detail": "WA on case 3/43",
                        "code": "print(0)\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "lcb_glm_cortheon.json").write_text(
        json.dumps(
            {
                "dataset": "LiveCodeBench v6 Hard",
                "model": "glm-5.3-flash",
                "arm": "cortheon",
                "results": [
                    {
                        "pid": "abc387_f",
                        "passed": True,
                        "title": "Count Arrays",
                        "detail": "AC 43/43 (1100ms)",
                        "code": "import sys\nprint(2025)\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # 3. Multi-arm 6-arm ablation JSON (vanilla_b1 + maj4_prose dup + pal_b1 distinct + cortheon_w1 Wave-1->Wave-2 flip)
    (results_dir / "official_aime2025_ablation_glm53.json").write_text(
        json.dumps(
            {
                "dataset": "MathArena/aime_2025",
                "model": "glm-5.3-flash",
                "arms": ["vanilla_b1", "maj4_prose", "pal_b1", "cortheon_w1", "cortheon_full"],
                "results": {
                    "vanilla_b1": [
                        {"id": "AIME25_02", "expected": "588", "predicted": "720", "correct": False, "detail": "WRONG"},
                        {"id": "AIME25_03", "expected": "16", "predicted": "11", "correct": False, "detail": "WRONG"},
                    ],
                    "maj4_prose": [
                        {"id": "AIME25_02", "expected": "588", "predicted": "720", "correct": False, "detail": "WRONG"},
                        {"id": "AIME25_03", "expected": "16", "predicted": "12", "correct": False, "detail": "WRONG"},
                    ],
                    "pal_b1": [
                        {"id": "AIME25_02", "expected": "588", "predicted": "999", "correct": False, "detail": "WRONG"},
                        {"id": "AIME25_03", "expected": "16", "predicted": "13", "correct": False, "detail": "WRONG"},
                    ],
                    "cortheon_w1": [
                        {"id": "AIME25_02", "expected": "588", "predicted": "580", "correct": False, "detail": "W1 WRONG"},
                        {"id": "AIME25_03", "expected": "16", "predicted": "14", "correct": False, "detail": "W1 WRONG"},
                    ],
                    "cortheon_full": [
                        {"id": "AIME25_02", "expected": "588", "predicted": "588", "correct": True, "detail": "OK (588)"},
                        {"id": "AIME25_03", "expected": "16", "predicted": "15", "correct": False, "detail": "WRONG"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    # 4. Live JSONL trace log with PII
    record_live_trace(
        results_dir / "flywheel.jsonl",
        prompt="Debug parser in /Users/alice/src/parse.py on 192.168.1.5",
        chosen="def parse(s): return s.strip()",
        rejected="def parse(s): return s",
        domain="code",
        model="qwen-3.8",
    )

    # 5. Non-result summary JSON must be ignored gracefully
    (results_dir / "summary.json").write_text(json.dumps({"benchmarks": []}), encoding="utf-8")

    pairs = harvest_from_results_dir(results_dir, datasets_dir=datasets_dir)

    # Expected:
    # - 1 from flywheel.jsonl
    # - 1 from aime_glm paired (AIME25_02; AIME25_01 both passed; AIME25_03 cortheon failed)
    # - 1 from lcb_glm paired (abc387_f with full code!)
    # - 3 from ablation (AIME25_02: vanilla_b1=720, cortheon_w1=580, pal_b1=999; maj4_prose=720 collapsed; AIME25_03 failed)
    assert len(pairs) == 6

    # Prompt joining worked for AIME25_02 and abc387_f
    aime_pairs = [p for p in pairs if "heptagon AFNBCEM" in p.prompt]
    assert len(aime_pairs) == 4  # 1 paired + 3 ablation (one is exact duplicate of paired, deduped on export!)
    assert any(p.metadata.get("flip_type") == "wave1_to_wave2" and p.rejected == "ANSWER: 580" for p in aime_pairs)
    assert any(p.rejected == "ANSWER: 999" for p in aime_pairs)

    lcb_pairs = [p for p in pairs if p.metadata.get("pid") == "abc387_f"]
    assert len(lcb_pairs) == 1
    assert "Count Arrays" in lcb_pairs[0].prompt
    assert lcb_pairs[0].chosen == "import sys\nprint(2025)"
    assert lcb_pairs[0].rejected == "print(0)"
    assert lcb_pairs[0].domain == "code"

    # Export DPO deduplicates the identical (heptagon, 588, 720) across paired + ablation!
    dpo_path = tmp_path / "out_dpo.jsonl"
    dpo_stats = export_dpo_jsonl(pairs, dpo_path)
    assert dpo_stats.total_input == 6
    assert dpo_stats.exported == 5
    assert dpo_stats.deduplicated == 1


def test_export_dpo_and_sft_jsonl_structure_dedup_and_summary(tmp_path: Path):
    pairs = [
        FlywheelPair(
            prompt="Solve /Users/alice/math.txt with server 192.168.1.5",
            chosen="ANSWER: 42",
            rejected="ANSWER: 41",
            domain="math",
            source="aime_ablation",
            model="glm-5.3",
            chosen_verified=True,
            rejected_reason="off by one",
            metadata={"tmp": "/private/var/folders/aa/bb/T/ff_1"},
        ),
        # Exact duplicate after PII scrubbing -> should be deduplicated in BOTH DPO and SFT
        FlywheelPair(
            prompt="Solve /Users/bob/math.txt with server 10.0.0.8",
            chosen="ANSWER: 42",
            rejected="ANSWER: 41",
            domain="math",
            source="aime_paired",
            model="glm-5.3",
            chosen_verified=True,
        ),
        # Same (prompt, chosen) with a DIFFERENT hard negative -> kept in DPO, deduplicated in SFT!
        FlywheelPair(
            prompt="Solve /Users/alice/math.txt with server 192.168.1.5",
            chosen="ANSWER: 42",
            rejected="ANSWER: 99",
            domain="math",
            source="aime_pal",
            model="glm-5.3",
            chosen_verified=True,
        ),
        # Verified SFT-only trace (empty rejected) -> skipped in DPO, exported in SFT
        FlywheelPair(
            prompt="Write binary search in Python.",
            chosen="import bisect\ndef bsearch(a, x): return bisect.bisect_left(a, x)",
            rejected="",
            domain="code",
            source="live_trace",
            model="qwen-3.8",
            chosen_verified=True,
        ),
        # Unverified trace -> skipped in BOTH DPO and SFT
        FlywheelPair(
            prompt="Unverified problem",
            chosen="maybe 7",
            rejected="maybe 8",
            domain="logic",
            source="unverified",
            model="qwen-3.8",
            chosen_verified=False,
        ),
        # Degenerate identical chosen==rejected -> skipped in DPO
        FlywheelPair(
            prompt="Degenerate",
            chosen="same",
            rejected="same",
            domain="code",
            source="dup",
            model="qwen-3.8",
            chosen_verified=True,
        ),
    ]

    dpo_file = tmp_path / "train_dpo.jsonl"
    sft_file = tmp_path / "train_sft.jsonl"

    dpo_stats = export_dpo_jsonl(pairs, dpo_file)
    sft_stats = export_sft_jsonl(pairs, sft_file)

    assert dpo_stats.total_input == 6
    assert dpo_stats.exported == 2
    assert dpo_stats.deduplicated == 1
    assert dpo_stats.skipped_unverified == 1
    assert dpo_stats.skipped_empty == 2
    assert dpo_stats.domain_counts == {"math": 2}

    dpo_rows = [json.loads(ln) for ln in dpo_file.read_text(encoding="utf-8").splitlines()]
    assert len(dpo_rows) == 2
    for row in dpo_rows:
        assert set(row.keys()) == {"prompt", "chosen", "rejected", "domain", "source"}
        assert "alice" not in row["prompt"]
        assert "bob" not in row["prompt"]
        assert "192.168.1.5" not in row["prompt"]
        assert "10.0.0.8" not in row["prompt"]
        assert "<HOME>/math.txt" in row["prompt"]
        assert "<PRIVATE_IP>" in row["prompt"]

    assert sft_stats.total_input == 6
    assert sft_stats.exported == 3  # math(42), code(bsearch), code(Degenerate)
    assert sft_stats.deduplicated == 2  # duplicate(bob) + second math hard-negative share (prompt, chosen)
    assert sft_stats.skipped_unverified == 1
    assert sft_stats.skipped_empty == 0
    assert sft_stats.domain_counts == {"code": 2, "math": 1}

    sft_rows = [json.loads(ln) for ln in sft_file.read_text(encoding="utf-8").splitlines()]
    assert len(sft_rows) == 3
    for row in sft_rows:
        assert list(row.keys()) == ["messages"]
        msgs = row["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user" and msgs[0]["content"]
        assert msgs[1]["role"] == "assistant" and msgs[1]["content"]
        assert "alice" not in msgs[0]["content"]

    dpo_summary = format_export_summary(dpo_stats)
    sft_summary = format_export_summary(sft_stats)
    assert "Exported DPO dataset" in dpo_summary
    assert "Harvested" in dpo_summary and "Deduplicated" in dpo_summary
    assert "math=2" in dpo_summary
    assert "mlx_lm.lora" in dpo_summary and "trl dpo" in dpo_summary

    assert "Exported SFT dataset" in sft_summary
    assert "code=2, math=1" in sft_summary
    assert "mlx_lm.lora" in sft_summary and "trl sft" in sft_summary


# ===========================================================================
# Part 2: CodeForge v6.1 Fast Unanimous Exit & Behavioral-Equivalence Tie-Break
# ===========================================================================


def _mk_res(content: str, tag: str = "") -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=60,
        reasoning_tokens=10,
        cached_tokens=0,
        latency_s=0.01,
        tag=tag,
    )


class _ScriptedCodeClient:
    """Offline mock TensorFoldClient tracking every LLM call tag and pairwise vote."""

    def __init__(self, responses_by_tag: dict[str, str], pairwise_answer: str = "A") -> None:
        self.responses_by_tag = responses_by_tag
        self.pairwise_answer = pairwise_answer
        self.call_tags: list[str] = []
        self.pairwise_calls: int = 0

    @property
    def total_llm_calls(self) -> int:
        # Each pairwise_vote executes 2 order-swapped LLM calls in production
        return len(self.call_tags) + 2 * self.pairwise_calls

    async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
        out: list[CompletionResult] = []
        for s in specs:
            self.call_tags.append(s.tag)
            reply = self.responses_by_tag.get(s.tag)
            if reply is None:
                # Prefix fallback for arbiters/auditors
                for prefix, val in self.responses_by_tag.items():
                    if prefix.endswith("*") and s.tag.startswith(prefix[:-1]):
                        reply = val
                        break
            if reply is None:
                reply = "VERDICT: PASS"
            out.append(_mk_res(reply, tag=s.tag))
        return out

    async def pairwise_vote(self, prompt_ab: str, prompt_ba: str, *, tag_prefix: str = "judge") -> str:
        self.pairwise_calls += 1
        return self.pairwise_answer


_STDIN_DOUBLE_PROBLEM = (
    "Read a single integer N from standard input and print 2 * N to standard output.\n\n"
    "### Sample Input 1\n```\n3\n```\n"
    "### Sample Output 1\n```\n6\n```\n\n"
    "### Sample Input 2\n```\n10\n```\n"
    "### Sample Output 2\n```\n20\n```\n"
)


def test_code_forge_v61_stdin_unanimous_fastpath_skips_audit_redteam_wave2_and_judge(tmp_path: Path):
    client = _ScriptedCodeClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": (
                "```python\nimport sys\n"
                "data = sys.stdin.read().split()\n"
                "n = int(data[0])\n"
                "print(n + n)\n```"
            ),
            "w1_edge_skeptic": (
                "```python\nimport sys\n"
                "val = int(sys.stdin.readline())\n"
                "print(2 * val)\n```"
            ),
            "w1_spec_oracle": (
                "```python\n"
                "def test_zero_and_neg():\n"
                "    assert run_solution('0\\n') == '0'\n"
                "    assert run_solution('-5\\n') == '-10'\n\n"
                "def diff_probe():\n"
                "    return {\n"
                "        'small': run_solution('7\\n'),\n"
                "        'large': run_solution('123456\\n'),\n"
                "    }\n"
                "```"
            ),
        }
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem.db")
    skills = SkillPack()
    trace = asyncio.run(solve_code(client, skills, mem, _STDIN_DOUBLE_PROBLEM, num_wave1_streams=4))  # type: ignore[arg-type]
    mem.close()

    assert trace.verified is True
    assert trace.unanimous_fastpath is True
    assert trace.wave2_triggered is False
    assert trace.redteam_tests_generated == 0
    assert client.pairwise_calls == 0
    assert trace.pairwise_tiebreak_skipped >= 1
    assert client.total_llm_calls <= 4
    assert client.call_tags == [
        "w1_radically_simple",
        "w1_first_principles",
        "w1_edge_skeptic",
        "w1_spec_oracle",
    ]
    assert not any(t.startswith("spec_audit") for t in client.call_tags)
    assert not any(t.startswith("s2_redteam") for t in client.call_tags)
    assert not any(t.startswith("w2_") for t in client.call_tags)


def test_code_forge_v61_stdin_divergence_and_failure_still_trigger_redteam_and_wave2(tmp_path: Path):
    # Control A: Candidates pass samples (3->6, 10->20) but DIVERGE on diff_probe (100)!
    # Must NOT fast-exit; MUST invoke Stage-2 red-team!
    div_client = _ScriptedCodeClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": (
                "```python\nimport sys\n"
                "n = int(sys.stdin.read().strip())\n"
                "print(n * 2 if n <= 50 else n * 2 + 1)\n```"
            ),
            "w1_edge_skeptic": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 3)\n```",
            "w1_spec_oracle": (
                "```python\n"
                "def diff_probe():\n"
                "    return {\n"
                "        'small': run_solution('7\\n'),\n"
                "        'large': run_solution('100\\n'),\n"
                "    }\n"
                "```"
            ),
            "s2_redteam_edge": (
                "```python\n"
                "def test_large_input():\n"
                "    assert run_solution('100\\n') == '200'\n"
                "```"
            ),
            "arbiter*": "Spec requires 2*N for all N.\nVERDICT: VALID",
        }
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_div.db")
    skills = SkillPack()
    trace_div = asyncio.run(solve_code(div_client, skills, mem, _STDIN_DOUBLE_PROBLEM, num_wave1_streams=4))  # type: ignore[arg-type]
    mem.close()

    assert trace_div.unanimous_fastpath is False
    assert "s2_redteam_edge" in div_client.call_tags
    assert trace_div.redteam_tests_generated >= 1
    assert trace_div.verified is True
    assert trace_div.winner_name == "radically_simple"

    # Control B: All Wave-1 candidates fail pinned sample tests -> MUST trigger red-team + Wave 2 repair!
    fail_client = _ScriptedCodeClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 3)\n```",
            "w1_first_principles": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) + 3)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(0)\n```",
            "w1_spec_oracle": "```python\ndef test_extra():\n    assert run_solution('4\\n') == '8'\n```",
            "s2_redteam_edge": "```python\ndef test_rt():\n    assert run_solution('5\\n') == '10'\n```",
            "arbiter*": "Valid.\nVERDICT: VALID",
            "w2_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w2_surgical_fix": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w2_clean_rewrite": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
        }
    )
    mem2 = EpisodicMemory(db_path=tmp_path / "mem_fail.db")
    trace_fail = asyncio.run(solve_code(fail_client, skills, mem2, _STDIN_DOUBLE_PROBLEM, num_wave1_streams=4))  # type: ignore[arg-type]
    mem2.close()

    assert trace_fail.unanimous_fastpath is False
    assert trace_fail.wave2_triggered is True
    assert trace_fail.verified is True
    assert trace_fail.recovered_by_redteam_or_wave2 is True
    assert trace_fail.winner_name.startswith("w2_")


def test_code_forge_v61_behavioral_fingerprint_tiebreak_and_non_stdin_audit(tmp_path: Path):
    # Direct unit tests on `_has_identical_behavioral_fingerprint`
    c_a = CodeCandidate(name="a", wave=1, code="x=1", raw_content="", diff_outputs={"k1": "10", "k2": "20"})
    c_b = CodeCandidate(name="b", wave=1, code="x=1\n# long", raw_content="", diff_outputs={"k1": "10", "k2": "20"})
    c_tmp1 = CodeCandidate(
        name="t1", wave=1, code="x=1", raw_content="", diff_outputs={"k1": "'/tmp/tmpAAA/a.csv'", "k2": "20"}
    )
    c_tmp2 = CodeCandidate(
        name="t2", wave=1, code="x=2", raw_content="", diff_outputs={"k1": "'/tmp/tmpBBB/a.csv'", "k2": "20"}
    )
    c_diff = CodeCandidate(name="d", wave=1, code="x=3", raw_content="", diff_outputs={"k1": "10", "k2": "99"})
    c_one = CodeCandidate(name="one", wave=1, code="x=4", raw_content="", diff_outputs={"k1": "10"})
    c_exc = CodeCandidate(
        name="exc", wave=1, code="x=5", raw_content="", diff_outputs={"k1": "10", "__probe_exc__": "RuntimeError"}
    )
    c_crash = CodeCandidate(
        name="cr", wave=1, code="x=6", raw_content="", diff_outputs={"k1": "'EXC:NameError'", "k2": "'EXC:NameError'"}
    )

    assert _has_identical_behavioral_fingerprint(c_a, c_b) is True
    assert _has_identical_behavioral_fingerprint(c_tmp1, c_tmp2) is True
    assert _has_identical_behavioral_fingerprint(c_a, c_diff) is False
    assert _has_identical_behavioral_fingerprint(c_one, c_one) is False
    assert _has_identical_behavioral_fingerprint(c_exc, c_exc) is False
    assert _has_identical_behavioral_fingerprint(c_crash, c_crash) is False

    # End-to-end non-stdin library spec:
    # 1. `_spec_compliance_audit` MUST STILL run on non-stdin API specs!
    # 2. Identical diff_outputs across >=2 keys MUST skip redundant `pairwise_vote`!
    lib_problem = "Implement `def double_val(x: int) -> int:` returning `2 * x`."
    compact_sol = "```python\ndef double_val(x: int) -> int:\n    return x * 2\n```"
    verbose_sol = (
        "```python\ndef double_val(x: int) -> int:\n"
        "    # Verbose equivalent implementation\n"
        "    result = x + x\n"
        "    return result\n```"
    )
    oracle_with_diff = (
        "```python\n"
        "def test_pos(): assert double_val(3) == 6\n"
        "def test_zero(): assert double_val(0) == 0\n"
        "def test_neg(): assert double_val(-4) == -8\n"
        "def diff_probe():\n"
        "    return {'a': double_val(1), 'b': double_val(5), 'c': double_val(-9)}\n"
        "```"
    )
    client_eq = _ScriptedCodeClient(
        {
            "w1_radically_simple": compact_sol,
            "w1_first_principles": verbose_sol,
            "w1_edge_skeptic": verbose_sol + "\n# v3",
            "w1_spec_oracle": oracle_with_diff,
            "spec_audit_edge": "All clauses met.\nVERDICT: PASS",
            "spec_audit_api": "Clean API.\nVERDICT: PASS",
        },
        pairwise_answer="B",
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_lib_eq.db")
    skills = SkillPack()
    trace_eq = asyncio.run(solve_code(client_eq, skills, mem, lib_problem, num_wave1_streams=4))  # type: ignore[arg-type]
    mem.close()

    assert trace_eq.verified is True
    assert trace_eq.unanimous_fastpath is True
    # Non-stdin MUST still invoke spec_compliance_audit!
    assert "spec_audit_edge" in client_eq.call_tags
    assert "spec_audit_api" in client_eq.call_tags
    # Identical behavioral fingerprints across 3 diff cases -> skip pairwise_vote and keep compact leader!
    assert client_eq.pairwise_calls == 0
    assert trace_eq.pairwise_tiebreak_skipped >= 1
    assert trace_eq.winner_name == "radically_simple"

    # Control: WITHOUT diff_outputs (empty diff_probe), pairwise_vote MUST still be called!
    oracle_no_diff = (
        "```python\n"
        "def test_pos(): assert double_val(3) == 6\n"
        "def test_zero(): assert double_val(0) == 0\n"
        "```"
    )
    client_no_diff = _ScriptedCodeClient(
        {
            "w1_radically_simple": compact_sol,
            "w1_first_principles": verbose_sol,
            "w1_edge_skeptic": compact_sol,
            "w1_spec_oracle": oracle_no_diff,
            "s2_redteam_edge": "```python\ndef test_rt(): assert double_val(9) == 18\n```",
        },
        pairwise_answer="B",
    )
    mem2 = EpisodicMemory(db_path=tmp_path / "mem_lib_nodiff.db")
    trace_nd = asyncio.run(solve_code(client_no_diff, skills, mem2, lib_problem, num_wave1_streams=4))  # type: ignore[arg-type]
    mem2.close()

    assert trace_nd.verified is True
    assert client_no_diff.pairwise_calls == 1
    assert trace_nd.pairwise_tiebreak_skipped == 0
    assert trace_nd.winner_name == "first_principles"


def test_code_forge_v61_stdin_two_of_three_passing_fast_exit_and_pinned_samples_only(tmp_path: Path):
    # Case 1: Only 2 of 3 Wave-1 candidates pass (3rd fails), 0 divergences across passing pair -> fast-exit in 4 calls!
    client_2of3 = _ScriptedCodeClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(n + n)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 999)\n```",
            "w1_spec_oracle": (
                "```python\n"
                "def diff_probe():\n"
                "    return {'a': run_solution('7\\n'), 'b': run_solution('15\\n')}\n"
                "```"
            ),
        }
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_2of3.db")
    skills = SkillPack()
    trace_2of3 = asyncio.run(solve_code(client_2of3, skills, mem, _STDIN_DOUBLE_PROBLEM, num_wave1_streams=4))  # type: ignore[arg-type]
    mem.close()

    assert trace_2of3.verified is True
    assert trace_2of3.unanimous_fastpath is True
    assert trace_2of3.wave2_triggered is False
    assert trace_2of3.redteam_tests_generated == 0
    assert client_2of3.pairwise_calls == 0
    assert client_2of3.total_llm_calls <= 4

    # Case 2: Pinned samples >= 2 with NO diff_probe -> still skips spec_audit, red-team, & Wave 2!
    client_pinned_only = _ScriptedCodeClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(n + n)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 999)\n```",
            "w1_spec_oracle": "```python\ndef test_extra():\n    assert run_solution('4\\n') == '8'\n```",
        },
        pairwise_answer="A",
    )
    mem2 = EpisodicMemory(db_path=tmp_path / "mem_pinned.db")
    trace_po = asyncio.run(solve_code(client_pinned_only, skills, mem2, _STDIN_DOUBLE_PROBLEM, num_wave1_streams=4))  # type: ignore[arg-type]
    mem2.close()

    assert trace_po.verified is True
    assert trace_po.unanimous_fastpath is True
    assert trace_po.wave2_triggered is False
    assert trace_po.redteam_tests_generated == 0
    assert not any(t.startswith("spec_audit") for t in client_pinned_only.call_tags)
    assert not any(t.startswith("s2_redteam") for t in client_pinned_only.call_tags)
    # Without diff_outputs, pairwise tie-break still runs once for the two distinct passing implementations
    assert client_pinned_only.pairwise_calls == 1

