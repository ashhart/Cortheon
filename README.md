# Cortheon v2 — Execution-Verified Consensus Runtime for Frozen Local Models

**Zero fine-tuning. Zero weight changes. Works on TensorFold, Ollama, llama.cpp, vLLM, and LM Studio.**

Cortheon lifts small, quantized local LLMs to frontier-level math, oracle-free coding, open-ended systems architecture, and autonomous engineering through one universal principle:

> **Parallel Diverse Execution → Synthesized Mechanical Verification & Behavioral Differential Consensus → Targeted Falsification / Repair ONLY on Disagreement or Failure.**

Even when the user provides **no test suite and no math answer key**, Cortheon synthesizes its own mechanical verifier on the fly — executable Python/SymPy/Z3 solvers (`ReasonForge`), Behavioral Differential Consensus + Starter-API Fidelity + Adversarial Red-Teaming (`CodeForge v5`), and auto-grounded Python physics/sizing/invariant calculators (`ThinkForge`).

---

## Live Measured Results (Three Frozen Local Model Families — Zero Weight Changes)

> **Full Benchmark Ledger & Per-Problem Result JSONs:** See **[`BENCHMARKS.md`](BENCHMARKS.md)** and **[`results/`](results/)** ([`results/summary.json`](results/summary.json)) for the complete per-problem logs, flips/regressions, calibration tables, and exact reproduction commands across every suite.

Every score below was measured on **100% pure local frozen weights** running on Apple Silicon (M5 Ultra & M3 Ultra, `enable_memory=False`, `force_mode=None`, zero cloud solver models) across three completely distinct model families: **`DeepSeek-V4.1-Flash-1017`**, **`GLM-5.3-Flash-TFQ3`**, and **`Qwen3.8-Flash-Next-TFQ3`**.

### 1. Official Untouched External Benchmarks (**8 for 8 Wins — 200 Official Runs Across 3 Model Families**)

Executed apples-to-apples via [`bench/run_official_eval.py`](bench/run_official_eval.py) and [`bench/run_livecodebench.py`](bench/run_livecodebench.py):

| Official Benchmark | Oracle at Solve Time? | Frozen Local Model | Vanilla | **Cortheon v2** | Delta | Avg Wall (Vanilla → Cortheon) |
|---|---|---|---:|---:|---:|---:|
| **`MathArena/aime_2025`** *(30/30)* | Synthesized PAL/SymPy/Z3 | **`DeepSeek-V4.1-Flash-1017`** | 20/30 (66.7%) | **27/30 (90.0%)** | **+23.3 pts** *(+7 problems)* | `113.7s` → `117.6s` |
| **`MathArena/aime_2025`** *(30/30)* | Synthesized PAL/SymPy/Z3 | **`GLM-5.3-Flash-TFQ3`** | 15/30 (50.0%) | **24/30 (80.0%)** | **+30.0 pts** *(+60% rel, +10 flips)* | `125.5s` → `175.8s` |
| **`MathArena/aime_2025`** *(30/30)* | Synthesized PAL/SymPy/Z3 | **`Qwen3.8-Flash-Next-TFQ3`** | 16/30 (53.3%) | **21/30 (70.0%)** | **+16.7 pts** *(0 regressions)* | `69.2s` → `80.5s` |
| **`LiveCodeBench v6 Hard`** *(20/20 AtCoder F/G)* | **Oracle-Free** *(prose only)* | **`DeepSeek-V4.1-Flash-1017`** | 11/20 (55.0%) | **13/20 (65.0%)** | **+10.0 pts** *(0 regressions)* | `73.6s` → `201.2s` |
| **`LiveCodeBench v6 Hard`** *(20/20 AtCoder F/G)* | **Oracle-Free** *(prose only)* | **`GLM-5.3-Flash-TFQ3`** | 8/20 (40.0%) | **11/20 (55.0%)** | **+15.0 pts** *(+37.5% rel)* | `124.7s` → `249.1s` |
| **`BigCodeBench-Hard v0.1.4`** *(30/30 complete)* | **Oracle-Free** *(docstring only)* | **`DeepSeek-V4.1-Flash-1017`** | 10/30 (33.3%) | **11/30 (36.7%)** | **+3.3 pts** | `44.4s` → `118.2s` |
| **`BigCodeBench-Hard v0.1.4`** *(30/30 instruct)* | **Oracle-Free** *(prompt only)* | **`Qwen3.8-Flash-Next-TFQ3`** | 6/30 (20.0%) | **8/30 (26.7%)** | **+6.7 pts** *(+33% rel)* | `57.9s` → `85.4s` |
| **`BigCodeBench-Hard v0.1.4`** *(Acid-10, v5.1)* | **Oracle-Free** *(docstring only)* | **`GLM-5.3-Flash-TFQ3`** | 1/10 (10.0%) | **2/10 (20.0%)** | **+10.0 pts** *(2× Vanilla, 0 regr)* | `17.2s` → `45.9s` |
| **Combined Official External (200 runs)** | **All 3 Model Families** | **DeepSeek + GLM-5.3 + Qwen** | **87/200 (43.5%)** | **117/200 (58.5%)** | **+15.0 pts (+34.5% rel)** | — |

- **`50 / 51 (98.0%)` Wave-1 PAL Consensus Calibration Across All 3 Model Families on AIME 2025:** `19/19 (100%)` on `DeepSeek-V4.1-Flash` + `18/18 (100%)` on `Qwen3.8-Flash-Next` + `13/14 (92.9%)` on `GLM-5.3-Flash` — fast-exiting cleanly without wasting Wave-2 compute!
- **`7.5×` Wave-1 Early-Abort Speedup (`271s → 36s`):** Combining `low`-effort PAL slots (`tb=500`), 20s exact-math sandbox budgets, atomic straggler cancellation the instant 2 independent PAL scripts agree, and `--parallel 8` lane headroom cut a 25-digit exact rational 7-server asymmetric Markov cover-time solve (`2630600114653531854763423 / 183488738276457059168800`) from `271s` to **`36s`** (`wave1_consensus (2 PAL)`, `queued=0.000s`).
- **Decisive Oracle-Free Algorithmic Lift (`LiveCodeBench v6 Hard`):** Across **40** hard AtCoder F/G competition problems (~43 judge tests each, zero hidden tests at solve time), Cortheon lifted `DeepSeek-V4.1` `55% → 65%` and `GLM-5.3` `40% → 55%` (**`+12.5 pts` combined**) by deterministically extracting verbatim sample I/O + synthesizing brute-force fuzzers, TLE stress tests, and zero-anchor Wave-2 rewrites.
- **Oracle-Free Open-Ended Systems, Physics & Security ([`bench/open_ended_bench.py`](bench/open_ended_bench.py), 10 tasks on `GLM-5.3-Flash-TFQ3`):** **`10/10 (100%)`** routing accuracy (`mode=think`), **`10/10 (100%)` `quant_verified=True`** (every open-ended blueprint mechanically Python-verified vs `0/10` Vanilla), **`83.5s` avg (25% faster than Vanilla's `111.4s`)**, flipping Vanilla's `OE09` network-shuffle physics impossibility failure (`25% → 75% PASS`).
- **Live Showdown on `DeepSeek-V4.1-Flash-1017` (`cortheon showdown`):** Vanilla `1/3` → **Cortheon `3/3 (100%)`** (Automaton Combinatorics `47.8s`, Weighted 4-Coloring `98.4s`, Exact Rational Parser `115.7s`).

### 2. 100-Task Multi-Domain Frontier Gauntlets (Hard 50 + Nightmare 50)

100 hand-engineered hard/nightmare tasks across 5 domains with **20 planted buggy starter controls** mechanically verified failing before running (`bench/run_gauntlet50.py`, `bench/run_gauntlet2.py`):

| Suite (50 tasks each) | Track Breakdown (Cortheon) | Vanilla (`tb=1500`) | Cortheon Pass-1 | **Cortheon Final** | Delta vs Vanilla |
|---|---|---:|---:|---:|---:|
| **Gauntlet-I (Hard 50)** | Math `10/10` · Logic `10/10` · Algo `10/10` · Debug `10/10` · Sys/Sec `10/10` | 25/50 (50.0%) | 46/50 (92.0%) | **50/50 (100.0%)** | **+50.0 pts** |
| **Gauntlet-II (Nightmare 50)** | Math `10/10` · Logic `10/10` · Debug `10/10` · Sys/Sec `9/10` · Algo `8/10` | 19/50 (38.0%) | 45/50 (90.0%) | **47/50 (94.0%)** | **+56.0 pts** |
| **100-Task Combined Total** | **Math `20/20` · Logic `20/20` · Debug `20/20` · Sys/Sec `19/20` · Algo `18/20`** | **44/100 (44.0%)** | **91/100 (91.0%)** | **97/100 (97.0%)** | **+53.0 pts** |

### 3. Autonomous Systems & Kernel Proving Ground (`ReplicaAgent` vs `VanillaReAct`)

Identical local weights, identical tools — comparing raw chat-history ReAct vs Cortheon's `ReplicaAgent` (`BrainState` working memory + `B=3` parallel thought forks + instant Python/Clang compile guards + independent clean-context Referee Gate using local `Qwen3.8-27B-TFQ3`):

| Challenge | VanillaReAct | **Cortheon `ReplicaAgent`** |
|---|---|---|
| **PG1 — Greenfield Crash-Safe CRC32 WAL DB** (`minidb/` from empty dir; torn-tail recovery & compaction) | FAIL (18 turns) | **PASS 4/4 in 8 turns** (`5/5` adversarial referee tests passed) |
| **PG2 — Vague 3-Incident Production Debug** (8-file async job queue + rate limiter, zero hints) | FAIL (18 turns) | **PASS 3/3** (Referee caught a subtle terminal-job queue bug & forced clean fix) |
| **PG3 — Real 100k+ LOC `TensorFold` Zig/Metal Audit** (read-only audit, mechanical citation check) | FAIL (`0` valid cites) | **PASS (`33` disk-verified `zig/src/*.zig` cites, `0` hallucinations)** |
| **PG4 — ARM64 NEON 4-Bit Affine GEMV Kernel** (immutable `clang -O3` exactness + mirrored benchmark) | FAIL (never compiled) | **PASS — `8.52×` verified speedup (`2.188ms → 0.257ms`), `max_err = 6.1e-5`** |

### 4. Multi-File Repo Engineering (`DevForge`) & Frontier-Killer 16

- **Frontier-Killer 16 (3-Arm Ablation):** Vanilla default `6/16 (37.5%)` · Vanilla `tb=1500` `7/16 (43.8%)` → **Cortheon `16/16 (100.0%)`**.
- **Multi-File `DevForge` (6-file `ledgerflow` repo):** `2/2` (`R1` 4 interlocking cross-file bugs + `R2` brand-new 5-file atomic batch settlement feature where Vanilla produced no valid patch).

---

## Architecture — Synthesized Mechanical Verification Across Every Domain

1. **Neuro-Symbolic Reasoning (`ReasonForge`):** `B=4` parallel executable Python / SymPy / Z3 scripts → exact equivalence consensus (`1e-12` precision guard + integer/prompt format gate + AST assert tiebreaker) → conditional `B=3` Disagreement Falsifier + Hybrid Wave-3 PAL + Analytical Proof consensus ONLY on disagreement.
2. **Behavioral Differential Consensus Coding (`CodeForge v5`):**
   - Single-shot `B=4` Wave 1 (`radically_simple` + `first_principles` + `edge_skeptic` in parallel with `spec_oracle` + `diff_probe()`).
   - **Oracle-Free Behavioral Differential Consensus:** Fingerprints every candidate on diverse `diff_probe()` inputs, filters out `NameError`/crasher noise, awards `diff_majority_hits`, and runs strict `B=2` Counterexample Adjudication on concrete `(case, out_A, out_B)` disagreements — preserving consensus strictly through final selection.
   - **Starter-Import Guard & AST API Fidelity:** Automatically preserves starter imports, scores declared starter-library usage (`starter_api_score`), enforces explicit `Raises:` coverage, and runs a parallel `B=2` Spec-Clause + API-Mockability Audit before fast-exiting.
   - **Native `stdin_mode` Competitive Harness:** Automatically detects CLI/stdin problems, injects verbatim immutable Sample Input/Output tests, runs isolated `run_solution()` subprocesses, and executes Zero-Anchor Wave-2/Wave-3 repairs + Helper-Aware Strict `B=2` Test Shield + Pre-Wave-3 Re-Audit.
3. **Executable-Grounded Open-Ended Architecture (`ThinkForge`):** `B=4` parallel specialist council (`architect`, `adversarial_breaker`, `pragmatist`, `quantifier`) + parallel **Universal Executable Verifier** (writes & runs Python scripts verifying exact sizing/bandwidth/storage/VRAM/RTT physics feasibility & protocol/quorum/security invariants, with 1-shot self-repair) → Clean-Context Critique → Master Synthesis.
4. **Autonomous Engineering (`ReplicaAgent` & `DevForge`):** Compacted `BrainState` + Pinned File Cache + Live AST Map + `B=3` Strategic Thought-Forking + Ephemeral Git Worktree Racing + Read-Only `shell=False` Lockdown + Independent Clean-Context Referee Gate.
5. **Universal Drop-In Proxy (`cortheon proxy`):** Exposes OpenAI (`/v1/chat/completions`, with transparent tool-turn passthrough for **OMP**, **Pi**, **OpenCode**, and **Cursor**) and Anthropic (`/v1/messages`) with SSE streaming & keepalive heartbeats.

---

## Install & Run in 30 Seconds

> **Agent Harness Quick Start (OMP, Pi, OpenCode, Claude Code, Cursor, TensorFold Native):** See **[`GETTING_STARTED.md`](GETTING_STARTED.md)** for copy-paste configs, isolated `omp --profile` setup, `Ctrl+P` Cortheon vs Raw toggling, and split-pane `tmux` live tracing.

Works with **TensorFold**, **Ollama** (`http://localhost:11434/v1`), **llama.cpp** (`http://localhost:8080/v1`), **vLLM**, or **LM Studio** (`http://localhost:1234/v1`).

```sh
uv pip install --python .venv/bin/python -e .

# 1. Run a 60-second live Vanilla vs Cortheon duel on YOUR own local model
.venv/bin/cortheon showdown --url http://127.0.0.1:8080/v1

# 2. Solve anything (auto-routes: math/logic -> ReasonForge, code -> CodeForge v5, design/audits -> ThinkForge)
.venv/bin/cortheon solve --url http://127.0.0.1:8080/v1 \
  "Design a multi-region 50k TPS payment ledger with 35ms inter-region RTT and <40ms p99 write latency."

# 3. Run full autonomous ReplicaAgent (greenfield builds, incident debugging, codebase audits, kernels)
.venv/bin/cortheon agent \
  --workspace /path/to/repo \
  --url http://127.0.0.1:8080/v1 \
  --referee-url http://127.0.0.1:8081/v1 \
  "Investigate why retries execute twice and fix with regression tests"

# 4. Run multi-file DevForge with ephemeral git worktree racing
.venv/bin/cortheon dev \
  --repo /path/to/repo --verify "pytest -q" --apply \
  --url http://127.0.0.1:8080/v1 \
  "Add atomic batch settlements with fee splitting"

# 5. Start the Universal OpenAI + Anthropic Proxy (drop-in for Claude Code, Cursor, OpenCode, SDKs)
.venv/bin/cortheon proxy \
  --upstream http://127.0.0.1:8080/v1 \
  --port 18652

# Point Claude Code / Anthropic SDK directly at local Cortheon:
# ANTHROPIC_BASE_URL=http://127.0.0.1:18652 ANTHROPIC_API_KEY=local claude
```

### Python SDK

```python
import asyncio
from cortheon import Cortheon, CortheonConfig

async def main():
    async with Cortheon(CortheonConfig(base_url="http://127.0.0.1:18654/v1")) as c:
        res = await c.solve("How many length-14 {A,B,C} strings avoid 'AA', 'BB', 'ABC' with #A > #B?")
        print(res.mode, res.answer, f"{res.wall_s:.1f}s")

asyncio.run(main())
```

---

## Core Modules & Reproducible Benchmarks

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — Quick Start guide for **OMP (`oh-my-pi`)**, **Pi**, **OpenCode**, **Claude Code**, **Cursor**, and **TensorFold Native** (`--cortheon --cortheon-exec --parallel 8`)
- [`BENCHMARKS.md`](BENCHMARKS.md) & [`results/`](results/) — Complete benchmark ledger + 23 per-problem result JSONs across 3 frozen local model families
- [`cortheon/`](cortheon/__init__.py) — Clean public SDK & `cortheon` CLI entrypoint (`Cortheon`, `CortheonConfig`, `CortheonResult`)
- [`flash_forge/code_forge.py`](flash_forge/code_forge.py) — `CodeForge v5` Behavioral Differential Consensus + Starter-API Fidelity + Native `stdin_mode` + Helper-Aware Strict `B=2` Shield
- [`flash_forge/reason_forge.py`](flash_forge/reason_forge.py) — `ReasonForge` Neuro-Symbolic `B=4` Python/SymPy/Z3 Engine + Hybrid Wave-3 Analytical Consensus
- [`flash_forge/think_forge.py`](flash_forge/think_forge.py) — `ThinkForge` Open-Ended Specialist Council + Universal Python Physics/Invariant Executable Verifier
- [`flash_forge/replica_agent.py`](flash_forge/replica_agent.py), [`brain_state.py`](flash_forge/brain_state.py), [`tools.py`](flash_forge/tools.py), [`referee.py`](flash_forge/referee.py) — Autonomous Frontier-Replica Cognitive Loop + Cross-Model Adversarial Referee Gate
- [`flash_forge/dev_agent.py`](flash_forge/dev_agent.py) & [`repo_index.py`](flash_forge/repo_index.py) — Multi-file ephemeral git worktree racer + AST repo map
- [`flash_forge/proxy.py`](flash_forge/proxy.py) — Universal OpenAI (`/v1/chat/completions`) + Anthropic (`/v1/messages`) streaming proxy
- [`flash_forge/memory.py`](flash_forge/memory.py) — Zero-weight SQLite + FTS5 BM25 Episodic Reflexion Memory
- [`bench/run_official_eval.py`](bench/run_official_eval.py), [`bench/run_livecodebench.py`](bench/run_livecodebench.py), [`bench/open_ended_bench.py`](bench/open_ended_bench.py) — Reproducible runners for Official `AIME 2025`, Official `LiveCodeBench v6 Hard`, Official `BigCodeBench-Hard v0.1.4`, and Oracle-Free Open-Ended Systems
