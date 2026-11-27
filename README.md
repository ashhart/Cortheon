# Cortheon v2 — Execution-Verified Parallel Consensus for Local Models

**Improves frozen local models on official exact-answer competition math (AIME 2025) by +16.7 to +30.0 points (pooled `51/90 → 72/90`, paired `+24 / −3`, exact McNemar `p = 4.92e-05`) at ~6–7 calls per problem. On a 30/30 compute-matched AIME 2025 ablation, Wave-1 execution consensus (`cortheon_w1`) matches `Maj@8` prose voting (`19/30`) at half the tokens (`0.49×`) and `1.74×` faster wall time while beating `Maj@4` (`16/30`) by `+10.0` points at `1.007×` tokens; full Cortheon reaches `21/30–24/30` at matched `Maj@8` tokens. Oracle-free coding gains on current 10–30-task subsets are positive (`+3.3 to +15.0 pts`, pooled `36/110 → 45/110`, `p = 0.049`) but statistically inconclusive per individual run (`p = 0.45–1.0`) pending full-dataset evaluations.**

Zero weight updates. Zero cloud APIs. Works with **TensorFold**, **oMLX**, **Ollama**, **llama.cpp**, **vLLM**, **SGLang**, **mlx-lm**, and **LM Studio** (including automatic reasoning-model headroom, `enable_thinking=False` routing, and scratchpad commit recovery on generic OpenAI backends — contributed in PR `#2` by `@SelfHostedSI`), with an OS-level sandbox (`sandbox-exec` on macOS / `bwrap` on Linux) enabled by default.

---

## Quick Start (60 Seconds)

```sh
# 1. Install Cortheon
git clone https://github.com/ashhart/Cortheon.git && cd Cortheon
uv venv .venv && uv pip install --python .venv/bin/python -e .

# 2. Verify your local backend, OS sandbox, and solvers in 2 seconds
.venv/bin/cortheon doctor --url http://127.0.0.1:8080/v1

# 3. Install /cortheon commands & register the native MCP server across your coding harnesses
.venv/bin/cortheon install-commands --harness all
.venv/bin/cortheon install-mcp --harness all --url http://127.0.0.1:8080/v1

# 4. Start the universal proxy (or serve TensorFold natively with --cortheon --cortheon-exec --parallel 8)
.venv/bin/cortheon proxy --upstream http://127.0.0.1:8080/v1 --port 18652 --streams 4
```

### Then use `/cortheon` inside your coding harness

Open **OMP**, **Pi**, **OpenCode**, **Claude Code**, or **Cursor** and type:

```text
/cortheon 7 servers sit in a ring (0..6). A token starts at server 1 (already visited) and moves +1 mod 7 w.p. 2/3 or -1 mod 7 w.p. 1/3, except at server 0 it teleports uniformly to one of 0..6. Find the exact rational expected ticks until all 7 servers are visited.
```

Or run a live side-by-side Vanilla vs Cortheon race on your own model:

```sh
.venv/bin/cortheon duel --url http://127.0.0.1:8080/v1 --preset ring
.venv/bin/cortheon showdown --url http://127.0.0.1:8080/v1
```

- **Full harness configs (OMP, Pi, OpenCode, Claude Code, Cursor, TensorFold Native, `tmux` cockpit):** [`GETTING_STARTED.md`](GETTING_STARTED.md)
- **8 copy-paste challenge prompts with mechanically verified ground truths:** [`CHALLENGE_PROMPTS.md`](CHALLENGE_PROMPTS.md)
- **Security & OS sandbox threat model:** [`SECURITY.md`](SECURITY.md)

---

## Measured Results (Reproducible, Paired Statistics & Compute-Matched Ablation)

> **Reproduce from a fresh clone:** run `python -m bench.fetch_datasets` to download official datasets into `bench/datasets/`, then `python -m bench.stats --dir results` to recompute every confidence interval and exact McNemar p-value directly from the committed per-problem JSONs in [`results/`](results/). See **[`BENCHMARKS.md`](BENCHMARKS.md)** for full details.

### 1. Compute-Matched 6-Arm Ablation on Full Official `MathArena/aime_2025` (30/30, `GLM-5.3-Flash-TFQ3`)

To isolate whether gains come from extra compute (`Maj@4` / `Maj@8`), from Python execution alone (`pal_b1`), from parallel execution consensus (`cortheon_w1`), or from Wave-2 falsification (`cortheon_full`), all 6 arms were run cleanly on all 30 official AIME 2025 problems (`0` timeouts, `8/8` streams completed on every `Maj@8` problem) with exact per-problem token, call, and latency logging ([`results/official_aime2025_ablation_glm53.json`](results/official_aime2025_ablation_glm53.json)):

| Arm | Accuracy `k/30` `[95% Wilson CI]` | Avg Wall | Avg Calls | Avg Total Tok | Paired vs `maj4_prose` | Paired vs `maj8_prose` |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla_b1` *(single prose call)* | `16/30` (**53.3%**) `[36.1, 69.8]` | `45.4s` | `1.00` | `2,781` | `+0.0 pts` (`p=1.00`) | `−10.0 pts` (`p=0.453`) |
| `maj4_prose` *(compute-matched `B=4` prose voting)* | `16/30` (**53.3%**) `[36.1, 69.8]` | `75.2s` | `4.00` | `6,504` | Baseline | `−10.0 pts` (`p=0.375`) |
| `maj8_prose` *(compute-matched `B=8` prose voting)* | `19/30` (**63.3%**) `[45.5, 78.1]` | `110.0s` | `8.00` | `13,314` | `+10.0 pts` (`p=0.375`) | Baseline |
| `pal_b1` *(single Python PAL script, `effort=low`)* | `8/30` (**26.7%**) `[14.2, 44.5]` | `19.4s` | `1.00` | `1,425` | `−26.7 pts` (`p=0.0215*`) | `−36.7 pts` (`p=0.0034**`) |
| **`cortheon_w1`** *(`B=4` PAL early-abort, no Wave 2)* | `19/30` (**63.3%**) `[45.5, 78.1]` | **`63.4s`** (`0.58× Maj@8`) | `4.07` | **`6,547`** (`0.49× Maj@8`) | **`+10.0 pts`** (`+5/−2`, `p=0.453`) | **`+0.0 pts`** (`+4/−4`, `p=1.00`) |
| **`cortheon_full`** *(Wave 1 + Wave 2, `effort=low`)* | `21/30` (**70.0%**) `[52.1, 83.3]` | `147.9s` | `6.23` | `14,396` (`1.08× Maj@8`) | **`+16.7 pts`** (`+8/−3`, `p=0.227`) | **`+6.7 pts`** (`+7/−5`, `p=0.774`) |
| **`cortheon_full`** *(default `tb=700/950`, prior 30/30 run)* | `24/30` (**80.0%**) `[62.7, 90.5]` | `175.8s` | `7.20` | `13,700` (`1.03× Maj@8`) | **`+26.7 pts`** (`+11/−3`, `p=0.057`) | **`+16.7 pts`** (`+7/−2`, `p=0.180`) |

**What the 6-arm ablation proves honestly:**
- **Plain `Maj@4` prose voting added `0.0` points** (`16/30 → 16/30`) at `2.34×` tokens. Scaling to **`Maj@8` (`13,314` tok, `110.0s`) reached `19/30`** but still suffered severe correlated wrong-answer collapses (`7/8` voted wrong on `AIME25_27` and `AIME25_29`, `6/8` on `AIME25_14`, `5/8` on `AIME25_18`).
- **A Python sandbox alone (`pal_b1`) performed significantly *worse* than Vanilla and `Maj@8`** (`8/30 = 26.7%`, `p = 0.039` vs Vanilla, `p = 0.0034` vs `Maj@8`) because single-shot model scripts frequently crash or time out on Olympiad problems.
- **Parallel execution consensus (`cortheon_w1`) matched `Maj@8` (`19/30`) at HALF the tokens (`6,547` vs `13,314` = `0.49×`) and `1.74×` faster wall time (`63.4s` vs `110.0s`)**, while beating same-token `Maj@4` (`6,504` tok) by `+10.0` points and running `16%` faster via Wave-1 early-abort.
- **Wave-2 falsification pushed past `Maj@8` at matched tokens:** `21/30` (`1.08×` `Maj@8` tokens with `effort=low`) and **`24/30 (80.0%)`** (`~1.03×` `Maj@8` tokens with default `tb=700/950`, `+7/−2` over `Maj@8`, catching `AIME25_18` and `AIME25_27` where 5/8 and 7/8 prose samples voted wrong).

> **Where Verification Works vs. Where It Doesn't (Honest Boundary):** Cortheon's gains do **not** require a benchmark oracle — on AIME 2025 it never sees an answer key and synthesizes its own Python/SymPy/Z3 checkers. However, it *does* require a problem with **executable or formal invariants**. On a 12-task internal ablation of pure open-ended engineering/security prose advice (`GLM-5.3-Flash-TFQ3`), `Maj@4` self-consistency (`11/12`) outperformed multi-stage adversarial prose falsification (`9/12`, `vanilla_b1` `10/12`). We reverted that prose-falsification experiment rather than shipping a regression — use Cortheon where execution can genuinely falsify wrong hypotheses, not for subjective prose generation.

### 2. Official External Benchmarks Across 3 Frozen Local Model Families

Computed directly from [`results/`](results/) via `python -m bench.stats --dir results` (`enable_memory=False`, `force_mode=None`):

| Official Benchmark | Frozen Local Model | Vanilla `k/n` `[95% CI]` | Cortheon `k/n` `[95% CI]` | Delta | Paired (`+fix / −brk`) | Exact McNemar `p` | Avg Calls / Tok (Cortheon) |
|---|---|---:|---:|---:|:---:|---:|---:|
| **`MathArena/aime_2025`** *(30/30)* | `DeepSeek-V4.1-Flash-1017` | 20/30 (66.7%) `[48.8, 80.8]` | **27/30 (90.0%)** `[74.4, 96.5]` | **+23.3 pts** | `+9 / −2` | `0.0654` | `6.3` calls · `17.1k` tok |
| **`MathArena/aime_2025`** *(30/30)* | `GLM-5.3-Flash-TFQ3` | 15/30 (50.0%) `[33.2, 66.8]` | **24/30 (80.0%)** `[62.7, 90.5]` | **+30.0 pts** | `+10 / −1` | **`0.0117*`** | `7.2` calls · `13.7k` tok |
| **`MathArena/aime_2025`** *(30/30)* | `Qwen3.8-Flash-Next-TFQ3` | 16/30 (53.3%) `[36.1, 69.8]` | **21/30 (70.0%)** `[52.1, 83.3]` | **+16.7 pts** | `+5 / −0` | `0.0625` | `6.8` calls · `15.4k` tok |
| **Pooled Official AIME 2025** *(90 runs)* | **3 model families** | **51/90 (56.7%)** `[46.4, 66.4]` | **72/90 (80.0%)** `[70.6, 87.0]` | **+23.3 pts** | **`+24 / −3`** | **`4.92e-05***`** | `6.8` calls · `15.4k` tok |
| **`LiveCodeBench v6 Hard`** *(20 AtCoder F/G, oracle-free)* | `DeepSeek-V4.1-Flash-1017` | 11/20 (55.0%) `[34.2, 74.2]` | 13/20 (65.0%) `[43.3, 81.9]` | +10.0 pts | `+2 / −0` | `0.5000` *(ns)* | `18.9` calls · `46.0k` tok |
| **`LiveCodeBench v6 Hard`** *(20 AtCoder F/G, oracle-free)* | `GLM-5.3-Flash-TFQ3` | 8/20 (40.0%) `[21.9, 61.3]` | 11/20 (55.0%) `[34.2, 74.2]` | +15.0 pts | `+5 / −2` | `0.4531` *(ns)* | `16.4` calls · `43.6k` tok |
| **`BigCodeBench-Hard v0.1.4`** *(30/148 complete, oracle-free)* | `DeepSeek-V4.1-Flash-1017` | 10/30 (33.3%) `[19.2, 51.2]` | 11/30 (36.7%) `[21.9, 54.5]` | +3.3 pts | `+2 / −1` | `1.0000` *(ns)* | `16.8` calls · `26.3k` tok |
| **`BigCodeBench-Hard v0.1.4`** *(30/148 instruct, oracle-free)* | `Qwen3.8-Flash-Next-TFQ3` | 6/30 (20.0%) `[9.5, 37.3]` | 8/30 (26.7%) `[14.2, 44.4]` | +6.7 pts | `+3 / −1` | `0.6250` *(ns)* | `20.6` calls · `27.0k` tok |
| **`BigCodeBench-Hard v0.1.4`** *(10/148 acid slice, oracle-free)* | `GLM-5.3-Flash-TFQ3` | 1/10 (10.0%) `[1.8, 40.4]` | 2/10 (20.0%) `[5.7, 51.0]` | +10.0 pts | `+1 / −0` | `1.0000` *(ns)* | `13.2` calls · `12.1k` tok |
| **Pooled Official Code Slices** *(110 runs)* | **3 model families** | **36/110 (32.7%)** `[24.7, 41.9]` | **45/110 (40.9%)** `[32.2, 50.3]` | **+8.2 pts** | **`+13 / −4`** | **`0.0490*`** | `17.9` calls · `31.9k` tok |

- **Wave-1 Consensus Calibration (AIME 2025):** `50/51 (98.0%)` accuracy when `>=2` independent Wave-1 PAL scripts agreed (`19/19` DeepSeek, `18/18` Qwen3.8, `13/14` GLM-5.3).
- **Honest Caveat on Coding & Backends:** Individual coding slices (10–30 tasks) are currently underpowered (`p = 0.45–1.0`, inconclusive per run) and cost `~13–21` calls per task. All published benchmark tables below used **TensorFold** on Apple Silicon (`thinking_budget` + `loop_guard`); generic OpenAI backends (`oMLX`, `vLLM`, `llama.cpp`, `Ollama`) are supported via PR `#2` (`@SelfHostedSI`, verified live on `oMLX` + `DeepSeek-V4.1-Flash` `3/3` Showdown), though full benchmark runs on Ollama/llama.cpp remain to be published.

### 3. Author-Authored Development & Regression Suites *(Not Held-Out External Benchmarks)*

These internal suites were authored during development to stress-test failure modes and prevent regressions (kept strictly separate from official external benchmarks above):
- **Gauntlet-I (Hard 50) & Gauntlet-II (Nightmare 50)** (`Qwen3.8-Flash-Next`): `44/100 → 97/100` (`+53 / −0`, see [`BENCHMARKS.md`](BENCHMARKS.md)).
- **Open-Ended Systems & Physics 10** (`GLM-5.3-Flash-TFQ3`): `9/10 → 9/10` pass rate, `10/10` Python-verified quantitative tables vs `0/10` Vanilla, `83.5s` avg vs `111.4s` Vanilla.
- **Autonomous Proving Ground (4 tasks, Experimental):** `0/4` VanillaReAct → `4/4` `ReplicaAgent` (WAL DB, async debug, codebase audit, NEON GEMV kernel).

---

## Core Capabilities & Scope

### Core (Proven & Production-Ready)
1. **Neuro-Symbolic Exact-Answer Solver (`ReasonForge`):** `B=2|4|8` parallel Python / SymPy / Z3 scripts with streaming Wave-1 early-abort (cancels lagging GPU slots the instant 2 independent scripts agree), arbitrary-precision rational equivalence, and conditional Wave-2 falsification.
2. **OS-Sandboxed Execution (`flash_forge/sandbox.py`):** macOS `sandbox-exec` / Linux `bwrap` enabled by default — blocks network sockets, outside-tempdir writes, and sensitive credential reads (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.omp`, `~/.pi`, `~/.config`), plus POSIX rlimits and hard timeouts. See [`SECURITY.md`](SECURITY.md).
3. **Zero-False-Positive Proof-Carrying Diff Breaker (`cortheon guard`):** Spawns clean-context adversaries in isolated ephemeral `git worktree`s inside the OS sandbox and **bans prose code reviews** — a bug is *only* reported if backed by an executed failing Python reproducer (optionally verified passing on `HEAD` and auto-fixed with `--auto-fix`).
4. **Native Zero-Dependency MCP Server (`cortheon mcp`) & One-Command Installer (`cortheon install-mcp`):** Stdio JSON-RPC 2.0 MCP server exposing `cortheon_verify`, `cortheon_guard`, `cortheon_spec_verify`, and `cortheon_doctor`, auto-registered non-destructively into Claude Code, Cursor, OpenCode, Codex, and Windsurf via `cortheon install-mcp --harness all`.
5. **Universal Drop-In Proxy (`cortheon proxy`), Diagnostics (`cortheon doctor`) & Live Progress (`flash_forge/progress.py`):** OpenAI + Anthropic proxy with transparent tool-turn passthrough, **Automatic Death-Spiral Rescue** (`B=3` graveyard backtracking after `>=2` consecutive failing tool/test runs), live per-slot streaming progress, 2-second sandbox/backend/throughput diagnostics (`cortheon doctor`), `--record-flywheel` auto-harvesting, and `--streams 2|4|8` cost/latency dial.
6. **All-Harness `/cortheon` Installer (`cortheon install-commands --harness all`):** Native `/cortheon` slash command & tool across **OMP**, **Pi**, **OpenCode**, **Claude Code**, **Cursor**, and **Codex**.

### Frontier Research Engines (New in v2.1–v2.2 — Offline Unit-Tested, Live Benchmarks Ongoing)
- **Streaming `CodeForge v7` (Sample-First All-4-Slot Wave 1 + Early-Abort + TLE Ranking):** Directly addresses what our 52-problem compute-matched LiveCodeBench v6 Hard ablation revealed (`best4_sample` beat pre-v7 `CodeForge` `18/32` vs `16/32` at 3.5× fewer calls): dedicates all 4 Wave-1 slots to diverse implementations whenever verbatim I/O samples exist, executes each stream via `asyncio.wait(FIRST_COMPLETED)`, cancels lagging GPU slots the instant `2` independent implementations pass samples unanimously, and ranks fast `O(N log N)` survivors ahead of slow TLE-risk candidates.
- **Speculative Execution-Verification (`cortheon spec`):** Lifts speculative decoding from token logits to OS-sandboxed program execution — drafts parallel Python/SymPy/Z3 scripts on a tiny fast local model (2B–9B), accepts immediately on `>=2` distinct sandboxed script consensus (`0` big-model calls!), and escalates to the big local/cloud model *only* on falsification (`bench/run_spec_verify_ablation.py`).
- **Self-Play Verification Flywheel (`--record-flywheel` + `cortheon export-dpo` / `export-sft`):** Automatically logs verified Wave-2 rescues during live use and harvests mechanically verified `(rejected, chosen)` preference pairs into deduplicated, PII-scrubbed DPO/SFT JSONL for local `mlx-lm` / `TRL` LoRA distillation.
- **Open-Ended Technical Council (`ThinkForge`) & Autonomous Repo Agents (`cortheon agent`, `cortheon dev`):** Experimental — use in containers or disposable clones.

---

## Python SDK & Module Map

```python
import asyncio
from cortheon import Cortheon, CortheonConfig

async def main():
    async with Cortheon(CortheonConfig(base_url="http://127.0.0.1:8080/v1", num_streams=4)) as c:
        res = await c.solve("How many length-14 {A,B,C} strings avoid 'AA', 'BB', 'ABC' with #A > #B?")
        print(res.mode, res.answer, f"{res.wall_s:.1f}s")

asyncio.run(main())
```

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — Harness setup (OMP, Pi, OpenCode, Claude Code, Cursor, Codex, TensorFold Native)
- [`CHALLENGE_PROMPTS.md`](CHALLENGE_PROMPTS.md) — 8 mechanically verified prompts to test Vanilla vs `/cortheon` live
- [`BENCHMARKS.md`](BENCHMARKS.md) & [`results/`](results/) — Complete benchmark ledger, paired stats, and 24 per-problem JSONs
- [`SECURITY.md`](SECURITY.md) — Default OS sandbox protections and threat model
- [`bench/fetch_datasets.py`](bench/fetch_datasets.py), [`bench/stats.py`](bench/stats.py), [`bench/run_aime_ablation.py`](bench/run_aime_ablation.py), [`bench/run_livecodebench.py`](bench/run_livecodebench.py), [`bench/run_spec_verify_ablation.py`](bench/run_spec_verify_ablation.py) — Fresh-clone dataset fetcher, exact McNemar/Wilson stats calculator, 6-arm AIME ablation, 3-arm LiveCodeBench ablation, and SpecVerify ablation
- [`flash_forge/sandbox.py`](flash_forge/sandbox.py), [`reason_forge.py`](flash_forge/reason_forge.py), [`spec_verify.py`](flash_forge/spec_verify.py), [`guard.py`](flash_forge/guard.py), [`mcp_server.py`](flash_forge/mcp_server.py), [`flywheel.py`](flash_forge/flywheel.py), [`code_forge.py`](flash_forge/code_forge.py), [`doctor.py`](flash_forge/doctor.py), [`proxy.py`](flash_forge/proxy.py), [`rescue.py`](flash_forge/rescue.py), [`duel.py`](flash_forge/duel.py), [`integrations.py`](flash_forge/integrations.py)
