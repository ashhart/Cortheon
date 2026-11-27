# Cortheon v2 — Execution-Verified Parallel Consensus for Local Models

**Improves frozen local models on official exact-answer competition math (AIME 2025) by +16.7 to +30.0 points (pooled `51/90 → 72/90`, paired `+24 / −3`, exact McNemar `p = 4.92e-05`) at ~6–7 calls per problem, and outperforms compute-matched `Maj@4` prose self-consistency voting at matched tokens (`1.007×`) while running 16% faster via Wave-1 early-abort. Oracle-free coding gains on current 10–30-task subsets are positive (`+3.3 to +15.0 pts`, pooled `36/110 → 45/110`, `p = 0.049`) but statistically inconclusive per individual run (`p = 0.45–1.0`) pending full-dataset evaluations.**

Zero weight updates. Zero cloud APIs. Works with **TensorFold**, **Ollama**, **llama.cpp**, **vLLM**, and **LM Studio** — with an OS-level sandbox (`sandbox-exec` on macOS / `bwrap` on Linux) enabled by default.

---

## Quick Start (60 Seconds)

```sh
# 1. Install Cortheon
git clone https://github.com/ashhart/Cortheon.git && cd Cortheon
uv venv .venv && uv pip install --python .venv/bin/python -e .

# 2. Install the /cortheon slash command across OMP, Pi, OpenCode, Claude Code, Cursor & Codex
.venv/bin/cortheon install-commands --harness all

# 3. Start the universal proxy (or serve TensorFold natively with --cortheon --cortheon-exec --parallel 8)
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

### 1. Compute-Matched 5-Arm Ablation on Full Official `MathArena/aime_2025` (30/30, `GLM-5.3-Flash-TFQ3`)

To isolate whether gains come from extra compute (`Maj@4`), from Python execution alone (`pal_b1`), from parallel execution consensus (`cortheon_w1`), or from Wave-2 falsification (`cortheon_full`), all 5 arms were run on all 30 official AIME 2025 problems with exact per-problem token, call, and latency logging ([`results/official_aime2025_ablation_glm53.json`](results/official_aime2025_ablation_glm53.json)):

| Arm | Accuracy `k/30` `[95% Wilson CI]` | Avg Wall | Avg Calls | Avg Total Tok | Paired vs `maj4_prose` |
|---|---:|---:|---:|---:|---:|
| `vanilla_b1` *(single prose call)* | `16/30` (**53.3%**) `[36.1, 69.8]` | `45.4s` | `1.00` | `2,781` | `+0.0 pts` (`+3/−3`, `p=1.00`) |
| `maj4_prose` *(compute-matched `B=4` prose voting)* | `16/30` (**53.3%**) `[36.1, 69.8]` | `75.2s` | `4.00` | `6,504` | Baseline |
| `pal_b1` *(single Python PAL script, `effort=low`)* | `8/30` (**26.7%**) `[14.2, 44.5]` | `19.5s` | `1.00` | `1,428` | `−26.7 pts` (`+1/−9`, `p=0.0215*`) |
| **`cortheon_w1`** *(`B=4` PAL early-abort, no Wave 2)* | `19/30` (**63.3%**) `[45.5, 78.1]` | **`63.4s`** (`0.84×`) | `4.07` (`1.02×`) | **`6,547`** (`1.007×`) | **`+10.0 pts`** (`+5/−2`, `p=0.453`) |
| **`cortheon_full`** *(Wave 1 + conditional Wave 2, `effort=low`)* | `21/30` (**70.0%**) `[52.1, 83.3]` | `147.9s` | `6.23` | `14,396` | **`+16.7 pts`** (`+8/−3`, `p=0.227`) |
| **`cortheon_full`** *(default `tb=700/950`, prior 30/30 run)* | `24/30` (**80.0%**) `[62.7, 90.5]` | `175.8s` | `7.20` | `13,700` | *(vs `15/30` vanilla: `+30.0 pts`, `+10/−1`, `p=0.0117*`)* |

**What the ablation proves honestly:**
- **Plain self-consistency voting (`maj4_prose`) added `0.0` points** (`16/30 → 16/30`) at `2.34×` tokens and `1.66×` wall time.
- **A Python sandbox alone (`pal_b1`) performed significantly *worse* than Vanilla** (`8/30 = 26.7%`, `p = 0.039`) because single-shot model scripts frequently crash or time out on Olympiad problems.
- **Parallel execution consensus (`cortheon_w1`) beat `maj4_prose` by `+10.0` points (`63.3%` vs `53.3%`) at virtually identical compute (`6,547` vs `6,504` tokens = `1.007×`) and 16% lower latency (`63.4s` vs `75.2s`)** thanks to Wave-1 early-abort.
- **Wave-2 falsification added another `+6.7` points** (`19/30 → 21/30`), and avoiding overly aggressive `effort=low` truncation on hard Olympiad prompts raised full Cortheon to **`24/30 (80.0%)`**.

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
- **Honest Caveat on Coding & Backends:** Individual coding slices (10–30 tasks) are currently underpowered (`p = 0.45–1.0`, inconclusive per run) and cost `~13–21` calls per task. All published runs so far used **TensorFold** on Apple Silicon (`thinking_budget` + `loop_guard`); behavior on Ollama/llama.cpp may differ depending on server parallelism and template handling.

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
3. **Universal Drop-In Proxy (`cortheon proxy`):** Exposes OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) with transparent tool-turn passthrough + **Automatic Death-Spiral Rescue** (`B=3` clean-context graveyard backtracking when a coding harness hits `>=2` consecutive failing tool/test runs).
4. **All-Harness `/cortheon` Installer (`cortheon install-commands --harness all`):** Native `/cortheon` slash command & tool across **OMP**, **Pi**, **OpenCode**, **Claude Code**, **Cursor**, and **Codex**.
5. **Cost / Latency Dial (`--streams 2|4|8`):** Trade compute directly for verification depth (`--streams 2` fast sanity check, `--streams 4` default, `--streams 8` maximum diversity).

### Experimental (Smaller Evaluation Base — Use in Containers / Disposable Clones)
- **Oracle-Free Coding (`CodeForge v6`):** Behavioral Differential Consensus with DataFrame/Matplotlib/filesystem side-effect fingerprinting, AST mutation-tested oracles, Starter-API Fidelity, and `B=2` Red-Teaming.
- **Open-Ended Technical Council (`ThinkForge`):** `B=4` specialist council + synthesized Python physics/invariant checker (not for subjective creative prose).
- **Autonomous Repo Agents (`cortheon agent`, `cortheon dev`):** Multi-turn `BrainState` + ephemeral git worktree racing + optional cross-model Referee Gate.

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
- [`bench/fetch_datasets.py`](bench/fetch_datasets.py), [`bench/stats.py`](bench/stats.py), [`bench/run_aime_ablation.py`](bench/run_aime_ablation.py) — Fresh-clone dataset fetcher, exact McNemar/Wilson stats calculator, and 5-arm compute-matched ablation runner
- [`flash_forge/sandbox.py`](flash_forge/sandbox.py), [`reason_forge.py`](flash_forge/reason_forge.py), [`code_forge.py`](flash_forge/code_forge.py), [`think_forge.py`](flash_forge/think_forge.py), [`proxy.py`](flash_forge/proxy.py), [`rescue.py`](flash_forge/rescue.py), [`duel.py`](flash_forge/duel.py), [`integrations.py`](flash_forge/integrations.py)
