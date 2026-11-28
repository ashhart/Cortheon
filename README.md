# Cortheon v2 — Execution-Verified Parallel Consensus for Local Models

**Improves frozen local models on official competition math (`MathArena/aime_2025`, 30/30) to `25/30 (83.3%)` on 3-bit `GLM-5.3-Flash-TFQ3` (`+30.0 pts` vs Vanilla `16/30`, paired `+10 / −1`, exact McNemar `p = 0.0117*`, replicated identically on all 30/30 problems; `+20.0 pts` over compute-matched `Maj@8` at *faster* wall-clock `107.8s` vs `110.0s` and *fewer* calls `7.67` vs `8.00`; pooled 3-model baseline `51/90 → 72/90`, `+24 / −3`, `p = 4.92e-05`), and on official oracle-free competitive coding (`LiveCodeBench v6 Hard`, 32 paired AtCoder ABC Hard problems, `GLM-5.3-Flash-TFQ3`) by `+34.4` points (`12/32 → 23/32 = 71.9%`) — reaching ZERO REGRESSIONS vs Vanilla (`+11 / −0`, exact McNemar `p = 0.00098***`) with `CodeForge v8`'s sound brute-force counterexample fuzzer (`8.28` calls) and matching `23/32` at `6.19` avg calls (`+12 / −1`, `p = 0.0034**`, `+5 / −0` over `best4_sample`, `+7 / −0` over `CodeForge v6`) with `CodeForge v7`. Smaller BigCodeBench-Hard slices (`10–30` tasks) remain positive (`+3.3` to `+10.0` pts) but statistically underpowered per run (`p = 0.625–1.0`).**

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

# 4. Start the universal proxy + Live Web Cockpit at http://127.0.0.1:18652/ui
.venv/bin/cortheon proxy --upstream http://127.0.0.1:8080/v1 --port 18652 --streams 4 --record-flywheel
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
| **`cortheon_full`** *(v2 default `tb=700/950`, prior 30/30)* | `24/30` (**80.0%**) `[62.7, 90.5]` | `175.8s` | `7.20` | `13,700` (`1.03× Maj@8`) | **`+26.7 pts`** (`+11/−3`, `p=0.057`) | **`+16.7 pts`** (`+7/−2`, `p=0.180`) |
| **`reasonforge_v5`** *(Fast Pair + Executable TIR Self-Repair)* | **`25/30` (`83.3%`)** `[66.4, 92.7]` | **`107.8s`** (`1.63×` faster than v2!) | **`7.67`** (`< Maj@8`) | `15,721` (`1.18× Maj@8`) | **`+30.0 pts`** (`+10/−1`, **`p=0.0117*`**) | **`+20.0 pts`** (`+8/−2`, `p=0.109`) |

**What the ablation proves honestly:**
- **Plain `Maj@4` prose voting added `0.0` points** (`16/30 → 16/30`) at `2.34×` tokens. Scaling to **`Maj@8` (`13,314` tok, `110.0s`, `8.00` calls) reached `19/30`** but still suffered severe correlated wrong-answer collapses (`7/8` voted wrong on `AIME25_27` and `AIME25_29`, `6/8` on `AIME25_14`, `5/8` on `AIME25_18`).
- **A Python sandbox alone (`pal_b1`) performed significantly *worse* than Vanilla and `Maj@8`** (`8/30 = 26.7%`, `p = 0.039` vs Vanilla, `p = 0.0034` vs `Maj@8`) because single-shot model scripts frequently crash or time out on Olympiad problems.
- **Parallel execution consensus (`cortheon_w1`) matched `Maj@8` (`19/30`) at HALF the tokens (`6,547` vs `13,314` = `0.49×`) and `1.74×` faster wall time (`63.4s` vs `110.0s`)**, while beating same-token `Maj@4` (`6,504` tok) by `+10.0` points.
- **`ReasonForge v5` (Executable TIR Self-Repair + Unanchored 50-digit `mpmath`/`numpy` Wave 2/3, [`results/official_aime2025_reasonforge_v5_glm53.json`](results/official_aime2025_reasonforge_v5_glm53.json)) hit a new all-time local 3-bit record of `25/30 (83.3%)` while cutting average wall time `1.63×` vs `v2` (`175.8s → 107.8s`)**: beating `vanilla_b1` and `Maj@4` by **`+30.0 pts` (`+10/−1`, `p = 0.0117*`)**, beating `cortheon_w1` with zero regressions (`+6/−0`, `p = 0.0313*`), beating `Maj@8` by **`+20.0 pts` (`+8/−2`) at *faster* wall time (`107.8s` vs `110.0s`) and *fewer* LLM calls (`7.67` vs `8.00`)**, and solving `AIME25_10` (`81`), `AIME25_14` (`60`), and `AIME25_25` (`907`) where `v2` failed.
- **Live 30/30 `SpecVerify` (`inkling-2-mini` 18.8 GiB drafter → `glm-5.3-flash-tfq3` verifier, [`results/official_aime2025_spec_verify_inkling_glm53.json`](results/official_aime2025_spec_verify_inkling_glm53.json)):** Reached **`20/30 (66.7%)`** `[48.8, 80.8]` — bypassing the big verifier completely on **`9/30 (30.0%)`** problems at **`9/9 (100.0%)` drafter precision** (`0` false accepts), cutting big-model calls **`42.9%` vs `Maj@8`** (`4.57` vs `8.00`, `11,879` vs `13,314` big tok) and **`26.7%` vs `cortheon_full` (`effort=low`)** (`17.5%` fewer big tok; paired `+2/−3`, `p=1.00`). *(Wall time averaged `253.4s` because the BF16 looped drafter ran at ~10 tok/s; quantized 50–100+ tok/s drafters or metered cloud verifiers get the full latency/cost win.)*

> **Where Verification Works vs. Where It Doesn't (Honest Boundary):** Cortheon's gains do **not** require a benchmark oracle — on AIME 2025 it never sees an answer key and synthesizes its own Python/SymPy/Z3 checkers. However, it *does* require a problem with **executable or formal invariants**. On a 12-task internal ablation of pure open-ended engineering/security prose advice (`GLM-5.3-Flash-TFQ3`), `Maj@4` self-consistency (`11/12`) outperformed multi-stage adversarial prose falsification (`9/12`, `vanilla_b1` `10/12`). We reverted that prose-falsification experiment rather than shipping a regression — use Cortheon where execution can genuinely falsify wrong hypotheses, not for subjective prose generation.

### 2. Compute-Matched `CodeForge v7` & `v8` Ablation on Official `LiveCodeBench v6 Hard` (32/32 Paired ABC Hard, `GLM-5.3-Flash-TFQ3`)

Every arm ran **oracle-free** on the exact same 32 official AtCoder ABC Hard problems (`abc387_f`–`abc398_d`, ~42 hidden judge tests each, `0` hidden tests ever seen at solve time) with full per-problem token, call, and latency logging ([`results/official_lcb_v6_hard_ablation_glm53.json`](results/official_lcb_v6_hard_ablation_glm53.json), [`results/official_lcb_v6_hard_codeforge_v8_glm53.json`](results/official_lcb_v6_hard_codeforge_v8_glm53.json)):

| Arm | Accuracy `k/32` `[95% Wilson CI]` | Avg Wall | Avg Calls | Avg Total Tok | Paired vs `vanilla` | Paired vs `codeforge_v7` |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla` *(single call, `tb=1500`)* | `12/32` (**37.5%**) `[22.9, 54.8]` | `74.9s` | `1.00` | `5,770` | Baseline | `−34.4 pts` (`+12/−1`, **`p=0.0034**`**) |
| `codeforge_v6` *(prior architecture)* | `16/32` (**50.0%**) `[33.6, 66.4]` | `272.4s` | `17.00` | `41,936` | `+12.5 pts` (`+7/−3`, `p=0.344`) | `−21.9 pts` (`+7/−0`, **`p=0.0156*`**) |
| `best4_sample` *(compute-matched `B=4` public-sample)* | `18/32` (**56.2%**) `[39.3, 71.8]` | `112.4s` | `3.81` | `11,398` | `+18.8 pts` (`+10/−4`, `p=0.180`) | `−15.6 pts` (`+5/−0`, `p=0.0625`) |
| **`codeforge_v7`** *(sample-first 4-slot + early-abort + TLE rank)* | **`23/32` (`71.9%`)** `[54.6, 84.4]` | **`165.3s`** | **`6.19`** | **`18,563`** | **`+34.4 pts`** (`+12/−1`, **`p=0.0034**`**) | **Speed / call-efficiency leader** |
| **`codeforge_v8`** *(+ inline TIR repair + sound brute-force fuzzer)* | **`23/32` (`71.9%`)** `[54.6, 84.4]` | `244.9s` | `8.28` | `25,158` | **`+34.4 pts`** (**`+11/−0`, `p=0.00098***`**) | `+0.0 pts` (`+1/−1`, `p=1.00`) |

- **Why `CodeForge v7` & `v8` flipped the ablation:** On this exact 32-problem set, lean `best4_sample` (`18/32`, `3.81` calls) originally beat `CodeForge v6` (`16/32`, `17.00` calls). Dedicating all 4 Wave-1 slots to diverse implementations whenever verbatim I/O samples exist, executing via `asyncio.wait(FIRST_COMPLETED)` with early-abort, and ranking fast `O(N log N)` runtime tiers ahead of shorter `O(N^2)` TLE candidates jumped pass rate by **`+21.9 pts` over v6 (`+7/−0`, `p=0.0156*`) while cutting calls `2.75×` (`17.00 → 6.19`) and tokens `2.26×` (`41,936 → 18,563`)**. **`CodeForge v8`** adds inline Wave-1 TIR sample-diff self-repair and a sample-gated sound-oracle brute-force counterexample fuzzer, wiping out `v7`'s lone regression (`abc394_f` → AC 43/43) for **zero regressions vs Vanilla (`+11/−0`, `p = 0.00098***`)** and rescuing hidden-test bugs live on `abc393_e` and `abc398_d` (`8.28` avg calls).

### 3. Official External Benchmarks Across 3 Frozen Local Model Families

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
- **Honest Caveat on Smaller Slices & Backends:** While `CodeForge v7` now achieves a statistically significant single-run coding lift on 32-problem LiveCodeBench v6 ABC Hard (`+34.4 pts`, `p=0.0034**` at `6.19` calls), the earlier pre-v7 10–30-task BigCodeBench-Hard slices above remain statistically underpowered per run (`p = 0.625–1.0`), and ultra-hard ARC tasks remain difficult for 3-bit models (`0/20 → 2/20` pre-v7). All published benchmark tables above used **TensorFold** on Apple Silicon (`thinking_budget` + `loop_guard`); generic OpenAI backends (`oMLX`, `vLLM`, `llama.cpp`, `Ollama`) are supported via PR `#2` (`@SelfHostedSI`, verified live on `oMLX` + `DeepSeek-V4.1-Flash` `3/3` Showdown), though full benchmark runs on Ollama/llama.cpp remain to be published.

### 4. Author-Authored Development & Regression Suites *(Not Held-Out External Benchmarks)*

These internal suites were authored during development to stress-test failure modes and prevent regressions (kept strictly separate from official external benchmarks above):
- **Gauntlet-I (Hard 50) & Gauntlet-II (Nightmare 50)** (`Qwen3.8-Flash-Next`): `44/100 → 97/100` (`+53 / −0`, see [`BENCHMARKS.md`](BENCHMARKS.md)).
- **Open-Ended Systems & Physics 10** (`GLM-5.3-Flash-TFQ3`): `9/10 → 9/10` pass rate, `10/10` Python-verified quantitative tables vs `0/10` Vanilla, `83.5s` avg vs `111.4s` Vanilla.
- **Autonomous Proving Ground (4 tasks, Experimental):** `0/4` VanillaReAct → `4/4` `ReplicaAgent` (WAL DB, async debug, codebase audit, NEON GEMV kernel).

---

## Core Capabilities & Scope

### Core (Proven & Production-Ready)
1. **Neuro-Symbolic Exact-Answer Solver (`ReasonForge`):** `B=2|4|8` parallel Python / SymPy / Z3 scripts with streaming Wave-1 early-abort (cancels lagging GPU slots the instant 2 independent scripts agree), arbitrary-precision rational equivalence, and conditional Wave-2 falsification.
2. **Streaming Sample-First Competitive Coder (`CodeForge v8`):** Dedicates all 4 Wave-1 slots to diverse implementations with inline TIR sample-diff self-repair, executes streams via `asyncio.wait(FIRST_COMPLETED)` with early-abort, ranks fast runtime tiers ahead of slow TLE-risk candidates, and runs a sample-gated sound-oracle brute-force counterexample fuzzer (`12/32 → 23/32 = 71.9%` on LiveCodeBench v6 Hard, zero regressions vs Vanilla `+11/−0`, `p=0.00098***`).
3. **OS-Sandboxed Execution (`flash_forge/sandbox.py`):** macOS `sandbox-exec` / Linux `bwrap` enabled by default — blocks network sockets, outside-tempdir writes, and sensitive credential/harness reads (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.config`, `~/.omp`, `~/.pi`, `~/.dsh`, `~/.claude`, `~/.codex`, `~/.cursor`), plus POSIX rlimits and hard timeouts. See [`SECURITY.md`](SECURITY.md).
4. **Polyglot Proof-Carrying Diff Breaker (`cortheon guard`, `--sarif`, `cortheon install-hooks`):** Spawns clean-context adversaries in isolated ephemeral `git worktree`s inside the OS sandbox across **Python, Node/TS, Go, and Shell** — **bans prose code reviews** (only alerts on executed failing reproducers, verified against `HEAD`, with `--auto-fix`, GitHub PR **SARIF 2.1.0** annotations, and non-destructive `pre-push` git hooks).
5. **Native Zero-Dependency MCP Server (`cortheon mcp`) & One-Command Installer (`cortheon install-mcp`):** Stdio JSON-RPC 2.0 MCP server exposing `cortheon_verify`, `cortheon_guard`, `cortheon_spec_verify`, and `cortheon_doctor` with live `notifications/progress` streaming (`_meta.progressToken`), `notifications/cancelled` + dead-pipe task/sandbox cancellation, and lazy web-stack loading; auto-registered non-destructively into Claude Code, Cursor, OpenCode, Codex, and Windsurf via `cortheon install-mcp --harness all`. *(Tip: set your MCP client's per-call timeout to at least `300s` for deep Olympiad/competitive-coding verifications.)*
6. **Universal Drop-In Proxy + Live Web Cockpit (`http://127.0.0.1:18652/ui`) & `cortheon doctor`:** OpenAI + Anthropic proxy with transparent tool-turn passthrough, **Automatic Death-Spiral Rescue** (`B=3` graveyard backtracking), a zero-CDN real-time SSE Web Cockpit streaming every slot race, early-abort, rescue, and 1-click DPO export, plus 2-second live diagnostics (`cortheon doctor`) and `--streams 2|4|8` cost/latency dial.
7. **All-Harness `/cortheon` Installer (`cortheon install-commands --harness all`):** Native `/cortheon` slash command & tool across **OMP**, **Pi**, **OpenCode**, **Claude Code**, **Cursor**, and **Codex**.

### Frontier Research Engines (First Live Receipts Published, Expanded Evals Ongoing)
- **Speculative Execution-Verification (`cortheon spec`):** Lifts speculative decoding from token logits to OS-sandboxed program execution — drafts parallel Python/SymPy/Z3 scripts on a tiny fast local model, accepts immediately on `>=2` distinct sandboxed script consensus (`0` big-model calls!), and escalates to the big local/cloud model *only* on falsification (first live 30/30 AIME 2025 run with `inkling-2-mini` → `glm-5.3-flash-tfq3`: **`9/9 = 100.0%` drafter precision**, `30%` zero-verifier bypass, `20/30` overall at `4.57` big calls).
- **Self-Play Verification Flywheel & `Cortheon Evolve` (`--record-flywheel` + `cortheon export-dpo` / `cortheon evolve`):** Automatically logs verified rescues during live use, weights contrastive curricula (`3×` disagreement falsifications, `2×` crash repairs, `1×` fast-consensus compression), gates candidate checkpoints with exact dual-regime McNemar non-regression tests (`single_shot` + `verified` + `anchor`), and merges top-K accepted `.npz` adapters with TIES sign-consensus + SHA-256 audits (first live zero-leakage `Qwen2.5-7B-Instruct-4bit` `mlx-lm` cycle on held-out 30/30 AIME 2025 in [`results/official_aime2025_evolve_qwen25_7b.json`](results/official_aime2025_evolve_qwen25_7b.json): `vanilla_b1` `0/30 → 3/30`, `+10.0 pts`, `+3/−0` `ACCEPTED`, directional `p=0.25`; `pal_b1` `2/30 → 2/30`, `+1/−1` `REJECTED` by the non-regression gate).
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
- [Rust ReasonForge CLI](rust/README.md) — Run the math solver or AIME voting across one or more inference endpoints.
- [`bench/fetch_datasets.py`](bench/fetch_datasets.py), [`bench/stats.py`](bench/stats.py), [`bench/run_aime_ablation.py`](bench/run_aime_ablation.py), [`bench/run_livecodebench.py`](bench/run_livecodebench.py), [`bench/run_spec_verify_ablation.py`](bench/run_spec_verify_ablation.py) — Fresh-clone dataset fetcher, exact McNemar/Wilson stats calculator, 6-arm AIME ablation, 3-arm LiveCodeBench ablation, and SpecVerify ablation
- [`flash_forge/sandbox.py`](flash_forge/sandbox.py), [`reason_forge.py`](flash_forge/reason_forge.py), [`spec_verify.py`](flash_forge/spec_verify.py), [`guard.py`](flash_forge/guard.py), [`mcp_server.py`](flash_forge/mcp_server.py), [`flywheel.py`](flash_forge/flywheel.py), [`evolve.py`](flash_forge/evolve.py), [`code_forge.py`](flash_forge/code_forge.py), [`doctor.py`](flash_forge/doctor.py), [`proxy.py`](flash_forge/proxy.py), [`rescue.py`](flash_forge/rescue.py), [`duel.py`](flash_forge/duel.py), [`integrations.py`](flash_forge/integrations.py)
