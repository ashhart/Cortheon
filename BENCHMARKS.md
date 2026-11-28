# Cortheon v2 — Complete Benchmark Ledger & Reproduction Guide

Every result in this ledger is backed by per-problem, PII-scrubbed JSON files in [`results/`](results/) (indexed in [`results/summary.json`](results/summary.json)).

## Methodology & Fairness Controls

- **100% Frozen Local Quantized Weights:** Every score below ran exclusively on local Apple Silicon hardware (Mac Studio M5 Ultra and Mac Studio M3 Ultra) using TensorFold — **zero fine-tuning, zero LoRA/weight updates, and zero cloud solver models**.
- **Strict Apples-to-Apples Arms:**
  - **Vanilla:** Single-stream generation against the exact same local endpoint with `thinking_budget=1500`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `seed=42`, `loop_guard=True`.
  - **Cortheon v2:** `enable_memory=False` (cold start, zero cross-problem memory leakage), `force_mode=None` (live automatic task routing), `num_streams=4`.
- **Three Independent Model Families Tested:**
  1. `deepseek-v4.1-flash-1017` (non-grammar, soft-effort server)
  2. `glm-5.3-flash-tfq3` (3-bit TFQ3 MoE + MTP drafter)
  3. `qwen3.8-flash-next-tfq3` (3-bit TFQ3 MoE with structured grammar)

---

## 1. Compute-Matched 6-Arm Ablation on Full Official `MathArena/aime_2025` (30/30, `GLM-5.3-Flash-TFQ3`)

Per-problem token-, call-, and latency-logged ablation across all 30 official AIME 2025 problems ([`results/official_aime2025_ablation_glm53.json`](results/official_aime2025_ablation_glm53.json), zero timeouts, all 8/8 streams completed on every `Maj@8` problem):

| Arm | Accuracy `k/30` `[95% Wilson CI]` | Avg Wall | Avg Calls | Avg Total Tok | Avg Reason Tok | McNemar `p` vs `vanilla_b1` | McNemar `p` vs `maj4_prose` | McNemar `p` vs `maj8_prose` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_b1` | `16/30` (**53.3%**) `[36.1, 69.8]` | `45.4s` | `1.00` | `2,781` | `1,002` | — | `1.0000` | `0.4531` |
| `maj4_prose` | `16/30` (**53.3%**) `[36.1, 69.8]` | `75.2s` | `4.00` | `6,504` | `1,821` | `1.0000` | — | `0.3750` |
| `maj8_prose` | `19/30` (**63.3%**) `[45.5, 78.1]` | `110.0s` | `8.00` | `13,314` | `3,760` | `0.4531` | `0.3750` | — |
| `pal_b1` | `8/30` (**26.7%**) `[14.2, 44.5]` | `19.4s` | `1.00` | `1,425` | `196` | `0.0386*` | `0.0215*` | `0.0034**` |
| **`cortheon_w1`** | `19/30` (**63.3%**) `[45.5, 78.1]` | **`63.4s`** | `4.07` | **`6,547`** | `1,013` | `0.5078` | `0.4531` | `1.0000` |
| **`cortheon_full`** *(effort=low)* | `21/30` (**70.0%**) `[52.1, 83.3]` | `147.9s` | `6.23` | `14,396` | `2,592` | `0.2266` | `0.2266` | `0.7744` |
| **`cortheon_full`** *(v2 default `tb=700/950`)* | `24/30` (**80.0%**) `[62.7, 90.5]` | `175.8s` | `7.20` | `13,700` | — | `0.0386*` | `0.0574` | `0.1797` |
| **`reasonforge_v5`** *(Fast Pair + TIR Self-Repair)* | **`25/30` (`83.3%`)** `[66.4, 92.7]` | **`107.8s`** | **`7.67`** | `15,721` | `2,759` | **`0.0117*`** | **`0.0117*`** | `0.1094` |

- `cortheon_w1` matches `Maj@8` (`19/30`) at **0.49× tokens** (`6,547` vs `13,314`) and **1.74× faster wall time** (`63.4s` vs `110.0s`), while outperforming same-token `Maj@4` (`6,504` tok) by `+10.0` points.
- **`ReasonForge v5`** ([`results/official_aime2025_reasonforge_v5_glm53.json`](results/official_aime2025_reasonforge_v5_glm53.json)) combines a 2-call Stage-1A fast pair (`pal_exact` + `pal_skeptic`, clearing easy problems in `2.4–12.2s`), zero-anchoring inline **Executable TIR Self-Repair** (fixing syntax/import/range errors in-place without leaking other branches' answers), dynamic sandbox package awareness, bounded-range format gating, and strictly unanchored 50-digit `mpmath`/`numpy` Wave-2/Wave-3 verification: reaching **`25/30 (83.3%)`** at **`107.8s` avg (`1.63×` faster than `v2`'s `175.8s`)**, beating `vanilla_b1` and `Maj@4` by **`+30.0 pts` (`+10/−1`, `p = 0.0117*`)**, beating `cortheon_w1` with **zero regressions (`+6/−0`, `p = 0.0313*`)**, and beating `Maj@8` by **`+20.0 pts` (`+8/−2`)** at faster wall-clock (`107.8s` vs `110.0s`) and fewer calls (`7.67` vs `8.00`).

### 1b. Live 30/30 `SpecVerify` Ablation on Official `MathArena/aime_2025` (`inkling-2-mini` → `glm-5.3-flash-tfq3`)

First end-to-end live Speculative Execution-Verification run pairing a ~10× smaller local drafter (`inkling-2-mini`, 18.8 GiB BF16) with `glm-5.3-flash-tfq3` as the big verifier across all 30 official AIME 2025 problems ([`results/official_aime2025_spec_verify_inkling_glm53.json`](results/official_aime2025_spec_verify_inkling_glm53.json)):

| Arm | Accuracy `k/30` `[95% Wilson CI]` | Drafter Accept *(Precision)* | Avg Big Calls *(vs `Maj@8` / `cortheon_full`)* | Avg Big Tok *(vs `Maj@8` / `cortheon_full`)* | Avg Drafter Tok | Avg Wall |
|---|---:|---:|---:|---:|---:|---:|
| **`spec_verify`** | **`20/30` (`66.7%`)** `[48.8, 80.8]` | **`9/30 (30.0%)`** *(`9/9 = 100.0%`)* | **`4.57`** *(`−42.9%` / `−26.7%`)* | **`11,879`** *(`−10.8%` / `−17.5%`)* | `6,243` | `253.4s` *(`110.0s` on accept)* |

- **Zero-FP mechanical gate:** All `9/9` problems accepted by the smaller drafter (`AIME25_01`, `03`, `04`, `06`, `08`, `16`, `17`, `22`, `26`) matched ground truth with `0` big-model calls/tokens (`100.0%` precision); `21/30` escalated cleanly (`11/21` solved).
- **Honest speed note:** Because `inkling-2-mini` ran BF16 looped weights at ~10 tok/s (vs `glm-5.3-flash-tfq3`'s ~48 tok/s), wall-clock latency was slower than direct big-model execution (`253.4s` vs `147.9s`), even while big-model calls dropped `26.7%` vs `cortheon_full` (`effort=low`) and `42.9%` vs `Maj@8`. Pairing `SpecVerify` with a quantized 50–100+ tok/s drafter (or a metered cloud verifier) yields both cost and wall-clock reductions.

---

## 2. Compute-Matched 4-Arm `CodeForge v7` Ablation on Official `LiveCodeBench v6 Hard` (32/32 Paired ABC Hard, `GLM-5.3-Flash-TFQ3`)

Every arm ran **oracle-free** on the exact same 32 official AtCoder ABC Hard problems (`abc387_f`–`abc398_d`, ~42 hidden judge tests each) with exact per-problem token, call, and latency logging ([`results/official_lcb_v6_hard_ablation_glm53.json`](results/official_lcb_v6_hard_ablation_glm53.json)):

| Arm | Accuracy `k/32` `[95% Wilson CI]` | Avg Wall | Avg Calls | Avg Total Tok | McNemar `p` vs `vanilla` | McNemar `p` vs `codeforge_v7` |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla` *(B=1, `tb=1500`)* | `12/32` (**37.5%**) `[22.9, 54.8]` | `74.9s` | `1.00` | `5,770` | — | **`0.0034**`** (`+12/−1`) |
| `codeforge_v6` *(prior architecture)* | `16/32` (**50.0%**) `[33.6, 66.4]` | `272.4s` | `17.00` | `41,936` | `0.3438` (`+7/−3`) | **`0.0156*`** (`+7/−0`) |
| `best4_sample` *(B=4 public-sample)* | `18/32` (**56.2%**) `[39.3, 71.8]` | `112.4s` | `3.81` | `11,398` | `0.1796` (`+10/−4`) | `0.0625` (`+5/−0`) |
| **`codeforge_v7`** *(sample-first + early-abort + TLE rank)* | **`23/32` (`71.9%`)** `[54.6, 84.4]` | **`165.3s`** | **`6.19`** | **`18,563`** | **`0.0034**`** (`+12/−1`) | — |

- **`CodeForge v7` strictly dominates both `CodeForge v6` (`+7/−0`, `p=0.0156*`) and compute-matched `best4_sample` (`+5/−0`, `p=0.0625`) with zero regressions**, while cutting LLM calls **`2.75×`** (`17.00 → 6.19`), total tokens **`2.26×`** (`41,936 → 18,563`), and wall time **`1.65×`** (`272.4s → 165.3s`) compared to v6.
- On a separate 20-problem AtCoder ARC Hard slice (`GLM-5.3-Flash-TFQ3`), ultra-hard ARC tasks remained challenging for 3-bit local weights (`0/20 → 2/20` pre-v7).

---

## 3. Official Untouched External Benchmarks (200 Paired Runs Across 3 Model Families)

> **Statistical honesty:** Math gains on full AIME 2025 (`51/90 → 72/90`, `+24/−3`, exact McNemar `p = 4.92e-05`) and `CodeForge v7` on 32-problem LiveCodeBench v6 ABC Hard (`12/32 → 23/32`, `+12/−1`, `p = 0.0034**`) are statistically significant individually. Earlier pre-v7 10–30-task BigCodeBench-Hard and 20-task LCB slices below remain positive but underpowered per individual run (`p = 0.45–1.0`). Recompute all committed stats directly with `python -m bench.stats --dir results`.

| Benchmark | Split & Judge | Model Family | Vanilla | **Cortheon v2** | Delta | Paired (`+fix/−brk`, Exact McNemar `p`) | Avg Wall (V → C) | Result JSONs |
|---|---|---|---:|---:|---:|---|---:|---|
| **`MathArena/aime_2025`** | All 30/30 official problems; exact integer match | `DeepSeek-V4.1-Flash-1017` | 20/30 (66.7%) | **27/30 (90.0%)** | **+23.3 pts** | `+9 / −2` (`p=0.0654`) | `113.7s → 117.6s` | [vanilla](results/official_aime2025_deepseek_vanilla.json) · [cortheon](results/official_aime2025_deepseek_cortheon.json) |
| **`MathArena/aime_2025`** | All 30/30 official problems; exact integer match | `GLM-5.3-Flash-TFQ3` | 15/30 (50.0%) | **24/30 (80.0%)** | **+30.0 pts** *(+60% rel)* | `+10 / −1` (**`p=0.0117*`**) | `125.5s → 175.8s` | [vanilla](results/official_aime2025_glm53_vanilla.json) · [cortheon](results/official_aime2025_glm53_cortheon.json) |
| **`MathArena/aime_2025`** | All 30/30 official problems; exact integer match | `Qwen3.8-Flash-Next-TFQ3` | 16/30 (53.3%) | **21/30 (70.0%)** | **+16.7 pts** | `+5 / −0` (`p=0.0625`) | `69.2s → 80.5s` | [vanilla](results/official_aime2025_qwen38_vanilla.json) · [cortheon](results/official_aime2025_qwen38_cortheon.json) |
| **`LiveCodeBench v6 Hard`** | 20 AtCoder F/G hard problems (~43 judge tests each; **oracle-free**) | `DeepSeek-V4.1-Flash-1017` | 11/20 (55.0%) | 13/20 (65.0%) | +10.0 pts | `+2 / −0` (`p=0.5000`, ns) | `73.6s → 201.2s` | [vanilla](results/official_lcb_v6_hard_deepseek_vanilla.json) · [cortheon](results/official_lcb_v6_hard_deepseek_cortheon.json) |
| **`LiveCodeBench v6 Hard`** | 20 AtCoder F/G hard problems (~43 judge tests each; **oracle-free**) | `GLM-5.3-Flash-TFQ3` | 8/20 (40.0%) | 11/20 (55.0%) | +15.0 pts | `+5 / −2` (`p=0.4531`, ns) | `124.7s → 249.1s` | [vanilla](results/official_lcb_v6_hard_glm53_vanilla.json) · [cortheon](results/official_lcb_v6_hard_glm53_cortheon.json) |
| **`bigcode/bigcodebench-hard v0.1.4`** | 30/148 macOS-verified tasks (`complete_prompt`; **oracle-free**) | `DeepSeek-V4.1-Flash-1017` | 10/30 (33.3%) | 11/30 (36.7%) | +3.3 pts | `+2 / −1` (`p=1.0000`, ns) | `44.4s → 118.2s` | [vanilla](results/official_bcb_hard_complete_deepseek_vanilla.json) · [cortheon](results/official_bcb_hard_complete_deepseek_cortheon.json) |
| **`bigcode/bigcodebench-hard v0.1.4`** | 30/148 macOS-verified tasks (`instruct_prompt`; **oracle-free**) | `Qwen3.8-Flash-Next-TFQ3` | 6/30 (20.0%) | 8/30 (26.7%) | +6.7 pts | `+3 / −1` (`p=0.6250`, ns) | `57.9s → 85.5s` | [vanilla](results/official_bcb_hard_instruct_qwen38_vanilla.json) · [cortheon](results/official_bcb_hard_instruct_qwen38_cortheon.json) |
| **`bigcode/bigcodebench-hard v0.1.4`** | 10/148 Acid slice (`complete_prompt`, `CodeForge v5.1`; **oracle-free**) | `GLM-5.3-Flash-TFQ3` | 1/10 (10.0%) | 2/10 (20.0%) | +10.0 pts | `+1 / −0` (`p=1.0000`, ns) | `17.2s → 45.9s` | [vanilla](results/official_bcb_hard_acid10_glm53_vanilla.json) · [cortheon](results/official_bcb_hard_acid10_glm53_cortheon.json) |
| **Pooled Official Math (90 runs)** | **All 3 Model Families** | **DeepSeek + GLM-5.3 + Qwen** | **51/90 (56.7%)** | **72/90 (80.0%)** | **+23.3 pts** | **`+24 / −3` (`p=4.92e-05***`)** | — | [`results/summary.json`](results/summary.json) |
| **Pooled Official Code Slices (110 runs)** | **All 3 Model Families** | **DeepSeek + GLM-5.3 + Qwen** | **36/110 (32.7%)** | **45/110 (40.9%)** | **+8.2 pts** | **`+13 / −4` (`p=0.0490*`)** | — | [`results/summary.json`](results/summary.json) |

### Wave-1 PAL Consensus Calibration on Official `MathArena/aime_2025`

Whenever Cortheon's Wave-1 executable Python/SymPy/Z3 scripts reached consensus (fast-exiting without Wave 2):

- `DeepSeek-V4.1-Flash-1017`: **`19 / 19 (100.0%)`**
- `Qwen3.8-Flash-Next-TFQ3`: **`18 / 18 (100.0%)`**
- `GLM-5.3-Flash-TFQ3`: **`13 / 14 (92.9%)`**
- **Combined 3-Model-Family Wave-1 Calibration:** **`50 / 51 (98.0%)`**

---

## 4. Author-Authored Development & Open-Ended Suites *(Not Held-Out External Benchmarks)*

> **Honest negative result on pure open-ended prose:** In a 12-task compute-matched internal ablation of open-ended engineering/security/systems prose prompts (`GLM-5.3-Flash-TFQ3`), `vanilla_b1` scored `10/12 (83.3%)`, `Maj@4` prose self-consistency scored `11/12 (91.7%)`, and multi-stage adversarial prose falsification scored `9/12 (75.0%)` at `16.4k` tokens. We reverted that experimental prose-falsification layer rather than shipping a regression — verification gains require executable or formal invariants, not extra prose critique.

### 3a. Open-Ended Systems, Physics & Security 10 (`bench/open_ended_bench.py`)

10 natural-language principal-engineer architecture, capacity-physics, distributed-quorum, and security audit prompts — **no test suite, no `ANSWER:` tag, completely oracle-free at solve time** — evaluated objectively on exact quantitative bounds and hard physics/security trap detection on `GLM-5.3-Flash-TFQ3`:

| Arm | Pass Rate | Routing Accuracy (`mode=think`) | Executable Python Verification (`quant_verified`) | Avg Wall Time | Result JSON |
|---|---:|---:|---:|---:|---|
| **Vanilla** (`tb=1500`) | 9/10 (90.0%) *(missed `OE09` 10Gbps shuffle physics)* | — | 0/10 (0%) | `111.4s` | [vanilla](results/open_ended_10_glm53_vanilla.json) |
| **Cortheon `ThinkForge`** | **9/10 (90.0%)** *(flipped `OE09` to PASS)* | **10/10 (100%)** | **10/10 (100%)** | **`83.5s` (25% faster!)** | [cortheon](results/open_ended_10_glm53_cortheon.json) |

---

### 3b. 100-Task Multi-Domain Frontier Gauntlets *(Development Set, `Qwen3.8-Flash-Next-TFQ3`)*

100 hand-engineered hard & nightmare tasks across 5 domains (`math`, `logic`, `algo`, `debug`, `sys/sec`), including **20 planted buggy starter controls** mechanically verified failing before running (`bench/run_gauntlet50.py`, `bench/run_gauntlet2.py`):

| Suite | Track Breakdown (Cortheon) | Vanilla (`tb=1500`) | Cortheon Pass-1 | **Cortheon Final** | Delta | Result JSONs |
|---|---|---:|---:|---:|---:|---|
| **Gauntlet-I (Hard 50)** | Math `10/10` · Logic `10/10` · Algo `10/10` · Debug `10/10` · Sys/Sec `10/10` | 25/50 (50.0%) | 46/50 (92.0%) | **50/50 (100.0%)** | **+50.0 pts** | [vanilla](results/gauntlet1_hard50_qwen38_vanilla.json) · [cortheon](results/gauntlet1_hard50_qwen38_cortheon.json) |
| **Gauntlet-II (Nightmare 50)** | Math `10/10` · Logic `10/10` · Debug `10/10` · Sys/Sec `9/10` · Algo `8/10` | 19/50 (38.0%) | 45/50 (90.0%) | **47/50 (94.0%)** | **+56.0 pts** | [vanilla](results/gauntlet2_nightmare50_qwen38_vanilla.json) · [cortheon](results/gauntlet2_nightmare50_qwen38_cortheon.json) |
| **Combined 100-Task Total** | **Math `20/20` · Logic `20/20` · Debug `20/20` · Sys/Sec `19/20` · Algo `18/20`** | **44/100 (44.0%)** | **91/100 (91.0%)** | **97/100 (97.0%)** | **+53.0 pts** | — |

---

## 5. Autonomous Engineering, Multi-File Repo & Live Showdown Benchmarks

- **Autonomous Proving Ground (`bench/replica_proving_ground.py`, `ReplicaAgent` vs `VanillaReAct` on identical frozen `Qwen3.8-Flash-Next-TFQ3` weights):**
  - `PG1` Greenfield Crash-Safe CRC32 WAL KV Database (`minidb/`): Vanilla `FAIL (18 turns)` → **Cortheon `PASS 4/4 in 8 turns`** (`5/5` independent referee tests passed).
  - `PG2` Vague 3-Incident Production Debug (8-file async job queue): Vanilla `FAIL (18 turns)` → **Cortheon `PASS 3/3`**.
  - `PG3` 100k+ LOC Read-Only Zig/Metal Codebase Audit: Vanilla `FAIL (0 valid cites)` → **Cortheon `PASS (33 disk-verified cites, 0 hallucinations)`**.
  - `PG4` ARM64 NEON 4-Bit Affine GEMV Kernel (`clang -O3` exactness + benchmark): Vanilla `FAIL (never compiled)` → **Cortheon `PASS — 8.52× verified speedup (2.188ms → 0.257ms), max_err = 6.1e-5`**.
- **Frontier-Killer 16 (`bench/run_ab.py`, 3-Arm Ablation on `Qwen3.8-Flash-Next-TFQ3`):** Vanilla default `6/16 (37.5%)` · Vanilla `tb=1500` `7/16 (43.8%)` → **Cortheon `16/16 (100.0%)`**.
- **Multi-File `DevForge` (`bench/repo_bench.py`, 6-file `ledgerflow` repo):** `2/2` (`R1` 4 interlocking bugs + `R2` 5-file atomic batch settlement feature).
- **Live Showdown (`cortheon showdown` on `DeepSeek-V4.1-Flash-1017`):** Vanilla `1/3` → **Cortheon `3/3 (100%)`**.
- **Wave-1 Early-Abort & Low-Effort PAL Latency Optimization (`GLM-5.3-Flash-TFQ3`):** On a 25-digit exact rational 7-server asymmetric Markov cover-time problem (`2630600114653531854763423 / 183488738276457059168800`), switching PAL slots to `reasoning_effort="low"` (`tb=500`), raising the exact-math sandbox timeout to `20s`, enabling atomic early-abort on `2/4` PAL consensus, and serving `--parallel 8` cut wall time **`7.5×`** (**`271s → 36s`**, `wave1_consensus (2 PAL)`, `queued=0.000s`, cancelling the 4th stream automatically).

---

## Reproducing Every Benchmark From a Fresh Clone

Fetch official datasets cleanly into `bench/datasets/` (no extra dependencies beyond `httpx`), run `bench/stats.py` to recompute every Wilson 95% CI and exact McNemar p-value directly from `results/`, or run the 6-arm compute-matched AIME 2025 ablation on your own server:

```sh
# 0. Download official datasets into bench/datasets/ + recompute exact paired statistics
python -m bench.fetch_datasets --dataset all
python -m bench.stats --dir results

# 1. Compute-matched 6-arm AIME 2025 ablation (vanilla_b1 vs maj4_prose vs maj8_prose vs pal_b1 vs cortheon_w1 vs cortheon_full)
python -m bench.run_aime_ablation --url http://127.0.0.1:8080/v1 --dataset bench/datasets/aime_2025.json --out bench/results/aime25_ablation.json

# 2. Official MathArena/aime_2025 (30/30)
python -m bench.run_official_eval --dataset aime2025 --url http://127.0.0.1:8080/v1 --arm vanilla --out bench/results/aime25_vanilla.json
python -m bench.run_official_eval --dataset aime2025 --url http://127.0.0.1:8080/v1 --arm forge   --out bench/results/aime25_cortheon.json

# 3. Official LiveCodeBench v6 Hard (20/20 AtCoder F/G)
python -m bench.run_livecodebench --url http://127.0.0.1:8080/v1 --arm vanilla --limit 20 --out bench/results/lcb_vanilla.json
python -m bench.run_livecodebench --url http://127.0.0.1:8080/v1 --arm forge   --limit 20 --out bench/results/lcb_cortheon.json

# 4. Official bigcode/bigcodebench-hard v0.1.4 (complete_prompt)
python -m bench.run_official_eval --dataset bigcodebench_hard_complete --url http://127.0.0.1:8080/v1 --arm vanilla --out bench/results/bcb_vanilla.json
python -m bench.run_official_eval --dataset bigcodebench_hard_complete --url http://127.0.0.1:8080/v1 --arm forge   --out bench/results/bcb_cortheon.json

# 5. Live side-by-side split-screen duel
cortheon duel --url http://127.0.0.1:8080/v1 --preset ring
cortheon showdown --url http://127.0.0.1:8080/v1
```
