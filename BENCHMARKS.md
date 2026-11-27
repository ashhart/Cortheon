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

## 1. Official Untouched External Benchmarks (**8 for 8 Wins — 200 Runs Across 3 Model Families**)

| Benchmark | Split & Judge | Model Family | Vanilla | **Cortheon v2** | Delta | Flips / Regressions | Avg Wall (V → C) | Result JSONs |
|---|---|---|---:|---:|---:|---|---:|---|
| **`MathArena/aime_2025`** | All 30/30 official problems; exact integer match | `DeepSeek-V4.1-Flash-1017` | 20/30 (66.7%) | **27/30 (90.0%)** | **+23.3 pts** | `+9` flips / `-2` regr | `113.7s → 117.6s` | [vanilla](results/official_aime2025_deepseek_vanilla.json) · [cortheon](results/official_aime2025_deepseek_cortheon.json) |
| **`MathArena/aime_2025`** | All 30/30 official problems; exact integer match | `GLM-5.3-Flash-TFQ3` | 15/30 (50.0%) | **24/30 (80.0%)** | **+30.0 pts** *(+60% rel)* | `+10` flips / `-1` regr | `125.5s → 175.8s` | [vanilla](results/official_aime2025_glm53_vanilla.json) · [cortheon](results/official_aime2025_glm53_cortheon.json) |
| **`MathArena/aime_2025`** | All 30/30 official problems; exact integer match | `Qwen3.8-Flash-Next-TFQ3` | 16/30 (53.3%) | **21/30 (70.0%)** | **+16.7 pts** | `+5` flips / `0` regr | `69.2s → 80.5s` | [vanilla](results/official_aime2025_qwen38_vanilla.json) · [cortheon](results/official_aime2025_qwen38_cortheon.json) |
| **`LiveCodeBench v6 Hard`** | 20 AtCoder F/G hard problems (~43 judge tests each; **oracle-free at solve time**) | `DeepSeek-V4.1-Flash-1017` | 11/20 (55.0%) | **13/20 (65.0%)** | **+10.0 pts** | `+2` (`abc391_f`, `abc394_f`) / `0` regr | `73.6s → 201.2s` | [vanilla](results/official_lcb_v6_hard_deepseek_vanilla.json) · [cortheon](results/official_lcb_v6_hard_deepseek_cortheon.json) |
| **`LiveCodeBench v6 Hard`** | 20 AtCoder F/G hard problems (~43 judge tests each; **oracle-free at solve time**) | `GLM-5.3-Flash-TFQ3` | 8/20 (40.0%) | **11/20 (55.0%)** | **+15.0 pts** *(+37.5% rel)* | `+5` (`abc387_f`, `abc389_e`, `abc390_e`, `abc391_g`, `abc394_g`) / `-2` | `124.7s → 249.1s` | [vanilla](results/official_lcb_v6_hard_glm53_vanilla.json) · [cortheon](results/official_lcb_v6_hard_glm53_cortheon.json) |
| **`bigcode/bigcodebench-hard v0.1.4`** | 30 macOS-verified tasks (`complete_prompt`; **oracle-free at solve time**) | `DeepSeek-V4.1-Flash-1017` | 10/30 (33.3%) | **11/30 (36.7%)** | **+3.3 pts** | `+2` / `-1` | `44.4s → 118.2s` | [vanilla](results/official_bcb_hard_complete_deepseek_vanilla.json) · [cortheon](results/official_bcb_hard_complete_deepseek_cortheon.json) |
| **`bigcode/bigcodebench-hard v0.1.4`** | 30 macOS-verified tasks (`instruct_prompt`; **oracle-free at solve time**) | `Qwen3.8-Flash-Next-TFQ3` | 6/30 (20.0%) | **8/30 (26.7%)** | **+6.7 pts** *(+33% rel)* | `+2` net | `57.9s → 85.5s` | [vanilla](results/official_bcb_hard_instruct_qwen38_vanilla.json) · [cortheon](results/official_bcb_hard_instruct_qwen38_cortheon.json) |
| **`bigcode/bigcodebench-hard v0.1.4`** | 10-task Acid slice (`complete_prompt`, `CodeForge v5.1`; **oracle-free**) | `GLM-5.3-Flash-TFQ3` | 1/10 (10.0%) | **2/10 (20.0%)** | **+10.0 pts** *(2× Vanilla)* | `+1` (`BigCodeBench_777`) / `0` regr | `17.2s → 45.9s` | [vanilla](results/official_bcb_hard_acid10_glm53_vanilla.json) · [cortheon](results/official_bcb_hard_acid10_glm53_cortheon.json) |
| **Combined Official External (200 runs)** | **All 3 Model Families** | **DeepSeek + GLM-5.3 + Qwen** | **87/200 (43.5%)** | **117/200 (58.5%)** | **+15.0 pts (+34.5% rel)** | **8/8 wins** | — | [`results/summary.json`](results/summary.json) |

### Wave-1 PAL Consensus Calibration on Official `MathArena/aime_2025`

Whenever Cortheon's Wave-1 executable Python/SymPy/Z3 scripts reached consensus (fast-exiting without Wave 2):

- `DeepSeek-V4.1-Flash-1017`: **`19 / 19 (100.0%)`**
- `Qwen3.8-Flash-Next-TFQ3`: **`18 / 18 (100.0%)`**
- `GLM-5.3-Flash-TFQ3`: **`13 / 14 (92.9%)`**
- **Combined 3-Model-Family Wave-1 Calibration:** **`50 / 51 (98.0%)`**

---

## 2. Oracle-Free Open-Ended Systems, Physics & Security (`bench/open_ended_bench.py`)

10 natural-language principal-engineer architecture, capacity-physics, distributed-quorum, and security audit prompts — **no test suite, no `ANSWER:` tag, completely oracle-free at solve time** — evaluated objectively on exact quantitative bounds and hard physics/security trap detection on `GLM-5.3-Flash-TFQ3`:

| Arm | Pass Rate | Routing Accuracy (`mode=think`) | Executable Python Verification (`quant_verified`) | Avg Wall Time | Result JSON |
|---|---:|---:|---:|---:|---|
| **Vanilla** (`tb=1500`) | 9/10 (90.0%) *(missed `OE09` 10Gbps shuffle physics)* | — | 0/10 (0%) | `111.4s` | [vanilla](results/open_ended_10_glm53_vanilla.json) |
| **Cortheon `ThinkForge`** | **9/10 (90.0%)** *(flipped `OE09` to PASS)* | **10/10 (100%)** | **10/10 (100%)** | **`83.5s` (25% faster!)** | [cortheon](results/open_ended_10_glm53_cortheon.json) |

---

## 3. 100-Task Multi-Domain Frontier Gauntlets (`Qwen3.8-Flash-Next-TFQ3`)

100 hand-engineered hard & nightmare tasks across 5 domains (`math`, `logic`, `algo`, `debug`, `sys/sec`), including **20 planted buggy starter controls** mechanically verified failing before running (`bench/run_gauntlet50.py`, `bench/run_gauntlet2.py`):

| Suite | Track Breakdown (Cortheon) | Vanilla (`tb=1500`) | Cortheon Pass-1 | **Cortheon Final** | Delta | Result JSONs |
|---|---|---:|---:|---:|---:|---|
| **Gauntlet-I (Hard 50)** | Math `10/10` · Logic `10/10` · Algo `10/10` · Debug `10/10` · Sys/Sec `10/10` | 25/50 (50.0%) | 46/50 (92.0%) | **50/50 (100.0%)** | **+50.0 pts** | [vanilla](results/gauntlet1_hard50_qwen38_vanilla.json) · [cortheon](results/gauntlet1_hard50_qwen38_cortheon.json) |
| **Gauntlet-II (Nightmare 50)** | Math `10/10` · Logic `10/10` · Debug `10/10` · Sys/Sec `9/10` · Algo `8/10` | 19/50 (38.0%) | 45/50 (90.0%) | **47/50 (94.0%)** | **+56.0 pts** | [vanilla](results/gauntlet2_nightmare50_qwen38_vanilla.json) · [cortheon](results/gauntlet2_nightmare50_qwen38_cortheon.json) |
| **Combined 100-Task Total** | **Math `20/20` · Logic `20/20` · Debug `20/20` · Sys/Sec `19/20` · Algo `18/20`** | **44/100 (44.0%)** | **91/100 (91.0%)** | **97/100 (97.0%)** | **+53.0 pts** | — |

---

## 4. Autonomous Engineering, Multi-File Repo & Live Showdown Benchmarks

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

Fetch official datasets cleanly into `bench/datasets/` (no extra dependencies beyond `httpx`), run `bench/stats.py` to recompute every Wilson 95% CI and exact McNemar p-value directly from `results/`, or run the 5-arm compute-matched AIME 2025 ablation on your own server:

```sh
# 0. Download official datasets into bench/datasets/ + recompute exact paired statistics
python -m bench.fetch_datasets --dataset all
python -m bench.stats --dir results

# 1. Compute-matched 5-arm AIME 2025 ablation (vanilla_b1 vs maj4_prose vs pal_b1 vs cortheon_w1 vs cortheon_full)
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
