---
name: cortheon
description: Execution-verified consensus runtime for frozen local models (TensorFold, Ollama, llama.cpp, vLLM, LM Studio). Exploits parallel role-seeded trajectories, neuro-symbolic Python/SymPy/Z3 sandboxing, Behavioral Differential Consensus, adversarial clean-context red-teaming, graveyard backtracking, and zero-weight episodic memory.
---

# Cortheon v2 — Local-First Frontier Reasoning Skill

Use this skill whenever running a fast quantized local model on hard coding, debugging, math, combinatorics, probability, open-ended systems architecture, or autonomous repository engineering tasks where single-stream generation collapses.

## Core Rules (Why This Beats Single-Stream Local Prompting)

1. **Never rely on pass@1.** Fire `B=4` role-seeded parallel streams over one shared cached prefix.
2. **Never trust mental simulation or self-reported confidence.** Offload arithmetic, combinatorics, Markov chains, and constraint puzzles to the Python + `sympy` + `z3` + `fractions` sandbox (`ReasonForge`), coding correctness to Behavioral Differential Consensus + modular test/fuzzer suites (`CodeForge v5`), and open-ended systems sizing/invariants to the Python executable verifier (`ThinkForge`).
3. **Keep every active instruction card under 250 tokens.** Load only the micro-card in `cards/` for the current phase (`contract`, `oracle`, `race_*`, `redteam_*`, `pal_*`, `falsifier_*`).
4. **Gate with a clean-context Adversarial Red-Teamer + Helper-Aware Strict `B=2` Arbiter.** Open a fresh context containing only `{problem, candidate}` to hunt concrete breaking inputs; validate kills through independent arbiters seeing all module helpers before promoting them.
5. **Hard-reset on dead ends (Graveyard Backtracking).** Never patch a broken patch 3+ times inside polluted chat history. Write compact 1-line tombstones (`approach -> exact failure`), wipe context, and launch Zero-Anchor Wave 2 fresh.
6. **Learn without weight updates.** Distill verified recoveries into SQLite FTS5 (`~/.flashforge/lessons.db`) and inject top-3 BM25 matches into future shared prefixes.

## Running Cortheon

Point Cortheon at any local OpenAI-compatible server (`TensorFold`, `Ollama`, `llama.cpp`, `vLLM`, `LM Studio`):

```sh
# 60-second live Vanilla vs Cortheon duel on your local model
cortheon showdown --url http://127.0.0.1:8080/v1

# Solve a hard math, coding, debugging, or systems-design prompt
cortheon solve --url http://127.0.0.1:8080/v1 "Your hard problem"

# Start drop-in OpenAI + Anthropic proxy (models: cortheon, cortheon:code, cortheon:reason, cortheon:think, cortheon:deep, cortheon:fast)
cortheon proxy --upstream http://127.0.0.1:8080/v1 --port 18652

# Patch a real repository inside ephemeral git worktrees
cortheon dev --repo /path/to/repo --verify "pytest -q" --apply --url http://127.0.0.1:8080/v1 "Fix ..."
```
