# Changelog

All notable changes to Cortheon are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.1] - 2026-11-27

### Added
- **Live MCP Progress & Task/Sandbox Cancellation (`#6`, reported by `@JordiPosthumus`):** `cortheon mcp` now reads `_meta.progressToken` and streams strictly increasing `notifications/progress` events from Cortheon's universal `on_event` hooks across `cortheon_verify`, `cortheon_guard`, and `cortheon_spec_verify`. Stdio `tools/call` runs in tracked concurrent `asyncio.Task`s scoped via `sandbox_scope()` so `notifications/cancelled` and closed client stdout pipes cancel in-flight verifications immediately and SIGKILL orphaned sandbox subprocess groups.
- **`--model` Flag on `cortheon solve` (PR `#4` by `@JordiPosthumus`):** Passes explicit model ID overrides through `ForgeConfig.model` for multi-model backends (LM Studio, vLLM).
- **Published 32/32 `CodeForge v7` LiveCodeBench v6 Hard & 30/30 AIME 2025 `SpecVerify` Receipts:** Committed PII-scrubbed `results/official_lcb_v6_hard_ablation_glm53.json` (`12/32 → 23/32 = 71.9%`, `+34.4 pts`, `p = 0.0034**`, `+5/−0` vs `best4_sample`, `+7/−0` vs `CodeForge v6` at `2.75×` fewer calls) and `results/official_aime2025_spec_verify_inkling_glm53.json` (`20/30`, `9/9 = 100.0%` drafter precision, `30%` zero-verifier bypass).

### Security
- **Agent-Harness Home Directory Sandbox Hardening (PR `#3` by `@JordiPosthumus`):** Added `~/.dsh`, `~/.claude`, `~/.codex`, and `~/.cursor` to macOS Seatbelt `sensitive_dirs` with regression tests preventing credential and transcript exfiltration.

### Fixed
- **Lazy FastAPI/Uvicorn Web Surface Loading (PR `#5` by `@JordiPosthumus`):** Deferred `fastapi`/`uvicorn` imports in `flash_forge/cli.py` and `cortheon/__init__.py` (via PEP 562 `__getattr__`) so minimal installs without the web stack still run `cortheon mcp` and `cortheon solve`, while keeping `fastapi`/`uvicorn` in default dependencies so Quick Start `cortheon proxy` continues to work out of the box.
- **Always-On Reasoning Model Fast-Route Recovery (`flash_forge/client.py`):** Added `reasoning_optional=False` handling (e.g. `inkling-2-mini`) so `enable_thinking=False` caps thinking budget/effort and adapts cleanly without thinking starvation.
- **SARIF `informationUri` & Cold CI Timeout Polish:** Pointed SARIF `informationUri` to `https://github.com/ashhart/Cortheon` and raised `clang -fsyntax-only` syntax-check timeout from `8s` to `25s` in `flash_forge/tools.py`.

## [2.3.0] - 2026-11-27

### Added
- **Live Web Cockpit (`/ui`) & Real-Time SSE Event Bus (`flash_forge/cockpit.py`):** Mounted directly onto `cortheon proxy` (`http://127.0.0.1:18652/ui`) — zero external CDNs, dark terminal-scientific UI streaming live parallel slot races, early-aborts, death-spiral rescues, SpecVerify savings, proven diff bugs, recent runs, and 1-click PII-scrubbed DPO JSONL export (`/v1/cortheon/flywheel/export`).
- **Polyglot `cortheon guard`, SARIF 2.1.0 & Git Hook Installer (`cortheon install-hooks`):** Auto-detects Python, Node.js/TypeScript, Go, and Shell diffs and executes adversarial reproducers inside the OS sandbox (`macos-seatbelt` / `linux-bwrap`), exports GitHub Code Scanning SARIF 2.1.0 (`--sarif`) for native PR inline diff annotations, and non-destructively installs/uninstalls `.git/hooks/pre-push` guard hooks.
- **Universal Live `on_event` Streaming:** Added fault-isolated per-slot `on_event` callbacks across `ReasonForge`, `SpecVerify`, `CodeForge v7`, and `cortheon guard`, powering both terminal progress (`flash_forge/progress.py`), OpenAI proxy SSE comment heartbeats, and the Live Web Cockpit.

## [2.2.0] - 2026-11-27

### Added
- **CodeForge v7 Streaming Sample-First Wave 1 & TLE Runtime Ranking:** Runs sample-test verification as Wave-1 candidates stream in, supports early-abort once sample-verified consensus is reached, and ranks passing candidates by measured execution runtime to reject TLE-prone solutions before Wave-2 escalation.
- **One-Command MCP Auto-Installer (`cortheon install-mcp`):** Automatically registers Cortheon's native stdio MCP server across **Claude Code**, **Cursor**, **OpenCode**, **Codex**, and **Windsurf**.
- **Live Flywheel DPO Auto-Harvesting (`--record-flywheel`):** Opt-in live recording of PII-scrubbed, mechanically verified rescue traces into the local DPO/SFT flywheel store during normal runs.
- **Live Per-Slot Progress Streaming (`flash_forge/progress.py`):** Zero-dependency, thread-safe, `NO_COLOR`-aware terminal renderer (`LiveProgressReporter`) mirroring TensorFold Native's per-slot trace (`start`, `slot_done`, `early_abort`, `wave2_start`, `drafter_accepted`, `escalated`, `bug_proven`, `done`), plus safe SSE comment heartbeat framing (`format_sse_comment`) and `make_progress_callback`.
- **CLI Version Flag (`cortheon --version`):** Reports the installed Cortheon package version for diagnostics and release verification.
- **Automated Release Packaging Workflow (`.github/workflows/release.yml`):** Tag-triggered (`v*`) and `workflow_dispatch` GitHub Actions pipeline that gates releases on the offline `pytest` suite, builds sdist and wheel distributions, validates metadata via `twine check`, smoke-tests `cortheon --version` and `cortheon --help` from the built wheel, publishes GitHub Releases with `dist/*` artifacts, and supports optional OIDC PyPI publishing gated on `vars.PUBLISH_TO_PYPI == 'true'`.

## [2.1.0] - 2026-11-27

### Added
- **`SpecVerify` (`cortheon spec`):** Speculative execution-verification engine that drafts parallel Python/SymPy/Z3 scripts on a tiny fast local model (2B–9B) inside the OS sandbox, accepts immediately on distinct-solver consensus with zero verifier calls, and escalates to the larger verifier model only on falsification (with `bench/run_spec_verify_ablation.py`).
- **Zero-False-Positive Proof-Carrying Diff Breaker (`cortheon guard`):** Spawns clean-context adversarial bug-hunters in isolated ephemeral `git worktree`s + OS sandbox, refuses prose-only reviews, reports findings only when backed by an executed failing Python reproducer (classified against `HEAD` as regression vs. pre-existing vs. new-code), and supports verified `--auto-fix` patch racing.
- **Zero-Dependency Stdio MCP Server (`cortheon mcp`):** JSON-RPC 2.0 Model Context Protocol server exposing `cortheon_verify`, `cortheon_guard`, `cortheon_spec_verify`, and `cortheon_doctor` with zero external MCP dependencies.
- **Self-Play Verification Flywheel (`cortheon export-dpo` / `cortheon export-sft`):** Harvests mechanically verified `(rejected, chosen)` hard-negative flips from benchmark results and live traces into deduplicated, PII-scrubbed JSONL datasets for local `mlx-lm` / `TRL` LoRA fine-tuning, plus CodeForge v6.1 fast unanimous Wave-1 exit.
- **Local Diagnostics (`cortheon doctor`):** 2-second environment probe checking the OS sandbox backend, Python/SymPy/Z3 solver stack, installed coding harnesses, backend capabilities, and live fast-route throughput, alongside FastAPI lifespan proxy updates and `llama.cpp`/Nemotron backend polish.
- **Compute-Matched 6-Arm AIME 2025 Ablation & LiveCodeBench Runner Upgrades:** Complete 30/30 `Maj@4` and `Maj@8` prose voting baselines vs. `pal_b1`, `cortheon_w1`, and `cortheon_full` (`results/official_aime2025_ablation_glm53.json`), plus Best-of-4 baseline, per-problem token logging, and sharding in the LiveCodeBench runner.

### Fixed
- **Generic OpenAI Reasoning-Model Headroom (PR `#2` by `@SelfHostedSI`):** Added automatic reasoning-model token headroom, `enable_thinking=False` routing, and scratchpad commit recovery across generic OpenAI-compatible backends (`oMLX`, `vLLM`, `llama.cpp`, `Ollama`).
- **Linux x86_64 CI `wilson_ci` Boundary Clamp:** Fixed floating-point clamp at exact `0.0` / `1.0` proportions in `bench/stats.py` on Linux x86_64 and made the SpecVerify ablation unit test self-contained for fresh-clone CI.

## [2.0.0] - 2026-11-26

### Added
- **Initial Cortheon v2 Runtime:** Execution-verified parallel consensus runtime for frozen local models featuring `ReasonForge` (parallel Python/SymPy/Z3 with Wave-1 streaming early-abort, hardened arbitrary-precision rational equivalence, and conditional Wave-2 falsification), `CodeForge` (parallel coding race, brute-force differential fuzzer, trace/mutation checks, and adversarial red-teaming), and `ThinkForge` (open-ended technical council), plus experimental `ReplicaAgent` and `DevForge`.
- **Default OS-Level Sandbox (`flash_forge/sandbox.py`):** Enabled `macos-seatbelt` (`sandbox-exec`) on macOS and `linux-bwrap` (`bubblewrap`) on Linux by default to block network sockets, outside-tempdir writes, and sensitive credential reads (`~/.ssh`, `~/.aws`, `~/.gnupg`, etc.), paired with POSIX rlimits and hard execution timeouts (`SECURITY.md`).
- **All-Harness `/cortheon` Installer (`cortheon install-commands`):** One-command installation of the `/cortheon` slash command and tool integration across **OMP**, **Pi**, **OpenCode**, **Claude Code**, **Cursor**, and **Codex**.
- **Live Side-by-Side Racing (`cortheon duel` & `cortheon showdown`):** Real-time Vanilla (`B=1`) vs. Cortheon (`B=4` early-abort) terminal racer and 60-second showcase presets (`CHALLENGE_PROMPTS.md`).
- **Universal Drop-In Proxy (`cortheon proxy`) & Automatic Death-Spiral Rescue:** OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) proxy with transparent tool-turn passthrough and automatic `B=3` graveyard backtracking when agentic tool loops stall.
- **Fresh-Clone Reproducibility & Statistics (`bench/fetch_datasets.py`, `bench/stats.py`):** Official dataset fetcher, exact McNemar p-values, Wilson confidence intervals, and full benchmark ledger (`BENCHMARKS.md`, `GETTING_STARTED.md`).

[2.2.0]: https://github.com/ashhart/Cortheon/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/ashhart/Cortheon/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/ashhart/Cortheon/releases/tag/v2.0.0
