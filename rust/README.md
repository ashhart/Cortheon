# Rust ReasonForge CLI

`cortheon-rs` runs the Cortheon 2.3.1 ReasonForge strategy and AIME voting from a standalone Tokio CLI. It streams completions from one or more OpenAI-compatible endpoints and assigns sample `i` to endpoint `i % endpoint_count`. Use endpoints that serve the same model. The ReasonForge arms read this checkout's prompt cards and reuse its Python answer extraction, symbolic equivalence checks and sandbox.

## Build and test

From the repository root on Linux, use Rust 1.85 or later and Python 3.11 or later.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
cargo build --locked --release --manifest-path rust/Cargo.toml
cargo test --locked --manifest-path rust/Cargo.toml -- --test-threads=1
```

The integration tests run both the Rust CLI and this checkout's Python reference implementation against local HTTP fixtures. They compare request bodies, predictions and grades for all six official AIME arms. Additional cases exercise Wave-2 and Wave-3 recovery, judging, progress events and client retries. The verifier tests execute known scripts through the Python sandbox. [Sandbox documentation](../SECURITY.md)

CI also exercises the verifier worker from outside the checkout with the installed wheel. The Debian container checks the subprocess resource limits. A native Ubuntu job runs the same check with bubblewrap and verifies that the script has its own network namespace.

## Run a vote

Start the included fixture in one terminal.

```sh
rust/target/release/cortheon-rs mock --scenario rust/examples/vote.json
```

From the repository root in a second terminal, run a vote. The fixture returns the canned answer 42.

```sh
rust/target/release/cortheon-rs run --endpoints http://127.0.0.1:18080/v1 \
  --arms maj_early --n 4 --bench single --problem 'What is six times seven?'
```

For a model-backed evaluation, supply the URL of your inference server. To grade AIME problems, fetch the dataset once and select a half-open range of rows.

```sh
python -m bench.fetch_datasets --dataset aime2025
rust/target/release/cortheon-rs run --endpoints http://127.0.0.1:8080/v1 \
  --arms vanilla_b1,maj4_prose,maj8_prose,pal_b1,cortheon_w1,cortheon_full \
  --bench aime2025 --start 0 --end 5 --progress-jsonl progress.jsonl
```

| Arm | Decision rule |
| --- | --- |
| `vanilla_b1` | Generate one prose solution and grade the extracted answer. |
| `maj4_prose`, `maj8_prose` | Use the four- or eight-sample seed and temperature schedule from the current Python benchmark. Rank by vote count, then by the lowest contributing sample temperature. |
| `pal_b1` | Generate one `pal_exact` script and verify it with a 20-second sandbox budget. |
| `cortheon_w1` | Run up to four Wave-1 roles. Check all results ready together before deciding whether to cancel outstanding work. |
| `cortheon_full` | Follow Wave 1 with conditional falsification, recovery and adjudication. |
| `maj` | Collect `--n` prose samples. Widths 4 and 8 select the corresponding official schedule. |
| `maj_early` | Cast 1–16 equally weighted AIME ballots. Cancel outstanding requests when the lead over the runner-up exceeds the number of outstanding ballots. A final tie returns no winner. |

`maj_early` counts an answer only when the model ends normally with an `ANSWER: <integer>` line in the 0–999 range. Truncated answers abstain. A terminal choice can trigger cancellation immediately. Counted streams must then supply token usage and the SSE `[DONE]` marker. A malformed counted stream invalidates the trial and excludes it from scoring. The arm uses a distinct seed per sample and requires streaming.

The single-sample and named majority controls use their specified widths even when mixed in one run. ReasonForge has four distinct Wave-1 roles. With `--n 8` it schedules those four, as the current Python solver does. Each row records its effective width in `samples`. The official comparison uses the benchmark defaults, including disabled episodic memory. [Python AIME runner](../bench/run_aime_ablation.py)

For ordinary solver and baseline requests, the client uses the reported context length and reasoning capabilities to choose a token budget. It can retry after a backend rejects optional fields, or after a reasoning-only response reaches its length limit. The strict ballot arm, an explicit `--max-tokens` cap and deterministic runs suppress that extra generation. All endpoints in a fan-out must report the same selected model ID.

## Output

The CLI writes per-problem results to `rust_results.json` and timestamped client events to `rust_events.jsonl`. Choose separate `--out` and `--events` paths for each run you want to keep. Single questions have no reference answer and produce a null score. Token totals are null when a request lacks server-reported usage. The event log retains the usage that individual requests did report.

After each completed problem and arm, the CLI flushes its events and atomically replaces the results file with all completed rows. The top-level `complete` field becomes true after the last requested row. If a verifier fails, the CLI records a `problem_error` trace and exits with an error. Earlier rows remain available in the checkpoint.

`--progress-jsonl PATH` writes live ReasonForge events for the upstream progress renderer, with an `arm` and `problem_id` on each line. A write failure produces an immediate warning. The solver finishes the benchmark and returns a nonzero status for the failed log. The result row also records whether the answer was verified and whether the solver recovered from disagreement. [Progress renderer](../flash_forge/progress.py)

Rust's `wall_s` includes orchestration, verification and grading. Python's official ablation timer stops at the end of inference or solving, depending on the arm. Use matching start and stop events for timing comparisons.

Cancellation latency measures the interval from consensus to the client dropping its last cancelled stream. `cancelled_stream_chunks` counts nonempty deltas already received on those streams. Server-side slot release, subsequent token generation and GPU idleness require separate server instrumentation.

Use `run --help` for model selection, sampling and capture options. `--capture-output` includes prompts, model output and available reasoning in the event file. `CORTHEON_API_KEY` supplies endpoint authentication.

To run away from the checkout, keep the built binary and `rust/verifier/worker.py` together and install the wheel built from this revision into the worker's Python environment. The wheel contains the shared AIME grading code and protocol cards. Pass `--worker-script /path/to/worker.py` and set `CORTHEON_SRC` to the directory containing the installed `flash_forge` package. You can locate that directory with `python -c 'from pathlib import Path; import flash_forge; print(Path(flash_forge.__file__).resolve().parent.parent)'`. For dataset runs also pass `--dataset /path/to/aime_2025.json`. The packaged source includes the original [MIT license](LICENSE) and [derivative notice](NOTICE).
