# Rust ReasonForge CLI

`cortheon-rs` runs Cortheon's math-solving strategy and AIME voting from a standalone Tokio CLI. It streams completions from one or more OpenAI-compatible endpoints and assigns sample `i` to endpoint `i % endpoint_count`. Use endpoints that serve the same model. The ReasonForge arms read this checkout's prompt cards and reuse its Python answer extraction, symbolic equivalence checks and sandbox.

## Build and test

From the repository root on Linux, use Rust 1.85 or later and Python 3.11 or later.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
cargo build --locked --release --manifest-path rust/Cargo.toml
cargo test --locked --manifest-path rust/Cargo.toml -- --test-threads=1
```

The integration tests start local HTTP fixtures. The verifier tests execute known scripts through the same Python sandbox used by the CLI. For model-generated code, review Cortheon's sandbox backends and host requirements before choosing an endpoint. [Sandbox documentation](../SECURITY.md)

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
  --arms cortheon_w1,maj,maj_early --n 4 --bench aime2025 --start 0 --end 5
```

| Arm | Decision rule |
| --- | --- |
| `cortheon_w1` | Execute up to four Wave-1 roles. Agreement among verified scripts can cancel the remaining work. |
| `cortheon_full` | Follow Wave 1 with conditional falsification and adjudication. |
| `maj` | Collect all prose samples, cluster answers and use the lowest sample temperature to break a vote-count tie. |
| `maj_early` | Cast 1–16 equally weighted AIME ballots. Cancel outstanding requests when the lead over the runner-up exceeds the number of outstanding ballots. A final tie returns no winner. |

`maj_early` counts an answer only when the model ends normally with an `ANSWER: <integer>` line in the 0–999 range. Truncated answers abstain. A terminal choice can trigger cancellation immediately. Counted streams must then supply token usage and the SSE `[DONE]` marker. A malformed counted stream invalidates the trial and excludes it from scoring. The arm uses a distinct seed per sample and requires streaming.

## Output

The CLI writes per-problem results to `rust_results.json` and timestamped client events to `rust_events.jsonl`. Choose separate `--out` and `--events` paths for each run you want to keep. Single questions have no reference answer and produce a null score. Token totals are null when a request lacks server-reported usage. The event log retains the usage that individual requests did report.

Cancellation latency measures the interval from consensus to the client dropping its last cancelled stream. `cancelled_stream_chunks` counts nonempty deltas already received on those streams. Server-side slot release, subsequent token generation and GPU idleness require separate server instrumentation.

Use `run --help` for model selection, sampling and capture options. `--capture-output` includes prompts, model output and available reasoning in the event file. `CORTHEON_API_KEY` supplies endpoint authentication. If you move a built binary away from its checkout, supply `--cortheon-src` and `--worker-script` to locate the Python package and verifier worker. The packaged source includes the original [MIT license](LICENSE) and [derivative notice](NOTICE).
