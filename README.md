# Cortheon

Cortheon is a lightweight runtime that helps a small local model reason,
discover, and complete work like a frontier model.

It runs alongside Pi, OpenCode, Codex, or any MCP-compatible host. The host
keeps control of the model, files, network, tools, and permissions. Cortheon
decides what evidence the task needs, compares explanations, connects facts
across sources, and checks whether the work is complete.

Sessions stay in memory. Cortheon does not retain project files or build a
knowledge database. When evidence is missing, it can make one focused follow-up
request. It withholds certification if the task remains unsettled. If the
runtime stops responding, the host continues and marks the result uncertified.

> Cortheon is alpha software. The project has not proved universal frontier
> parity.

## Install in three steps

You need Python 3.11 or newer, Git, and one supported host. Cortheon does not
use Docker or a second model.

### 1. Clone and install

```bash
git clone https://github.com/ashhart/Cortheon.git
cd Cortheon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
cortheon --version
```

Keep the virtual environment active while running Cortheon commands. In a new
terminal, return to the repository and run `source .venv/bin/activate`.

### 2. Connect your host

Pick the host you use.

For Pi:

```bash
cortheon install --host pi
cortheon doctor --host pi
```

Restart Pi, then run these commands inside Pi:

```text
/cortheon enable
/cortheon status
```

For OpenCode:

```bash
cortheon install --host opencode
cortheon doctor --host opencode
```

Restart OpenCode. Cortheon starts when the next task begins.

For Codex:

```bash
cortheon install --host codex
cortheon doctor --host codex
```

Start a new Codex task. Cortheon starts with the task.

For another MCP host:

```bash
cortheon configure --host generic
```

Copy the printed JSON into the host's MCP configuration and restart the host.
Generic MCP mode is cooperative. Pi and OpenCode can enforce the reasoning and
completion checks more reliably.

### 3. Run a task and check it

Ask the model to complete a real task. After the first task, replace `HOST`
with `pi`, `opencode`, or `codex` and run:

```bash
cortheon doctor --host HOST --require-runtime
cortheon conformance --host HOST
cortheon results
```

For generic MCP, run `cortheon conformance --host generic` instead. The
`results` command reports content-free counters for the current runtime. It
never includes prompts, answers, file contents, or URLs.

Remove an integration with `cortheon uninstall --host HOST`. Pi and OpenCode
also support `--scope project` if you want Cortheon enabled for one repository
only.

## The goal

Cortheon aims to make a small model perform like a frontier system by improving
how it reasons, finds current information, combines evidence, and verifies its
work. That is a benchmark target, not a product claim.

Cortheon cannot add knowledge to a model's weights. It can help the model work
from fresh evidence and expose unsupported conclusions.

We test the goal with blind paired runs. The same small model attempts hidden
tasks with and without Cortheon. A frontier model receives the same tasks and
grading rules. Every run uses one model endpoint. Cortheon does not stack
models or delegate difficult reasoning to a second model.

We measure verified completion, supported new deductions, recovery after bad
evidence, false approvals, false blocks, latency, tool calls, crashes, and
repeated-request loops.

## Models tested

| Model | Host | Without Cortheon | With Cortheon | Result |
| --- | --- | --- | --- | --- |
| Qwen3.5 0.8B, 8-bit | OpenCode / oMLX | Not retained | Smoke run | Lifecycle only; ungraded |
| Qwen3.5 4B MLX, 8-bit | Generic MCP / oMLX | Yes | Yes | Retained public pilot, 24 August 2026 |
| Qwen3.5 9B MLX, 8-bit | Generic MCP / oMLX | Yes | Yes | Old diagnostic; invalid for claims |

The fresh 4B run used one public case per operator and three repetitions per
arm:

| Operator | Full Cortheon | Operator removed | Equal-budget placebo |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 3/3 | 2/3 | 0/3 |
| Discriminating evidence | 3/3 | 3/3 | 0/3 |
| Cross-source derivation | 3/3 | 0/3 | 0/3 |
| Adaptive stopping | 3/3 | 0/3 | 0/3 |
| Contradiction revision | 3/3 | 0/3 | 0/3 |

Full Cortheon was correct in 15 of 15 cells. The operator-removed arms were
correct in 5 of 15, and the placebo was correct in 0 of 15. All 45 cells were
identity-bound, transcript-valid, and safe, with no timeout. Five
operator-removed cells were withheld. The configured budgets matched, but the
realized compute did not.

These are public development cases, not hidden independent evidence. The run
does not isolate lift from discriminating evidence because that operator's
removed arm also scored 3/3. Every development gate remains false. See the
[retained reports and chain roots](benchmarks/frozen/operator_lift_qwen35_4b_20260824/SUMMARY.md).

## Run a fair before-and-after test

Use the same host, local model, quantization, settings, repository state, and
prompt in two fresh sessions. Run the baseline first. For Pi, leave Cortheon
disabled. For OpenCode or Codex, run the baseline before installing Cortheon.
Then enable or install Cortheon and repeat the prompt without changing it.

Pick work where extra reasoning can change the outcome. Good tests include a
bug spread across several files, a question that needs current primary sources,
or a conclusion that must combine two documents. Record correctness, completed
tests, working citations, elapsed time, tool calls, and any repeated request.

After the assisted run, inspect Cortheon's content-free counters:

```bash
cortheon results
```

The counters cover the current runtime process. For several tasks, capture the
output before and after each task. `sessions_completed` and
`hook_turns_certified` show successful gates. `completion_withheld`,
`hook_uncertified_releases`, and `controller_zero_gain_stops` expose failures
and bounded stops. Cortheon never puts prompts, answers, file contents, or URLs
in this report.

Share this short report with the test group. Redact project names and secrets:

```text
Host and version:
Model ID and quantization:
Hardware:
Exact task type:
Baseline result:
Cortheon result:
Tests or citations checked:
Elapsed time and tool calls:
Repeated requests or stalls:
Cortheon results output:
```

If `doctor`, `conformance`, or `results` reports a runtime identity mismatch,
an older Cortheon process still owns port 8743. Close the hosts using Cortheon,
stop that old `cortheon serve` process, and restart the selected host. The
commands fail instead of accepting metrics or conformance from the wrong build.

Generic MCP has no Cortheon-owned host file, so remove that configuration entry
yourself. Pi and OpenCode keep one `.cortheon.bak` before the first
configuration change and reject malformed or symlinked configuration files.

## Generic MCP reference

Run the MCP server with `cortheon-mcp` or `cortheon mcp`. The default exposes
session lifecycle tools with fixed limits. Add `--advanced` for debugging.

```json
{
  "mcpServers": {
    "cortheon": {
      "command": "cortheon-mcp"
    }
  }
}
```

Native plugins start their memory-only loopback transport automatically. For
adapter debugging, it can also be run directly on `127.0.0.1:8743`:

```bash
cortheon serve
```

Set `CORTHEON_COGNITIVE_TOKEN` if another local process can reach the port.

## What certification means

- A code change needs a diff and tests run by the host.
- A numerical answer needs primary data or a reproducible calculation.
- Current research needs recent sources and a check for conflicts.
- A private-record claim needs direct evidence from those records.
- Sites repeating the same underlying report count as one source.

Certification means the supplied evidence supports the stated result. It does
not prove that every source is truthful. Cortheon reports the assurance the
evidence earned instead of treating every completed tool call as proof.

Cortheon limits evidence per session and quarantines instructions found inside
evidence. Sessions disappear after completion, abandonment, or expiry.

For Codex, nested web work reaches hooks as one outer execution receipt.
Cortheon does not use that evidence for published benchmark claims.

## Package limits

Release tests enforce these limits:

| Property | Limit |
|---|---:|
| Third-party runtime dependencies | 0 |
| Wheel size | 241,000 bytes |
| Source distribution size | 222,000 bytes |
| Installed console commands | 2 |
| Runtime project database | None |

The wheel contains the runtime and host adapters. It excludes benchmark and
experimental modules.

## Development and release

```bash
python3 -m pip install -e '.[test]'
PYTHONPATH=src pytest -q
ruff check src tests setup.py
```

Run the source-bound release check with:

```bash
scripts/verify-release --status
```

The check fails if source files change while it runs. It records the source
digest, test results, package contents, and artifact hashes.

Operator-lift runs keep small release records instead of source material or
model responses. Their hash chain proves continuity only when someone stores
the root hash independently. See
[the release-record documentation](docs/operator-lift-release.md).

## Proof status

Cortheon does not yet have a valid universal-frontier result. Older runs are
development evidence, not proof of the main claim.

A publishable result still needs preregistered hidden tasks, isolated runs of
the same model, named reduced Cortheon conditions, external grading, and every
failure counted. It also needs live replication across Pi, OpenCode, Codex,
and generic MCP hosts, followed by an independent repeat outside this project.
