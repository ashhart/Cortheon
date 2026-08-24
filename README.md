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
| Qwen3.5 4B MLX, 8-bit | Generic MCP / oMLX | Yes | Yes | Five public development contrasts |
| Qwen3.5 9B MLX, 8-bit | Generic MCP / oMLX | Yes | Yes | Paired diagnostic; invalid for claims |

The 4B development pilots used one public case cluster per operator and three
repetitions per arm:

| Operator | Full Cortheon | Operator removed | Equal-budget placebo |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 3/3 | 1/3 | 0/3 |
| Discriminating evidence | 3/3 | 3/3 | 0/3 |
| Cross-source derivation | 3/3 | 0/3 | 0/3 |
| Adaptive stopping | 3/3 | 0/3 | 0/3 |
| Contradiction revision | 3/3 | 1/3 | 0/3 |

All 45 cells were identity-bound and safe. These are public development cases,
not hidden independent evidence. They show substrate amplification on these
cases; they do not prove general lift or frontier parity.

## Install

Cortheon requires Python 3.11 or newer. Install it from a repository checkout:

```bash
python3 -m pip install .
cortheon --version
```

Cortheon is not yet available from a package registry.

## Connect a host

### Pi

```bash
cortheon install --host pi
cortheon doctor --host pi
```

Restart Pi, then enable Cortheon for the session:

```text
/cortheon enable
/cortheon status
```

Use `/cortheon disable` to turn it off.

### OpenCode

```bash
cortheon install --host opencode
cortheon doctor --host opencode
```

Restart OpenCode. Its adapter starts with the host and has no per-session
command.

### Codex

```bash
cortheon install --host codex
cortheon doctor --host codex
```

Start a new Codex chat. The installer binds the plugin to Cortheon's Python
environment and checks the protocol version and source fingerprint.

### Generic MCP

```bash
cortheon configure --host generic
cortheon conformance --host generic
```

`configure` prints an MCP entry but does not edit the host configuration. This
mode is cooperative because a generic MCP server cannot intercept every host
tool or the final answer.

Add `--scope project` to Pi or OpenCode install, doctor, and uninstall commands
for a project-only integration.

## Diagnose and manage Cortheon

Run `cortheon doctor --host HOST` to inspect an adapter and runtime identity.
Run `cortheon conformance --host HOST` for the protocol smoke test. Replace
`HOST` with `pi`, `opencode`, `codex`, or `generic`.

Native adapters start the memory-only runtime when needed. `doctor` therefore
accepts a missing runtime unless you add `--require-runtime`. When one is
running, it checks the service name, protocol, package version, and source
fingerprint.

Remove an integration with `cortheon uninstall --host HOST`. Generic MCP has no
Cortheon-owned host file, so remove that entry yourself. Pi and OpenCode keep
one `.cortheon.bak` before the first configuration change and reject malformed
or symlinked configuration files.

## Generic MCP

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
| Wheel size | 240,000 bytes |
| Source distribution size | 220,000 bytes |
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
