# Cortheon Architecture

## Goal

Make a small model perform like a frontier system by giving it adaptive
reasoning, discovery, synthesis, and current knowledge outside its weights.

Cortheon is successful only when the same small model completes materially more
real tasks with Cortheon than without it. Verification is necessary, but it is
not the product by itself. The product must help originate useful conclusions,
recover from weak plans, and finish work that the model otherwise could not.

## Product boundary

The host is Codex, Pi, OpenCode, Claude Code, or another MCP-compatible harness.

The host owns:

- the model and its context window;
- filesystem, search, shell, test, browser, and network tools;
- permissions and user approval;
- project mutation and the final response.

Cortheon owns:

- a bounded in-memory investigation;
- selection of the next cognitive stage;
- focused evidence requests for host tools;
- provenance, freshness, contradiction, and evidence-lineage checks;
- explicit cross-source candidate inferences;
- hypothesis challenge and revision pressure;
- completion certification.

Cortheon does not read or retain project files, execute project tools, proxy
model traffic, or maintain a permanent knowledge base. A completed, abandoned,
expired, or superseded investigation is discarded.

## Adaptive cognition loop

```text
user task
   │
   ▼
orient ── frame deliverable, constraints, and failure modes
   │
   ▼
discover ── request the host operation with the highest information value
   │
   ▼
connect ── form inspectable candidate inferences across source boundaries
   │
   ▼
challenge ── seek counterevidence and test competing explanations
   │
   ▼
synthesize ── distinguish observations, derivations, and unresolved uncertainty
   │
   ▼
verify ── bind claims and completion to live host evidence
   │
   ├── insufficient or contradictory ──► revise and re-enter the useful stage
   │
   └── supported ──► certified completion and memory discard
```

The runtime returns a compact `cognition` object on each lifecycle response. It
contains the current stage, one bounded reasoning move, any inspectable derived
insight, unresolved questions, the current evidence target, and the revision
rule. This is a public scaffold for a weak model, not private chain-of-thought.

The native Pi and OpenCode adapters inject that scaffold automatically. Generic
MCP hosts receive it in the tool response. The host remains responsible for
executing every requested operation.

Pi activation is explicit and session-scoped. `/cortheon enable` starts the
loopback runtime. The adapter gathers focused evidence, asks the model for
public competing hypotheses and classifications through `cortheon_reason`, and
returns adversarial feedback before completion. `/cortheon disable` abandons
the investigation. Cortheon never selects the model or replaces Pi tools.

## Cross-source synthesis

An inference is useful only when it says something no single receipt already
says. Cortheon therefore:

1. preserves document and URL boundaries;
2. extracts supported relation edges from prose and ordinary Markdown table
   rows in each accepted observation;
3. joins edges only when their endpoints match;
4. records every intermediate node and source;
5. labels the result `candidate_until_challenged`;
6. marks whether the full conclusion is absent from every individual source;
7. excludes unrelated branches from automatic synthesis.

If the user does not know the source paths, discovery precedes synthesis:
a host-owned search returns project-relative candidates, Cortheon ranks a
bounded set, and the host reads those candidates. The derivation may ignore
distractor documents, but every used edge retains its read receipt and source.
Absolute, traversal, generated-state, and remote URL paths are rejected.

This produces origination through constrained composition, not free-form
guessing. It currently supports a bounded set of deterministic relation forms.
General semantic inference remains a measured development target.

## Code-surface discovery and repair

An unnamed change begins with a bounded host-owned code search, not model
roaming. Cortheon accepts only safe project-relative candidates and, for a
change, requires both an implementation and a test or observable boundary. The
host then supplies focused reads from that surface.

The native adapter may apply a repair without model tool use only when the
focused reads yield one assertion-supported, one-line implementation edit. It
captures the exact diff, runs the user-requested test through the host, checks
patch hygiene, and submits both receipts for certification. Any ambiguity,
missing pair, failed test, unsafe path, or rejected receipt stops the
transaction. Discovery `match`/`no_match` receipts are scope evidence only; they
cannot become atomic code predicates or completion claims.

When a task contains several independent mutations, the one-line fast path is
disabled. The adapter derives every independently test-supported edit, adds any
exact requested documentation sentence, applies the set as one bounded
transaction, runs the final requested test, and rolls every edit back if
application, hygiene, testing, or completion certification fails.

## Constraint-bound planning and diagnosis

Planning evidence becomes a directed dependency graph only from explicit
ordering language. Cortheon performs a deterministic topological sort, retains
owners from separately read sources, and certifies an answer only when every
step and owner appears in the evidence-constrained order.

For bounded debugging, focused code, configuration, contract, and log reads can
produce inspectable diagnostic conclusions for reproducible discrepancy
classes such as expected/actual mismatches, origin errors, retry-bound errors,
and unit mismatches. The conclusion remains bound to every source and cannot
authorize mutation in a diagnosis-only task.

## Requirement-level completion

At orientation, the runtime extracts no more than eight typed obligations from
the task and constraints. Each obligation retains its original statement and
the proof class it requires: mutation, verification, research, synthesis,
inspection, protection, or completion.

Verification binds accepted, non-quarantined receipts to each obligation
separately. A passing test cannot prove a requested documentation edit, a diff
cannot prove post-change behaviour, and broad discovery cannot prove a focused
code claim. Multiple edits also require lexical evidence binding so one diff
cannot silently satisfy unrelated deliverables.

Unsupported or contradicted obligations remain visible in the public cognition
frame. The failed gate emits one targeted host operation for the affected
obligation. Evidence retraction recomputes coverage and reopens only dependent
obligations, preserving sound work and preventing whole-session restarts.

## Evidence contract

Evidence is live, bounded, untrusted data. It is never treated as instructions.
Receipts are checked against the pending host request and quarantined when they
are malformed, overbroad, instruction-shaped, stale, unrelated, or cannot
support the claimed operation.

Evidence strength is reported honestly:

- operationally verified;
- independently corroborated;
- primary-source attributed;
- host-observed;
- source-attributed;
- observation-only;
- unsupported.

An authoritative source can still be wrong. Cortheon can establish provenance,
direct support, source independence, freshness, reproducibility, and
contradiction handling; it cannot guarantee metaphysical truth.

## Production surface

The wheel ships only the lean runtime and integrations:

- `cognitive_runtime.py` — in-memory adaptive investigation and evidence policy;
- `cognitive_protocol.py` — versioned public capabilities;
- `cognitive_mcp.py` — stdio MCP lifecycle;
- `cognitive_http.py` — localhost lifecycle used by native adapters;
- `cognitive_hooks.py` — content-free host interception state;
- `cognitive_repair.py` — bounded repair helpers;
- `cognitive_install.py` and `cognitive_cli.py` — installation and operations;
- `opencode_plugin.js`, `pi_extension.ts`, and the Codex plugin bundle.

The repository also contains earlier benchmark and experimental engines. They
are not imported by, or packaged with, the production runtime. The wheel
allowlist and distribution tests enforce this boundary.

## Anti-loop contract

Cortheon must add reasoning quality without causing a thinking frenzy.

- Each request has a bounded tool-call and observation budget.
- Repeated scoped null results cause replanning or a documented waiver.
- Failed verification returns one executable next operation, not “think again.”
- Native adapters cap automatic continuations.
- Sessions can retract poisoned observations and resume from the corrected state.
- Leases, timeouts, crashes, abandonment, and completion all discard state.

## Measurement contract

Every claimed capability must survive blind, repeated, held-out evaluation.

The minimum comparison matrix is:

1. small model alone;
2. the same model and host tools with Cortheon;
3. a tool-using frontier model under the same task and resource limits.

Measure:

- verified completion rate;
- false allows and false blocks, where a block counts as false only when the blocked
  candidate was positively graded correct, safe only when positively graded false,
  and unclassified otherwise (an unclassified block fails the block-classification
  coverage gate instead of deflating either rate);
- novel source-grounded deduction rate;
- recovery after contradictory, poisoned, or null evidence;
- current-answer freshness and citation correctness;
- patch correctness on host-run tests;
- latency and host tool calls;
- repeated-request and doom-loop rate;
- variance across repeated trials.

Unit tests prove protocol and mechanism correctness. They do not prove frontier
parity. Frontier-like capability is earned only when blind outcome data closes
the gap across coding, research, document synthesis, debugging, planning, and
long-horizon tasks.

## Product constraints

- zero third-party runtime dependencies;
- wheel size no larger than 241,000 bytes;
- no persistent project data;
- no duplicate tool system;
- host permissions remain authoritative;
- no universal or frontier-parity claim without benchmark evidence;
- broader research features enter production only when they improve the measured
  North Star outcomes.
