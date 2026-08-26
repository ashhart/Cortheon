# Operator-lift full campaign: Qwen3.5-4B (2026-08-26)

Full 540-cell development run of the sealed operator-lift instrument with a
real local model through the evaluator-owned generic MCP host. Development
scope only: `claim_scope=development_operator_lift_execution_only`,
`development_gate_passes=false`, event authenticity requires the external
chain root, and promotion needs the two external reviewer signoffs. This is a
measurement, not a claim.

## Run

- model: `mlx-community--Qwen3.5-4B-MLX-8bit` served by local oMLX.
- design: 60 sealed cases, 3 conditions (full, without-operator ablation,
  equal-budget placebo), 3 repetitions each; 540 cells completed, zero
  timeouts, all records schema-validated.
- 9 cells carry `terminal_status=transport_error` (generic host subprocess
  transport failures, evaluator-attested) and are invalid, not scored.

## Per-cluster results (correct / scheduled)

| Operator | full | placebo | operator removed |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 33/36 | 0/36 | 34/36 |
| Discriminating evidence | 29/36 | 0/36 | 35/36 |
| Contradiction revision | 18/36 | 0/36 | 17/36 |
| Cross-source derivation | 34/36 | 0/36 | 15/36 |
| Adaptive stopping | 18/36 | 0/36 | 0/36 |
| Total | 132/180 | 0/180 | 101/180 |

Total correct 233/540, delivered 454/540 (84%).

## Honest reading

The full arm beat the placebo 132/180 against 0/180, consistent with the
dev smoke. Removing derivation or stopping costs the runtime measurable
ground (34 to 15, and 18 to 0), while removing framing, discrimination, or
revision leaves this cluster family flat or slightly higher, meaning those
operators do not isolate lift on this bank. The nine invalid cells are all
transport errors, not wrong answers. No proof gate passes: development
accounting is complete, but this is the input to the preregistered P6
same-model qualification, not a result of it.
