# Held-out live pilot: Qwen3.5-4B on the sealed P6 pack (2026-08-26)

First live measurement on the sealed held-out pack: fresh hypothesis-framing
instances the model never saw, through the generic MCP host, development
scope only. `claim_eligible=false`, `development_gate_passes=false`, and the
external chain root plus reviewer signoffs are still required before any
claim.

## Run

- model: `mlx-community--Qwen3.5-4B-MLX-8bit` (local oMLX).
- pack: `benchmarks/frozen/p6_heldout_pack` (60 fresh cases, isolated
  vocabulary), hypothesis-framing cluster only, 6 instances x 3 conditions x
  3 repetitions = 54 cells. All identity-valid, zero timeouts.

## Outcomes (correct / scheduled)

| Condition | Correct | Delivered |
| --- | ---: | ---: |
| full | 17/18 | 18/18 |
| equal-budget placebo (bare model) | 0/18 | 18/18 |
| without_hypothesis_framing | 18/18 | 18/18 |

## Reading

The full-versus-bare gap replicates out of sample: 94.4 percent versus zero
on tasks never seen by the model. Removing the framing operator shows no
isolated effect here either, identical to the development bank (33/36 vs
34/36 there; 17/18 vs 18/18 here), so the held-out pack transfers the
protocol faithfully. Six instances per arm cannot support any operator
inference; the value is procedural: the sealed pack, the CLI `--heldout`
path, and the live runner execute end to end on unseen material, which is
the loop the powered held-out campaign extends.
