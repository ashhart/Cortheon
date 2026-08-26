# Held-out live pilot, all operators: Qwen3.5-4B on the sealed P6 pack (2026-08-26)

Live measurements on the sealed held-out pack (fresh instances the model
never saw) across every operator cluster, development scope only.
`claim_eligible=false`, `development_gate_passes=false`; the external chain
root and reviewer signoffs are still required before any claim.

## Runs

- model: `mlx-community--Qwen3.5-4B-MLX-8bit` (local oMLX).
- pack: `benchmarks/frozen/p6_heldout_pack` (60 fresh cases, isolated
  vocabulary).
- per operator: 3 cases x 3 conditions (full, placebo, operator removed) x 3
  repetitions = 27 cells; 5 operators = 135 cells. All identity-valid, zero
  timeouts.

## Outcomes (correct / scheduled)

| Operator | full | placebo | operator removed |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 7/9 | 0/9 | 9/9 |
| Discriminating evidence | 9/9 | 0/9 | 9/9 |
| Contradiction revision | 6/9 | 0/9 | 0/9 |
| Cross-source derivation | 9/9 | 0/9 | 6/9 |
| Adaptive stopping | 0/9 | 0/9 | 0/9 |
| Total | 31/45 | 0/45 | 24/45 |

Combined with the earlier 6-cluster hypothesis pilot (17/18), the
full-versus-bare gap persists on every held-out operator family.

## Reading

Out of sample the structure holds and sharpens: removal of the framing and
discrimination operators again shows no isolated effect; removal of
derivation and of revision costs ground (9 to 6 and 6 to 0); adaptive
stopping does not transfer as-is to the held-out instances (0/9 correct, and
the full arm delivers only 4/9), a genuine transfer gap to investigate
before the powered campaign, not a reason to weaken any gate. Three cases
per arm cannot support operator inferences; the purpose here is the
procedure: the sealed pack and the `--heldout` run path now execute every
operator's protocol out of sample, which is the loop the P6 campaign
extends.
