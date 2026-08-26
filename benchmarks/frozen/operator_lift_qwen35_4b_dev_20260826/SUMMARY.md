# Operator-lift live development smoke: Qwen3.5-4B (2026-08-26)

A live development-only slice of the sealed 60-case operator-lift instrument,
run with a real local model through the evaluator-owned generic MCP host. This
is a development smoke, not a claim. `claim_eligible=false`, every proof gate
remains false, and no promotion follows from it.

## Run

- model: `mlx-community--Qwen3.5-4B-MLX-8bit` served by local oMLX loopback.
- instrument: the sealed 60-case bank, hypothesis-framing cluster only,
  three repetitions per condition (full / without_hypothesis_framing /
  equal_budget_placebo), 9 cells, all identity-valid, no timeout.
- total runtime about 2.4 minutes for the nine cells.

## Outcomes

| Operator | Full | Operator removed | Equal-budget placebo |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 3/3 | 3/3 | 0/3 |

- `full`: 3/3 correct, 3/3 delivered, 2-3 model steps each.
- without_hypothesis_framing: 3/3 correct, 3/3 delivered, 1 step each.
- equal_budget_placebo: 0/3 correct, 3/3 delivered (the same bare model with
  no runtime).

## Honest reading

The live full arm beat the placebo 3/3 versus 0/3, consistent with the
retained development pilot. As in that pilot, removing the hypothesis-framing
operator did not measurably hurt this cluster (removed also 3/3), and three
cells per arm cannot support any operator-level inference. The point of this
run is procedural: the sealed instrument, the frozen case bank, the generic
host, and the runtime now execute end to end with a real model, producing
schema-14 records with identity validation, which is the loop the larger
campaign depends on. Run the nine-cell smoke on settled bytes, then the
540-cell instrument, before any operator conclusions.
