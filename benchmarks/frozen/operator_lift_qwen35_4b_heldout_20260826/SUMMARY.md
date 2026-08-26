# Held-out live pilot, all operators: Qwen3.5-4B on the sealed P6 pack (2026-08-26)

Development measurements on cases from the sealed held-out pack. The retained
`run.json`, `report.json`, and `release.json` form one sealed 54-cell
hypothesis-framing pilot. Despite its historical filename,
`release_all_operators.json` is an explicitly
unsealed diagnostic concatenation of five independent 27-cell release chains;
it is useful for debugging but cannot be presented as one auditable release.
Neither artifact is claim-eligible.

## Runs

- model: `mlx-community--Qwen3.5-4B-MLX-8bit` (local oMLX).
- pack: `benchmarks/frozen/p6_heldout_pack` (60 fresh cases, isolated
  vocabulary).
- sealed release: 6 hypothesis-framing cases x 3 conditions x 3 repetitions =
  54 cells.
- unsealed diagnostic: 3 cases x 3 conditions x 3 repetitions = 27 cells per
  operator, concatenated across 5 independent chains (135 records total).

## Outcomes (correct / scheduled)

| Operator | full | placebo | operator removed |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 7/9 | 0/9 | 9/9 |
| Discriminating evidence | 9/9 | 0/9 | 9/9 |
| Contradiction revision | 6/9 | 0/9 | 0/9 |
| Cross-source derivation | 9/9 | 0/9 | 6/9 |
| Adaptive stopping | 0/9 | 0/9 | 0/9 |
| Total | 31/45 | 0/45 | 24/45 |

The table above comes from the unsealed diagnostic. It is not a release result
and must not be combined with the sealed 54-cell pilot as though both shared a
single schedule, manifest, or hash chain.

## Reading

Out of sample the structure holds and sharpens: removal of the framing and
discrimination operators again shows no isolated effect; removal of
derivation and of revision costs ground (9 to 6 and 6 to 0); adaptive
stopping does not transfer as-is to the held-out instances (0/9 correct, and
the full arm delivers only 4/9), a genuine transfer gap before the powered
campaign. Root cause, traced and corrected in two parts: the held-out
stopping pack's expected-action order did not align with the runtime's
ascending cost order (fixed by cost-aligning the pack, with a compliance
test), and the replay's probe receipts used the filePath receipt key when
the runtime's execution ledger reads args.path (fixed and tested). A
residual replay iteration nuance on this one family remains for the live
runner to adjudicate; the P6 campaign itself runs live, not through the
replay, and its stopping hold-outs are now contractually aligned. Three
cases per arm cannot support operator inferences; the purpose here is the
procedure: the sealed pack and the `--heldout` run path now execute every
operator's protocol out of sample, which is the loop the P6 campaign
extends.
