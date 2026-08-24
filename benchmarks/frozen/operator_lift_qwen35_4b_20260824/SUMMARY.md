# Qwen3.5 4B public operator pilot, 24 August 2026

This run used `mlx-community--Qwen3.5-4B-MLX-8bit` through a local oMLX
endpoint and Cortheon's generic MCP evaluator. It ran one public development
case for each operator, with three repetitions of full Cortheon, the named
operator removed, and a placebo with the same configured limits.

| Operator | Full | Operator removed | Placebo |
| --- | ---: | ---: | ---: |
| Hypothesis framing | 3/3 | 2/3 | 0/3 |
| Discriminating evidence | 3/3 | 3/3 | 0/3 |
| Cross-source derivation | 3/3 | 0/3 | 0/3 |
| Adaptive stopping | 3/3 | 0/3 | 0/3 |
| Contradiction revision | 3/3 | 0/3 | 0/3 |

Across all operators, full Cortheon was correct and delivered in 15 of 15
cells. The operator-removed arms were correct in 5 of 15 cells and delivered
in 10. The placebo was correct in 0 of 15 cells and delivered in all 15. All
45 cells were safe, identity-bound, and transcript-valid. There were no
timeouts. The run used 305,143 tokens and took 783.53 seconds in total.

The configured budgets matched, but realized compute did not. Full Cortheon
used 125,148 tokens, 35 model steps, and 55 tool calls. The operator-removed
arms used 76,540 tokens, 19 steps, and 36 calls. The placebo used 103,455
tokens, 63 steps, and 33 calls.

| Operator | Chain root |
| --- | --- |
| Hypothesis framing | `d99c227df8b8663f2b693395aaff9ffcd7574714b84f6c61e17a53f137a67428` |
| Discriminating evidence | `9a5754fea49ad85068a944476706c7920e7f0547cffd11860d7c3b4f7a27446b` |
| Cross-source derivation | `d8666f559c919d88fa206c52f1b8a16d00c46fc449567fa97fbe7d2f902822e6` |
| Adaptive stopping | `fa28a9c894a70932633a2df7510e226901e2a21925a9c2a68d06d1b1adf95ff0` |
| Contradiction revision | `f09ba234c06aded2891b0c28a238ecf1af772667eed001bda5ef6e1fa87ab674` |

Each operator directory contains `run.json`, `report.json`, and `release.json`.
Verify a directory with its chain root:

```bash
python -m cortheon.operator_lift.cli verify-release \
  --release benchmarks/frozen/operator_lift_qwen35_4b_20260824/OPERATOR/release.json \
  --report benchmarks/frozen/operator_lift_qwen35_4b_20260824/OPERATOR/report.json \
  --run benchmarks/frozen/operator_lift_qwen35_4b_20260824/OPERATOR/run.json \
  --expected-chain-root CHAIN_ROOT
```

This is a small public development pilot. The three repetitions of one case
are not independent clusters. Every report therefore has
`development_gate_passes: false` and `pilot_claim_eligible: false`. The run
supports more testing. It does not prove general lift or frontier parity.
