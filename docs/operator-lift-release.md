# Operator-lift release records

Finalized repository-only runs retain `run.json`, `report.json`, and
`release.json`. Raw model output, evaluator traces, checkpoints, and case
material are private recovery state and are removed after finalization. The
retained files contain fixed enums, counts, verdicts, opaque case commitments,
and a run-bound hash chain.

The commitments identify the public development pack. They do not hide known
cases. Hidden packs need evaluator-issued opaque identifiers.

The chain detects corruption. It supports an authenticity claim only after its
root is pinned outside the run directory. Verify a published root and the full
run descriptor with:

```bash
python -m cortheon.operator_lift.cli verify-release \
  --release OUTPUT/release.json \
  --report OUTPUT/report.json \
  --run OUTPUT/run.json \
  --expected-chain-root PUBLISHED_SHA256
```

Replay checks identities, schedule coverage, measurements, operator and
placebo results, and the report digest. It revalidates evaluator-recorded
grades. It cannot re-grade semantic correctness without private answers.
