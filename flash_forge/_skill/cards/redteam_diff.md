Hunt for subtle algorithmic or differential bugs between the candidate implementation and the true specification.
Rules:
1. If an alternative candidate is shown, find inputs where Candidate 1 and Candidate 2 DISAGREE, determine which one violates the spec, and assert the spec-correct result.
2. Write a compact brute-force / invariant checker on nasty adversarial inputs (ties, degenerate geometry, pathological graphs, non-trivial nesting, boundary sizes) inside `def test_<name>():` functions.
3. Be 100% certain every assertion strictly follows the problem specification.
4. Output ONLY one fenced ```python``` block with `def test_...():` functions.
