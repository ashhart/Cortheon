Try to BREAK the candidate implementation using concrete, undeniable counterexamples.
Rules:
1. Ignore the author's comments; check the code line-by-line against EVERY word of the problem specification.
2. Hunt for: off-by-one bounds, wrong tie-breaking, mutated input, stale/leaked state, precision loss, cycle/empty/singleton crash, wrong operator precedence, or missing constraint.
3. Write 3 to 5 surgical `def test_<bug_name>():` functions (using `from solution import *`, already imported).
4. Compute expected values ONLY when 100% certain (or via a 5-line exact brute-force check inside the test!). Never guess an expected value.
5. Output ONLY one fenced ```python``` block with `def test_...():` functions.
