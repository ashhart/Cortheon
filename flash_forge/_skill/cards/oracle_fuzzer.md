Write ONLY a concise BRUTE-FORCE DIFFERENTIAL FUZZER (`<= 75 lines`) to stress-test `solution.py` (`from solution import *` is already imported).
Rules:
1. Write a dead-simple, 100%-obviously-correct brute-force reference `_brute_ref(...)` for small inputs (slow $O(N^3)$ / naive model is ideal—clarity beats speed).
2. Write `def test_brute_differential_fuzz():` testing `solution` vs `_brute_ref` on 40 seeded random small cases (`random.Random(1337)`) covering negatives, zero, empty/singleton, duplicates, exact ties, degenerate items, and nested overlaps.
3. Write `def test_extreme_boundaries():` checking 1–2 obvious huge/negative cases (e.g. `+/- 10^9`).
4. Output ONLY one fenced ```python``` block. No top-level calls.
