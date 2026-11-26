Write a CONCISE modular Python test + differential probe suite (<= 120 lines total, no docstrings) for `solution.py` (imported via `from solution import *`).
Rules:
1. Write 4 to 6 crisp `def test_<name>():` functions with 100% indisputable assertions: spec examples, EVERY explicit `Raises:` exception condition, empty/blank inputs (e.g. `""`, `[]`), and round-trip/structural invariants (NEVER guess complex hand-computed traces).
2. ALSO write ONE `def diff_probe() -> dict:` function that runs 6 to 10 diverse inputs (empty/blank args, singleton, boundary/nested-subdir/duplicate/mock/exception cases) WITHOUT asserting expected values, returning `{case_name: repr_or_value}` (catch exceptions per case as `f"EXC:{type(e).__name__}"`, use `tempfile.TemporaryDirectory()` for file I/O). This fingerprints subtle behavioral differences across candidates!
3. Output ONLY one complete fenced ```python``` block. No top-level calls.
