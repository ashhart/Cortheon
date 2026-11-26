Derive the solution from first principles before coding.
Rules:
- Prove the core invariant and state transition / loop termination carefully in thinking.
- Eliminate every hidden quadratic bottleneck, recursion-depth risk, or precision trap.
- Double-check tie-breaking, boundary indices, and empty/degenerate inputs.
- Honor EVERY starter import, Requirement, exact signature, and return format directly using idiomatic mock-friendly stdlib calls (`os.path.exists`, `glob.glob`, `os.makedirs(..., exist_ok=True)`, explicit file/zip modes, `threading.Thread`) without unrequested library swaps or extra exceptions.
- Keep code clean and dense: NO multi-line docstrings, ASCII banners, or redundant comments.
- Output ONLY one complete, self-contained fenced ```python``` block containing the exact requested definitions.
