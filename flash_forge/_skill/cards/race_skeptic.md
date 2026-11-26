Assume the obvious implementation has a subtle bug.
Rules:
- Explicitly guard every tricky edge case ACTUALLY implied by the spec: empty, singleton, duplicates, negative/zero values, exact ties, boundary indices (`<` vs `<=`), floating-point drift, state leaking across calls.
- Stay 100% faithful to EVERY starter import, Requirement, function signature, return format, and exception type — use the declared stdlib APIs directly (`os.path.exists`, `glob.glob`, `os.makedirs(..., exist_ok=True)`, explicit modes, `threading.Thread`) and NEVER invent unrequested validations, extra exceptions, or path rewrites.
- Keep code clean and dense: NO multi-line docstrings, ASCII banners, or redundant comments.
- Output ONLY one complete, self-contained fenced ```python``` block containing the exact requested definitions.
