Re-read EVERY word of the problem for traps ("at least" vs "exactly", inclusive vs exclusive, conditional event, symmetry, base cases).
Rules:
- Keep thinking brief—translate EVERY sentence/constraint directly into compact Python (<85 lines) using `Fraction`/`sympy`/`z3`/`int`.
- Assert EVERY constraint from the prompt holds on the final solution, NEVER assert equality against a hand-calculated guess or random simulation.
- Print `VERIFIED`, then `ANSWER: <exact_value>` on the LAST line of stdout.
- Output ONLY one concise fenced ```python``` block.
