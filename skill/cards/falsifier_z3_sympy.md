Wave-1 candidates DISAGREED or FAILED. Discard prior derivations and model the entire problem from scratch using `z3` (`z3.Int`/`z3.Real`/`z3.Optimize()` for constraints, LP, zero-sum games, scheduling) OR `sympy` + `fractions.Fraction` (for exact probability/algebra/combinatorics).
Rules:
- Never use floating-point arithmetic for exact answers.
- Avoid every failure in the graveyard.
- Assert solution validity (and uniqueness if applicable) inside the script, print `VERIFIED`, and print `ANSWER: <exact_value>` on the LAST line.
- Output ONLY one complete fenced ```python``` block.
