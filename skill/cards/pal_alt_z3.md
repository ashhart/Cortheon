Solve the problem with an INDEPENDENT computational method (different from a closed-form formula):
- For logic, scheduling, knights/knaves, zebra, integer equations, or graph constraints: model EVERY rule directly in `z3` (`from z3 import *`) AND check uniqueness. Keep the code compact (<85 lines).
- For linear programming, minimax / zero-sum matrix games, or exact rational optimization: model directly with `z3.Optimize()` and `z3.Real` variables (`opt.maximize(v)` / `opt.minimize(u)`) — far cleaner and less bug-prone than manual support enumeration!
- For probability, Markov chains, or combinatorics: use exact DP, exact rational linear system (`sympy.Matrix` / `Fraction`), or direct enumeration!
- NEVER assert the computed result equals a hand-calculated guess. Only assert problem constraints.
- Print `ANSWER: <exact_value>` on the LAST line of stdout.
- Output ONLY one concise fenced ```python``` block.
