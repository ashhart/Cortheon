"""50-Test Frontier Gauntlet II (Nightmare Tier) — Part 1: 25 Exact Math, Logic/CSP & Systems-Trap Tests (H01-H20, H41-H45)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Gauntlet2Problem:
    pid: str
    track: Literal["math_nm", "logic_nm", "algo_nm", "debug_nm", "sys_sec_nm"]
    forge_mode: Literal["reason", "code"]
    title: str
    prompt: str
    expected: str = ""
    hidden_tests: str = ""
    buggy_starter: str = ""


PART1_PROBLEMS: list[Gauntlet2Problem] = [
    # =========================================================================
    # TRACK 1: NIGHTMARE OLYMPIAD MATH & ALGORITHMIC NUMBER THEORY (H01-H10)
    # =========================================================================
    Gauntlet2Problem(
        pid="H01_noncoprime_crt_huge",
        track="math_nm",
        forge_mode="reason",
        title="Generalized Non-Coprime CRT Smallest Solution > 10^12",
        prompt=(
            "Find the SMALLEST integer x > 10^12 (1000000000000) satisfying ALL six non-coprime congruences:\n"
            "  x ≡ 11 (mod 24)\n"
            "  x ≡ 11 (mod 36)\n"
            "  x ≡ 41 (mod 42)\n"
            "  x ≡ 11 (mod 54)\n"
            "  x ≡ 27 (mod 56)\n"
            "  x ≡ 20 (mod 63)\n"
            "Give the exact integer x, ending with `ANSWER: <integer>`."
        ),
        expected="1000000000523",
    ),
    Gauntlet2Problem(
        pid="H02_lucas_binom_10e18",
        track="math_nm",
        forge_mode="reason",
        title="Lucas Theorem Binomial Coefficient at N ~ 7.5*10^17 (mod 1000003)",
        prompt=(
            "Compute the exact binomial coefficient:\n"
            "  C(750005120009120000, 310002100003690000) mod 1000003\n"
            "(Note: 1000003 is prime, and N is ~7.5 * 10^17 — use Lucas's Theorem!).\n"
            "Give the exact integer in [0, 1000002], ending with `ANSWER: <integer>`."
        ),
        expected="739078",
    ),
    Gauntlet2Problem(
        pid="H03_cond_markov_hitting_time",
        track="math_nm",
        forge_mode="reason",
        title="Conditional Expected Hitting Time Given Absorption at WIN (Doob h-Transform)",
        prompt=(
            "A particle walks on integer states {0, 1, 2, 3, 4, 5, 6}, where 0 (LOSE) and 6 (WIN) are absorbing states.\n"
            "From any transient state i in {1, 2, 3, 4, 5}, at each step it moves:\n"
            "- to i + 1 with probability 1/3,\n"
            "- to i - 1 with probability 1/2,\n"
            "- stays at i with probability 1/6.\n"
            "Starting at state 2, what is the exact CONDITIONAL expected number of steps until absorption, "
            "CONDITIONED ON the event that the particle is absorbed at state 6 (WIN) rather than state 0?\n"
            "(Careful: NOT the unconditional expected steps to absorption!)\n"
            "Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="1560/133",
    ),
    Gauntlet2Problem(
        pid="H04_matrix_exp_forbidden_strings",
        track="math_nm",
        forge_mode="reason",
        title="Length-10^15 Constrained Strings Automaton Matrix Exp (mod 10^9+7)",
        prompt=(
            "How many strings of length N = 10^15 (1000000000000000) over the 4-letter alphabet {A, B, C, D} satisfy BOTH:\n"
            "1. None of the 2-letter substrings 'AB', 'BC', or 'CA' appear anywhere in the string.\n"
            "2. No 3 consecutive characters are identical (i.e., none of 'AAA', 'BBB', 'CCC', 'DDD' appear).\n"
            "Compute the exact count modulo 1000000007. End with `ANSWER: <integer>`."
        ),
        expected="959237563",
    ),
    Gauntlet2Problem(
        pid="H05_pell_d61_fundamental",
        track="math_nm",
        forge_mode="reason",
        title="Fermat's Pell Equation x^2 - 61*y^2 = 1 Minimal Positive x",
        prompt=(
            "Find the SMALLEST positive integer x for which there exists a positive integer y satisfying Pell's equation:\n"
            "  x^2 - 61 * y^2 = 1.\n"
            "(Note: naive linear search over y will time out — use exact continued-fraction convergents of sqrt(61)!)\n"
            "Give the exact integer x, ending with `ANSWER: <integer>`."
        ),
        expected="1766319049",
    ),
    Gauntlet2Problem(
        pid="H06_mobius_squarefree_10e9",
        track="math_nm",
        forge_mode="reason",
        title="Exact Count of Squarefree Integers in [1, 10^9] via Mobius Function",
        prompt=(
            "How many positive integers n with 1 <= n <= 10^9 (1000000000) are SQUAREFREE "
            "(i.e., not divisible by p^2 for any prime p)?\n"
            "(Note: N = 10^9 is far too large for an O(N) sieve — use exact Mobius inclusion-exclusion in O(sqrt(N))!)\n"
            "Give the exact integer count, ending with `ANSWER: <integer>`."
        ),
        expected="607927124",
    ),
    Gauntlet2Problem(
        pid="H07_cubic_roots_symmetric_poly",
        track="math_nm",
        forge_mode="reason",
        title="Exact Integer Symmetric Polynomial of Cubic Roots (9th Powers + Quartic Diffs)",
        prompt=(
            "Let r1, r2, r3 be the three (possibly complex) roots of the cubic polynomial:\n"
            "  x^3 - 5*x^2 + 3*x - 7 = 0.\n"
            "Compute the EXACT integer value of:\n"
            "  (r1^9 + r2^9 + r3^9) + ((r1 - r2)^4 + (r1 - r3)^4 + (r2 - r3)^4).\n"
            "(Use exact Newton identities / elementary symmetric polynomials — do NOT rely on float64 root approximations!)\n"
            "Give the exact integer, ending with `ANSWER: <integer>`."
        ),
        expected="1074565",
    ),
    Gauntlet2Problem(
        pid="H08_seven_card_full_house_prob",
        track="math_nm",
        forge_mode="reason",
        title="Exact 7-Card Poker Full-House Multiplicity Probability",
        prompt=(
            "Seven cards are dealt uniformly at random without replacement from a standard 52-card deck "
            "(13 ranks, 4 suits each).\n"
            "What is the EXACT probability that the 7-card hand has:\n"
            "- maximum rank multiplicity EQUAL to 3 (so NO four-of-a-kind!), AND\n"
            "- second-highest rank multiplicity AT LEAST 2 (i.e., rank count multiset is {3,3,1}, {3,2,2}, or {3,2,1,1})?\n"
            "Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="726/27965",
    ),
    Gauntlet2Problem(
        pid="H09_kirchhoff_multigraph_trees",
        track="math_nm",
        forge_mode="reason",
        title="Kirchhoff Matrix-Tree Exact Spanning Trees on 9-Vertex Multigraph",
        prompt=(
            "Consider an undirected multigraph on 9 labeled vertices {0, 1, 2, 3, 4, 5, 6, 7, 8} with the following "
            "16 edge bundles given as `(u, v, multiplicity)` (each parallel edge between u and v is distinguishable):\n"
            "  (0,1,2), (0,2,1), (0,3,3), (1,2,2), (1,4,1), (2,3,1), (2,5,2), (3,6,2),\n"
            "  (4,5,3), (4,7,1), (5,6,1), (5,8,2), (6,8,3), (7,8,2), (0,8,1), (1,6,1).\n"
            "How many distinct labeled spanning trees does this multigraph have?\n"
            "Give the exact integer, ending with `ANSWER: <integer>`."
        ),
        expected="209314",
    ),
    Gauntlet2Problem(
        pid="H10_euclidean_floor_sum_10e12",
        track="math_nm",
        forge_mode="reason",
        title="Exact Euclidean Floor Sum up to N = 10^12 (mod 10^9+7)",
        prompt=(
            "Compute the exact value of:\n"
            "  sum_{k=0}^{10^12 - 1} floor((73 * k + 19) / 97)  mod 1000000007\n"
            "(where 10^12 = 1000000000000 — impossible by O(N) loop; use exact Euclidean lattice-point floor_sum!).\n"
            "Give the exact integer in [0, 1000000006], ending with `ANSWER: <integer>`."
        ),
        expected="575143907",
    ),

    # =========================================================================
    # TRACK 2: NIGHTMARE LOGIC, SMT/CSP, GAME THEORY & GRAPH OPTIMIZATION (H11-H20)
    # =========================================================================
    Gauntlet2Problem(
        pid="H11_coprime_taxicab_diophantine",
        track="logic_nm",
        forge_mode="reason",
        title="Constrained Coprime Taxicab Diophantine Minimum Sum",
        prompt=(
            "Find positive integers 1 <= a < b < c < d <= 80 satisfying ALL three conditions:\n"
            "1. a^3 + d^3 = b^3 + c^3\n"
            "2. gcd(a, b, c, d) = 1\n"
            "3. (a + b + c + d) is divisible by 7.\n"
            "What is the SMALLEST possible value of the sum a + b + c + d? End with `ANSWER: <integer>`."
        ),
        expected="42",
    ),
    Gauntlet2Problem(
        pid="H12_blocked_weighted_8queens",
        track="logic_nm",
        forge_mode="reason",
        title="Maximum-Weight 8-Queens Placement with Blocked Squares",
        prompt=(
            "Place 8 non-attacking queens on an 8x8 chessboard (0-indexed rows 0..7, cols 0..7) so no two queens share "
            "a row, column, or diagonal, and NO queen is placed on the 7 blocked cells:\n"
            "  {(0,2), (2,4), (3,1), (4,5), (5,6), (6,7), (7,0)}.\n"
            "Cell weight matrix W[row][col]:\n"
            "  r0: [12,  7, 19,  4, 15,  8, 11,  6]\n"
            "  r1: [ 9, 14,  5, 18,  3, 16, 10, 13]\n"
            "  r2: [17,  6, 11,  9, 20,  4, 15,  8]\n"
            "  r3: [ 5, 19,  8, 12,  7, 14,  3, 17]\n"
            "  r4: [14, 10, 16,  6, 13, 19,  9,  5]\n"
            "  r5: [ 8, 15,  4, 17, 11,  7, 20, 12]\n"
            "  r6: [11,  3, 18, 10,  6, 15,  8, 19]\n"
            "  r7: [16, 13,  7, 14, 12,  9, 17,  4]\n"
            "What is the exact MAXIMUM total weight of a valid 8-queen placement? End with `ANSWER: <integer>`."
        ),
        expected="102",
    ),
    Gauntlet2Problem(
        pid="H13_rcpsp_capacity6_makespan",
        track="logic_nm",
        forge_mode="reason",
        title="Resource-Constrained Project Scheduling (RCPSP, 8 Jobs, Capacity 6)",
        prompt=(
            "Schedule 8 non-preemptive jobs J0..J7 starting at integer times s_i >= 0.\n"
            "Each job has (duration, resource_demand):\n"
            "  J0:(3,3), J1:(4,4), J2:(2,2), J3:(5,3), J4:(3,4), J5:(4,2), J6:(3,3), J7:(2,2).\n"
            "Constraints:\n"
            "1. Precedence (u -> v means s_u + dur_u <= s_v): J0->J2, J0->J3, J1->J4, J2->J5, J3->J6, J4->J6, J5->J7, J6->J7.\n"
            "2. Renewable resource limit: at EVERY integer time unit t (when job i runs during [s_i, s_i + dur_i)), "
            "the total resource demand of all concurrently active jobs MUST be <= 6.\n"
            "What is the exact MINIMUM possible makespan max_i(s_i + dur_i)? End with `ANSWER: <integer>`."
        ),
        expected="20",
    ),
    Gauntlet2Problem(
        pid="H14_min_cost_max_flow_15",
        track="logic_nm",
        forge_mode="reason",
        title="Exact Minimum-Cost Flow (15 Units on 8-Node Directed Network)",
        prompt=(
            "Send EXACTLY 15 units of flow from source node 0 to sink node 7 in an 8-node directed network (nodes 0..7) "
            "with 17 directed edges given as `(u, v, capacity, unit_cost)`:\n"
            "  (0,1,8,2), (0,2,10,5), (0,3,6,4), (1,4,6,3), (1,5,5,7), (2,1,4,1),\n"
            "  (2,4,5,2), (2,5,7,4), (3,2,5,1), (3,5,4,3), (3,6,6,6), (4,7,9,5),\n"
            "  (4,6,4,2), (5,4,3,1), (5,7,8,3), (5,6,5,2), (6,7,10,2).\n"
            "Flow conservation holds at all intermediate nodes 1..6 and 0 <= f_e <= capacity_e for every edge.\n"
            "What is the exact MINIMUM total flow cost? End with `ANSWER: <integer>`."
        ),
        expected="156",
    ),
    Gauntlet2Problem(
        pid="H15_zero_sum_4x4_mixed_value",
        track="logic_nm",
        forge_mode="reason",
        title="Exact Rational Minimax Value of 4x4 Zero-Sum Matrix Game (No Saddle Point)",
        prompt=(
            "Two players play a simultaneous zero-sum matrix game with payoff matrix A (Row player maximizes expected payoff, "
            "Column player minimizes expected payoff):\n"
            "  Row 0: [ 3, -2,  4, -1]\n"
            "  Row 1: [-1,  5, -3,  2]\n"
            "  Row 2: [ 2, -4,  1,  6]\n"
            "  Row 3: [-3,  2,  5, -2]\n"
            "This game has NO pure-strategy saddle point. Compute the EXACT game value v = max_p min_q p^T A q "
            "over mixed strategies (probability distributions p, q on the 4 actions).\n"
            "Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="159/178",
    ),
    Gauntlet2Problem(
        pid="H16_min_cost_4coloring_12v",
        track="logic_nm",
        forge_mode="reason",
        title="Exact Min-Cost Proper 4-Coloring of 12-Vertex Chorded Graph",
        prompt=(
            "Assign each vertex v in {0..11} a color c_v in {0, 1, 2, 3} such that NO two adjacent vertices share a color.\n"
            "24 undirected edges:\n"
            "  (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(6,8),(8,10),(10,6),(7,9),(9,11),(11,7),\n"
            "  (0,6),(1,7),(2,8),(3,9),(4,10),(5,11),(0,3),(1,4),(2,5),(6,7),(8,9),(10,11).\n"
            "Per-vertex color cost table C[v] = [cost_c0, cost_c1, cost_c2, cost_c3]:\n"
            "  0:[4,9,7,11],  1:[8,3,6,10],  2:[5,12,4,9],  3:[11,6,8,3],\n"
            "  4:[7,4,10,6],  5:[9,8,3,7],   6:[6,11,5,8],  7:[10,5,9,4],\n"
            "  8:[3,7,11,6],  9:[8,4,6,12], 10:[12,6,4,5], 11:[5,9,7,3].\n"
            "What is the exact MINIMUM total coloring cost? End with `ANSWER: <integer>`."
        ),
        expected="54",
    ),
    Gauntlet2Problem(
        pid="H17_six_house_zebra_deduction",
        track="logic_nm",
        forge_mode="reason",
        title="6-House Unique Multi-Attribute Zebra Logic CSP",
        prompt=(
            "Six houses stand in a row numbered 1 (left) to 6 (right). Each house has a distinct Name "
            "{Alice, Bob, Cara, Dan, Eve, Fay}, a distinct Pet {Cat, Dog, Fox, Owl, Rat, Yak}, and a distinct Drink "
            "{Coffee, Juice, Milk, Soda, Tea, Water}. Clues:\n"
            "1. Bob lives exactly 2 houses to the right of Alice (Bob = Alice + 2).\n"
            "2. Cara owns the Owl.\n"
            "3. The Cat owner lives immediately adjacent to (|pos1 - pos2| == 1) the Tea drinker.\n"
            "4. Milk is drunk in house 3.\n"
            "5. Dan lives in house 1.\n"
            "6. Coffee is drunk immediately to the right of the Dog owner (Coffee = Dog + 1).\n"
            "7. Eve drinks Water.\n"
            "8. Fay lives immediately adjacent to the Fox owner.\n"
            "9. The Yak lives in house 6.\n"
            "10. Bob drinks Juice.\n"
            "11. Alice owns the Cat.\n"
            "12. The Rat lives strictly to the left of the Fox (Rat < Fox).\n"
            "13. Soda is drunk strictly to the right of Coffee (Soda > Coffee).\n"
            "14. Fay does NOT live in house 6.\n"
            "15. The Dog lives in house 1.\n"
            "16. Eve lives strictly to the right of Cara (Eve > Cara).\n"
            "Who owns the Fox, and what does the owner of the Rat drink? Reply as `Name,Drink`, ending with `ANSWER: Name,Drink`."
        ),
        expected="Bob,Milk",
    ),
    Gauntlet2Problem(
        pid="H18_min_weighted_feedback_arc_set",
        track="logic_nm",
        forge_mode="reason",
        title="NP-Hard Exact Minimum Weighted Feedback Arc Set (9 Vertices, 20 Edges)",
        prompt=(
            "Consider a weighted directed graph on 9 vertices {0..8} with 20 directed edges `(u, v, weight)`:\n"
            "  (0,1,7), (1,2,6), (2,0,5), (1,3,8), (3,4,4), (4,1,9), (2,4,3), (4,5,7),\n"
            "  (5,2,6), (3,6,5), (6,7,8), (7,3,4), (5,7,6), (7,8,9), (8,5,5), (6,8,3),\n"
            "  (8,0,7), (0,6,4), (2,7,2), (4,8,5).\n"
            "You must DELETE a subset of edges so that the remaining directed graph is ACYCLIC (a DAG, zero directed cycles).\n"
            "What is the exact MINIMUM total weight of deleted edges (Feedback Arc Set)? End with `ANSWER: <integer>`."
        ),
        expected="27",
    ),
    Gauntlet2Problem(
        pid="H19_weighted_4jug_dijkstra",
        track="logic_nm",
        forge_mode="reason",
        title="4-Jug Minimum Total Liters Poured (Dijkstra vs BFS Trap)",
        prompt=(
            "You have 4 unmarked jugs with capacities (17, 13, 8, 5) liters, initially holding (17, 0, 0, 0) liters.\n"
            "In one move, you may pour water from any jug i into any other jug j until either jug i is empty OR jug j is full "
            "(no water may be spilled or added from outside; total water stays 17L). "
            "The COST of pouring k liters is k (NOT 1 per move!).\n"
            "What is the exact MINIMUM TOTAL LITERS POURED across all moves to reach ANY state where at least one jug "
            "holds EXACTLY 7 liters? End with `ANSWER: <integer>`."
        ),
        expected="15",
    ),
    Gauntlet2Problem(
        pid="H20_product_automaton_len35_mod7",
        track="logic_nm",
        forge_mode="reason",
        title="Length-35 Binary Strings Containing '101', Avoiding '111', Divisible by 7",
        prompt=(
            "How many binary strings of exact length 35 (leading zeros allowed, alphabet {0, 1}) satisfy ALL three conditions:\n"
            "1. Contains '101' as a contiguous substring at least once.\n"
            "2. Does NOT contain '111' as a contiguous substring anywhere.\n"
            "3. When interpreted as an unsigned base-2 integer, its value is divisible by 7 (i.e., value % 7 == 0).\n"
            "(Note: 2^35 is far too large to brute-force — use product-automaton DP!)\n"
            "Give the exact count modulo 1000000007, ending with `ANSWER: <integer>`."
        ),
        expected="293513061",
    ),

    # =========================================================================
    # TRACK 5 (PART A): NIGHTMARE SYSTEMS / PROTOCOL EXECUTION TRAPS (H41-H45)
    # =========================================================================
    Gauntlet2Problem(
        pid="H41_python_descriptor_generator_trap",
        track="sys_sec_nm",
        forge_mode="reason",
        title="Python Data vs Non-Data Descriptor + Generator send()/throw()/finally Trace",
        prompt=(
            "What exact string does the following Python 3 code produce for `res`?\n"
            "```python\n"
            "class DataDesc:\n"
            "    def __get__(self, obj, cls=None): return 'DD'\n"
            "    def __set__(self, obj, val): obj._dd = val\n"
            "class NonDataDesc:\n"
            "    def __get__(self, obj, cls=None): return 'ND'\n"
            "class Base:\n"
            "    x = DataDesc()\n"
            "    y = NonDataDesc()\n"
            "b = Base()\n"
            "b.__dict__['x'] = 'inst_x'\n"
            "b.__dict__['y'] = 'inst_y'\n"
            "def gen():\n"
            "    log = []\n"
            "    try:\n"
            "        v = yield 'init'\n"
            "        log.append(f'got:{v}')\n"
            "        try:\n"
            "            yield 'mid'\n"
            "        except ValueError:\n"
            "            log.append('caught')\n"
            "            yield 'after'\n"
            "    finally:\n"
            "        log.append('fin')\n"
            "    return '|'.join(log)\n"
            "g = gen()\n"
            "o1 = next(g)\n"
            "o2 = g.send(7)\n"
            "o3 = g.throw(ValueError())\n"
            "try:\n"
            "    next(g)\n"
            "except StopIteration as e:\n"
            "    ret = e.value\n"
            "res = f'{b.x},{b.y},{o1},{o2},{o3},{ret}'\n"
            "```\n"
            "Give the exact string value of `res` (no quotes), ending with `ANSWER: <string>`."
        ),
        expected="DD,inst_y,init,mid,after,got:7|caught|fin",
    ),
    Gauntlet2Problem(
        pid="H42_ieee754_ulp_subnormal_bankers",
        track="sys_sec_nm",
        forge_mode="reason",
        title="IEEE-754 Binary64 ULP Distance + Subnormal Bit Pattern + Bankers Rounding",
        prompt=(
            "Using standard IEEE-754 binary64 (`float` in Python 3, `struct.pack('>d', x)` as unsigned 64-bit integer bits):\n"
            "1. Let `u_dist_1` be the unsigned integer bit difference `bits(1.0 + 2**-50) - bits(1.0)`.\n"
            "2. Let `u_half_1` be `(bits(1.0) - bits(0.5)) // (1 << 40)`.\n"
            "3. Let `sub = math.nextafter(0.0, 1.0)` (smallest positive subnormal), and let `u_sub = bits(sub * 1024)`.\n"
            "4. Let `bank = sum(round(k + 0.5) for k in range(10))` using Python 3 round-half-to-even.\n"
            "Compute the exact integer sum `u_half_1 + u_dist_1 + u_sub + bank`. End with `ANSWER: <integer>`."
        ),
        expected="5174",
    ),
    Gauntlet2Problem(
        pid="H43_sqlite_recursive_cte_windows",
        track="sys_sec_nm",
        forge_mode="reason",
        title="SQLite Recursive CTE + Window LAG/Running SUM with NULLs",
        prompt=(
            "Given SQLite table `tx(id INT, acct TEXT, amt INT)` populated with:\n"
            "  (1,'A',10), (2,'A',NULL), (3,'A',25), (4,'A',-5),\n"
            "  (5,'B',30), (6,'B',15), (7,'B',NULL), (8,'B',20).\n"
            "Evaluate this exact SQLite query:\n"
            "```sql\n"
            "WITH RECURSIVE fib(n, a, b) AS (\n"
            "  SELECT 1, 1, 1\n"
            "  UNION ALL\n"
            "  SELECT n + 1, b, a + b FROM fib WHERE n < 8\n"
            "),\n"
            "win AS (\n"
            "  SELECT id, acct,\n"
            "         COALESCE(amt, 0) AS v,\n"
            "         LAG(amt, 1, -1) OVER (PARTITION BY acct ORDER BY id) AS prev_amt,\n"
            "         SUM(amt) OVER (PARTITION BY acct ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS r2\n"
            "  FROM tx\n"
            ")\n"
            "SELECT (SELECT SUM(a) FROM fib) * 100\n"
            "     + (SELECT SUM(COALESCE(prev_amt, 7) + COALESCE(r2, 3)) FROM win);\n"
            "```\n"
            "Give the exact integer result, ending with `ANSWER: <integer>`."
        ),
        expected="5667",
    ),
    Gauntlet2Problem(
        pid="H44_16bit_isa_carry_loop_trace",
        track="sys_sec_nm",
        forge_mode="reason",
        title="20-Iteration 16-Bit Register/Carry/ROR Mini-ISA Exact Hex State",
        prompt=(
            "Simulate a 16-bit unsigned CPU with registers `R0 = 0x1357`, `R1 = 0x2468`, `R2 = 20`, `R3 = 0x0001`, "
            "and carry flag `C = 0`.\n"
            "While `R2 != 0`, execute these 6 steps in order using exact 16-bit unsigned arithmetic (`& 0xFFFF`):\n"
            "  1. `s = R0 + R1 + C`; set `C = 1 if s > 0xFFFF else 0`; set `R0 = s & 0xFFFF`.\n"
            "  2. `R1 = ((R1 >> 3) | ((R1 << 13) & 0xFFFF)) & 0xFFFF` (16-bit rotate right by 3).\n"
            "  3. `R1 = R1 ^ R0`.\n"
            "  4. Unsigned borrow check BEFORE updating R3: `C = 1 if R3 < R2 else 0`; then `R3 = (R3 - R2) & 0xFFFF`.\n"
            "  5. `R0 = R0 ^ R3`.\n"
            "  6. `R2 = (R2 - 1) & 0xFFFF`.\n"
            "After the loop finishes, give `R0,R1` as two 4-digit lowercase hex values separated by a comma "
            "(e.g. `01ab,ff02`), ending with `ANSWER: <4hex>,<4hex>`."
        ),
        expected="0696,ef93",
    ),
    Gauntlet2Problem(
        pid="H45_c11_union_endian_promotion_trap",
        track="sys_sec_nm",
        forge_mode="reason",
        title="C11 Little-Endian Union + Struct Layout + Integer Promotion Trap",
        prompt=(
            "Under standard C11 on a 64-bit little-endian LP64 / AAPCS64 platform (`int` is 32-bit, natural alignment, no packing):\n"
            "```c\n"
            "union U { uint32_t u32; uint16_t u16[2]; uint8_t b[4]; };\n"
            "struct S { uint8_t tag; uint16_t arr[3]; union U u; uint64_t val; uint8_t end; };\n"
            "// Runtime values:\n"
            "union U x; x.u32 = 0x12345678u; x.b[1] ^= 0xFFu;\n"
            "uint16_t lo = x.u16[0];\n"
            "int16_t neg = -5; uint16_t u = 3;\n"
            "int cmp = (neg < u) ? 1 : 0;\n"
            "int32_t neg32 = -5; uint32_t u32 = 3u;\n"
            "int cmp32 = (neg32 < u32) ? 1 : 0;\n"
            "printf(\"%zu,%zu,%04x,%d,%d\\n\", sizeof(struct S), offsetof(struct S, val), (unsigned)lo, cmp, cmp32);\n"
            "```\n"
            "Give the exact printed output string (5 comma-separated fields), ending with `ANSWER: <string>`."
        ),
        expected="32,16,a978,1,0",
    ),
]
