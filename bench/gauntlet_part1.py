"""50-Test Frontier Gauntlet — Part 1: 26 Exact Math, Logic/CSP, Optimization & Systems-Trace Tests (G01-G20, G41-G46)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GauntletProblem:
    pid: str
    track: Literal["math", "logic_opt", "algo_code", "debug_code", "systems_sec"]
    forge_mode: Literal["reason", "code"]
    title: str
    prompt: str
    expected: str = ""
    hidden_tests: str = ""
    buggy_starter: str = ""


PART1_PROBLEMS: list[GauntletProblem] = [
    # =========================================================================
    # TRACK 1: HARD OLYMPIAD / AIME MATH, NUMBER THEORY & PROBABILITY (G01-G10)
    # =========================================================================
    GauntletProblem(
        pid="G01_biased_coupon_collector",
        track="math",
        forge_mode="reason",
        title="Biased 6-Sided Die Full Collection Expected Rolls",
        prompt=(
            "A biased 6-sided die lands on faces 1, 2, 3, 4, 5, 6 with exact probabilities "
            "1/12, 1/12, 2/12, 2/12, 3/12, 3/12 respectively. "
            "What is the exact expected number of rolls required to observe ALL 6 distinct faces at least once? "
            "Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="6671/330",
    ),
    GauntletProblem(
        pid="G02_huge_n_linear_recurrence",
        track="math",
        forge_mode="reason",
        title="Non-Homogeneous Linear Recurrence at N = 10^15 (mod 10^9+7)",
        prompt=(
            "Define a sequence by a_0 = 1, a_1 = 2, a_2 = 5, and for n >= 3:\n"
            "  a_n = 3*a_{n-1} + 2*a_{n-2} - a_{n-3} + 7.\n"
            "Compute a_{10^15} mod 1000000007 (where 10^15 = 1000000000000000). "
            "Give the exact integer in [0, 1000000006], ending with `ANSWER: <integer>`."
        ),
        expected="913437523",
    ),
    GauntletProblem(
        pid="G03_bounded_parity_compositions",
        track="math",
        forge_mode="reason",
        title="Bounded Integer Compositions with Parity & Inequality Constraints",
        prompt=(
            "How many 4-tuples of integers (x1, x2, x3, x4) satisfy ALL of:\n"
            "- x1 + x2 + x3 + x4 = 120\n"
            "- 3 <= x1 <= 35, 5 <= x2 <= 40, 0 <= x3 <= 30, 10 <= x4 <= 50\n"
            "- x1 + x2 is even\n"
            "- x3 != x4\n"
            "Give the exact integer count, ending with `ANSWER: <integer>`."
        ),
        expected="4124",
    ),
    GauntletProblem(
        pid="G04_portal_random_walk",
        track="math",
        forge_mode="reason",
        title="Absorbing Random Walk with Asymmetric Drift & Teleport",
        prompt=(
            "A particle walks on integer positions {0, 1, 2, 3, 4, 5, 6}, where 0 and 6 are absorbing states. "
            "From any position i in {1, 2, 3, 4, 5}, at each step it moves:\n"
            "- to i + 1 with probability 1/2,\n"
            "- to i - 1 with probability 1/3,\n"
            "- teleports to position 1 with probability 1/6 (if already at i = 1, it stays at 1).\n"
            "Starting at position 3, what is the exact expected number of steps until absorption at 0 or 6? "
            "Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="1460/179",
    ),
    GauntletProblem(
        pid="G05_divisors_totient_congruence",
        track="math",
        forge_mode="reason",
        title="Number Theory: Divisor Count + Totient Divisibility + Congruence",
        prompt=(
            "Find the SMALLEST positive integer n > 10000 such that ALL three conditions hold:\n"
            "1. n has EXACTLY 24 positive divisors.\n"
            "2. Euler's totient phi(n) is divisible by 16.\n"
            "3. n ≡ 7 (mod 11).\n"
            "Give the exact integer, ending with `ANSWER: <integer>`."
        ),
        expected="10050",
    ),
    GauntletProblem(
        pid="G06_two_test_factory_bayes",
        track="math",
        forge_mode="reason",
        title="Multi-Factory Two-Test Conditional Bayes",
        prompt=(
            "A product is made by three factories: Factory A makes 50% (2% defective), "
            "Factory B makes 30% (5% defective), and Factory C makes 20% (10% defective).\n"
            "A randomly chosen item undergoes two conditionally independent diagnostic tests (given true defective status):\n"
            "- Test 1 detects a defective item with probability 90%, and has a 5% false-positive rate on non-defective items.\n"
            "- Test 2 detects a defective item with probability 80%, and has a 10% false-positive rate on non-defective items.\n"
            "Given that the item tests POSITIVE on Test 1 AND NEGATIVE on Test 2, what is the exact probability it came from Factory C? "
            "Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="52/227",
    ),
    GauntletProblem(
        pid="G07_sprague_grundy_subtraction",
        track="math",
        forge_mode="reason",
        title="Sprague-Grundy Losing Positions Count for Subtraction Game {1,3,4}",
        prompt=(
            "Two players play an impartial subtraction game with 3 piles of stones (a, b, c). "
            "On each turn, a player chooses one pile and removes 1, 3, or 4 stones (normal play: last player to move wins).\n"
            "How many unordered triples of integers (a, b, c) with 1 <= a <= b <= c <= 25 are P-positions "
            "(second-player winning states, i.e., Sprague-Grundy XOR sum == 0)? "
            "Give the exact integer, ending with `ANSWER: <integer>`."
        ),
        expected="742",
    ),
    GauntletProblem(
        pid="G08_exact_10x10_determinant",
        track="math",
        forge_mode="reason",
        title="Exact Integer Determinant of 10x10 GCD+Mod Structured Matrix",
        prompt=(
            "Let M be the 10 x 10 matrix (0-indexed rows i = 0..9, cols j = 0..9) defined by:\n"
            "  M[i][j] = gcd(i + 1, j + 1) + (2 if i == j else 0) + ((i * j) % 3).\n"
            "Compute the EXACT integer determinant det(M). End with `ANSWER: <integer>`."
        ),
        expected="40608000",
    ),
    GauntletProblem(
        pid="G09_dihedral_proper_necklaces",
        track="math",
        forge_mode="reason",
        title="Burnside/Dihedral Proper 3-Colored Necklaces of Length 12",
        prompt=(
            "How many distinct circular necklaces of length 12 can be formed using beads of 3 available colors "
            "such that NO two adjacent beads have the same color, where two necklaces are considered identical if one "
            "can be obtained from the other by rotation OR reflection (flipping over, i.e., dihedral group D_12)? "
            "Give the exact integer, ending with `ANSWER: <integer>`."
        ),
        expected="224",
    ),
    GauntletProblem(
        pid="G10_coprime_cubes_sum_huge_n",
        track="math",
        forge_mode="reason",
        title="Sum of k^3 Coprime to 30 up to N = 10^9 (mod 998244353)",
        prompt=(
            "Compute the exact value of:\n"
            "  sum(k^3 for 1 <= k <= 10^9 if gcd(k, 30) == 1) mod 998244353.\n"
            "(Note: 10^9 = 1000000000 — too large for a naive O(N) loop!). "
            "Give the exact integer in [0, 998244352], ending with `ANSWER: <integer>`."
        ),
        expected="839394368",
    ),

    # =========================================================================
    # TRACK 2: MULTI-CONSTRAINT LOGIC, CSP & COMBINATORIAL OPTIMIZATION (G11-G20)
    # =========================================================================
    GauntletProblem(
        pid="G11_cryptarithm_cross_roads",
        track="logic_opt",
        forge_mode="reason",
        title="Unique Cryptarithm CROSS + ROADS = DANGER",
        prompt=(
            "Solve the base-10 cryptarithm: CROSS + ROADS = DANGER.\n"
            "Each distinct letter maps to a distinct digit 0..9, and leading letters (C, R, D) cannot be 0.\n"
            "What is the exact 6-digit integer value of DANGER? End with `ANSWER: <integer>`."
        ),
        expected="158746",
    ),
    GauntletProblem(
        pid="G12_truth_liar_alternator",
        track="logic_opt",
        forge_mode="reason",
        title="5-Person Truth-Teller / Liar / Alternator Deduction",
        prompt=(
            "Five people (A, B, C, D, E) each belong to one of three types:\n"
            "- Truth-teller (TT): both statements are TRUE.\n"
            "- Liar (LL): both statements are FALSE.\n"
            "- Alternator: either TF (1st statement TRUE, 2nd FALSE) or FT (1st FALSE, 2nd TRUE).\n"
            "There is EXACTLY 1 Truth-teller, EXACTLY 1 Liar, and 3 Alternators (with at least one TF and at least one FT).\n"
            "Statements (1st, then 2nd):\n"
            "- A: (1) 'B is the Liar.'  (2) 'D is the Truth-teller.'\n"
            "- B: (1) 'A is the Truth-teller.'  (2) 'E's first statement is false.'\n"
            "- C: (1) 'D is an Alternator.'  (2) 'A and E are both Alternators.'\n"
            "- D: (1) 'C is the Liar.'  (2) 'B is NOT an Alternator.'\n"
            "- E: (1) 'C is the Truth-teller.'  (2) 'D's first statement is true.'\n"
            "Who is the Truth-teller and who is the Liar? Reply as `TruthTeller,Liar` (e.g. `A,B`), ending with `ANSWER: X,Y`."
        ),
        expected="C,B",
    ),
    GauntletProblem(
        pid="G13_asymmetric_tsptw",
        track="logic_opt",
        forge_mode="reason",
        title="Asymmetric TSP with Hard Arrival Time Windows (7 Nodes)",
        prompt=(
            "A courier starts at depot 0 at time t = 0, must visit each customer 1..6 EXACTLY once, and return to depot 0.\n"
            "Directed travel-time/distance matrix dist[i][j] (rows 0..6, cols 0..6):\n"
            "  0: [0,  5, 12,  8, 15,  9, 14]\n"
            "  1: [6,  0,  7, 10,  9, 13,  8]\n"
            "  2: [11, 8,  0,  4,  6, 12,  7]\n"
            "  3: [9, 12,  5,  0,  8,  6, 11]\n"
            "  4: [14, 7,  6,  9,  0,  5,  4]\n"
            "  5: [8, 11, 10,  5,  6,  0,  7]\n"
            "  6: [13, 9,  8, 12,  5,  6,  0]\n"
            "Hard time windows [earliest, latest] for customers 1..6 (service time = 0; if arriving before earliest, "
            "courier waits until earliest before departing; arriving after latest is forbidden):\n"
            "  1:[4,15], 2:[10,22], 3:[12,26], 4:[18,32], 5:[20,38], 6:[25,45].\n"
            "Minimize the TOTAL TRAVEL DISTANCE (sum of dist[u][v] along the 7-edge tour, NOT including waiting time!).\n"
            "Give the exact minimum total travel distance, ending with `ANSWER: <integer>`."
        ),
        expected="42",
    ),
    GauntletProblem(
        pid="G14_multidim_conflict_knapsack",
        track="logic_opt",
        forge_mode="reason",
        title="14-Item 2D Knapsack with Conflicts & Group-At-Most-1",
        prompt=(
            "Select a subset of 14 items (0-indexed 0..13), each given as (weight, volume, value):\n"
            "  0:(12,8,28), 1:(10,9,25), 2:(8,6,19), 3:(15,12,37), 4:(7,5,17),\n"
            "  5:(9,11,24), 6:(11,7,26), 7:(6,8,16), 8:(14,10,34), 9:(5,4,13),\n"
            "  10:(13,9,31), 11:(8,7,21), 12:(10,6,23), 13:(9,8,22).\n"
            "Constraints:\n"
            "1. Total weight <= 50 AND total volume <= 40.\n"
            "2. Incompatible pairs (cannot pick BOTH): (0,3), (1,8), (4,10), (6,12), (2,11).\n"
            "3. At most ONE item may be chosen from the group {3, 8, 10}.\n"
            "What is the exact MAXIMUM possible total value? End with `ANSWER: <integer>`."
        ),
        expected="124",
    ),
    GauntletProblem(
        pid="G15_killer_latin_square",
        track="logic_opt",
        forge_mode="reason",
        title="5x5 Killer-Cage Latin Square Weighted Diagonal",
        prompt=(
            "Fill a 5x5 grid G[r][c] (0-indexed r,c in 0..4) with integers in {1, 2, 3, 4, 5} such that:\n"
            "- Every row and every column contains each of {1, 2, 3, 4, 5} exactly once.\n"
            "- Fixed cells: G[0][0]=1, G[1][1]=2, G[2][2]=4, G[0][2]=2.\n"
            "- Cage sums (list of (r,c) cells -> exact sum):\n"
            "  [(0,0),(0,1)]->5, [(0,2),(1,2)]->7, [(0,3),(0,4)]->8, [(1,0),(2,0)]->7,\n"
            "  [(1,1),(2,1)]->3, [(1,3),(1,4)]->4, [(2,2),(2,3)]->9, [(2,4),(3,4)]->6,\n"
            "  [(3,0),(4,0)]->7, [(3,1),(3,2)]->4, [(3,3),(4,3)]->6, [(4,1),(4,2)]->8, [(4,4)]->1.\n"
            "Compute the weighted main-diagonal sum: sum((i + 1) * G[i][i] for i in range(5)). "
            "End with `ANSWER: <integer>`."
        ),
        expected="30",
    ),
    GauntletProblem(
        pid="G16_constrained_assignment",
        track="logic_opt",
        forge_mode="reason",
        title="6x6 Constrained Min-Cost Bipartite Assignment",
        prompt=(
            "Assign 6 workers (0..5) bijectively to 6 tasks (0..5) using cost matrix cost[worker][task]:\n"
            "  w0: [9,  12,  7, 15, 11,  8]\n"
            "  w1: [10,  6, 14,  8,  9, 13]\n"
            "  w2: [7,  11,  8,  6, 15, 10]\n"
            "  w3: [13,  9,  6, 11,  7, 12]\n"
            "  w4: [8,  14, 10,  7, 12,  6]\n"
            "  w5: [11,  8, 12,  9,  6,  9]\n"
            "Constraints:\n"
            "1. Forbidden (worker, task) pairs: (0,2), (1,1), (2,3), (3,2), (4,5), (5,4).\n"
            "2. Worker 0 and Worker 1 CANNOT both receive tasks of cost >= 10.\n"
            "What is the exact MINIMUM total assignment cost? End with `ANSWER: <integer>`."
        ),
        expected="47",
    ),
    GauntletProblem(
        pid="G17_circuit_stuck_at_fault",
        track="logic_opt",
        forge_mode="reason",
        title="Combinational Logic Circuit Stuck-At Fault Diagnosis",
        prompt=(
            "A combinational digital circuit has 4 binary inputs (a, b, c, d) in {0, 1} and 6 gates:\n"
            "  g1 = a AND b\n"
            "  g2 = c XOR d\n"
            "  g3 = g1 OR g2\n"
            "  g4 = (NOT b) AND c\n"
            "  g5 = g3 XOR g4\n"
            "  out = g5 AND (a OR d)\n"
            "EXACTLY ONE gate among {g1, g2, g3, g4, g5, out} has a single stuck-at fault (its output is permanently "
            "forced to 0 or permanently forced to 1 regardless of its inputs). Observed faulty outputs `out`:\n"
            "  (1,1,0,0) -> 0\n"
            "  (1,0,1,0) -> 0\n"
            "  (0,1,1,1) -> 1\n"
            "  (1,1,1,1) -> 0\n"
            "  (0,0,0,1) -> 0\n"
            "Identify the unique faulty gate and its stuck value as `gate=val` (e.g. `g2=0`), ending with `ANSWER: gate=val`."
        ),
        expected="g4=1",
    ),
    GauntletProblem(
        pid="G18_changeover_setup_makespan",
        track="logic_opt",
        forge_mode="reason",
        title="2-Machine Precedence + Type-Changeover Setup Makespan",
        prompt=(
            "Schedule 6 non-preemptive jobs J0..J5 on 2 identical machines (integer start times >= 0).\n"
            "Each job has (duration, type):\n"
            "  J0:(4,A), J1:(5,B), J2:(3,A), J3:(6,B), J4:(4,A), J5:(5,B).\n"
            "Constraints:\n"
            "1. Precedence (u -> v means finish(u) <= start(v)): J0->J3, J1->J2, J2->J4, J3->J5, J4->J5.\n"
            "2. Changeover setup time: if two jobs run on the SAME machine and have DIFFERENT types (A then B, or B then A), "
            "there must be a gap of at least 2 time units between them (finish(first) + 2 <= start(second)). "
            "Same-type jobs on the same machine have 0 setup gap.\n"
            "What is the exact MINIMUM possible makespan? End with `ANSWER: <integer>`."
        ),
        expected="17",
    ),
    GauntletProblem(
        pid="G19_keys_doors_maze_bfs",
        track="logic_opt",
        forge_mode="reason",
        title="6x6 Grid Shortest Path with Keys (a,b,c) & Doors (A,B,C)",
        prompt=(
            "Find the shortest number of 4-directional steps (up/down/left/right) from 'S' to 'E' in this 6x6 grid:\n"
            "  Row 0: S..a#.\n"
            "  Row 1: ###.#.\n"
            "  Row 2: b.A.#c\n"
            "  Row 3: ###B#.\n"
            "  Row 4: ......\n"
            "  Row 5: ###.CE\n"
            "Rules: '#' is an impassable wall; '.' is open; lowercase 'a','b','c' are keys picked up on entry; "
            "uppercase 'A','B','C' are locked doors passable ONLY after picking up the corresponding lowercase key.\n"
            "Give the exact minimum number of steps, ending with `ANSWER: <integer>`."
        ),
        expected="16",
    ),
    GauntletProblem(
        pid="G20_two_end_minimax_game",
        track="logic_opt",
        forge_mode="reason",
        title="Exact Optimal Minimax Score Difference (Take 1 or 2 From Either End)",
        prompt=(
            "Two players play a zero-sum game on the array:\n"
            "  [4, -2, 7, 3, -5, 9, 1, -4, 8, 2, -3, 6]\n"
            "Players alternate turns (Player 1 goes first). On a turn, a player removes EITHER 1 element OR 2 elements "
            "from the LEFT end, OR 1 element OR 2 elements from the RIGHT end (if >= 2 elements remain), adding the "
            "removed element(s)' values to their score. The game ends when the array is empty.\n"
            "Under optimal play by both players, what is the exact final score difference (Player 1 score minus Player 2 score)? "
            "End with `ANSWER: <integer>`."
        ),
        expected="6",
    ),

    # =========================================================================
    # TRACK 5 (PART A): ADVERSARIAL SYSTEMS / SEMANTICS EXECUTION TRAPS (G41-G46)
    # =========================================================================
    GauntletProblem(
        pid="G41_python_semantics_trap",
        track="systems_sec",
        forge_mode="reason",
        title="Python MRO + Mutable Default + try/finally Return + Late Closure Trace",
        prompt=(
            "What exact string does the following Python 3 code produce for `res`?\n"
            "```python\n"
            "class A:\n"
            "    def f(self): return ['A']\n"
            "class B(A):\n"
            "    def f(self): return ['B'] + super().f()\n"
            "class C(A):\n"
            "    def f(self): return ['C'] + super().f()\n"
            "class D(B, C):\n"
            "    def f(self): return ['D'] + super().f()\n"
            "def acc(x, buf=[]):\n"
            "    buf.append(x)\n"
            "    return len(buf)\n"
            "def tricky(n):\n"
            "    try:\n"
            "        return n * 10\n"
            "    finally:\n"
            "        if n % 2 == 0:\n"
            "            return n + 1\n"
            "funcs = [lambda y: y + i for i in range(3)]\n"
            "res = f\"{''.join(D().f())},{acc(1)+acc(2)},{tricky(4)+tricky(3)},{sum(fn(1) for fn in funcs)}\"\n"
            "```\n"
            "Give the exact value of `res` (no quotes), ending with `ANSWER: <string>`."
        ),
        expected="DBCA,3,35,9",
    ),
    GauntletProblem(
        pid="G42_arx_uint32_simulation",
        track="systems_sec",
        forge_mode="reason",
        title="100-Round 32-Bit Unsigned ARX Mixer Exact Hex State",
        prompt=(
            "Start with 32-bit unsigned integers x = 0x12345678 and y = 0x9ABCDEF0.\n"
            "For r = 0, 1, ..., 99 (100 rounds), perform exact 32-bit unsigned wraparound arithmetic (mod 2^32):\n"
            "  x = (ROR32(x, 7) + y) & 0xFFFFFFFF\n"
            "  x = x ^ ((r * 0x9E3779B9) & 0xFFFFFFFF)\n"
            "  y = ROL32(y, 5) ^ x\n"
            "where ROR32(v, k) = ((v >> k) | (v << (32 - k))) & 0xFFFFFFFF and ROL32(v, k) = ((v << k) | (v >> (32 - k))) & 0xFFFFFFFF.\n"
            "What is the final value of `x` as an 8-digit lowercase hex string (no `0x` prefix)? End with `ANSWER: <8hex>`."
        ),
        expected="7e2dea04",
    ),
    GauntletProblem(
        pid="G43_ieee754_cancellation_error",
        track="systems_sec",
        forge_mode="reason",
        title="IEEE-754 Binary64 Catastrophic Cancellation Drift",
        prompt=(
            "Consider summing the 50,000-element sequence left-to-right:\n"
            "  seq = [1e15] + [0.1] * 49998 + [-1e15]\n"
            "Let `acc_f` be the result of left-to-right IEEE-754 binary64 (Python `float`) addition starting from `0.0`, "
            "and let `acc_exact` be the exact mathematical sum of `10^15 + 49998*(1/10) - 10^15`.\n"
            "Compute `round(abs(acc_exact - acc_f))` (nearest integer). End with `ANSWER: <integer>`."
        ),
        expected="1250",
    ),
    GauntletProblem(
        pid="G44_c_struct_abi_layout",
        track="systems_sec",
        forge_mode="reason",
        title="C11 / 64-Bit Struct Alignment, Padding & Offsetof",
        prompt=(
            "Under standard C11 natural alignment rules on 64-bit platforms (LP64 / AAPCS64, no `#pragma pack`):\n"
            "```c\n"
            "struct Inner { uint8_t a; uint32_t b; uint8_t c; };\n"
            "struct Outer { uint8_t tag; struct Inner in; uint64_t big; uint16_t tail; };\n"
            "```\n"
            "Give `sizeof(struct Outer),offsetof(struct Outer, big)` as two comma-separated integers (e.g. `24,12`), "
            "ending with `ANSWER: X,Y`."
        ),
        expected="32,16",
    ),
    GauntletProblem(
        pid="G45_sql_left_join_null_count",
        track="systems_sec",
        forge_mode="reason",
        title="SQL LEFT JOIN + NULL + COUNT(*) vs COUNT(col) Exact Execution",
        prompt=(
            "Given SQL tables:\n"
            "  dept(id, name): (1,'Eng'), (2,'Ops'), (3,'Lab'), (4,'QA')\n"
            "  emp(id, dept_id, bonus): (1,1,100), (2,1,NULL), (3,1,200), (4,2,NULL), (5,2,NULL), (6,3,50), (7,NULL,500)\n"
            "Evaluate the standard SQL (SQLite) query:\n"
            "```sql\n"
            "SELECT SUM(c_star * 10 + c_bonus) FROM (\n"
            "  SELECT d.id, COUNT(*) AS c_star, COUNT(e.bonus) AS c_bonus\n"
            "  FROM dept d LEFT JOIN emp e ON d.id = e.dept_id\n"
            "  GROUP BY d.id\n"
            ");\n"
            "```\n"
            "Give the exact integer result, ending with `ANSWER: <integer>`."
        ),
        expected="73",
    ),
    GauntletProblem(
        pid="G46_regex_shortest_distinguishing",
        track="systems_sec",
        forge_mode="reason",
        title="Shortest Distinguishing String Between Two Tricky Regexes",
        prompt=(
            "Consider two full-string regular expressions over the alphabet {a, b}:\n"
            "  R1 = ^(a|ab)*b$\n"
            "  R2 = ^a*(ab)*b$\n"
            "Find the SHORTEST string (breaking ties by lexicographical order: 'a' < 'b') over {a, b} "
            "for which R1 and R2 DISAGREE (one matches and the other does not). End with `ANSWER: <string>`."
        ),
        expected="abab",
    ),
]
