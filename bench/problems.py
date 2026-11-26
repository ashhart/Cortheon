"""16-problem Frontier-Killer Benchmark Suite (4 categories x 4 problems).

Categories:
  M1..M4: Hard Math, Probability, Combinatorics & Number Theory (exact answers)
  L1..L4: Multi-Constraint Logic, CSP & Optimal Scheduling (Z3-verified unique answers)
  C1..C4: Hard Algorithmic & Edge-Case Coding (comprehensive hidden test suites)
  D1..D4: Adversarial Multi-Bug Debugging (buggy starter modules + hidden test suites)

Zero-leak design: FlashForge sees ONLY `problem.prompt`, NEVER `expected` or `hidden_tests`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BenchProblem:
    pid: str
    category: Literal["math", "logic", "algo_code", "debug_code"]
    forge_mode: Literal["reason", "code"]
    title: str
    prompt: str
    expected: str = ""          # Exact canonical answer for reason tasks
    hidden_tests: str = ""      # Hidden pytest-style assertions for code tasks
    buggy_starter: str = ""     # For D1..D4 planted-control verification


PROBLEMS: list[BenchProblem] = [
    # =========================================================================
    # CATEGORY 1: HARD MATH, PROBABILITY & COMBINATORICS (M1..M4)
    # =========================================================================
    BenchProblem(
        pid="M1_markov_pattern_race",
        category="math",
        forge_mode="reason",
        title="6-Sided Die Non-Transitive Pattern Race",
        prompt=(
            "A fair 6-sided die (faces 1..6) is rolled repeatedly until either Alice's pattern "
            "(6, 6, 5) or Bob's pattern (6, 5, 6) appears as three consecutive rolls. "
            "Whoever's pattern appears first wins.\n"
            "What is the exact probability that Alice wins? "
            "Give your final answer as a reduced fraction `p/q` (for example `3/7`), ending with `ANSWER: p/q`."
        ),
        expected="6/11",
    ),
    BenchProblem(
        pid="M2_restricted_automaton_count",
        category="math",
        forge_mode="reason",
        title="Automaton-Constrained Sequence Count (L=14)",
        prompt=(
            "How many strings of length 14 over the alphabet {A, B, C} satisfy ALL of the following?\n"
            "1. No substring 'AA'.\n"
            "2. No substring 'BB'.\n"
            "3. No substring 'ABC'.\n"
            "4. The total count of 'A' characters is STRICTLY GREATER than the total count of 'B' characters.\n"
            "Give the exact integer, ending with `ANSWER: <integer>`."
        ),
        expected="54662",
    ),
    BenchProblem(
        pid="M3_crt_prime_digitsum",
        category="math",
        forge_mode="reason",
        title="CRT + Prime Digit-Sum Filter (N <= 200,000)",
        prompt=(
            "Find the exact sum of all positive integers n with 1 <= n <= 200000 such that:\n"
            "- n ≡ 17 (mod 29)\n"
            "- n ≡ 23 (mod 37)\n"
            "- n ≡ 5 (mod 13)\n"
            "- The sum of the base-10 digits of n is a prime number.\n"
            "Give the exact integer sum, ending with `ANSWER: <integer>`."
        ),
        expected="202251",
    ),
    BenchProblem(
        pid="M4_bayesian_modified_urn",
        category="math",
        forge_mode="reason",
        title="Two-Stage Bayesian Urn Update Without Replacement",
        prompt=(
            "There are 5 urns labeled k = 1, 2, 3, 4, 5. Urn k initially contains k red balls and (6 - k) blue balls "
            "(so 6 balls total per urn).\n"
            "Step 1: Choose one of the 5 urns uniformly at random.\n"
            "Step 2: Draw 4 balls WITHOUT replacement from the chosen urn. You observe exactly 2 red balls and 2 blue balls.\n"
            "Step 3: Return those 4 balls to the SAME urn, add 1 brand-new red ball to that urn (now 7 balls total), "
            "and draw 2 balls WITHOUT replacement from it.\n"
            "Conditioned on observing 2 red and 2 blue balls in Step 2, what is the exact probability that BOTH balls "
            "drawn in Step 3 are red? Give a reduced fraction `p/q`, ending with `ANSWER: p/q`."
        ),
        expected="44/147",
    ),
    # =========================================================================
    # CATEGORY 2: MULTI-CONSTRAINT LOGIC, CSP & SCHEDULING (L1..L4)
    # =========================================================================
    BenchProblem(
        pid="L1_knights_knaves_spies_6",
        category="logic",
        forge_mode="reason",
        title="6-Person Knights, Knaves & Spies Deduction",
        prompt=(
            "Six people (A, B, C, D, E, F) are each one of three types: Knight (always tells the truth), "
            "Knave (always lies), or Spy (may tell the truth or lie). "
            "There are EXACTLY 2 Knights, 2 Knaves, and 2 Spies.\n"
            "Statements:\n"
            "- A says: 'B is a Knight and C is not a Spy.'\n"
            "- B says: 'Exactly one of D and E is a Knave.'\n"
            "- C says: 'A and F are of the SAME type.'\n"
            "- D says: 'B is a Spy or E is a Knight.'\n"
            "- E says: 'Neither A nor C is a Knave.'\n"
            "- F says: 'If D is a Knave, then B is a Knight; and I am not a Spy.'\n"
            "Additional known facts:\n"
            "- F and A are NOT both Spies.\n"
            "- At least one of {A, D} is a Knight.\n"
            "- C is NOT a Knave.\n"
            "Which two people are the Knights? Give their letters in alphabetical order separated by a comma "
            "(e.g. `A,B`), ending with `ANSWER: X,Y`."
        ),
        expected="D,F",
    ),
    BenchProblem(
        pid="L2_job_shop_mutex_makespan",
        category="logic",
        forge_mode="reason",
        title="2-Machine Precedence + Mutual-Exclusion Makespan",
        prompt=(
            "Six non-preemptive jobs J0..J5 must be scheduled on 2 identical machines with integer start times >= 0.\n"
            "Durations: J0=4, J1=6, J2=5, J3=3, J4=7, J5=4.\n"
            "Constraints:\n"
            "1. Each machine runs at most one job at a time.\n"
            "2. Precedence (u -> v means u must finish at or before v starts): "
            "J0->J2, J0->J3, J1->J4, J2->J5, J3->J5, J4->J5.\n"
            "3. Cross-machine mutual exclusion (their execution intervals [start, end) CANNOT overlap even on different machines): "
            "(J1, J2) cannot overlap, and (J3, J4) cannot overlap.\n"
            "4. Release time: J1 cannot start before time 2 (start(J1) >= 2).\n"
            "What is the exact MINIMUM possible makespan (completion time of the last finishing job)? "
            "End with `ANSWER: <integer>`."
        ),
        expected="19",
    ),
    BenchProblem(
        pid="L3_five_office_zebra",
        category="logic",
        forge_mode="reason",
        title="5-Office 20-Attribute Interlocking Logic Puzzle",
        prompt=(
            "Five engineers (Ava, Ben, Cam, Dia, Eli) occupy 5 consecutive offices numbered 1 to 5 from left to right. "
            "Each uses a distinct language (C, Go, Py, Rs, Zig), drinks a distinct drink (Coffee, Matcha, Soda, Tea, Water), "
            "and arrives at a distinct hour (7, 8, 9, 10, 11).\n"
            "Clues:\n"
            "1. The Water drinker is in office 1.\n"
            "2. Ava drinks Matcha.\n"
            "3. The Py engineer drinks Coffee.\n"
            "4. Ben is immediately to the right of the person arriving at 7 (office(Ben) = office(7) + 1).\n"
            "5. Cam is immediately to the right of the Tea drinker (office(Cam) = office(Tea) + 1).\n"
            "6. Eli arrives at 9.\n"
            "7. The C engineer arrives at 8.\n"
            "8. The Go engineer arrives at 10.\n"
            "9. The Rs engineer drinks Soda.\n"
            "10. The Matcha drinker is adjacent to (office differs by 1 from) the Zig engineer.\n"
            "11. Dia's office number is strictly less than Ava's office number.\n"
            "12. The person arriving at 10 has a strictly smaller office number than the person arriving at 11.\n"
            "13. The Tea drinker is in office 3.\n"
            "14. The C engineer is adjacent to the Zig engineer.\n"
            "15. The Coffee drinker has a strictly smaller office number than the Soda drinker.\n"
            "Who is the Go engineer, and at what hour do they arrive? Reply as `Name,Hour` (e.g. `Ava,8`), "
            "ending with `ANSWER: Name,Hour`."
        ),
        expected="Dia,10",
    ),
    BenchProblem(
        pid="L4_weighted_constrained_coloring",
        category="logic",
        forge_mode="reason",
        title="Constrained Weighted 4-Coloring Optimization",
        prompt=(
            "Consider an undirected graph on 8 vertices labeled 0..7 with these 15 edges:\n"
            "(0,1), (0,2), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,6), (5,7), (6,7), (0,7), (2,6).\n"
            "Assign each vertex v in {0..7} an integer color c[v] in {1, 2, 3, 4} subject to ALL constraints:\n"
            "1. Adjacent vertices must have different colors (c[u] != c[v]).\n"
            "2. Vertex 0 must have color 1 (c[0] == 1).\n"
            "3. For every edge (u, v) where BOTH u and v are odd numbers, |c[u] - c[v]| != 3.\n"
            "4. No color may be used more than 2 times across the 8 vertices.\n"
            "Minimize the weighted cost: sum((v + 1) * c[v] for v in range(8)).\n"
            "What is the exact minimum possible weighted cost? End with `ANSWER: <integer>`."
        ),
        expected="88",
    ),
    # =========================================================================
    # CATEGORY 3: HARD ALGORITHMIC & EDGE-CASE CODING (C1..C4)
    # =========================================================================
    BenchProblem(
        pid="C1_exact_expr_eval",
        category="algo_code",
        forge_mode="code",
        title="Exact Rational Expression Parser (Unary/Power Precedence)",
        prompt=(
            "Implement a Python function `eval_expr(expr: str) -> Fraction` (returning `fractions.Fraction`) "
            "that parses and evaluates a mathematical expression string with EXACT rational arithmetic.\n"
            "Specification:\n"
            "- Numbers: non-negative integer literals (e.g. `42`) and decimal literals (e.g. `0.1`, `.5`, `12.34`) "
            "parsed EXACTLY as `Fraction` (e.g. `0.1 + 0.2 == Fraction(3, 10)`, never binary float).\n"
            "- Binary operators: `+`, `-`, `*`, `/` (exact rational division), `^` (exponentiation).\n"
            "- Precedence (lowest to highest):\n"
            "  1. `+`, `-` (left-associative)\n"
            "  2. `*`, `/` (left-associative)\n"
            "  3. Unary `+` and `-` (NOTE: lower precedence than `^`! So `-2^2 == -4`, `-2^3 == -8`, `--3 == 3`, "
            "AND unary operators may appear in the exponent of `^`, so `2^-3 == Fraction(1, 8)` and `2^-2^2 == Fraction(1, 16)`!)\n"
            "  4. `^` (RIGHT-associative! So `2^3^2 == 2^(3^2) == 512`, and `(-2)^2 == 4`.)\n"
            "  5. Parentheses `(...)`.\n"
            "- Exponentiation rules: exponent must evaluate to an exact integer (denominator == 1); otherwise raise `ValueError`. "
            "Define `0^0 == Fraction(1, 1)`. Raising `0` to a negative integer raises `ValueError`.\n"
            "- Division by zero raises `ValueError` (or `ZeroDivisionError` is NOT acceptable — must raise `ValueError`).\n"
            "- Any invalid syntax, empty/whitespace string, unknown character, mismatched parens, or trailing tokens raises `ValueError`.\n"
            "Output ONLY the Python code defining `eval_expr`."
        ),
        hidden_tests='''
from fractions import Fraction
import pytest

def test_basic_and_exact_decimals():
    assert eval_expr("0.1 + 0.2") == Fraction(3, 10)
    assert isinstance(eval_expr("3 + 4"), Fraction)
    assert eval_expr("  .5 * 4 - 1.25 ") == Fraction(3, 4)
    assert eval_expr("10 / 3 * 3") == Fraction(10, 1)

def test_unary_vs_power_precedence():
    assert eval_expr("-2^2") == Fraction(-4, 1)
    assert eval_expr("(-2)^2") == Fraction(4, 1)
    assert eval_expr("-2^3^2") == Fraction(-512, 1)
    assert eval_expr("2^3^2") == Fraction(512, 1)
    assert eval_expr("2^-3") == Fraction(1, 8)
    assert eval_expr("2^-2^2") == Fraction(1, 16)
    assert eval_expr("2^--3") == Fraction(8, 1)
    assert eval_expr("-2^-2^2") == Fraction(-1, 16)
    assert eval_expr("3 - -2^2") == Fraction(7, 1)
    assert eval_expr("+ - + 5") == Fraction(-5, 1)
    assert eval_expr("0^0") == Fraction(1, 1)
    assert eval_expr("(1/2)^-3") == Fraction(8, 1)

def test_complex_nested():
    assert eval_expr("((1 + 2)^2 - 5) / (2^-2 + 0.75) + -3^2") == Fraction(-5, 1)
    assert eval_expr("100 / 5 / 2 - 3 - 2") == Fraction(5, 1)

def test_errors():
    for bad in ["", "   ", "1 / 0", "0^-1", "4^(1/2)", "2 + * 3", "(1 + 2", "1 + 2)", "1 2", "2^", "abc", "1.2.3", "."]:
        try:
            res = eval_expr(bad)
            raise AssertionError(f"Expected ValueError for {bad!r}, got {res!r}")
        except ValueError:
            pass
''',
    ),
    BenchProblem(
        pid="C2_sweep_line_union_area",
        category="algo_code",
        forge_mode="code",
        title="Exact Sweep-Line Rectangle Union Area & Interior Max Overlap Depth",
        prompt=(
            "Implement `union_area_and_max_depth(rects: list[tuple[int, int, int, int]]) -> tuple[int, int]`.\n"
            "Each item in `rects` is `(x1, y1, x2, y2)` representing an axis-aligned rectangle `[x1, x2] x [y1, y2]` "
            "with arbitrary integer coordinates (can be negative, up to +/- 10^9).\n"
            "Rules:\n"
            "- Ignore degenerate rectangles where `x1 >= x2` or `y1 >= y2` (they contribute 0 to both area and depth).\n"
            "- Return `(total_union_area, max_overlap_depth)`:\n"
            "  * `total_union_area`: exact integer area of the union of all valid rectangles.\n"
            "  * `max_overlap_depth`: maximum integer number of valid rectangles whose OPEN INTERIORS "
            "`(x1, x2) x (y1, y2)` share a common point. Rectangles that only touch along an edge or at a corner "
            "do NOT overlap in open interior (depth stays 1, not 2!). Duplicate valid rectangles DO overlap.\n"
            "- If no valid rectangles exist, return `(0, 0)`.\n"
            "- Do NOT mutate the input list `rects`.\n"
            "- Must handle up to 1,500 rectangles in under 1.0 second.\n"
            "Output ONLY the Python code defining `union_area_and_max_depth`."
        ),
        hidden_tests='''
import random, time

def _brute(rects):
    valid = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rects if x1 < x2 and y1 < y2]
    if not valid:
        return (0, 0)
    xs = sorted({x for r in valid for x in (r[0], r[2])})
    ys = sorted({y for r in valid for y in (r[1], r[3])})
    area = 0
    max_d = 0
    for i in range(len(xs) - 1):
        mx = (xs[i] + xs[i+1]) / 2.0
        dx = xs[i+1] - xs[i]
        for j in range(len(ys) - 1):
            my = (ys[j] + ys[j+1]) / 2.0
            dy = ys[j+1] - ys[j]
            d = sum(1 for x1, y1, x2, y2 in valid if x1 < mx < x2 and y1 < my < y2)
            if d > 0:
                area += dx * dy
                if d > max_d:
                    max_d = d
    return (area, max_d)

def test_degenerate_and_touching():
    assert union_area_and_max_depth([]) == (0, 0)
    assert union_area_and_max_depth([(0, 0, 0, 5), (3, 2, 1, 5), (0, 5, 5, 5)]) == (0, 0)
    # Touching along edge & corner only -> depth MUST be 1
    assert union_area_and_max_depth([(0, 0, 2, 2), (2, 0, 4, 2), (0, 2, 2, 4), (2, 2, 4, 4)]) == (16, 1)
    # Duplicates
    assert union_area_and_max_depth([(-5, -5, 5, 5), (-5, -5, 5, 5), (-5, -5, 5, 5)]) == (100, 3)

def test_huge_negative_coords():
    rects = [(-10**9, -10**9, 10**9, 10**9), (0, 0, 10**9, 10**9)]
    assert union_area_and_max_depth(rects) == (4 * 10**18, 2)

def test_differential_fuzz():
    rng = random.Random(12345)
    for _ in range(30):
        n = rng.randint(1, 14)
        rects = []
        for _ in range(n):
            x1 = rng.randint(-6, 6)
            y1 = rng.randint(-6, 6)
            x2 = x1 + rng.randint(-2, 6)
            y2 = y1 + rng.randint(-2, 6)
            rects.append((x1, y1, x2, y2))
        snap = list(rects)
        got = union_area_and_max_depth(rects)
        exp = _brute(snap)
        assert rects == snap, "Input list mutated!"
        assert got == exp, f"Mismatch on {snap}: got {got}, exp {exp}"

def test_scale_performance():
    rng = random.Random(99)
    rects = [(rng.randint(-10000, 10000), rng.randint(-10000, 10000), 0, 0) for _ in range(1500)]
    rects = [(x, y, x + rng.randint(1, 500), y + rng.randint(1, 500)) for x, y, _, _ in rects]
    t0 = time.perf_counter()
    area, depth = union_area_and_max_depth(rects)
    dt = time.perf_counter() - t0
    assert area > 0 and depth >= 1
    assert dt < 1.5, f"Too slow: {dt:.2f}s"
''',
    ),
    BenchProblem(
        pid="C3_mvcc_snapshot_store",
        category="algo_code",
        forge_mode="code",
        title="Snapshot-Isolation MVCC Key-Value Store",
        prompt=(
            "Implement a Python class `MVCCStore` providing Snapshot Isolation with First-Committer-Wins "
            "write-write conflict detection.\n"
            "Exact API:\n"
            "- `begin() -> int`: starts a new active transaction whose `snapshot_ts` equals the store's current "
            "committed version counter (initially 0). Returns a unique integer `tx_id`.\n"
            "- `get(tx_id: int, key: str)`: returns the value visible to `tx_id`: if `tx_id` has buffered a write "
            "for `key`, return that buffered value (`None` if `tx_id` deleted `key`); otherwise return the latest "
            "committed value for `key` with `commit_ts <= tx_id.snapshot_ts`, or `None` if absent/deleted.\n"
            "- `put(tx_id: int, key: str, value) -> None`: buffers setting `key` to `value`. NOTE: falsy values "
            "like `0`, `False`, `\"\"`, `[]`, `{}` are VALID stored values and must NOT be treated as missing/deleted!\n"
            "- `delete(tx_id: int, key: str) -> None`: buffers deleting `key`.\n"
            "- `abort(tx_id: int) -> None`: discards `tx_id`'s buffered writes and marks `tx_id` closed.\n"
            "- `commit(tx_id: int) -> bool`:\n"
            "  * Write-set consists of EVERY key touched by `put` or `delete` in `tx_id` (even if `put` then `delete`, "
            "or `delete` on a non-existent key, or `put` of the same value).\n"
            "  * If ANY key in `tx_id`'s write-set has a committed version (put or delete) with `commit_ts > tx_id.snapshot_ts`, "
            "there is a write-write conflict: mark `tx_id` closed (aborted), discard all its writes, and return `False`.\n"
            "  * Otherwise, commit ALL buffered writes atomically at a new version `commit_ts = current_version + 1` "
            "(for read-only transactions with empty write-set, commit succeeds and returns `True` without creating conflicts), "
            "mark `tx_id` closed, and return `True`.\n"
            "- Calling `get`, `put`, `delete`, `commit`, or `abort` with an unknown or already closed (committed/aborted) "
            "`tx_id` MUST raise `ValueError`.\n"
            "Output ONLY the Python code defining `MVCCStore`."
        ),
        hidden_tests='''
def test_snapshot_and_falsy_values():
    s = MVCCStore()
    t1 = s.begin()
    s.put(t1, "zero", 0)
    s.put(t1, "empty", "")
    s.put(t1, "flag", False)
    s.put(t1, "lst", [])
    assert s.get(t1, "zero") == 0
    assert s.get(t1, "flag") is False
    assert s.commit(t1) is True

    t2 = s.begin()
    t3 = s.begin()
    assert s.get(t2, "zero") == 0
    assert s.get(t2, "empty") == ""
    assert s.get(t2, "flag") is False
    assert s.get(t2, "lst") == []

    # t2 modifies zero and deletes flag
    s.put(t2, "zero", 99)
    s.delete(t2, "flag")
    assert s.get(t2, "flag") is None
    assert s.commit(t2) is True

    # t3 snapshot isolation must still see old values!
    assert s.get(t3, "zero") == 0
    assert s.get(t3, "flag") is False
    assert s.commit(t3) is True  # read-only commits cleanly

    t4 = s.begin()
    assert s.get(t4, "zero") == 99
    assert s.get(t4, "flag") is None

def test_write_write_conflicts_and_deletes():
    s = MVCCStore()
    t0 = s.begin()
    s.put(t0, "a", 1)
    s.put(t0, "b", 2)
    assert s.commit(t0) is True

    t1 = s.begin()
    t2 = s.begin()
    s.put(t1, "a", 10)
    s.delete(t2, "a")  # delete conflicts with put!
    assert s.commit(t1) is True
    assert s.commit(t2) is False

    # Conflict on delete of non-existent key vs put
    t3 = s.begin()
    t4 = s.begin()
    s.delete(t3, "ghost")
    s.put(t4, "ghost", "boo")
    assert s.commit(t3) is True
    assert s.commit(t4) is False

    # Same-value put still counts in write-set
    t5 = s.begin()
    t6 = s.begin()
    s.put(t5, "b", 2)
    s.put(t6, "b", 200)
    assert s.commit(t5) is True
    assert s.commit(t6) is False

    # Put then delete in same tx still conflicts
    t7 = s.begin()
    t8 = s.begin()
    s.put(t7, "b", 5)
    s.delete(t7, "b")
    s.put(t8, "b", 77)
    assert s.commit(t8) is True
    assert s.commit(t7) is False

def test_disjoint_writes_and_closed_tx_errors():
    s = MVCCStore()
    t1 = s.begin()
    t2 = s.begin()
    s.put(t1, "x", 1)
    s.put(t2, "y", 2)
    assert s.commit(t1) is True
    assert s.commit(t2) is True  # disjoint keys -> no conflict!

    t3 = s.begin()
    assert s.get(t3, "x") == 1 and s.get(t3, "y") == 2
    s.abort(t3)

    for fn in [
        lambda: s.get(t1, "x"),
        lambda: s.put(t1, "x", 9),
        lambda: s.delete(t1, "x"),
        lambda: s.commit(t1),
        lambda: s.abort(t1),
        lambda: s.get(t3, "x"),
        lambda: s.get(999999, "x"),
    ]:
        try:
            fn()
            raise AssertionError("Expected ValueError on closed/unknown tx_id")
        except ValueError:
            pass
''',
    ),
    BenchProblem(
        pid="C4_lex_min_constrained_path",
        category="algo_code",
        forge_mode="code",
        title="Lexicographically-Smallest Resource-Constrained Shortest Walk",
        prompt=(
            "Implement `min_cost_lex_path(n: int, edges: list[tuple[int, int, int, int]], start: int, goal: int, max_time: int) -> tuple[int, list[int]] | None`.\n"
            "You are given a directed graph with vertices `0..n-1` and edges `(u, v, cost, duration)` where "
            "`cost >= 0` and `duration >= 0` (NOTE: zero-cost and/or zero-duration edges, parallel edges between "
            "the same `(u, v)`, self-loops, and cycles MAY all exist!).\n"
            "Find a valid walk `[v0, v1, ..., vk]` with `v0 == start`, `vk == goal`, and total duration `<= max_time` that:\n"
            "1. Minimizes total `cost`,\n"
            "2. Breaks ties by minimizing total `duration`,\n"
            "3. Breaks remaining ties by returning the LEXICOGRAPHICALLY SMALLEST vertex list `path` "
            "(standard Python list comparison: `[0, 1, 3] < [0, 2, 3]` and `[0, 1] < [0, 1, 1]`).\n"
            "Return `(min_cost, best_path)`, or `None` if no valid walk exists (including if `max_time < 0`).\n"
            "Note: if `start == goal` and `max_time >= 0`, the 0-edge walk `[start]` has cost 0, duration 0, and is lex smallest, so return `(0, [start])`.\n"
            "Must NOT infinite-loop on 0-cost/0-duration cycles, and must handle `n=80, max_time=120` in < 1.0s.\n"
            "Output ONLY the Python code defining `min_cost_lex_path`."
        ),
        hidden_tests='''
import time

def test_basic_and_ties():
    assert min_cost_lex_path(3, [], 1, 1, 0) == (0, [1])
    assert min_cost_lex_path(3, [], 1, 1, -1) is None
    assert min_cost_lex_path(3, [(0, 1, 1, 5)], 0, 1, 4) is None

    # Tie on cost (10), different duration -> smaller duration wins even if lex larger!
    edges = [
        (0, 1, 5, 6), (1, 3, 5, 4),  # cost 10, dur 10, path [0,1,3]
        (0, 2, 5, 2), (2, 3, 5, 3),  # cost 10, dur 5,  path [0,2,3]
    ]
    assert min_cost_lex_path(4, edges, 0, 3, 15) == (10, [0, 2, 3])

    # Tie on cost (10) AND duration (5) -> lex smallest wins!
    edges2 = [
        (0, 2, 5, 2), (2, 3, 5, 3),
        (0, 1, 4, 2), (1, 3, 6, 3),
    ]
    assert min_cost_lex_path(4, edges2, 0, 3, 5) == (10, [0, 1, 3])

def test_zero_cycles_and_lex_prefix_trap():
    # Classic trap: node 0->1 (cost 0, dur 0), 1->0 (cost 0, dur 0), 1->2 (cost 0, dur 0), 0->2 (cost 0, dur 0)
    # Walks 0->2: [0, 2] vs [0, 1, 2] -> [0, 1, 2] < [0, 2] lexicographically!
    # Wait: [0, 1, 0, 1, 2] < [0, 1, 2]!! HOLD ON! Let's check simple vs walk with 0-cost/0-dur cycle:
    # Let's test 0-cost/0-dur cycle that does NOT improve lex (higher vertex id in cycle: 0->3->0, 0->2)
    edges = [
        (0, 3, 0, 0), (3, 0, 0, 0), (3, 3, 0, 0),
        (0, 2, 1, 1), (3, 2, 1, 1),
    ]
    # [0, 2] cost 1 dur 1 vs [0, 3, 2] cost 1 dur 1 -> [0, 2] < [0, 3, 2]! Must not loop on 0<->3!
    assert min_cost_lex_path(4, edges, 0, 2, 10) == (1, [0, 2])

def test_pareto_non_dominated_tradeoff():
    # Path A to node 2: cheap (cost 1) but slow (dur 9) -> cannot finish 2->4 (dur 3) within max_time=10
    # Path B to node 2: pricier (cost 5) but fast (dur 4) -> finishes 2->4 (cost 2, dur 3) -> total (7, 7)
    # Path C direct 0->4: cost 9, dur 1
    edges = [
        (0, 1, 1, 5), (1, 2, 0, 4),   # to 2: (cost=1, dur=9)
        (0, 3, 5, 2), (3, 2, 0, 2),   # to 2: (cost=5, dur=4)
        (2, 4, 2, 3),
        (0, 4, 9, 1),
        # Parallel edges between 3 and 2: worse parallel edge should not hide better one
        (3, 2, 10, 10),
    ]
    assert min_cost_lex_path(5, edges, 0, 4, 10) == (7, [0, 3, 2, 4])
    assert min_cost_lex_path(5, edges, 0, 4, 12) == (3, [0, 1, 2, 4])

def test_same_cost_lex_via_longer_hop_count():
    # [0, 1, 2, 4] has cost 6, dur 6; [0, 3, 4] has cost 6, dur 6 -> [0, 1, 2, 4] < [0, 3, 4]
    edges = [
        (0, 3, 3, 3), (3, 4, 3, 3),
        (0, 1, 2, 2), (1, 2, 2, 2), (2, 4, 2, 2),
    ]
    assert min_cost_lex_path(5, edges, 0, 4, 6) == (6, [0, 1, 2, 4])

def test_scale():
    edges = []
    for i in range(79):
        edges.append((i, i + 1, 2, 1))
        if i + 2 < 80:
            edges.append((i, i + 2, 3, 2))
        if i % 5 == 0 and i > 0:
            edges.append((i, i - 1, 1, 1))
    t0 = time.perf_counter()
    res = min_cost_lex_path(80, edges, 0, 79, 100)
    dt = time.perf_counter() - t0
    assert res is not None and res[0] == 119
    assert dt < 1.5, f"Too slow: {dt:.2f}s"
''',
    ),
    # =========================================================================
    # CATEGORY 4: ADVERSARIAL MULTI-BUG DEBUGGING (D1..D4)
    # =========================================================================
    BenchProblem(
        pid="D1_debug_ttl_lru_cache",
        category="debug_code",
        forge_mode="code",
        title="Fix 5 Interlocking Bugs in TTLLRUCache",
        buggy_starter='''from collections import OrderedDict

class TTLLRUCache:
    def __init__(self, capacity: int, clock=None):
        self.capacity = capacity
        self.clock = clock or (lambda: 0.0)
        self._data = OrderedDict()  # key -> (value, expires_at)
        self._hits = 0
        self._misses = 0

    def get(self, key):
        if key not in self._data:
            self._misses += 1
            return None
        val, exp = self._data[key]
        if exp is not None and self.clock() > exp:
            del self._data[key]
            self._misses += 1
            return None
        self._data.move_to_end(key)
        self._hits += 1
        return val

    def put(self, key, value, ttl=None):
        now = self.clock()
        exp = (now + ttl) if ttl is not None else None
        if key in self._data:
            self._data[key] = (value, exp)
            return
        if len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = (value, exp)

    def size(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}
''',
        prompt=(
            "Fix ALL bugs in the following `TTLLRUCache` implementation so it obeys every rule of the specification below.\n\n"
            "SPECIFICATION:\n"
            "1. `TTLLRUCache(capacity: int, clock)`: `capacity >= 0` (raise `ValueError` if `capacity < 0`). "
            "If `capacity == 0`, `put` is a no-op and never crashes, `get` always misses and returns `None`, `size()` is 0.\n"
            "2. Expiration boundary: an entry with `ttl` set at time `now` expires AT `now + ttl` (so if `clock() >= expires_at`, "
            "it is EXPIRED!). If `ttl is not None and ttl <= 0`, the item is immediately expired (if `key` already existed, "
            "remove it; do not store it and do not evict other live entries!).\n"
            "3. `put(key, value, ttl=None)`: MUST purge expired entries BEFORE checking whether capacity is full so live entries "
            "are NEVER evicted to make room when expired entries exist! Updating an existing live key MUST refresh it as most-recently-used "
            "(move to MRU end) AND update its value & expiration (`ttl=None` means no expiration).\n"
            "4. `get(key)`: returns stored value (note: falsy values like `0`, `False`, `\"\"`, `None`? Support storing any value including `0`/`False`/`\"\"`; "
            "missing or expired returns `None` and increments `misses` by 1; live hit refreshes LRU order and increments `hits` by 1).\n"
            "5. `size() -> int`: returns the count of CURRENTLY LIVE (non-expired) entries.\n"
            "6. `stats() -> dict`: returns `{\"hits\": int, \"misses\": int}`.\n\n"
            "BUGGY CODE:\n"
            "```python\n"
            "from collections import OrderedDict\n\n"
            "class TTLLRUCache:\n"
            "    def __init__(self, capacity: int, clock=None):\n"
            "        self.capacity = capacity\n"
            "        self.clock = clock or (lambda: 0.0)\n"
            "        self._data = OrderedDict()  # key -> (value, expires_at)\n"
            "        self._hits = 0\n"
            "        self._misses = 0\n\n"
            "    def get(self, key):\n"
            "        if key not in self._data:\n"
            "            self._misses += 1\n"
            "            return None\n"
            "        val, exp = self._data[key]\n"
            "        if exp is not None and self.clock() > exp:\n"
            "            del self._data[key]\n"
            "            self._misses += 1\n"
            "            return None\n"
            "        self._data.move_to_end(key)\n"
            "        self._hits += 1\n"
            "        return val\n\n"
            "    def put(self, key, value, ttl=None):\n"
            "        now = self.clock()\n"
            "        exp = (now + ttl) if ttl is not None else None\n"
            "        if key in self._data:\n"
            "            self._data[key] = (value, exp)\n"
            "            return\n"
            "        if len(self._data) >= self.capacity:\n"
            "            self._data.popitem(last=False)\n"
            "        self._data[key] = (value, exp)\n\n"
            "    def size(self) -> int:\n"
            "        return len(self._data)\n\n"
            "    def stats(self) -> dict:\n"
            "        return {\"hits\": self._hits, \"misses\": self._misses}\n"
            "```\n"
            "Output ONLY the complete fixed Python code defining `TTLLRUCache`."
        ),
        hidden_tests='''
def test_expiry_boundary_and_purge_before_evict():
    now = [10.0]
    c = TTLLRUCache(2, clock=lambda: now[0])
    c.put("a", 1, ttl=5.0)   # expires at 15.0
    c.put("b", 2, ttl=20.0)  # expires at 30.0
    now[0] = 15.0            # exact boundary -> 'a' is EXPIRED!
    assert c.size() == 1
    # Putting 'c' must purge expired 'a' and NOT evict live 'b'!
    c.put("c", 3, ttl=20.0)
    assert c.get("b") == 2
    assert c.get("c") == 3
    assert c.get("a") is None
    assert c.stats() == {"hits": 2, "misses": 1}

def test_update_refreshes_lru_and_nonpositive_ttl():
    now = [0.0]
    c = TTLLRUCache(2, clock=lambda: now[0])
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)          # MUST move 'a' to MRU so 'b' is now LRU!
    c.put("c", 3)           # MUST evict 'b', keeping 'a' and 'c'!
    assert c.get("a") == 10
    assert c.get("b") is None
    assert c.get("c") == 3
    # Non-positive TTL deletes existing key and does not evict live keys
    c.put("a", 99, ttl=0)
    assert c.size() == 1
    assert c.get("a") is None
    c.put("d", 4, ttl=-5)
    assert c.size() == 1
    assert c.get("c") == 3

def test_zero_and_negative_capacity():
    c0 = TTLLRUCache(0, clock=lambda: 0.0)
    c0.put("x", 1)
    assert c0.size() == 0
    assert c0.get("x") is None
    assert c0.stats() == {"hits": 0, "misses": 1}
    try:
        TTLLRUCache(-1)
        raise AssertionError("Expected ValueError for negative capacity")
    except ValueError:
        pass
''',
    ),
    BenchProblem(
        pid="D2_debug_cron_matcher",
        category="debug_code",
        forge_mode="code",
        title="Fix 5 Classic POSIX/Vixie Cron Bugs in next_cron_run",
        buggy_starter='''from datetime import datetime, timedelta

def _parse_field(field: str, lo: int, hi: int) -> set[int]:
    vals = set()
    for part in field.split(","):
        if part == "*":
            vals.update(range(lo, hi + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            vals.update(range(lo, hi + 1, step))
        elif "-" in part:
            a, b = map(int, part.split("-"))
            vals.update(range(a, b + 1))
        else:
            vals.add(int(part))
    return vals

def next_cron_run(expr: str, after: datetime) -> datetime:
    fields = expr.split()
    mins = _parse_field(fields[0], 0, 59)
    hrs = _parse_field(fields[1], 0, 23)
    doms = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    dows = _parse_field(fields[4], 0, 6)
    cur = after.replace(second=0, microsecond=0)
    while True:
        # Python weekday(): Mon=0..Sun=6; cron: Sun=0, Mon=1..Sat=6
        cron_dow = (cur.weekday() + 1) % 7
        if cur.month in months and cur.day in doms and cron_dow in dows and cur.hour in hrs and cur.minute in mins:
            return cur
        cur += timedelta(minutes=1)
''',
        prompt=(
            "Fix ALL bugs in the following 5-field POSIX/Vixie cron scheduler `next_cron_run(expr: str, after: datetime) -> datetime`.\n\n"
            "SPECIFICATION:\n"
            "1. `expr` has exactly 5 whitespace-separated fields: `minute hour dom month dow` "
            "(valid ranges: minute 0-59, hour 0-23, dom 1-31, month 1-12, dow 0-7 where BOTH `0` and `7` mean Sunday, 1=Mon..6=Sat).\n"
            "2. Each field is a comma-separated list of items supporting `*`, `*/step`, `N`, `A-B`, `A-B/step`, and `N/step` "
            "(meaning `N..hi` by `step`). Validate syntax & bounds: raise `ValueError` for wrong field count, step <= 0, "
            "out-of-range numbers, reversed ranges `A > B`, or non-numeric garbage.\n"
            "3. Strictly AFTER `after`: the returned datetime MUST have `second=0, microsecond=0` and satisfy `result > after`. "
            "(E.g., if `after = 12:00:00` and expr matches 12:00, return the NEXT matching minute after 12:00; "
            "if `after = 12:00:01` and expr matches 12:01, return 12:01:00!)\n"
            "4. Standard Vixie/POSIX DOM vs DOW semantics:\n"
            "   - If BOTH `dom` and `dow` are unrestricted (`*`), every day matches.\n"
            "   - If ONLY ONE of `dom` or `dow` is restricted (not `*` — note `*/2` IS restricted!), match that restricted field.\n"
            "   - If BOTH `dom` and `dow` are restricted (neither raw field is `*`), a day matches if EITHER `dom` matches OR `dow` matches (logical OR, NOT AND!).\n"
            "5. Leap years & impossible dates: MUST find Feb 29 on leap years (searching across up to 8 years cleanly in milliseconds, "
            "NOT minute-by-minute over 8 years!), and MUST raise `ValueError` if no valid calendar date can EVER match (e.g. `0 0 31 2 *` or `0 0 31 4,6 *`).\n\n"
            "BUGGY CODE:\n"
            "```python\n"
            "from datetime import datetime, timedelta\n\n"
            "def _parse_field(field: str, lo: int, hi: int) -> set[int]:\n"
            "    vals = set()\n"
            "    for part in field.split(\",\"):\n"
            "        if part == \"*\":\n"
            "            vals.update(range(lo, hi + 1))\n"
            "        elif part.startswith(\"*/\"):\n"
            "            step = int(part[2:])\n"
            "            vals.update(range(lo, hi + 1, step))\n"
            "        elif \"-\" in part:\n"
            "            a, b = map(int, part.split(\"-\"))\n"
            "            vals.update(range(a, b + 1))\n"
            "        else:\n"
            "            vals.add(int(part))\n"
            "    return vals\n\n"
            "def next_cron_run(expr: str, after: datetime) -> datetime:\n"
            "    fields = expr.split()\n"
            "    mins = _parse_field(fields[0], 0, 59)\n"
            "    hrs = _parse_field(fields[1], 0, 23)\n"
            "    doms = _parse_field(fields[2], 1, 31)\n"
            "    months = _parse_field(fields[3], 1, 12)\n"
            "    dows = _parse_field(fields[4], 0, 6)\n"
            "    cur = after.replace(second=0, microsecond=0)\n"
            "    while True:\n"
            "        cron_dow = (cur.weekday() + 1) % 7\n"
            "        if cur.month in months and cur.day in doms and cron_dow in dows and cur.hour in hrs and cur.minute in mins:\n"
            "            return cur\n"
            "        cur += timedelta(minutes=1)\n"
            "```\n"
            "Output ONLY the complete fixed Python code defining `next_cron_run`."
        ),
        hidden_tests='''
from datetime import datetime
import time

def test_strictly_after_and_steps():
    # Exact boundary must advance!
    assert next_cron_run("0 12 * * *", datetime(2025, 1, 1, 12, 0, 0)) == datetime(2025, 1, 2, 12, 0)
    # Non-zero seconds rounds up to next minute
    assert next_cron_run("1 12 * * *", datetime(2025, 1, 1, 12, 0, 1)) == datetime(2025, 1, 1, 12, 1)
    assert next_cron_run("0 12 * * *", datetime(2025, 1, 1, 12, 0, 1)) == datetime(2025, 1, 2, 12, 0)
    # Range with step: 10-20/4 -> 10, 14, 18
    assert next_cron_run("10-20/4 9 * * *", datetime(2025, 1, 1, 9, 10, 0)) == datetime(2025, 1, 1, 9, 14)
    # N/step: 50/5 -> 50, 55
    assert next_cron_run("50/5 * * * *", datetime(2025, 1, 1, 9, 50, 0)) == datetime(2025, 1, 1, 9, 55)

def test_vixie_dom_or_dow_and_sunday_7():
    # 2025-01-01 is Wednesday (dow=3). 2025-01-05 is Sunday (dow=0 or 7).
    # Restrict BOTH dom=15 and dow=7 (Sunday): OR semantics -> must hit Jan 5 (Sun), NOT Jan 15!
    assert next_cron_run("0 8 15 * 7", datetime(2025, 1, 1, 0, 0)) == datetime(2025, 1, 5, 8, 0)
    # Only dow restricted: 2025-01-05 is Sunday
    assert next_cron_run("30 6 * * 7", datetime(2025, 1, 1, 0, 0)) == datetime(2025, 1, 5, 6, 30)
    # Range ending at 7: 5-7 = Fri(5), Sat(6), Sun(0)
    assert next_cron_run("0 9 * * 5-7", datetime(2025, 1, 4, 10, 0)) == datetime(2025, 1, 5, 9, 0)

def test_leap_year_fast_and_impossible_dates():
    t0 = time.perf_counter()
    assert next_cron_run("0 0 29 2 *", datetime(2025, 3, 1, 0, 0)) == datetime(2028, 2, 29, 0, 0)
    assert time.perf_counter() - t0 < 0.5, "Leap year jump too slow!"
    # 2096 -> 2104 (2100 is NOT a leap year!)
    assert next_cron_run("0 0 29 2 *", datetime(2096, 3, 1, 0, 0)) == datetime(2104, 2, 29, 0, 0)

    for bad in ["0 0 31 2 *", "0 0 31 4,6,9,11 *", "0 0 * *", "60 * * * *", "*/0 * * * *", "10-5 * * * *", "a * * * *"]:
        try:
            r = next_cron_run(bad, datetime(2025, 1, 1, 0, 0))
            raise AssertionError(f"Expected ValueError for {bad!r}, got {r}")
        except ValueError:
            pass
''',
    ),
    BenchProblem(
        pid="D3_debug_glob_matcher",
        category="debug_code",
        forge_mode="code",
        title="Fix Exponential Backtracking & Semantics in Filepath Glob Matcher",
        buggy_starter='''def glob_match(pattern: str, path: str) -> bool:
    def _rec(pi: int, si: int) -> bool:
        if pi == len(pattern) and si == len(path):
            return True
        if pi == len(pattern):
            return False
        if pattern[pi:pi+2] == "**":
            return _rec(pi + 2, si) or (si < len(path) and _rec(pi, si + 1))
        if pattern[pi] == "*":
            return _rec(pi + 1, si) or (si < len(path) and path[si] != "/" and _rec(pi, si + 1))
        if pattern[pi] == "?":
            return si < len(path) and path[si] != "/" and _rec(pi + 1, si + 1)
        if pattern[pi] == "[":
            end = pattern.find("]", pi + 1)
            if end != -1 and si < len(path):
                cls = pattern[pi+1:end]
                neg = cls.startswith("!")
                if neg:
                    cls = cls[1:]
                matched = path[si] in cls
                if neg:
                    matched = not matched
                return matched and _rec(end + 1, si + 1)
        return si < len(path) and pattern[pi] == path[si] and _rec(pi + 1, si + 1)
    return _rec(0, 0)
''',
        prompt=(
            "Fix ALL bugs and performance cliffs in the following filepath glob matcher `glob_match(pattern: str, path: str) -> bool`.\n\n"
            "SPECIFICATION:\n"
            "1. `*` matches 0 or more characters EXCEPT `/`.\n"
            "2. `?` matches any SINGLE character EXCEPT `/`.\n"
            "3. `**` as a FULL path segment (`**`, `**/...`, `.../**`, `.../**/...`) matches ZERO or more path segments! "
            "Crucially:\n"
            "   - `**/foo.py` MUST match BOTH `foo.py` (0 dirs) AND `a/b/foo.py`!\n"
            "   - `a/**/b` MUST match `a/b`, `a/x/b`, and `a/x/y/b`, and MUST NOT match `a/xb` or `ax/b`!\n"
            "   - `a/**` MUST match `a`, `a/`, `a/b`, `a/b/c`, and MUST NOT match `ab/c`!\n"
            "   - `**` standing inside a segment like `a**b` acts like two `*`s (cannot cross `/`, so `a**b` does NOT match `a/b`).\n"
            "4. Character classes `[...]`: matches a SINGLE character EXCEPT `/`! (Even negated `[!a-z]` MUST NOT match `/`!)\n"
            "   - Supports ranges `a-z`, negation `[!...]` or `[^...]`, literal `]` when immediately after `[` or `[!` (e.g. `[]a]` matches `]` or `a`, `[!]a]` matches anything except `]`, `a`, `/`), and literal `-` at the start or end of class (e.g. `[-a]` or `[a-]`).\n"
            "   - Unclosed `[` (no closing `]`) is treated as literal `[`.\n"
            "5. Polynomial time (NO exponential backtracking!): pathological patterns like `a/**/**/**/**/z` or `*a*a*a*a*a*a*a*b` "
            "against 60-character non-matching paths must finish in < 0.05 seconds.\n\n"
            "BUGGY CODE:\n"
            "```python\n"
            "def glob_match(pattern: str, path: str) -> bool:\n"
            "    def _rec(pi: int, si: int) -> bool:\n"
            "        if pi == len(pattern) and si == len(path):\n"
            "            return True\n"
            "        if pi == len(pattern):\n"
            "            return False\n"
            "        if pattern[pi:pi+2] == \"**\":\n"
            "            return _rec(pi + 2, si) or (si < len(path) and _rec(pi, si + 1))\n"
            "        if pattern[pi] == \"*\":\n"
            "            return _rec(pi + 1, si) or (si < len(path) and path[si] != \"/\" and _rec(pi, si + 1))\n"
            "        if pattern[pi] == \"?\":\n"
            "            return si < len(path) and path[si] != \"/\" and _rec(pi + 1, si + 1)\n"
            "        if pattern[pi] == \"[\":\n"
            "            end = pattern.find(\"]\", pi + 1)\n"
            "            if end != -1 and si < len(path):\n"
            "                cls = pattern[pi+1:end]\n"
            "                neg = cls.startswith(\"!\")\n"
            "                if neg:\n"
            "                    cls = cls[1:]\n"
            "                matched = path[si] in cls\n"
            "                if neg:\n"
            "                    matched = not matched\n"
            "                return matched and _rec(end + 1, si + 1)\n"
            "        return si < len(path) and pattern[pi] == path[si] and _rec(pi + 1, si + 1)\n"
            "    return _rec(0, 0)\n"
            "```\n"
            "Output ONLY the complete fixed Python code defining `glob_match`."
        ),
        hidden_tests='''
import time

def test_globstar_segments():
    assert glob_match("**/foo.py", "foo.py") is True
    assert glob_match("**/foo.py", "a/b/foo.py") is True
    assert glob_match("**/foo.py", "barfoo.py") is False
    assert glob_match("a/**/b", "a/b") is True
    assert glob_match("a/**/b", "a/x/b") is True
    assert glob_match("a/**/b", "a/x/y/b") is True
    assert glob_match("a/**/b", "a/xb") is False
    assert glob_match("a/**/b", "ax/b") is False
    assert glob_match("a/**", "a") is True
    assert glob_match("a/**", "a/b/c") is True
    assert glob_match("a/**", "ab/c") is False
    # Inline ** cannot cross '/'
    assert glob_match("a**b", "axxxb") is True
    assert glob_match("a**b", "a/b") is False
    assert glob_match("**", "a/b/c") is True
    assert glob_match("**", "") is True

def test_char_classes():
    assert glob_match("[a-z].py", "m.py") is True
    assert glob_match("[a-z].py", "M.py") is False
    assert glob_match("[!a-z]", "5") is True
    assert glob_match("[!a-z]", "/") is False  # NEVER matches '/'!
    assert glob_match("[]a]", "]") is True
    assert glob_match("[]a]", "a") is True
    assert glob_match("[!]a]", "b") is True
    assert glob_match("[!]a]", "]") is False
    assert glob_match("[!]a]", "/") is False
    assert glob_match("[-a]", "-") is True
    assert glob_match("[a-]", "-") is True
    assert glob_match("[abc", "[abc") is True

def test_no_exponential_blowup():
    t0 = time.perf_counter()
    assert glob_match("*a*a*a*a*a*a*a*a*b", "a" * 55 + "c") is False
    assert glob_match("a/**/**/**/**/**/z", "a/" + "x/" * 25 + "y") is False
    assert time.perf_counter() - t0 < 0.2
''',
    ),
    BenchProblem(
        pid="D4_debug_dag_scheduler",
        category="debug_code",
        forge_mode="code",
        title="Fix 5 Interlocking Bugs in Priority + Exclusive-Group DAG Scheduler",
        buggy_starter='''import heapq

def schedule_dag(tasks: dict[str, dict], workers: int) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    # Check cycles from roots
    indeg = {k: len(v.get("deps", [])) for k, v in tasks.items()}
    children = {k: [] for k in tasks}
    for k, v in tasks.items():
        for d in v.get("deps", []):
            children[d].append(k)
    ready = [(tasks[k].get("priority", 0), k) for k, deg in indeg.items() if deg == 0]
    heapq.heapify(ready)
    running = []  # (end_time, task_id)
    schedule = {}
    t = 0
    while ready or running:
        while ready and len(running) < workers:
            _, tid = heapq.heappop(ready)
            dur = tasks[tid]["duration"]
            schedule[tid] = (t, t + dur)
            heapq.heappush(running, (t + dur, tid))
        if not running:
            break
        t = running[0][0]
        while running and running[0][0] == t:
            _, done_id = heapq.heappop(running)
            for ch in children[done_id]:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    heapq.heappush(ready, (tasks[ch].get("priority", 0), ch))
    return {"schedule": schedule, "makespan": t}
''',
        prompt=(
            "Fix ALL bugs in the following event-driven parallel DAG scheduler `schedule_dag(tasks: dict[str, dict], workers: int) -> dict`.\n\n"
            "SPECIFICATION:\n"
            "Input: `tasks` maps `task_id -> {\"duration\": int, \"deps\": list[str], \"priority\": int (default 0), \"exclusive_group\": str | None (default None)}`.\n"
            "1. Validation: raise `ValueError` if `workers <= 0`, if any `duration < 0`, if any dependency is missing from `tasks`, "
            "or if ANY cycle exists (including self-loops or disconnected cycles where in-degree > 0 for every cycle node!).\n"
            "2. Empty `tasks == {}` returns `{\"schedule\": {}, \"makespan\": 0}`.\n"
            "3. Dispatch priority at any timestamp `t`:\n"
            "   - Among all ready (all deps finished at time `<= t`) and eligible tasks, pick HIGHER `priority` FIRST, "
            "breaking ties by LEXICOGRAPHICALLY SMALLEST `task_id`.\n"
            "4. `exclusive_group` mutual exclusion:\n"
            "   - If a task has `exclusive_group` set to a non-None string `g`, AT MOST ONE task with `exclusive_group == g` "
            "may be executing concurrently at any open time interval (even across different workers!).\n"
            "   - If the highest-priority ready task is blocked by an active `exclusive_group`, the scheduler MUST skip it for now "
            "and dispatch the next highest-priority eligible ready task if a worker is free! (Wait-free work-conserving dispatch.)\n"
            "5. Zero-duration tasks (`duration == 0`):\n"
            "   - When eligible and dispatched at time `t` (when a worker is free — or even better: a zero-duration task starts and finishes "
            "instantaneously at `t`, `(t, t)`, immediately frees its worker & exclusive_group at `t`, and immediately unlocks its dependents at `t`!).\n"
            "   - Note: a zero-duration task with `exclusive_group == g` CANNOT run while a positive-duration task with group `g` is actively running, "
            "because the positive task holds group `g` on `[s, e)`.\n"
            "6. Output: `{\"schedule\": {task_id: (start_time, end_time)}, \"makespan\": max_end_time}`.\n\n"
            "BUGGY CODE:\n"
            "```python\n"
            "import heapq\n\n"
            "def schedule_dag(tasks: dict[str, dict], workers: int) -> dict:\n"
            "    if workers <= 0:\n"
            "        raise ValueError(\"workers must be positive\")\n"
            "    indeg = {k: len(v.get(\"deps\", [])) for k, v in tasks.items()}\n"
            "    children = {k: [] for k in tasks}\n"
            "    for k, v in tasks.items():\n"
            "        for d in v.get(\"deps\", []):\n"
            "            children[d].append(k)\n"
            "    ready = [(tasks[k].get(\"priority\", 0), k) for k, deg in indeg.items() if deg == 0]\n"
            "    heapq.heapify(ready)\n"
            "    running = []\n"
            "    schedule = {}\n"
            "    t = 0\n"
            "    while ready or running:\n"
            "        while ready and len(running) < workers:\n"
            "            _, tid = heapq.heappop(ready)\n"
            "            dur = tasks[tid][\"duration\"]\n"
            "            schedule[tid] = (t, t + dur)\n"
            "            heapq.heappush(running, (t + dur, tid))\n"
            "        if not running:\n"
            "            break\n"
            "        t = running[0][0]\n"
            "        while running and running[0][0] == t:\n"
            "            _, done_id = heapq.heappop(running)\n"
            "            for ch in children[done_id]:\n"
            "                indeg[ch] -= 1\n"
            "                if indeg[ch] == 0:\n"
            "                    heapq.heappush(ready, (tasks[ch].get(\"priority\", 0), ch))\n"
            "    return {\"schedule\": schedule, \"makespan\": t}\n"
            "```\n"
            "Output ONLY the complete fixed Python code defining `schedule_dag`."
        ),
        hidden_tests='''
def test_validation_and_disconnected_cycle():
    assert schedule_dag({}, 2) == {"schedule": {}, "makespan": 0}
    # Disconnected cycle alongside valid root!
    bad_tasks = {
        "ok": {"duration": 2, "deps": []},
        "c1": {"duration": 1, "deps": ["c2"]},
        "c2": {"duration": 1, "deps": ["c1"]},
    }
    for bad, w in [
        (bad_tasks, 2),
        ({"a": {"duration": 1, "deps": ["missing"]}}, 1),
        ({"a": {"duration": -1, "deps": []}}, 1),
        ({"a": {"duration": 1, "deps": ["a"]}}, 1),
        ({"a": {"duration": 1, "deps": []}}, 0),
    ]:
        try:
            r = schedule_dag(bad, w)
            raise AssertionError(f"Expected ValueError, got {r}")
        except ValueError:
            pass

def test_priority_order_and_work_conserving_exclusive_group():
    tasks = {
        "a": {"duration": 5, "deps": [], "priority": 10, "exclusive_group": "gpu"},
        "b": {"duration": 3, "deps": [], "priority": 8,  "exclusive_group": "gpu"},  # blocked by 'a' at t=0!
        "c": {"duration": 4, "deps": [], "priority": 1,  "exclusive_group": None},   # MUST run at t=0 on worker 2!
        "d": {"duration": 2, "deps": ["a"], "priority": 10, "exclusive_group": "gpu"},
    }
    res = schedule_dag(tasks, workers=2)
    # t=0: 'a' (prio 10, gpu) starts [0,5); 'b' blocked on gpu -> 'c' starts [0,4)!
    # t=4: 'c' finishes; 'b' still blocked by 'a' until t=5!
    # t=5: 'a' finishes; ready = {'d' prio 10, 'b' prio 8} -> 'd' beats 'b' -> 'd' [5,7), then 'b' [7,10)!
    assert res["schedule"] == {"a": (0, 5), "c": (0, 4), "d": (5, 7), "b": (7, 10)}
    assert res["makespan"] == 10

def test_zero_duration_chain_and_lex_tiebreak():
    tasks = {
        "z0": {"duration": 0, "deps": [], "priority": 5},
        "a_child": {"duration": 3, "deps": ["z0"], "priority": 5},
        "b_sib": {"duration": 3, "deps": [], "priority": 5},
    }
    # At t=0 on 1 worker: b_sib vs z0 (same prio 5 -> 'b_sib' < 'z0' lex!).
    # Wait: test z0 having higher prio so it runs at t=0 and IMMEDIATELY unlocks a_child at t=0 without burning a worker slot!
    tasks2 = {
        "z0": {"duration": 0, "deps": [], "priority": 10},
        "a_child": {"duration": 3, "deps": ["z0"], "priority": 5},
        "b_sib": {"duration": 3, "deps": [], "priority": 5},
    }
    res = schedule_dag(tasks2, workers=1)
    # z0 runs [0,0], frees worker immediately at t=0, unlocks a_child (prio 5) which beats b_sib (prio 5) lexicographically!
    assert res["schedule"] == {"z0": (0, 0), "a_child": (0, 3), "b_sib": (3, 6)}
    assert res["makespan"] == 6

def test_duplicate_deps_in_list():
    tasks = {
        "a": {"duration": 2, "deps": []},
        "b": {"duration": 3, "deps": ["a", "a"]},  # duplicate dep must not stall indeg!
    }
    res = schedule_dag(tasks, workers=1)
    assert res["schedule"] == {"a": (0, 2), "b": (2, 5)}
    assert res["makespan"] == 5
''',
    ),
]
