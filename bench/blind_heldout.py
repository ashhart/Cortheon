"""20-Test Frozen Blind Held-Out External Benchmark (10 Official AIME 2024 + 10 Unseen Hard Systems/Algo Coding).

STRICT REFEREE RULE:
  - Engine (`flash_forge/`) MUST be 100% FROZEN before running this suite.
  - Pure cold-start PASS-1 ONLY (`enable_memory=False`, zero post-run tuning!).
  - Supports running on ANY OpenAI/TensorFold model endpoint (`--url`) for cross-model proof!
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.code_forge import CodeForgeTrace
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.reason_forge import ReasonForgeTrace, extract_answer_from_prose
from flash_forge.sandbox import extract_python_code

from .run_ab import grade_code, grade_reason


@dataclass(frozen=True)
class HeldOutProblem:
    pid: str
    track: Literal["aime2024", "unseen_code"]
    forge_mode: Literal["reason", "code"]
    title: str
    prompt: str
    expected: str = ""
    hidden_tests: str = ""


HELDOUT_20: list[HeldOutProblem] = [
    # =========================================================================
    # TRACK A: 10 OFFICIAL AIME 2024 (I & II) COMPETITION MATH PROBLEMS (B01-B10)
    # =========================================================================
    HeldOutProblem(
        pid="B01_aime24_I_01",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 I #1 (Aya Walk & Coffee Shop)",
        prompt=(
            "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. "
            "When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, "
            "including t minutes spent in the coffee shop. When she walks s + 2 kilometers per hour, "
            "the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. "
            "Suppose Aya walks at s + 1/2 kilometers per hour. Find the number of minutes the walk takes her, "
            "including the t minutes spent in the coffee shop. End with `ANSWER: <integer>`."
        ),
        expected="204",
    ),
    HeldOutProblem(
        pid="B02_aime24_I_03",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 I #3 (Alice & Bob 1-or-4 Token Game)",
        prompt=(
            "Alice and Bob play the following game. A stack of n tokens lies before them. "
            "The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens "
            "from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or "
            "equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game "
            "regardless of Alice's play. End with `ANSWER: <integer>`."
        ),
        expected="809",
    ),
    HeldOutProblem(
        pid="B03_aime24_I_04",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 I #4 (Lottery Conditional Grand Prize)",
        prompt=(
            "Jen enters a lottery by picking 4 distinct numbers from S = {1, 2, 3, ..., 10}. "
            "4 numbers are randomly chosen from S. She wins a prize if at least two of her numbers were among the 4 "
            "randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. "
            "The probability of her winning the grand prize given that she won a prize is m/n, where m and n are "
            "relatively prime positive integers. Find m + n. End with `ANSWER: <integer>`."
        ),
        expected="116",
    ),
    HeldOutProblem(
        pid="B04_aime24_I_08",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 I #8 (Red/Blue Regular Octagon Rotation)",
        prompt=(
            "Each vertex of a regular octagon is independently colored either red or blue with equal probability 1/2. "
            "The probability that the octagon can then be rotated so that all of the blue vertices end up at positions "
            "where there were originally red vertices is m/n, where m and n are relatively prime positive integers. "
            "What is m + n? End with `ANSWER: <integer>`."
        ),
        expected="371",
    ),
    HeldOutProblem(
        pid="B05_aime24_I_13",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 I #13 (Least Prime p with p^2 | n^4 + 1)",
        prompt=(
            "Let p be the least prime number for which there exists a positive integer n such that n^4 + 1 is "
            "divisible by p^2. Find the least positive integer m such that m^4 + 1 is divisible by p^2. "
            "End with `ANSWER: <integer>`."
        ),
        expected="110",
    ),
    HeldOutProblem(
        pid="B06_aime24_II_02",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 II #2 (Symmetric Cubic Triples a+b+c=300)",
        prompt=(
            "Find the number of ordered triples of nonnegative integers (a, b, c) satisfying "
            "a + b + c = 300 and a^2*b + a^2*c + b^2*a + b^2*c + c^2*a + c^2*b = 6,000,000. "
            "End with `ANSWER: <integer>`."
        ),
        expected="601",
    ),
    HeldOutProblem(
        pid="B07_aime24_II_06",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 II #6 (2024 Subsets Max Element Set A)",
        prompt=(
            "Alice chooses a set A of positive integers. Then Bob lists all finite nonempty sets B of positive integers "
            "with the property that the maximum element of B belongs to A. Bob's list has 2024 sets. "
            "Find the sum of the elements of A. End with `ANSWER: <integer>`."
        ),
        expected="55",
    ),
    HeldOutProblem(
        pid="B08_aime24_II_07",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 II #7 (4-Digit Number Changing Any Digit to 1 Divisible by 7)",
        prompt=(
            "Let N be the greatest four-digit positive integer with the property that whenever one of its digits "
            "is changed to 1, the resulting number is divisible by 7. Let Q and R be the quotient and remainder, "
            "respectively, when N is divided by 1000. Find Q + R. End with `ANSWER: <integer>`."
        ),
        expected="699",
    ),
    HeldOutProblem(
        pid="B09_aime24_II_14",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 II #14 (Least Base b with >10 b-eautiful Integers)",
        prompt=(
            "Call a positive integer n 'b-eautiful' if it has exactly two digits when expressed in base b >= 2 "
            "(so b <= n < b^2) and these two digits sum to sqrt(n). For example, 81 is 13-eautiful because "
            "81 = 6*13 + 3 and 6 + 3 = 9 = sqrt(81). Find the least integer b >= 2 for which there are "
            "more than ten b-eautiful integers. End with `ANSWER: <integer>`."
        ),
        expected="211",
    ),
    HeldOutProblem(
        pid="B10_aime24_I_12",
        track="aime2024",
        forge_mode="reason",
        title="AIME 2024 I #12 (Max Real Part on Circle |z| = 4)",
        prompt=(
            "Find the largest possible real part of (75 + 117i)*z + (96 + 144i)/z "
            "where z is a complex number with |z| = 4. End with `ANSWER: <integer>`."
        ),
        expected="540",
    ),

    # =========================================================================
    # TRACK B: 10 UNSEEN HARD ALGORITHMIC, CRYPTO & SYSTEMS CODING TASKS (B11-B20)
    # =========================================================================
    HeldOutProblem(
        pid="B11_fenwick_binary_lifting",
        track="unseen_code",
        forge_mode="code",
        title="O(log N) Fenwick Tree with Binary-Lifting Lower-Bound Search",
        prompt=(
            "Implement class `OrderFenwick(n: int)` managing non-negative integer counts on indices `0..n-1` (`n >= 1`, else `ValueError`):\n"
            "- `add(idx: int, delta: int) -> None`: adds `delta` to index `idx` (raise `ValueError` if resulting count at `idx` would be `< 0`, or `IndexError` if out of range).\n"
            "- `prefix_sum(idx: int) -> int`: sum of counts on `0..idx` inclusive (`-1` returns `0`).\n"
            "- `range_sum(l: int, r: int) -> int`: sum on half-open `[l, r)` (`0 <= l <= r <= n`).\n"
            "- `find_by_order(k: int) -> int`: returns the SMALLEST index `i` such that `prefix_sum(i) > k` (0-indexed `k`-th item in multiset!) in strict `O(log N)` using Fenwick binary lifting! Raise `IndexError` if `k < 0` or `k >= total_count`.\n"
            "- Must handle `N = 10000` and `20000` ops fast (`< 0.8s`).\n"
            "Output ONLY Python code defining `OrderFenwick`."
        ),
        hidden_tests=r'''
import random, time

def test_fenwick_fuzz_and_perf():
    rng = random.Random(11)
    n = 32
    ft = OrderFenwick(n)
    arr = [0] * n
    for _ in range(300):
        idx = rng.randrange(n)
        d = rng.randint(-3, 5)
        if arr[idx] + d < 0:
            try:
                ft.add(idx, d); raise AssertionError("Expected ValueError for negative count")
            except ValueError:
                pass
        else:
            ft.add(idx, d); arr[idx] += d
        tot = sum(arr)
        if tot > 0:
            k = rng.randrange(tot)
            s = 0; exp = -1
            for i, c in enumerate(arr):
                s += c
                if s > k: exp = i; break
            assert ft.find_by_order(k) == exp
        l = rng.randrange(n); r = rng.randint(l, n)
        assert ft.range_sum(l, r) == sum(arr[l:r])
    big = OrderFenwick(10000)
    t0 = time.perf_counter()
    for i in range(10000): big.add(i, 1)
    for i in range(10000): assert big.find_by_order(i) == i
    assert (time.perf_counter() - t0) < 0.8
''',
    ),
    HeldOutProblem(
        pid="B12_iterative_2sat_solver",
        track="unseen_code",
        forge_mode="code",
        title="Iterative O(N+M) 2-SAT Solver via Tarjan SCC (N=5000)",
        prompt=(
            "Implement `solve_2sat(num_vars: int, clauses: list[tuple[int, int]]) -> dict[int, bool] | None`.\n"
            "- Variables are `1..num_vars`. Each clause `(a, b)` means `(lit(a) OR lit(b))` where positive `v` is `x_v` and negative `-v` is `NOT x_v`.\n"
            "- Return a complete assignment `{1: bool, ..., num_vars: bool}` satisfying ALL clauses, or `None` if UNSAT.\n"
            "- MUST be ITERATIVE (no `RecursionError` on a 5,000-variable implication chain!) and run in `O(V + E)` (`< 0.8s` for `N=5000`).\n"
            "Output ONLY Python code defining `solve_2sat`."
        ),
        hidden_tests=r'''
import random, time

def test_2sat_unsat_fuzz_and_deep_chain():
    assert solve_2sat(0, []) == {}
    # Unsat: (x1 OR x2), (-x1 OR x2), (x1 OR -x2), (-x1 OR -x2)
    assert solve_2sat(2, [(1, 2), (-1, 2), (1, -2), (-1, -2)]) is None
    # Deep 5000-var chain: x1 -> x2 -> ... -> x5000 and x1 forced True
    n = 5000
    clauses = [(1, 1)] + [(-i, i + 1) for i in range(1, n)]
    t0 = time.perf_counter()
    sol = solve_2sat(n, clauses)
    assert sol is not None and all(sol[i] is True for i in range(1, n + 1))
    assert (time.perf_counter() - t0) < 0.8
    # Random planted 2-SAT
    rng = random.Random(12)
    truth = {i: rng.choice([True, False]) for i in range(1, 201)}
    cls = []
    for _ in range(500):
        u, v = rng.sample(range(1, 201), 2)
        lu = u if rng.random() < 0.5 else -u
        lv = v if rng.random() < 0.5 else -v
        if not ((lu > 0 and truth[lu]) or (lu < 0 and not truth[-lu]) or (lv > 0 and truth[lv]) or (lv < 0 and not truth[-lv])):
            lu = -lu
        cls.append((lu, lv))
    m = solve_2sat(200, cls)
    assert m is not None and len(m) == 200
    for a, b in cls:
        va = m[a] if a > 0 else (not m[-a])
        vb = m[b] if b > 0 else (not m[-b])
        assert va or vb
''',
    ),
    HeldOutProblem(
        pid="B13_iterative_tree_lca_dist",
        track="unseen_code",
        forge_mode="code",
        title="Iterative O(log N) Binary-Lifting / O(1) Euler-Tour Tree LCA & Distance (N=5000)",
        prompt=(
            "Implement class `TreeLCA(n: int, edges: list[tuple[int, int]], root: int = 0)` on a tree of vertices `0..n-1`:\n"
            "- `lca(u: int, v: int) -> int`: exact lowest common ancestor of `u` and `v`.\n"
            "- `dist(u: int, v: int) -> int`: exact edge distance between `u` and `v`.\n"
            "- `kth_ancestor(u: int, k: int) -> int | None`: `k`-th ancestor of `u` (`k == 0` -> `u`; returns `None` if `k > depth(u)`; raise `ValueError` if `k < 0`).\n"
            "- MUST be ITERATIVE (zero `RecursionError` on a 5,000-node path!) and handle `N=5000` build + 5,000 queries in `< 0.8s`.\n"
            "Output ONLY Python code defining `TreeLCA`."
        ),
        hidden_tests=r'''
import random, time

def test_lca_fuzz_and_deep_path():
    # Deep 5000-node path rooted at 0
    n = 5000
    edges = [(i, i + 1) for i in range(n - 1)]
    t0 = time.perf_counter()
    tl = TreeLCA(n, edges, root=0)
    assert tl.lca(4999, 2500) == 2500
    assert tl.dist(100, 4900) == 4800
    assert tl.kth_ancestor(4999, 0) == 4999
    assert tl.kth_ancestor(4999, 4999) == 0
    assert tl.kth_ancestor(4999, 5000) is None
    for i in range(5000):
        assert tl.lca(i, n - 1) == i
    assert (time.perf_counter() - t0) < 0.8

    # Random 80-node tree vs naive parent walk
    rng = random.Random(13)
    parent = {0: -1}; es = []
    for v in range(1, 80):
        p = rng.randrange(v); parent[v] = p; es.append((p, v))
    t2 = TreeLCA(80, es, root=0)
    def anc_list(u):
        out = []
        while u != -1: out.append(u); u = parent[u]
        return out
    for _ in range(200):
        u, v = rng.randrange(80), rng.randrange(80)
        au, av = anc_list(u), anc_list(v)
        set_v = set(av)
        exp_lca = next(x for x in au if x in set_v)
        assert t2.lca(u, v) == exp_lca
        assert t2.dist(u, v) == au.index(exp_lca) + av.index(exp_lca)
''',
    ),
    HeldOutProblem(
        pid="B14_gfp_poly_div_gcd",
        track="unseen_code",
        forge_mode="code",
        title="Exact Polynomial Division & Monic GCD over Finite Field GF(p)",
        prompt=(
            "Implement exact polynomial arithmetic over finite prime field `GF(p)` (polynomials represented as ascending coefficient lists `[c0, c1, ..., cd]` where `c_i` is coeff of `x^i`, canonicalized with NO trailing zeros — zero polynomial is `[]`!):\n"
            "- `poly_divmod_mod_p(a: list[int], b: list[int], p: int) -> tuple[list[int], list[int]]`: returns canonical `(q, r)` over `GF(p)` with `a ≡ q*b + r (mod p)` and `len(r) < len(b)` (raise `ZeroDivisionError` if `b` is zero mod `p`).\n"
            "- `poly_gcd_mod_p(a: list[int], b: list[int], p: int) -> list[int]`: returns the unique **MONIC** (leading coefficient == `1`) GCD polynomial over `GF(p)` (or `[]` if both `a` and `b` are zero mod `p`).\n"
            "- All returned coefficients MUST be canonical ints in `[0, p - 1]` with NO trailing zeros. Handle arbitrary prime `p` (including large primes like `10**9 + 7`).\n"
            "Output ONLY Python code defining `poly_divmod_mod_p` and `poly_gcd_mod_p`."
        ),
        hidden_tests=r'''
def _mul(a, b, p):
    if not a or not b: return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    while out and out[-1] == 0: out.pop()
    return out

def _add(a, b, p):
    n = max(len(a), len(b))
    out = [((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)]
    while out and out[-1] == 0: out.pop()
    return out

def test_gfp_divmod_and_monic_gcd():
    p = 1000000007
    # Build a = g * u, b = g * v where g = x^2 + 3x + 5 (monic!), u = 7x + 2, v = 11x + 9 (coprime!)
    g = [5, 3, 1]
    a = _mul(g, [2, 7], p)
    b = _mul(g, [9, 11], p)
    assert poly_gcd_mod_p(a, b, p) == g
    assert poly_gcd_mod_p([], [], p) == []
    assert poly_gcd_mod_p([0, 0, 5], [], 7) == [0, 0, 1]  # monic!
    q, r = poly_divmod_mod_p(_add(_mul([3, 1, 2], [4, 5], 13), [7]), [4, 5], 13)
    assert _add(_mul(q, [4, 5], 13), r, 13) == _add(_mul([3, 1, 2], [4, 5], 13), [7])
    assert r == [7]
    try:
        poly_divmod_mod_p([1, 2], [0, 13], 13); raise AssertionError("Expected ZeroDivisionError")
    except ZeroDivisionError:
        pass
''',
    ),
    HeldOutProblem(
        pid="B15_orset_crdt_merge",
        track="unseen_code",
        forge_mode="code",
        title="Add-Wins Observed-Remove Set (OR-Set) CRDT with Commutative/Idempotent Merge",
        prompt=(
            "Implement an **Add-Wins Observed-Remove Set (OR-Set) CRDT** class `ORSet(replica_id: str)`:\n"
            "- `add(elem: str) -> None`: generates a globally unique tag `(self.replica_id, counter)` and records it for `elem`.\n"
            "- `remove(elem: str) -> None`: observes and tombstones ONLY the currently visible tags of `elem` on this replica (does NOT suppress a concurrent `add(elem)` with a fresh unseen tag on another replica!).\n"
            "- `elements() -> set[str]`: returns the set of all currently present elements (those having at least one live tag not in tombstones).\n"
            "- `merge(other: ORSet) -> None`: merges `other`'s state into `self` (union of add-tags, union of tombstones) without mutating `other`! Must satisfy full CRDT semilattice laws:\n"
            "  * **Commutativity:** `A.merge(B)` == `B.merge(A)`\n"
            "  * **Associativity:** `(A ⊔ B) ⊔ C == A ⊔ (B ⊔ C)`\n"
            "  * **Idempotence:** `A.merge(A)` leaves `A` unchanged\n"
            "  * **Add-Wins under Concurrency:** If Replica 1 and Replica 2 both have `'x'`, then R1 calls `remove('x')` while concurrently R2 calls `add('x')` (new tag!), merging R1 and R2 MUST keep `'x'` present! And re-adding after remove works cleanly!\n"
            "- `clone(new_replica_id: str) -> ORSet`: deep-copies current state for a forked replica with a fresh `new_replica_id`.\n"
            "Output ONLY Python code defining `ORSet`."
        ),
        hidden_tests=r'''
def test_orset_add_wins_and_semilattice_laws():
    r1 = ORSet("r1")
    r1.add("x"); r1.add("y")
    r2 = r1.clone("r2")
    r3 = r1.clone("r3")

    # Concurrent remove on r1 vs add on r2 -> ADD WINS for 'x'!
    r1.remove("x")
    r2.add("x")
    # Only r1 removes 'y' (no concurrent add) -> 'y' stays removed!
    r1.remove("y")
    # r3 adds 'z'
    r3.add("z")

    # Commutativity & Associativity check
    m_123 = r1.clone("m1"); m_123.merge(r2); m_123.merge(r3)
    m_321 = r3.clone("m2"); m_321.merge(r2); m_321.merge(r1)
    assert m_123.elements() == {"x", "z"}
    assert m_321.elements() == {"x", "z"}
    # Idempotence
    m_123.merge(r1); m_123.merge(r2); m_123.merge(m_123)
    assert m_123.elements() == {"x", "z"}
    # Re-remove after merge removes BOTH tags cleanly
    m_123.remove("x")
    assert m_123.elements() == {"z"}
    m_123.add("x")
    assert m_123.elements() == {"x", "z"}
''',
    ),
    HeldOutProblem(
        pid="B16_dinic_max_flow",
        track="unseen_code",
        forge_mode="code",
        title="Exact Dinic Max-Flow (Parallel/Anti-Parallel Edges, Large Capacities, N=100)",
        prompt=(
            "Implement `max_flow(n: int, edges: list[tuple[int, int, int]], s: int, t: int) -> int`.\n"
            "- Directed network on nodes `0..n-1` with `(u, v, cap)` (`cap >= 0`; parallel edges AND anti-parallel `(u, v)` + `(v, u)` edges allowed; ignore self-loops `u == v`; if `s == t` return `0`).\n"
            "- Handle huge integer capacities up to `10^12` (so naive unit-augmenting Ford-Fulkerson would time out — use Dinic's algorithm with level graphs + current-arc pointer!).\n"
            "- Handle `N = 100`, `550+` edges in `< 0.4s`.\n"
            "Output ONLY Python code defining `max_flow`."
        ),
        hidden_tests=r'''
import random, time

def test_dinic_anti_parallel_huge_cap_and_scale():
    assert max_flow(2, [(0, 1, 10)], 0, 0) == 0
    # Classic Ford-Fulkerson killer with capacity 10^9 + anti-parallel + parallel edges!
    M = 10**9
    edges = [(0, 1, M), (0, 2, M), (1, 2, 1), (2, 1, 1), (1, 3, M), (2, 3, M), (0, 1, 5)]
    t0 = time.perf_counter()
    assert max_flow(4, edges, 0, 3) == 2 * M
    assert (time.perf_counter() - t0) < 0.2

    # Bipartite max matching check: 50x50 complete-ish layered graph
    rng = random.Random(16)
    es = []
    S, T = 0, 101
    for i in range(1, 51): es.append((S, i, 1))
    for j in range(51, 101): es.append((j, T, 1))
    # Guarantee a perfect matching + random extra edges
    perm = list(range(51, 101)); rng.shuffle(perm)
    for i in range(1, 51): es.append((i, perm[i - 1], 1))
    for _ in range(400): es.append((rng.randint(1, 50), rng.randint(51, 100), 1))
    t1 = time.perf_counter()
    assert max_flow(102, es, S, T) == 50
    assert (time.perf_counter() - t1) < 0.4
''',
    ),
    HeldOutProblem(
        pid="B17_jsonc_comment_trailing_comma_parser",
        track="unseen_code",
        forge_mode="code",
        title="Strict JSONC Parser (// and /* */ Comments, String Escapes, Trailing Commas)",
        prompt=(
            "Implement `parse_jsonc(text: str)` supporting JSON with Comments (`// line`, `/* block */`) and trailing commas in arrays/objects:\n"
            "- MUST NOT corrupt `//`, `/*`, `*/`, or `,` appearing INSIDE JSON string literals (including escaped quotes `\\\"` and escaped backslashes `\\\\\"` right before the closing quote!).\n"
            "- Allow single trailing comma before `]` or `}` (even separated by comments/whitespace: `[1, 2, /* hi */ ]`), but REJECT double commas `[1, , 2]` or leading commas `[, 1]` with `ValueError`!\n"
            "- Reject unclosed block comments `/* ...` and invalid JSON syntax with `ValueError`.\n"
            "- Return standard Python `json.loads`-equivalent objects (`dict`, `list`, `str`, `int`, `float`, `bool`, `None`).\n"
            "Output ONLY Python code defining `parse_jsonc`."
        ),
        hidden_tests=r'''
def test_jsonc_strings_escapes_comments_and_errors():
    src = r"""
    // leading line comment
    {
      "url": "https://example.com/a/*not_comment*/?x=1", // inline comment
      "escaped": "end with backslash \\", /* block
         comment spanning lines */
      "quote": "say \"// hi\", ok",
      "arr": [1, 2, /* c */ ],
      "nested": {"k": true, "n": null,},
    }
    """
    got = parse_jsonc(src)
    assert got == {
        "url": "https://example.com/a/*not_comment*/?x=1",
        "escaped": "end with backslash \\",
        "quote": 'say "// hi", ok',
        "arr": [1, 2],
        "nested": {"k": True, "n": None},
    }
    for bad in ['/* unclosed', '[1, , 2]', '[, 1]', '{"a": 1,,}', '{1: 2}']:
        try:
            parse_jsonc(bad); raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
''',
    ),
    HeldOutProblem(
        pid="B18_shamir_secret_sharing_m127",
        track="unseen_code",
        forge_mode="code",
        title="Exact Shamir Secret Sharing over Mersenne Prime Field GF(2^127 - 1)",
        prompt=(
            "Implement exact cryptographic Shamir's Secret Sharing over prime `PRIME = (1 << 127) - 1`:\n"
            "- `split_secret(secret: int, n: int, k: int, seed: int | None = None) -> list[tuple[int, int]]`:\n"
            "  * Validate `0 <= secret < PRIME`, `2 <= k <= n`, else raise `ValueError`.\n"
            "  * Construct a degree `k - 1` polynomial `f(x) = secret + a_1 x + ... + a_{k-1} x^{k-1} (mod PRIME)` "
            "(using `random.Random(seed)` if `seed is not None`, else `secrets.randbelow(PRIME)`).\n"
            "  * Return `[(i, f(i) % PRIME) for i in range(1, n + 1)]`.\n"
            "- `reconstruct_secret(shares: list[tuple[int, int]]) -> int`:\n"
            "  * Exact Lagrange interpolation at `x = 0` modulo `PRIME` (ZERO floats — use exact modular inverse `pow(den, -1, PRIME)`!).\n"
            "  * Validate `len(shares) >= 2`, all `x` coordinates distinct and `1 <= x < PRIME` (else raise `ValueError`).\n"
            "Output ONLY Python code defining `PRIME`, `split_secret`, and `reconstruct_secret`."
        ),
        hidden_tests=r'''
from itertools import combinations

def test_shamir_exact_127bit_lagrange_and_validation():
    P = (1 << 127) - 1
    secret = (1 << 126) + 98765432109876543210987654321
    shares = split_secret(secret, n=6, k=4, seed=42)
    assert len(shares) == 6
    # EVERY 4-subset, 5-subset, and full 6-subset MUST reconstruct exact 127-bit secret!
    for sub in list(combinations(shares, 4)) + list(combinations(shares, 5)) + [tuple(shares)]:
        assert reconstruct_secret(list(sub)) == secret
    # 3 shares (< k=4) must NOT reconstruct secret (for deterministic non-degenerate polynomial)
    assert reconstruct_secret(shares[:3]) != secret
    for fn in [
        lambda: split_secret(-1, 5, 3),
        lambda: split_secret(P, 5, 3),
        lambda: split_secret(5, 3, 4),
        lambda: reconstruct_secret([(1, 10), (1, 20)]),
    ]:
        try:
            fn(); raise AssertionError("Expected ValueError")
        except ValueError:
            pass
''',
    ),
    HeldOutProblem(
        pid="B19_optimistic_mini_stm",
        track="unseen_code",
        forge_mode="code",
        title="Optimistic Software Transactional Memory (Read-Set Validation + Serializable Commit)",
        prompt=(
            "Implement class `MiniSTM` providing **Serializable Optimistic Concurrency Control (OCC)** over integer-versioned TVars:\n"
            "- `set_initial(key: str, value) -> None`: sets initial committed state before transactions.\n"
            "- `begin() -> int`: starts a transaction `tx_id` at current global version.\n"
            "- `read(tx: int, key: str)`: returns local uncommitted write if `tx` wrote `key`, else records `(key, observed_version)` in `tx`'s **read-set** and returns the committed value at `tx`'s snapshot (or `None` if absent). (Raises `KeyError` if `tx` is not active.)\n"
            "- `write(tx: int, key: str, value) -> None`: buffers write locally in `tx` (raises `KeyError` if `tx` inactive).\n"
            "- `commit(tx: int) -> None`:\n"
            "  * **Serializable Read-Set Validation (prevents Write Skew!!):** For EVERY key in `tx`'s read-set AND write-set, verify its current committed version `<= tx.start_version` (i.e. NOT modified by ANY concurrent transaction that committed after `tx.begin()`!).\n"
            "  * If ANY read-set OR write-set key changed concurrently, ABORT `tx` cleanly (discarding all writes) and raise `RuntimeError`!\n"
            "  * Otherwise increment global version and apply all writes atomically.\n"
            "- `abort(tx: int) -> None`: aborts `tx` cleanly.\n"
            "Output ONLY Python code defining `MiniSTM`."
        ),
        hidden_tests=r'''
def test_stm_blocks_write_skew_and_allows_disjoint():
    stm = MiniSTM()
    stm.set_initial("doc_a", True)
    stm.set_initial("doc_b", True)

    # Classic Write-Skew scenario: both t1 and t2 read BOTH doc_a & doc_b, then t1 writes doc_a=False, t2 writes doc_b=False!
    t1 = stm.begin(); t2 = stm.begin()
    assert stm.read(t1, "doc_a") is True and stm.read(t1, "doc_b") is True
    assert stm.read(t2, "doc_a") is True and stm.read(t2, "doc_b") is True
    stm.write(t1, "doc_a", False)
    stm.write(t2, "doc_b", False)
    stm.commit(t1)
    # t2 read doc_a which t1 modified -> Serializable OCC MUST reject t2!
    try:
        stm.commit(t2)
        raise AssertionError("Write-skew slipped past STM read-set validation!")
    except RuntimeError:
        pass
    t3 = stm.begin()
    assert stm.read(t3, "doc_a") is False and stm.read(t3, "doc_b") is True

    # Truly disjoint transactions both commit cleanly!
    t4 = stm.begin()
    stm.write(t3, "x", 1); stm.write(t4, "y", 2)
    stm.commit(t3); stm.commit(t4)
    t5 = stm.begin()
    assert stm.read(t5, "x") == 1 and stm.read(t5, "y") == 2
''',
    ),
    HeldOutProblem(
        pid="B20_ipv4_cidr_lpm_router",
        track="unseen_code",
        forge_mode="code",
        title="O(32) Binary-Trie IPv4 CIDR Longest-Prefix-Match Router Table",
        prompt=(
            "Implement class `CIDRRouter` using a 32-bit binary radix/prefix trie for exact IPv4 Longest-Prefix Match (LPM):\n"
            "- `add_route(cidr: str, next_hop: str) -> None`: adds or updates route for IPv4 CIDR `'A.B.C.D/P'` (`0 <= P <= 32`; masks host bits cleanly so `'10.1.2.3/8'` is equivalent to `'10.0.0.0/8'`; raise `ValueError` on invalid IPv4/prefix).\n"
            "- `remove_route(cidr: str) -> bool`: removes exact prefix route and returns `True`, or `False` if absent (do NOT delete child/longer prefixes beneath it!).\n"
            "- `lookup(ip: str) -> str | None`: returns `next_hop` of the LONGEST matching prefix (`0..32`, `/32` beats `/24` beats `/8` beats `/0` default!), or `None` if no route matches. Raise `ValueError` on invalid IPv4.\n"
            "- Performance: insert 2,000 prefixes + run 10,000 lookups in `< 0.8s` (O(32) trie — NOT O(R) linear scan!).\n"
            "Output ONLY Python code defining `CIDRRouter`."
        ),
        hidden_tests=r'''
import time

def test_cidr_lpm_hostbits_delete_and_perf():
    rt = CIDRRouter()
    assert rt.lookup("8.8.8.8") is None
    rt.add_route("0.0.0.0/0", "default_gw")
    rt.add_route("10.1.2.3/8", "corp_net")     # host bits masked -> 10.0.0.0/8!
    rt.add_route("10.20.0.0/16", "eng_vlan")
    rt.add_route("10.20.30.40/32", "host_box")

    assert rt.lookup("8.8.8.8") == "default_gw"
    assert rt.lookup("10.99.1.1") == "corp_net"
    assert rt.lookup("10.20.5.5") == "eng_vlan"
    assert rt.lookup("10.20.30.40") == "host_box"
    # Removing /16 must NOT remove /32 child or /8 parent!
    assert rt.remove_route("10.20.0.0/16") is True
    assert rt.remove_route("10.20.0.0/16") is False
    assert rt.lookup("10.20.5.5") == "corp_net"
    assert rt.lookup("10.20.30.40") == "host_box"

    # Scale: 2000 routes + 10000 lookups MUST be O(32) fast!
    t0 = time.perf_counter()
    big = CIDRRouter()
    big.add_route("0.0.0.0/0", "gw")
    for i in range(2000):
        big.add_route(f"10.{(i >> 8) & 255}.{i & 255}.0/24", f"r{i}")
    for i in range(10000):
        idx = i % 2000
        assert big.lookup(f"10.{(idx >> 8) & 255}.{idx & 255}.42") == f"r{idx}"
    assert (time.perf_counter() - t0) < 0.8
    for bad in ["999.1.1.1/24", "10.0.0.0/33", "10.0.0.0/-1", "not_an_ip"]:
        try:
            rt.add_route(bad, "x"); raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
''',
    ),
]


def verify_suite() -> None:
    assert len(HELDOUT_20) == 20
    assert len({p.pid for p in HELDOUT_20}) == 20
    for p in HELDOUT_20:
        if p.forge_mode == "reason":
            assert p.expected.strip(), f"{p.pid} missing expected"
        else:
            tree = ast.parse(p.hidden_tests)
            assert any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in tree.body)
    print("All 20 Blind Held-Out problems (10 AIME 2024 + 10 Unseen Hard Code) VERIFIED CLEAN!")


async def run_one_vanilla(client: TensorFoldClient, prob: HeldOutProblem) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": prob.prompt}],
            tag=f"heldout_vanilla_{prob.pid}",
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=42,
            enable_thinking=True,
            thinking_budget=1500,
            loop_guard=True,
        )
    )
    wall = time.perf_counter() - t0
    if prob.forge_mode == "reason":
        ext = extract_answer_from_prose(res.content)
        if not ext and res.content:
            lines = [ln.strip() for ln in res.content.strip().splitlines() if ln.strip()]
            ext = lines[-1] if lines else ""
        passed, detail = grade_reason(ext, prob.expected)
    else:
        passed, detail = grade_code(extract_python_code(res.content), prob.hidden_tests)
    return {
        "arm": "vanilla_budgeted",
        "pid": prob.pid,
        "track": prob.track,
        "passed": passed,
        "detail": detail,
        "wall_s": round(wall, 2),
    }


async def run_one_forge(base_url: str, prob: HeldOutProblem, streams: int = 4) -> dict[str, Any]:
    cfg = ForgeConfig(base_url=base_url, num_streams=streams, deep=False, enable_memory=False)
    async with FlashForge(cfg) as ff:
        res = await ff.solve(prob.prompt, force_mode=None, learn=False)
    if prob.forge_mode == "reason":
        ans = res.answer if res.mode == "reason" else extract_answer_from_prose(res.answer)
        passed, detail = grade_reason(ans, prob.expected)
        note = res.raw_trace.consensus_type if isinstance(res.raw_trace, ReasonForgeTrace) else f"routed={res.mode}"
    else:
        code = res.code or extract_python_code(res.answer)
        passed, detail = grade_code(code, prob.hidden_tests)
        note = res.raw_trace.winner_name if isinstance(res.raw_trace, CodeForgeTrace) else f"routed={res.mode}"
    return {
        "arm": "flash_forge_v3",
        "pid": prob.pid,
        "track": prob.track,
        "passed": passed,
        "detail": detail,
        "wall_s": round(res.wall_s, 2),
        "note": note,
        "routed_match": res.mode == prob.forge_mode,
        "stats": res.token_stats,
    }


async def run_suite(base_url: str, arm: str, pids: list[str] | None, concurrency: int, out_path: Path) -> None:
    verify_suite()
    selected = [p for p in HELDOUT_20 if not pids or p.pid in pids]
    async with TensorFoldClient(base_url=base_url) as client:
        model_id = client.caps.model_id
        print(f"=== BLIND HELD-OUT EVAL (FROZEN ENGINE) | model={model_id} | arm={arm} | n={len(selected)} ===")
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        rows_by_pid: dict[str, dict[str, Any]] = {}
        done = 0

        async def _w(prob: HeldOutProblem) -> None:
            nonlocal done
            async with sem:
                row = await (run_one_vanilla(client, prob) if arm == "vanilla" else run_one_forge(base_url, prob))
                async with lock:
                    done += 1
                    rows_by_pid[prob.pid] = row
                    mark = "PASS" if row["passed"] else "FAIL"
                    print(f"[{done:02d}/{len(selected):02d}] {mark:4s} | {prob.pid:<36} | {row['wall_s']:5.1f}s | {row['detail'][:70]}")
                    sys.stdout.flush()
                    ordered = [rows_by_pid[p.pid] for p in selected if p.pid in rows_by_pid]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        json.dumps(
                            {
                                "suite": "blind_heldout_20",
                                "model": model_id,
                                "arm": arm,
                                "completed": len(ordered),
                                "passed": sum(1 for r in ordered if r["passed"]),
                                "results": ordered,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

        await asyncio.gather(*[_w(p) for p in selected])
    ordered = [rows_by_pid[p.pid] for p in selected]
    for tr in ("aime2024", "unseen_code"):
        sub = [r for r in ordered if r["track"] == tr]
        if sub:
            p = sum(1 for r in sub if r["passed"])
            print(f"  {tr:<12}: {p}/{len(sub)} ({100*p/len(sub):.1f}%)")
    tot = sum(1 for r in ordered if r["passed"])
    print(f"  OVERALL     : {tot}/{len(ordered)} ({100*tot/max(1,len(ordered)):.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=["vanilla", "forge"], default="forge")
    p.add_argument("--pids", nargs="*", default=None)
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--out", default="bench/results/blind_heldout.json")
    args = p.parse_args()
    if args.verify_only:
        verify_suite()
        return
    conc = args.concurrency or (1 if args.arm == "forge" else 4)
    asyncio.run(run_suite(args.url, args.arm, args.pids, conc, Path(args.out)))


if __name__ == "__main__":
    main()
