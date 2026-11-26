"""50-Test Frontier Gauntlet II (Nightmare Tier) — Part 2: 25 Brutal Algo, Multi-Bug Debug & Security Tests (H21-H40, H46-H50)."""

from __future__ import annotations

from .gauntlet2_part1 import Gauntlet2Problem


PART2_PROBLEMS: list[Gauntlet2Problem] = [
    # =========================================================================
    # TRACK 3: NIGHTMARE ALGORITHMS, COMPILERS, DATA STRUCTURES & DISTRIBUTED (H21-H30)
    # =========================================================================
    Gauntlet2Problem(
        pid="H21_dpll_sat_solver",
        track="algo_nm",
        forge_mode="code",
        title="DPLL SAT Solver with Unit Propagation & Pure Literal Elimination",
        prompt=(
            "Implement `solve_sat(num_vars: int, clauses: list[list[int]]) -> dict[int, bool] | None` (NO external SAT libs like `z3`/`pysat`!).\n"
            "- Variables are `1..num_vars`. Positive int `v` means `x_v = True`; negative `-v` means `x_v = False`.\n"
            "- `clauses` is a CNF list of disjunctions (all clauses must be satisfied).\n"
            "- Edge cases:\n"
            "  * Empty `clauses == []`: SAT (return dict mapping every `1..num_vars` to bool, or `{}` if `num_vars == 0`).\n"
            "  * Any empty clause `[]` in `clauses`: UNSAT (`return None`).\n"
            "  * Tautological clauses (containing both `v` and `-v`) and duplicate literals in a clause must be handled cleanly.\n"
            "  * Must return a COMPLETE assignment dict with keys `1..num_vars` (every value a `bool`) satisfying ALL clauses, or `None` if UNSAT.\n"
            "- Must use unit propagation + backtracking (fast enough to solve 50-variable 3-SAT & 4-pigeon-3-hole UNSAT in < 0.5s!).\n"
            "Output ONLY Python code defining `solve_sat` (do NOT import `z3` or `pysat`)."
        ),
        hidden_tests=r'''
import ast, inspect, random, time

def test_no_z3_or_pysat():
    tree = ast.parse(inspect.getsource(solution))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name not in ("z3", "pysat") for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] not in ("z3", "pysat")

def _verify(num_vars, clauses, model):
    assert isinstance(model, dict) and set(model.keys()) == set(range(1, num_vars + 1))
    assert all(isinstance(v, bool) for v in model.values())
    for cl in clauses:
        assert any((lit > 0 and model[lit]) or (lit < 0 and not model[-lit]) for lit in cl)

def test_edges_pigeonhole_and_planted_50var():
    assert solve_sat(0, []) == {}
    _verify(3, [], solve_sat(3, []))
    assert solve_sat(2, [[1], []]) is None
    _verify(2, [[1, -1], [2, 2]], solve_sat(2, [[1, -1], [2, 2]]))

    # Pigeonhole PHP(4, 3): 4 pigeons in 3 holes -> UNSAT!
    def var(p, h): return p * 3 + h + 1
    php = [[var(p, 0), var(p, 1), var(p, 2)] for p in range(4)]
    for h in range(3):
        for p1 in range(4):
            for p2 in range(p1 + 1, 4):
                php.append([-var(p1, h), -var(p2, h)])
    t0 = time.perf_counter()
    assert solve_sat(12, php) is None

    # Planted 50-variable 3-SAT instance (180 clauses)
    rng = random.Random(21)
    truth = {v: rng.choice([True, False]) for v in range(1, 51)}
    clauses = []
    for _ in range(180):
        vs = rng.sample(range(1, 51), 3)
        cl = [v if rng.random() < 0.5 else -v for v in vs]
        if not any((l > 0 and truth[l]) or (l < 0 and not truth[-l]) for l in cl):
            cl[0] = -cl[0]
        clauses.append(cl)
    m = solve_sat(50, clauses)
    assert m is not None
    _verify(50, clauses, m)
    assert (time.perf_counter() - t0) < 0.8
''',
    ),
    Gauntlet2Problem(
        pid="H22_canonical_huffman_codec",
        track="algo_nm",
        forge_mode="code",
        title="Self-Contained Canonical Huffman Binary Codec with Strict Framing & Integrity",
        prompt=(
            "Implement `huffman_encode(data: bytes) -> bytes` and `huffman_decode(blob: bytes) -> bytes` (NO `zlib`/`gzip`/`bz2`/`lzma`!).\n"
            "Requirements:\n"
            "- Self-contained binary format: `huffman_decode(huffman_encode(data)) == data` for EVERY possible `bytes` input:\n"
            "  * Empty bytes `b''`,\n"
            "  * Single byte `b'A'` and single repeated symbol `b'Z' * 5000` (degenerate 1-symbol Huffman tree!),\n"
            "  * All 256 byte values `bytes(range(256))`, arbitrary random binary data.\n"
            "- Must genuinely compress skewed data: `len(huffman_encode(b'aaaaabbc' * 1000)) < 2500` bytes (8000 bytes -> <2500 bytes!).\n"
            "- Strict error handling: `huffman_decode` MUST raise `ValueError` on truncated/corrupted blobs or trailing junk bytes!\n"
            "Output ONLY Python code defining `huffman_encode` and `huffman_decode` (do NOT import `zlib`/`gzip`/`bz2`/`lzma`)."
        ),
        hidden_tests=r'''
import ast, inspect, random

def test_no_builtin_compressors():
    tree = ast.parse(inspect.getsource(solution))
    banned = {"zlib", "gzip", "bz2", "lzma"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name not in banned for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] not in banned

def test_roundtrip_compression_and_corruption():
    rng = random.Random(22)
    samples = [
        b"", b"A", b"Z" * 5000, b"\x00\xff" * 200, bytes(range(256)),
        b"aaaaabbc" * 1000, bytes(rng.getrandbits(8) for _ in range(1500)),
    ]
    for s in samples:
        enc = huffman_encode(s)
        assert isinstance(enc, bytes)
        assert huffman_decode(enc) == s
    # Compression ratio check
    skewed = b"aaaaabbc" * 1000
    assert len(huffman_encode(skewed)) < 2500
    # Corruption / truncation checks
    enc_ok = huffman_encode(b"hello world! " * 30)
    for bad in [b"", enc_ok[:1], enc_ok[:-3], enc_ok + b"\x00\x01\x02"]:
        try:
            res = huffman_decode(bad)
            if bad == b"" and res == b"":
                raise AssertionError("Empty blob without header should raise ValueError")
            raise AssertionError(f"Expected ValueError for corrupted blob len={len(bad)}")
        except ValueError:
            pass
''',
    ),
    Gauntlet2Problem(
        pid="H23_expr_compiler_and_stack_vm",
        track="algo_nm",
        forge_mode="code",
        title="Zero-Eval Expression Lexer, Parser & Evaluator (Chained Comparisons, Ternary, Short-Circuit)",
        prompt=(
            "Implement `eval_expr(src: str, env: dict[str, int] | None = None) -> int` WITHOUT using `eval`, `exec`, `compile`, or `ast`!\n"
            "Support Python-exact integer expression semantics with full operator precedence (highest -> lowest):\n"
            "1. Parentheses `(...)`, integer literals, boolean literals `True`/`False` (treated as `1`/`0`), variable identifiers looked up in `env` (unknown var -> `NameError` or `ValueError`, raise `ValueError`!).\n"
            "2. Power `**` — RIGHT-associative (`2 ** 3 ** 2 == 512`), AND unary `-`/`+` binds LOOSER on the left of `**` than on the right (`-2 ** 2 == -4`, `2 ** -0 == 1`)!\n"
            "3. Unary `+`, `-` (chainable: `--5 == 5`).\n"
            "4. Multiplicative `*`, `//` (Python floor division: `-7 // 2 == -4`!), `%` (Python modulo: `-7 % 2 == 1`!); division/mod by 0 raises `ZeroDivisionError`.\n"
            "5. Additive `+`, `-`.\n"
            "6. Chained comparisons `<`, `<=`, `>`, `>=`, `==`, `!=` returning `1` or `0`: `1 < x < 10` evaluates `x` ONCE and checks `(1 < x) and (x < 10)`!\n"
            "7. Boolean `not`, then `and`, then `or` (returning `1` or `0`? Wait: return Python `int` truth value `1`/`0` for comparisons & `not`, and for `a and b` / `a or b` return the exact Python short-circuit operand value, e.g. `0 or 5 == 5`, `0 and (1 // 0) == 0`, `1 or (1 // 0) == 1`!).\n"
            "8. Right-associative ternary `A if COND else B` (MUST short-circuit — untaken branch like `10 if 1 else (1 // 0)` MUST NOT raise `ZeroDivisionError`!).\n"
            "Raise `ValueError` on any syntax error or unknown variable.\n"
            "Output ONLY Python code defining `eval_expr`."
        ),
        hidden_tests=r'''
import ast, inspect

def test_no_eval_exec_ast():
    tree = ast.parse(inspect.getsource(solution))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name != "ast" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "ast"
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id not in ("eval", "exec", "compile")

def test_precedence_short_circuit_chained_and_errors():
    assert eval_expr("2 ** 3 ** 2") == 512
    assert eval_expr("-2 ** 2") == -4
    assert eval_expr("(-2) ** 2") == 4
    assert eval_expr("-7 // 2") == -4
    assert eval_expr("-7 % 2") == 1
    assert eval_expr("1 < x < 10", {"x": 5}) == 1
    assert eval_expr("1 < x < 10", {"x": 10}) == 0
    assert eval_expr("1 < 2 == 2 < 3") == 1
    assert eval_expr("0 and (1 // 0)") == 0
    assert eval_expr("7 or (1 // 0)") == 7
    assert eval_expr("42 if 1 < 2 else (1 // 0)") == 42
    assert eval_expr("1 if 0 else 2 if 0 else 3") == 3
    assert eval_expr("not 0 and 3 + 4 * 2 == 11") == 1
    for bad in ["1 +", "(2 + 3", "unknown_var + 1", "1 2", ""]:
        try:
            eval_expr(bad, {})
            raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
    try:
        eval_expr("5 // 0")
        raise AssertionError("Expected ZeroDivisionError")
    except ZeroDivisionError:
        pass
''',
    ),
    Gauntlet2Problem(
        pid="H24_persistent_segment_tree",
        track="algo_nm",
        forge_mode="code",
        title="O(log N) Persistent Segment Tree (Range Sum + Range Min Across 5000 Versions)",
        prompt=(
            "Implement class `PersistentSegTree` supporting immutable versioned point updates & range queries in `O(log N)` time and space per update (NO O(N) array copies!):\n"
            "- `__init__(arr: list[int])`: builds version `0` from `arr` (do NOT mutate `arr`; allow `len(arr) >= 1`, raise `ValueError` if empty).\n"
            "- `set(version: int, idx: int, val: int) -> int`: creates a NEW version from `version` with `arr[idx] = val`, returning the new integer version ID (`1, 2, ...`). Old versions remain 100% unchanged!\n"
            "- `add(version: int, idx: int, delta: int) -> int`: creates a NEW version from `version` adding `delta` to `arr[idx]`, returning new version ID.\n"
            "- `range_sum(version: int, l: int, r: int) -> int`: sum over half-open `[l, r)` (`0 <= l <= r <= n`; returns `0` if `l == r`).\n"
            "- `range_min(version: int, l: int, r: int) -> int`: minimum over half-open `[l, r)` (`0 <= l < r <= n`, else raise `ValueError`/`IndexError`).\n"
            "- Invalid `version` or out-of-bounds `idx`/`l`/`r` raises `IndexError` or `ValueError`.\n"
            "- Performance: `N = 2000`, `4000` branching updates + `4000` queries MUST complete in `< 0.8s`!\n"
            "Output ONLY Python code defining `PersistentSegTree`."
        ),
        hidden_tests=r'''
import random, time

def test_branching_versions_fuzz_and_perf():
    rng = random.Random(24)
    init = [rng.randint(-50, 50) for _ in range(40)]
    pst = PersistentSegTree(list(init))
    vers = [list(init)]
    for _ in range(150):
        base = rng.randint(0, len(vers) - 1)
        idx = rng.randint(0, 39)
        if rng.random() < 0.5:
            val = rng.randint(-50, 50)
            nv = pst.set(base, idx, val)
            arr = list(vers[base]); arr[idx] = val
        else:
            delta = rng.randint(-20, 20)
            nv = pst.add(base, idx, delta)
            arr = list(vers[base]); arr[idx] += delta
        assert nv == len(vers)
        vers.append(arr)
        qv = rng.randint(0, len(vers) - 1)
        l = rng.randint(0, 39); r = rng.randint(l + 1, 40)
        assert pst.range_sum(qv, l, r) == sum(vers[qv][l:r])
        assert pst.range_min(qv, l, r) == min(vers[qv][l:r])
        assert pst.range_sum(qv, l, l) == 0

    # Performance stress: O(N) copy would take seconds / OOM!
    big = PersistentSegTree([i % 100 for i in range(2000)])
    t0 = time.perf_counter()
    v = 0
    for i in range(4000):
        v = big.add(i // 2, (i * 17) % 2000, 1)
    tot = 0
    for i in range(2000):
        tot += big.range_min(i * 2, i % 500, (i % 500) + 500)
    assert isinstance(tot, int) and (time.perf_counter() - t0) < 1.2
''',
    ),
    Gauntlet2Problem(
        pid="H25_exact_min_edit_diff_patch",
        track="algo_nm",
        forge_mode="code",
        title="Exact Minimum-Edit-Distance Line Diff, Apply & Reverse Patch Engine",
        prompt=(
            "Implement three functions (do NOT use `difflib` — compute an EXACT minimum-edit-script diff!):\n"
            "1. `diff_lines(a: list[str], b: list[str]) -> list[tuple[str, str]]`: returns a sequence of ops `('=', line)`, `('-', line)`, `('+', line)` transforming `a` into `b` with STRICTLY MINIMUM number of edits (`count('-') + count('+') == len(a) + len(b) - 2*LCS(a, b)`!).\n"
            "2. `apply_diff(a: list[str], ops: list[tuple[str, str]]) -> list[str]`: applies `ops` to `a`, returning `b`. Raises `ValueError` if any `'='` or `'-'` op does NOT match the next line of `a`, or if unconsumed lines remain in `a`, or on invalid op tag.\n"
            "3. `reverse_diff(b: list[str], ops: list[tuple[str, str]]) -> list[str]`: reverses `ops` on `b` to recover `a` (raising `ValueError` on mismatch!).\n"
            "- Never mutate input lists.\n"
            "Output ONLY Python code defining `diff_lines`, `apply_diff`, and `reverse_diff`."
        ),
        hidden_tests=r'''
import ast, inspect, random

def test_no_difflib():
    tree = ast.parse(inspect.getsource(solution))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name != "difflib" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "difflib"

def _lcs_len(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]

def test_exact_optimality_roundtrip_and_errors():
    rng = random.Random(25)
    vocab = ["a", "b", "c", "d", "e"]
    cases = [([], []), ([], ["x", "y"]), (["x", "y"], []), (["a", "b", "a"], ["b", "a", "b"])]
    for _ in range(30):
        a = [rng.choice(vocab) for _ in range(rng.randint(0, 25))]
        b = [rng.choice(vocab) for _ in range(rng.randint(0, 25))]
        cases.append((a, b))
    for a, b in cases:
        ops = diff_lines(list(a), list(b))
        edits = sum(1 for tag, _ in ops if tag in ("-", "+"))
        min_edits = len(a) + len(b) - 2 * _lcs_len(a, b)
        assert edits == min_edits, f"Suboptimal diff: {edits} > {min_edits}"
        assert apply_diff(a, ops) == b
        assert reverse_diff(b, ops) == a
    try:
        apply_diff(["a", "b"], [("=", "a")])  # unconsumed 'b'!
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
    try:
        apply_diff(["a"], [("-", "z")])  # mismatch!
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
''',
    ),
    Gauntlet2Problem(
        pid="H26_exact_sutherland_hodgman_area",
        track="algo_nm",
        forge_mode="code",
        title="Exact Rational Concave-x-Convex Polygon Clipping (2*Area as Fraction)",
        prompt=(
            "Implement `clip_polygon_area2(subject: list[tuple[int, int]], clip_convex: list[tuple[int, int]]) -> Fraction`.\n"
            "- `subject`: vertices of a simple polygon (may be CONCAVE, and may be given in EITHER CW or CCW winding order!).\n"
            "- `clip_convex`: vertices of a convex polygon (may ALSO be given in EITHER CW or CCW order!).\n"
            "- Return the EXACT non-negative `fractions.Fraction` equal to `2 * Area(subject ∩ clip_convex)` using exact rational Sutherland-Hodgman polygon clipping (ZERO floats — exact `Fraction` intersections only!).\n"
            "- Return `Fraction(0)` if either polygon has `< 3` vertices, has zero area, or their intersection is empty / degenerate (point or segment).\n"
            "- Do NOT mutate input lists.\n"
            "Output ONLY Python code defining `clip_polygon_area2`."
        ),
        hidden_tests=r'''
from fractions import Fraction

def test_concave_cw_ccw_and_exact_rationals():
    # L-shaped concave subject in [0,4]x[0,4] minus [2,4]x[2,4] (area = 12, 2*area = 24)
    l_shape = [(0,0), (4,0), (4,2), (2,2), (2,4), (0,4)]
    sq = [(0,0), (4,0), (4,4), (0,4)]
    assert clip_polygon_area2(l_shape, sq) == Fraction(24)
    assert clip_polygon_area2(list(reversed(l_shape)), list(reversed(sq))) == Fraction(24)

    # Clip L-shape with triangle (0,0)-(4,0)-(0,4):
    # Triangle x+y <= 4 covers all of [0,2]x[0,2] (area 4) + half of [2,4]x[0,2] (area 2) + half of [0,2]x[2,4] (area 2) = 8 -> 2*area = 16!
    tri = [(0,0), (4,0), (0,4)]
    assert clip_polygon_area2(l_shape, tri) == Fraction(16)

    # Non-integer exact rational intersection:
    # Square [0,3]x[0,3] clipped by triangle (0,0)-(3,1)-(1,3):
    # Triangle is inside square -> 2*area = |(3*3 - 1*1)| = 8
    # Shift triangle by (-1,-1): (-1,-1)-(2,0)-(0,2) intersected with [0,3]x[0,3]:
    # Segment (-1,-1)->(2,0) hits x=0 at (0, -2/3) and y=0 at (2,0).
    # Segment (-1,-1)->(0,2) hits y=0 at (-2/3, 0) and x=0 at (0,2).
    # Wait: let's check exact triangle (0,0)-(2,0)-(0,2) clipped by x+2y <= 2 -> vertices (0,0),(2,0),(0,1) -> 2*area=2.
    # Better exact non-integer intersection: two triangles:
    # T1 = (0,0),(3,0),(0,3) (x>=0,y>=0,x+y<=3) and T2 = (0,0),(3,1),(1,3)? Already inside!
    # Try T1 = (0,0),(2,0),(0,2) and T2 = (0,1),(2,1),(1,-1)?
    # Better: square [0,1]x[0,1] clipped by triangle (0,0),(2,1),(1,2) ->
    # edges of T inside unit square: y = x/2 hits x=1 at (1,1/2); y = 2x hits y=1 at (1/2,1);
    # intersection is polygon (0,0),(1,1/2),(1,1),(1/2,1) -> area = 1 - 1/4 - 1/4 = 1/2 -> 2*area = 1!
    unit = [(0,0),(1,0),(1,1),(0,1)]
    wedge = [(0,0),(2,1),(1,2)]
    assert clip_polygon_area2(unit, wedge) == Fraction(1)
    # Clip unit square by half-plane triangle (0,0),(3,1),(1,3):
    # hits (1,1/3) and (1/3,1) -> area = 1 - 1/6 - 1/6 = 2/3 -> 2*area = 4/3!
    wedge2 = [(0,0),(3,1),(1,3)]
    assert clip_polygon_area2(unit, wedge2) == Fraction(4, 3)
    assert isinstance(clip_polygon_area2(unit, wedge2), Fraction)
    # Disjoint -> 0
    assert clip_polygon_area2(unit, [(5,5),(6,5),(5,6)]) == Fraction(0)
''',
    ),
    Gauntlet2Problem(
        pid="H27_raft_log_state_machine",
        track="algo_nm",
        forge_mode="code",
        title="Raft Consensus Log Replication & Election State Machine (Figure-8 Safe)",
        prompt=(
            "Implement deterministic class `RaftNode` faithfully obeying the Raft paper (Ongaro & Ousterhout):\n"
            "- `__init__(node_id: str, cluster_size: int)`: starts as `'follower'`, `current_term = 0`, `voted_for = None`, "
            "`log = []` (1-indexed entries `(term: int, cmd: str)`, so `log[i-1]` is Raft log index `i`), `commit_index = 0`.\n"
            "- `handle_request_vote(term: int, candidate_id: str, last_log_index: int, last_log_term: int) -> tuple[int, bool]`:\n"
            "  * If `term < current_term`: reply `(current_term, False)`.\n"
            "  * If `term > current_term`: set `current_term = term`, `role = 'follower'`, `voted_for = None`.\n"
            "  * Grant vote iff (`voted_for is None` or `voted_for == candidate_id`) AND candidate's log is at least as up-to-date "
            "(higher `last_log_term`, OR equal `last_log_term` and `last_log_index >= len(self.log)`). Record `voted_for = candidate_id` if granted.\n"
            "- `handle_append_entries(term: int, leader_id: str, prev_log_index: int, prev_log_term: int, entries: list[tuple[int, str]], leader_commit: int) -> tuple[int, bool]`:\n"
            "  * Reject if `term < current_term`. If `term >= current_term`: update `current_term = term`, step down to `'follower'` (clear `voted_for` ONLY if `term > old_term`).\n"
            "  * Log Matching Check: if `prev_log_index > 0` and (`prev_log_index > len(log)` or `log[prev_log_index - 1][0] != prev_log_term`): return `(current_term, False)`.\n"
            "  * CRITICAL RAFT RULE: Iterate incoming `entries` starting at `prev_log_index + 1`: ONLY truncate existing log from index `idx` onward if an existing entry at `idx` has a DIFFERENT term! NEVER truncate a longer matching suffix when receiving an out-of-order/stale subset AppendEntries!\n"
            "  * If `leader_commit > commit_index`: `commit_index = min(leader_commit, len(log))`.\n"
            "- `become_leader(term: int) -> None`: sets `current_term = term`, `role = 'leader'`, `voted_for = node_id`.\n"
            "- `client_append(cmd: str) -> int`: leader appends `(current_term, cmd)` and returns its 1-based log index (raise `RuntimeError` if not `'leader'`).\n"
            "- `advance_leader_commit(match_indices: dict[str, int]) -> int`: CRITICAL FIGURE-8 RULE! Given replicated `match_indices` across peers (include leader's own `len(log)`), advance `commit_index` to the highest `N > commit_index` replicated on a strict majority (`> cluster_size // 2`) **ONLY IF `log[N - 1][0] == current_term`**! Never commit an old-term entry directly by counting replicas!\n"
            "Output ONLY Python code defining `RaftNode`."
        ),
        hidden_tests=r'''
def test_raft_invariants_stale_rpc_and_figure8():
    # 1. Log matching + stale AppendEntries MUST NOT truncate matching suffix!
    f = RaftNode("n2", cluster_size=3)
    assert f.handle_append_entries(1, "n1", 0, 0, [(1, "a"), (1, "b"), (1, "c")], 0) == (1, True)
    assert len(f.log) == 3
    # Stale/out-of-order heartbeat carrying only index 1..2 with MATCHING terms must preserve index 3!
    assert f.handle_append_entries(1, "n1", 0, 0, [(1, "a"), (1, "b")], 2) == (1, True)
    assert f.log == [(1, "a"), (1, "b"), (1, "c")] and f.commit_index == 2
    # Conflicting term at index 2 MUST truncate index 2..3 and append new entry!
    assert f.handle_append_entries(2, "n3", 1, 1, [(2, "x")], 1) == (2, True)
    assert f.log == [(1, "a"), (2, "x")] and f.commit_index == 2  # commit_index never decreases!

    # 2. Election up-to-date check (higher term wins over longer lower-term log!)
    assert f.handle_request_vote(3, "cand_old", last_log_index=10, last_log_term=1) == (3, False)
    assert f.handle_request_vote(3, "cand_new", last_log_index=2, last_log_term=2) == (3, True)
    assert f.handle_request_vote(3, "cand_other", last_log_index=5, last_log_term=3) == (3, False)

    # 3. Figure-8 Safety Rule: leader in term 4 CANNOT commit term-2 entry directly even on majority!
    l = RaftNode("l1", cluster_size=5)
    l.log = [(1, "k1"), (2, "k2")]
    l.become_leader(4)
    # 4/5 nodes have index 2, but index 2 has term 2 != current_term 4!
    assert l.advance_leader_commit({"n2": 2, "n3": 2, "n4": 2, "n5": 0}) == 0
    # Once leader appends a term-4 entry at index 3 and replicates it to majority (3/5 incl leader), commit jumps to 3!
    idx3 = l.client_append("k3")
    assert idx3 == 3
    assert l.advance_leader_commit({"n2": 3, "n3": 3, "n4": 0, "n5": 0}) == 3
    assert l.commit_index == 3
''',
    ),
    Gauntlet2Problem(
        pid="H28_lsm_store_tombstones_snapshots",
        track="algo_nm",
        forge_mode="code",
        title="Multi-Level LSM-Tree KV Store (Memtable Flush, Tombstone Shadowing, Compaction, Snapshots)",
        prompt=(
            "Implement class `LSMStore(memtable_limit: int = 4)`:\n"
            "- `put(key: str, value: str) -> None`: writes into active memtable; when `len(memtable) >= memtable_limit`, flushes it as a NEW immutable sorted SSTable level (newest SSTable overrides older SSTables!).\n"
            "- `delete(key: str) -> None`: writes a TOMBSTONE into active memtable (may trigger flush!). Deleting a key stored in an older SSTable MUST shadow it so `get(key)` returns `None` and `scan()` omits it!\n"
            "- `get(key: str) -> str | None`: checks memtable -> newest SSTable -> ... -> oldest SSTable; returns `None` if absent or shadowed by tombstone.\n"
            "- `scan(lo: str, hi: str) -> list[tuple[str, str]]`: returns sorted list of all live `(key, value)` pairs with `lo <= key < hi` (deduplicated newest-wins, tombstones excluded!).\n"
            "- `compact() -> None`: merges ALL SSTables (preserving memtable untouched) into ONE sorted SSTable (dropping overwritten versions AND dropping tombstones cleanly since no older SSTables remain beneath it!).\n"
            "- `snapshot() -> int` and `get_at(snap_id: int, key: str) -> str | None` and `scan_at(snap_id: int, lo: str, hi: str) -> list[tuple[str, str]]`: "
            "snapshots MUST stay 100% isolated even across subsequent `put()`, `delete()`, flushes, AND `compact()`!\n"
            "Output ONLY Python code defining `LSMStore`."
        ),
        hidden_tests=r'''
import random

def test_lsm_tombstones_compaction_and_snapshot_isolation():
    db = LSMStore(memtable_limit=3)
    for k, v in [("a","1"),("b","2"),("c","3"),("d","4"),("e","5")]:
        db.put(k, v)
    s1 = db.snapshot()
    db.delete("b")      # tombstone shadows SSTable 'b'
    db.put("c", "30")   # overwrite shadows SSTable 'c'
    db.put("f", "6")    # trigger another flush
    db.delete("d")
    s2 = db.snapshot()
    db.compact()
    db.put("b", "200")  # resurrect 'b' after compaction
    db.delete("a")

    assert db.get("a") is None and db.get("b") == "200" and db.get("c") == "30" and db.get("d") is None
    assert db.scan("a", "z") == [("b", "200"), ("c", "30"), ("e", "5"), ("f", "6")]
    # Snapshots MUST remain completely immune to later writes & compact()!
    assert db.get_at(s1, "b") == "2" and db.get_at(s1, "c") == "3" and db.get_at(s1, "f") is None
    assert db.scan_at(s1, "a", "z") == [("a","1"),("b","2"),("c","3"),("d","4"),("e","5")]
    assert db.get_at(s2, "b") is None and db.get_at(s2, "c") == "30" and db.get_at(s2, "d") is None
    assert db.scan_at(s2, "a", "f") == [("a","1"),("c","30"),("e","5")]
''',
    ),
    Gauntlet2Problem(
        pid="H29_hungarian_min_weight_matching",
        track="algo_nm",
        forge_mode="code",
        title="O(N^3) Exact Hungarian Min-Weight Bipartite Matching (Rectangular + Negative Costs)",
        prompt=(
            "Implement `hungarian_min_cost(cost: list[list[int]]) -> tuple[int, list[int]]` (NO `scipy`!).\n"
            "- Given an `m x n` integer matrix `cost` with `0 <= m <= n` (may contain NEGATIVE, zero, and huge positive integers!):\n"
            "- Assign each row `i in 0..m-1` to a DISTINCT column `col_of_row[i] in 0..n-1` minimizing `sum(cost[i][col_of_row[i]])`.\n"
            "- Return `(min_total_cost, col_of_row)`. For `m == 0`, return `(0, [])`. Raise `ValueError` if `m > n` or ragged rows.\n"
            "- Do NOT mutate `cost`. Must be true `O(m^2 * n)` Hungarian / Kuhn-Munkres (handle `60 x 60` in `< 0.4s`!).\n"
            "Output ONLY Python code defining `hungarian_min_cost` (do NOT import `scipy`)."
        ),
        hidden_tests=r'''
import ast, inspect, itertools, random, time

def test_no_scipy():
    tree = ast.parse(inspect.getsource(solution))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name != "scipy" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "scipy"

def test_exact_small_rectangular_negative_and_scale60():
    assert hungarian_min_cost([]) == (0, [])
    rng = random.Random(29)
    for m, n in [(1, 1), (3, 3), (3, 5), (4, 6), (5, 5)]:
        mat = [[rng.randint(-50, 50) for _ in range(n)] for _ in range(m)]
        snap = [list(r) for r in mat]
        got_cost, cols = hungarian_min_cost(mat)
        assert mat == snap
        assert len(cols) == m and len(set(cols)) == m and all(0 <= c < n for c in cols)
        assert sum(mat[i][cols[i]] for i in range(m)) == got_cost
        brute = min(sum(mat[i][perm[i]] for i in range(m)) for perm in itertools.permutations(range(n), m))
        assert got_cost == brute

    # Scale 60x60 with known exact unique minimum (shifted diagonal + large random noise)
    N = 60
    perm = list(range(N)); rng.shuffle(perm)
    u = [rng.randint(-200, 200) for _ in range(N)]
    v = [rng.randint(-200, 200) for _ in range(N)]
    mat60 = [[u[i] + v[j] + (0 if j == perm[i] else rng.randint(1, 500)) for j in range(N)] for i in range(N)]
    t0 = time.perf_counter()
    c60, cols60 = hungarian_min_cost(mat60)
    assert c60 == sum(u) + sum(v) and cols60 == perm and (time.perf_counter() - t0) < 0.8
    try:
        hungarian_min_cost([[1, 2], [3, 4], [5, 6]])
        raise AssertionError("Expected ValueError for m > n")
    except ValueError:
        pass
''',
    ),
    Gauntlet2Problem(
        pid="H30_unix_cron_next_trigger",
        track="algo_nm",
        forge_mode="code",
        title="Strict Standard UNIX 5-Field Cron Next-Run Calculator (OR-Semantics, Leap Years, Step Ranges)",
        prompt=(
            "Implement `next_cron_run(expr: str, after: datetime) -> datetime` (NO third-party `croniter`!).\n"
            "Standard 5-field UNIX cron: `minute(0-59) hour(0-23) dom(1-31) month(1-12) dow(0-7, both 0 and 7 mean Sunday; 1=Mon..6=Sat)`.\n"
            "Rules:\n"
            "- Each field is comma-separated items: `*`, `*/step`, `N`, `A-B`, `A-B/step` (`step >= 1`, `A <= B`, all values strictly within field bounds — else raise `ValueError`!). Reject != 5 fields.\n"
            "- Return the EARLIEST `datetime` STRICTLY AFTER `after` (with `second=0, microsecond=0`, preserving `after.tzinfo`) matching `expr`.\n"
            "- CRITICAL STANDARD UNIX CRON DOM/DOW RULE:\n"
            "  * If BOTH `dom` and `dow` are restricted (i.e., NEITHER is unrestricted `*`; note `*/1` is treated as restricted or match? Use standard rule: field string == `'*'` is unrestricted!), then a day matches if `dom` matches **OR** `dow` matches!\n"
            "  * If either `dom == '*'` or `dow == '*'` (or both), then BOTH must match (AND semantics).\n"
            "- Must handle leap years cleanly (e.g., `0 0 29 2 *` after `2024-03-01` -> `2028-02-29 00:00`!) and skip impossible dates like Feb 30 without crashing or looping minute-by-minute for 4 years (fast `< 0.1s`!).\n"
            "- Raise `ValueError` if an expression can NEVER match any calendar date (e.g. `0 0 31 2 *`).\n"
            "Output ONLY Python code defining `next_cron_run`."
        ),
        hidden_tests=r'''
from datetime import datetime, timezone
import time

def test_cron_unix_or_leap_year_and_validation():
    # Basic step & strict > after
    assert next_cron_run("*/15 9-17 * * 1-5", datetime(2025, 1, 3, 17, 0, 0)) == datetime(2025, 1, 6, 9, 0)
    assert next_cron_run("0 12 * * 7", datetime(2025, 1, 4, 12, 0)) == datetime(2025, 1, 5, 12, 0)  # Sunday = 7!
    # Standard UNIX OR-semantics when BOTH dom and dow are restricted:
    # 2025-01-05 is Sunday (dow=0), 2025-01-15 is Wednesday (dom=15)
    assert next_cron_run("30 8 15 * 0", datetime(2025, 1, 4, 0, 0)) == datetime(2025, 1, 5, 8, 30)
    assert next_cron_run("30 8 15 * 0", datetime(2025, 1, 13, 0, 0)) == datetime(2025, 1, 15, 8, 30)
    # Leap year jump across 4 years MUST be instant (< 0.1s)!
    t0 = time.perf_counter()
    nxt = next_cron_run("0 0 29 2 *", datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc))
    assert nxt == datetime(2028, 2, 29, 0, 0, tzinfo=timezone.utc)
    assert (time.perf_counter() - t0) < 0.2
    # Invalid expressions & impossible calendar date (Feb 31)
    for bad in ["* * * *", "60 * * * *", "* 24 * * *", "0 0 31 2 *", "*/0 * * * *", "10-5 * * * *"]:
        try:
            next_cron_run(bad, datetime(2025, 1, 1, 0, 0))
            raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
''',
    ),

    # =========================================================================
    # TRACK 4: NIGHTMARE MULTI-BUG DEBUGGING WITH PLANTED CONTROLS (H31-H40)
    # =========================================================================
    Gauntlet2Problem(
        pid="H31_debug_mvcc_snapshot_isolation",
        track="debug_nm",
        forge_mode="code",
        title="Fix 5 Interlocking Bugs in MVCC Snapshot-Isolation KV Store",
        buggy_starter='''class MVCCStore:
    def __init__(self):
        self.data = {}
        self.tx_id = 0
        self.active = {}

    def begin(self) -> int:
        self.tx_id += 1
        self.active[self.tx_id] = {}
        return self.tx_id

    def get(self, tx: int, key: str):
        if key in self.active[tx]:
            return self.active[tx][key]
        return self.data.get(key)

    def put(self, tx: int, key: str, value) -> None:
        self.active[tx][key] = value

    def delete(self, tx: int, key: str) -> None:
        self.active[tx][key] = None

    def commit(self, tx: int) -> None:
        buf = self.active.pop(tx)
        for k, v in buf.items():
            if v is None: self.data.pop(k, None)
            else: self.data[k] = v

    def abort(self, tx: int) -> None:
        self.active.pop(tx, None)
''',
        prompt=(
            "Fix ALL bugs in `MVCCStore` to implement genuine **MVCC Snapshot Isolation** with First-Committer-Wins:\n"
            "1. `begin() -> int`: starts a transaction capturing a read snapshot at the current global commit version `snap_ver`.\n"
            "2. `get(tx, key)`: returns uncommitted local write in `tx` (distinguishing local DELETE tombstone from `value=None` or absent!), "
            "otherwise the latest committed version of `key` with `commit_ver <= tx.snap_ver` (returning `None` if absent/deleted at that snapshot — NEVER seeing commits that happened AFTER `tx.begin()`!).\n"
            "3. `put(tx, key, value)` and `delete(tx, key)`: buffer writes locally in `tx` (raise `KeyError` if `tx` is not active!).\n"
            "4. `commit(tx) -> None`: **First-Committer-Wins Write-Write Conflict Detection!** If ANY key written/deleted by `tx` was committed by ANOTHER transaction at `ver > tx.snap_ver`, MUST abort `tx` (discarding all its writes atomically!) and raise `RuntimeError`! Otherwise assign `commit_ver = global_ver + 1` and commit atomically.\n"
            "5. `abort(tx) -> None`: discards local buffer and marks `tx` inactive (subsequent ops on committed/aborted `tx` raise `KeyError`).\n"
            "BUGGY CODE:\n```python\n"
            "class MVCCStore:\n"
            "    def __init__(self): self.data, self.tx_id, self.active = {}, 0, {}\n"
            "    def begin(self): self.tx_id += 1; self.active[self.tx_id] = {}; return self.tx_id\n"
            "    def get(self, tx, key): return self.active[tx].get(key, self.data.get(key))\n"
            "    def put(self, tx, key, value): self.active[tx][key] = value\n"
            "    def delete(self, tx, key): self.active[tx][key] = None\n"
            "    def commit(self, tx):\n"
            "        for k, v in self.active.pop(tx).items():\n"
            "            if v is None: self.data.pop(k, None)\n"
            "            else: self.data[k] = v\n"
            "    def abort(self, tx): self.active.pop(tx, None)\n```\n"
            "Output ONLY fixed Python code defining `MVCCStore`."
        ),
        hidden_tests=r'''
def test_mvcc_snapshot_isolation_and_first_committer_wins():
    db = MVCCStore()
    t0 = db.begin()
    db.put(t0, "x", 10); db.put(t0, "y", 20); db.commit(t0)

    t1 = db.begin()
    t2 = db.begin()
    db.put(t1, "x", 99)
    db.delete(t1, "y")
    # Read-your-own-writes + tombstone inside t1; t2 isolated!
    assert db.get(t1, "x") == 99 and db.get(t1, "y") is None
    assert db.get(t2, "x") == 10 and db.get(t2, "y") == 20
    db.commit(t1)
    # Snapshot isolation: t2 MUST STILL see old x=10, y=20!
    assert db.get(t2, "x") == 10 and db.get(t2, "y") == 20
    # Write-write conflict on 'x' -> t2 MUST raise RuntimeError AND rollback its write to 'z'!
    db.put(t2, "z", 777)
    db.put(t2, "x", 55)
    try:
        db.commit(t2)
        raise AssertionError("Expected RuntimeError on write-write conflict")
    except RuntimeError:
        pass
    t3 = db.begin()
    assert db.get(t3, "x") == 99 and db.get(t3, "y") is None and db.get(t3, "z") is None
    # Non-conflicting concurrent writes both succeed!
    t4 = db.begin()
    db.put(t3, "a", 1); db.put(t4, "b", 2)
    db.commit(t3); db.commit(t4)
    t5 = db.begin()
    assert db.get(t5, "a") == 1 and db.get(t5, "b") == 2
    try:
        db.get(t1, "x"); raise AssertionError("Expected KeyError on closed tx")
    except KeyError:
        pass
''',
    ),
    Gauntlet2Problem(
        pid="H32_debug_avl_balanced_tree",
        track="debug_nm",
        forge_mode="code",
        title="Fix 5 Bugs in AVL Balanced BST (Double Rotations, Delete Rebalance, Range Query)",
        buggy_starter='''class AVLTree:
    class _N:
        def __init__(self, k): self.k, self.l, self.r, self.h = k, None, None, 1
    def __init__(self): self.root = None; self._size = 0
    def _h(self, n): return n.h if n else 0
    def _rot_r(self, y):
        x = y.l; y.l = x.r; x.r = y
        return x  # BUG: heights never updated!
    def insert(self, key: int) -> bool:
        def _ins(n):
            if not n: self._size += 1; return self._N(key)
            if key < n.k: n.l = _ins(n.l)
            elif key > n.k: n.r = _ins(n.r)
            return n
        self.root = _ins(self.root); return True
    def delete(self, key: int) -> bool: return False
    def contains(self, key: int) -> bool: return False
    def range_query(self, lo: int, hi: int) -> list[int]: return []
    def height(self) -> int: return self._h(self.root)
    def __len__(self) -> int: return self._size
''',
        prompt=(
            "Fix ALL bugs in `AVLTree` (strictly balanced BST set of unique integers):\n"
            "1. `insert(key: int) -> bool`: inserts `key` and returns `True` if newly added, `False` if already present (no duplicates!). Maintains strict AVL height balance `|h(left) - h(right)| <= 1` via LL/RR/LR/RL rotations.\n"
            "2. `delete(key: int) -> bool`: removes `key` and returns `True` if present, `False` otherwise. Rebalances ALL ancestors back to root!\n"
            "3. `contains(key: int) -> bool`, `__len__() -> int`, `height() -> int` (0 for empty, 1 for single node).\n"
            "4. `range_query(lo: int, hi: int) -> list[int]`: sorted list of all stored keys `lo <= k <= hi` (pruning subtrees outside `[lo, hi]`).\n"
            "5. `is_valid_avl() -> bool`: verifies BST ordering, cached heights, AND balance factor `in {-1, 0, 1}` at every node.\n"
            "BUGGY CODE:\n```python\n"
            "class AVLTree:\n"
            "    class _N:\n"
            "        def __init__(self, k): self.k, self.l, self.r, self.h = k, None, None, 1\n"
            "    def __init__(self): self.root = None; self._size = 0\n"
            "    def _h(self, n): return n.h if n else 0\n"
            "    def _rot_r(self, y): x = y.l; y.l = x.r; x.r = y; return x\n"
            "    def insert(self, key):\n"
            "        def _ins(n):\n"
            "            if not n: self._size += 1; return self._N(key)\n"
            "            if key < n.k: n.l = _ins(n.l)\n"
            "            elif key > n.k: n.r = _ins(n.r)\n"
            "            return n\n"
            "        self.root = _ins(self.root); return True\n```\n"
            "Output ONLY fixed Python code defining `AVLTree`."
        ),
        hidden_tests=r'''
import math, random

def test_avl_differential_fuzz_and_balance():
    rng = random.Random(32)
    t = AVLTree()
    ref = set()
    # Worst-case monotonic inserts MUST stay O(log N) height!
    for x in range(256):
        assert t.insert(x) is True
        ref.add(x)
    assert t.insert(100) is False
    assert t.height() <= 10 and t.is_valid_avl() is True
    for _ in range(400):
        op = rng.choice(["ins", "del", "has", "rng"])
        x = rng.randint(-50, 300)
        if op == "ins":
            got = t.insert(x); exp = x not in ref; ref.add(x)
            assert got is exp
        elif op == "del":
            got = t.delete(x); exp = x in ref; ref.discard(x)
            assert got is exp
        elif op == "has":
            assert t.contains(x) is (x in ref)
        else:
            lo = rng.randint(-50, 250); hi = lo + rng.randint(-5, 60)
            assert t.range_query(lo, hi) == sorted(v for v in ref if lo <= v <= hi)
        assert len(t) == len(ref)
    assert t.is_valid_avl() is True
    assert t.height() <= int(1.45 * math.log2(len(ref) + 2)) + 1
''',
    ),
    Gauntlet2Problem(
        pid="H33_debug_aho_corasick",
        track="debug_nm",
        forge_mode="code",
        title="Fix 5 Bugs in Aho-Corasick Multi-Pattern Automaton (Dictionary Suffix Links + Overlaps)",
        buggy_starter='''class AhoCorasick:
    def __init__(self, patterns: list[str]):
        self.goto = [{}]
        self.fail = [0]
        self.out = [[]]
        for p in patterns:
            s = 0
            for ch in p:
                if ch not in self.goto[s]:
                    self.goto[s][ch] = len(self.goto)
                    self.goto.append({}); self.fail.append(0); self.out.append([])
                s = self.goto[s][ch]
            self.out[s].append(p)
        # BUG: BFS never merges output links from failure states, fails on root transitions!
    def search(self, text: str) -> list[tuple[int, int, str]]:
        res = []
        s = 0
        for i, ch in enumerate(text):
            while s and ch not in self.goto[s]:
                s = self.fail[s]
            s = self.goto[s].get(ch, 0)
            for p in self.out[s]:
                res.append((i - len(p) + 1, i + 1, p))
        return res
''',
        prompt=(
            "Fix ALL bugs in `AhoCorasick(patterns: list[str])`:\n"
            "1. Ignore empty patterns `\"\"` and deduplicate duplicate patterns in `patterns` (each distinct matching pattern reported ONCE per `(start, end)` span!).\n"
            "2. Build full BFS failure links AND propagate output/dictionary suffix links (`out[v] += out[fail[v]]`) so overlapping sub-patterns (e.g., `'he'` inside `'she'`, `'her'` inside `'hers'`, `'a'`, `'aa'`, `'aaa'` inside `'aaaa'`) are ALL found!\n"
            "3. `search(text: str) -> list[tuple[int, int, str]]`: returns all matches as `(start, end, pattern)` (`text[start:end] == pattern`), sorted by `(start, end, pattern)`.\n"
            "4. Must run in `O(|text| + |patterns| + |matches|)` linear time (handle 50,000-char text in `< 0.3s`!).\n"
            "BUGGY CODE:\n```python\n"
            "class AhoCorasick:\n"
            "    def __init__(self, patterns):\n"
            "        self.goto, self.fail, self.out = [{}], [0], [[]]\n"
            "        for p in patterns:\n"
            "            s = 0\n"
            "            for ch in p:\n"
            "                if ch not in self.goto[s]:\n"
            "                    self.goto[s][ch] = len(self.goto); self.goto.append({}); self.fail.append(0); self.out.append([])\n"
            "                s = self.goto[s][ch]\n"
            "            self.out[s].append(p)\n"
            "    def search(self, text):\n"
            "        res, s = [], 0\n"
            "        for i, ch in enumerate(text):\n"
            "            while s and ch not in self.goto[s]: s = self.fail[s]\n"
            "            s = self.goto[s].get(ch, 0)\n"
            "            for p in self.out[s]: res.append((i - len(p) + 1, i + 1, p))\n"
            "        return res\n```\n"
            "Output ONLY fixed Python code defining `AhoCorasick`."
        ),
        hidden_tests=r'''
import random, time

def test_aho_overlaps_suffix_links_dedup_and_perf():
    ac = AhoCorasick(["he", "she", "his", "hers", "he", ""])
    got = ac.search("ushers")
    exp = [(1, 4, "she"), (2, 4, "he"), (2, 6, "hers")]
    assert got == exp

    # Heavy nested overlap fuzz against naive reference
    rng = random.Random(33)
    pats = ["a", "aa", "aaa", "aba", "ba", "bab", "ababa", "b"]
    text = "".join(rng.choice("ab") for _ in range(200))
    ac2 = AhoCorasick(pats)
    ref = []
    for p in set(pats):
        start = 0
        while True:
            idx = text.find(p, start)
            if idx == -1: break
            ref.append((idx, idx + len(p), p))
            start = idx + 1
    ref.sort()
    assert ac2.search(text) == ref

    # Linear-time check on 50k chars
    ac3 = AhoCorasick(["needle", "eed", "abcde", "zzzzz"])
    big = "abcd" * 12500 + "needle"
    t0 = time.perf_counter()
    hits = ac3.search(big)
    assert hits == [(50000, 50006, "needle"), (50001, 50004, "eed")]
    assert (time.perf_counter() - t0) < 0.5
''',
    ),
    Gauntlet2Problem(
        pid="H34_debug_rfc3629_utf8_stream",
        track="debug_nm",
        forge_mode="code",
        title="Fix Strict RFC-3629 Streaming UTF-8 Decoder (Overlongs, Surrogates, Split Bytes)",
        buggy_starter='''class UTF8StreamDecoder:
    def __init__(self):
        self.buf = bytearray()
    def feed(self, data: bytes) -> str:
        # BUG: ignores multi-byte boundaries, accepts overlongs/surrogates/invalid!
        return "".join(chr(b) for b in data)
    def finish(self) -> str:
        return ""
''',
        prompt=(
            "Implement/fix a strict **RFC-3629 streaming UTF-8 decoder** `UTF8StreamDecoder` WITHOUT using `bytes.decode` or `codecs`!\n"
            "- `feed(data: bytes) -> str`: processes arbitrary byte chunks (even 1 byte at a time!) and returns the decoded `str` of all COMPLETE valid Unicode codepoints decoded so far (buffering trailing incomplete 1..3 bytes for the next `feed()`).\n"
            "- `finish() -> str`: raises `ValueError` if any incomplete multi-byte sequence remains buffered; otherwise returns `\"\"`.\n"
            "- STRICT RFC-3629 VALIDATION (MUST raise `ValueError` immediately upon encountering an invalid byte!):\n"
            "  * Reject stray continuation bytes (`0x80..0xBF`), invalid lead bytes (`0xC0`, `0xC1`, `0xF5..0xFF`).\n"
            "  * Reject ALL overlong encodings (`2-byte < U+0080`, `3-byte < U+0800`, `4-byte < U+10000`, e.g. `0xE0 0x80 0xAF`).\n"
            "  * Reject UTF-16 surrogates (`U+D800..U+DFFF`, i.e., `0xED 0xA0..0xBF ...`).\n"
            "  * Reject codepoints `> U+10FFFF` (e.g., `0xF4 0x90 0x80 0x80`).\n"
            "  * Reject non-continuation byte (`not 0x80..0xBF`) inside a multi-byte sequence.\n"
            "BUGGY CODE:\n```python\n"
            "class UTF8StreamDecoder:\n"
            "    def __init__(self): self.buf = bytearray()\n"
            "    def feed(self, data: bytes) -> str: return ''.join(chr(b) for b in data)\n"
            "    def finish(self) -> str: return ''\n```\n"
            "Output ONLY fixed Python code defining `UTF8StreamDecoder` (do NOT call `.decode(` or import `codecs`)."
        ),
        hidden_tests=r'''
import inspect

def test_no_builtin_decode():
    src = inspect.getsource(solution)
    assert ".decode(" not in src and "import codecs" not in src and "codecs." not in src

def test_streaming_1_2_3_4_byte_and_rfc3629_attacks():
    sample = "A\u00e9\u20ac\U0001f680\ud7ff\ue000\U0010ffff!"
    raw = sample.encode("utf-8")
    dec = UTF8StreamDecoder()
    out = "".join(dec.feed(raw[i:i+1]) for i in range(len(raw)))
    out += dec.finish()
    assert out == sample

    # Invalid sequences MUST raise ValueError!
    invalid_cases = [
        b"\x80",                    # stray continuation
        b"\xc0\xaf",                # overlong '/'
        b"\xc1\xbf",                # overlong 2-byte
        b"\xe0\x80\xaf",            # overlong 3-byte
        b"\xe0\x9f\xbf",            # overlong 3-byte U+07FF
        b"\xed\xa0\x80",            # surrogate U+D800
        b"\xed\xbf\xbf",            # surrogate U+DFFF
        b"\xf0\x80\x80\xaf",        # overlong 4-byte
        b"\xf4\x90\x80\x80",        # > U+10FFFF
        b"\xf5\x80\x80\x80",        # invalid lead
        b"\xe2\x82X",               # broken continuation
    ]
    for bad in invalid_cases:
        d = UTF8StreamDecoder()
        try:
            d.feed(bad); d.finish()
            raise AssertionError(f"Accepted invalid UTF-8: {bad!r}")
        except ValueError:
            pass

    # Truncated sequence caught by finish()
    d2 = UTF8StreamDecoder()
    assert d2.feed(b"\xf0\x9f\x9a") == ""
    try:
        d2.finish(); raise AssertionError("Expected ValueError on truncated stream")
    except ValueError:
        pass
''',
    ),
    Gauntlet2Problem(
        pid="H35_debug_bencode_strict_codec",
        track="debug_nm",
        forge_mode="code",
        title="Fix Strict Canonical BitTorrent Bencode Encoder & Decoder",
        buggy_starter='''def bencode(obj) -> bytes:
    if isinstance(obj, int): return f"i{obj}e".encode()
    if isinstance(obj, bytes): return f"{len(obj)}:".encode() + obj
    if isinstance(obj, list): return b"l" + b"".join(bencode(x) for x in obj) + b"e"
    if isinstance(obj, dict): return b"d" + b"".join(bencode(k) + bencode(v) for k, v in obj.items()) + b"e"
    raise TypeError("unsupported")

def bdecode(data: bytes):
    def parse(i):
        if data[i:i+1] == b"i":
            j = data.index(b"e", i)
            return int(data[i+1:j]), j + 1
        return b"", i + 1
    return parse(0)[0]
''',
        prompt=(
            "Fix ALL bugs in strict BitTorrent Bencode `bencode(obj) -> bytes` and `bdecode(data: bytes)`:\n"
            "1. Supported types: `int` (REJECT `bool` with `TypeError` — in Python `isinstance(True, int)` is `True`!!), `bytes`, `list`, `dict[bytes, Any]`.\n"
            "2. `bencode`: dict keys MUST be `bytes` (else `TypeError`), sorted in STRICT raw lexicographical byte order!\n"
            "3. `bdecode`: strict canonical BEP-3 validation (raise `ValueError` on ANY violation!):\n"
            "   - Integers `i<num>e`: reject leading zeros (`i03e`, `i00e`), negative zero (`i-0e`), plus sign (`i+1e`), empty (`ie`), non-digits.\n"
            "   - Byte strings `<len>:<raw>`: reject negative length, leading zeros in length (`01:a`, though `0:` for empty `b''` is valid!), or truncated payload.\n"
            "   - Dicts `d...e`: keys MUST be byte strings, STRICTLY increasing raw byte order (reject unsorted OR duplicate keys!).\n"
            "   - Reject trailing garbage bytes after the top-level value, empty input, or unclosed containers.\n"
            "BUGGY CODE:\n```python\n"
            "def bencode(obj) -> bytes:\n"
            "    if isinstance(obj, int): return f'i{obj}e'.encode()\n"
            "    if isinstance(obj, bytes): return f'{len(obj)}:'.encode() + obj\n"
            "    if isinstance(obj, list): return b'l' + b''.join(bencode(x) for x in obj) + b'e'\n"
            "    if isinstance(obj, dict): return b'd' + b''.join(bencode(k) + bencode(v) for k, v in obj.items()) + b'e'\n"
            "    raise TypeError('unsupported')\n```\n"
            "Output ONLY fixed Python code defining `bencode` and `bdecode`."
        ),
        hidden_tests=r'''
def test_bencode_canonical_and_strict_rejections():
    obj = {b"z": [0, -42, b"", b"\x00\xff"], b"a": {b"b": 1, b"aa": 2}}
    enc = bencode(obj)
    assert enc == b"d1:ad2:aai2e1:bi1ee1:zli0ei-42e0:2:\x00\xffee"
    assert bdecode(enc) == obj
    try:
        bencode(True); raise AssertionError("bool must raise TypeError")
    except TypeError:
        pass
    try:
        bencode({"str_key": 1}); raise AssertionError("non-bytes dict key must raise TypeError")
    except TypeError:
        pass
    for bad in [
        b"", b"i03e", b"i-0e", b"i+5e", b"ie", b"01:a", b"5:abc",
        b"d1:bi1e1:ai2ee",        # unsorted dict keys!
        b"d1:ai1e1:ai2ee",        # duplicate dict keys!
        b"di1ei2ee",              # non-bytes dict key!
        b"i1eEXTRA",              # trailing garbage!
        b"li1e",                  # unclosed list!
    ]:
        try:
            r = bdecode(bad); raise AssertionError(f"Expected ValueError for {bad!r}, got {r!r}")
        except ValueError:
            pass
''',
    ),
    Gauntlet2Problem(
        pid="H36_debug_tracing_gc_finalizers",
        track="debug_nm",
        forge_mode="code",
        title="Fix Tracing Mark-and-Sweep GC (Cyclic Garbage, WeakRefs, Finalizer Resurrection)",
        buggy_starter='''class TracingGC:
    def __init__(self):
        self.refs = {}
        self.roots = set()
        self.finalizers = {}
        self.weak_refs = {}
        self._next_id = 1

    def alloc(self, finalizer=None) -> int:
        oid = self._next_id; self._next_id += 1
        self.refs[oid] = set()
        if finalizer: self.finalizers[oid] = finalizer
        return oid

    def add_root(self, oid: int) -> None: self.roots.add(oid)
    def remove_root(self, oid: int) -> None: self.roots.discard(oid)
    def add_ref(self, src: int, dst: int) -> None: self.refs[src].add(dst)
    def remove_ref(self, src: int, dst: int) -> None: self.refs[src].discard(dst)
    def weak_ref(self, target: int): return lambda: target if target in self.refs else None
    def is_alive(self, oid: int) -> bool: return oid in self.refs
    def collect(self) -> list[int]:
        # BUG: only frees nodes with in-degree 0 (leaks cycles & breaks finalizer resurrection!)
        in_deg = {k: 0 for k in self.refs}
        for s, ds in self.refs.items():
            for d in ds:
                if d in in_deg: in_deg[d] += 1
        dead = [k for k, d in in_deg.items() if d == 0 and k not in self.roots]
        for k in dead: del self.refs[k]
        return sorted(dead)
''',
        prompt=(
            "Fix ALL bugs in `TracingGC` (a simulated tracing Mark-and-Sweep Garbage Collector with WeakRefs & Finalizer Resurrection!):\n"
            "- `alloc(finalizer=None) -> int`: allocates object ID (`1, 2, ...`); optional `finalizer(gc, oid)` callback.\n"
            "- `add_root(oid)`, `remove_root(oid)`, `add_ref(src, dst)`, `remove_ref(src, dst)`, `is_alive(oid) -> bool`.\n"
            "- `weak_ref(target: int)`: returns a callable `w()` returning `target` while `target` is alive, or `None` once collected (weak refs NEVER keep an object alive!).\n"
            "- `collect() -> list[int]` (returns sorted list of object IDs ACTUALLY freed in this cycle):\n"
            "  1. **Mark Phase:** Reachability DFS/BFS from `self.roots` through `self.refs` (automatically detects cyclic garbage!).\n"
            "  2. **Finalizer Phase (Java/Python-accurate Resurrection!):** For unreachable objects that still have an UNRUN finalizer:\n"
            "     - Run each such finalizer `fn(self, oid)` ONCE and mark it finalized (a finalizer MUST NEVER run twice on the same `oid`, even if resurrected and later collected!).\n"
            "     - During the finalizer, the callback may RESURRECT `oid` (or objects reachable from `oid`) by calling `add_root` or `add_ref` from a live object!\n"
            "     - Therefore, after running finalizers, RE-MARK from `self.roots`! Any object reachable after re-mark SURVIVES this cycle (is NOT freed and its weak refs remain alive!).\n"
            "  3. **Sweep Phase:** Free all objects still unreachable after the post-finalizer re-mark, deleting them from `self.refs` and clearing their `weak_ref` targets.\n"
            "BUGGY CODE:\n```python\n"
            "class TracingGC:\n"
            "    def __init__(self): self.refs, self.roots, self.finalizers, self._next_id = {}, set(), {}, 1\n"
            "    def alloc(self, finalizer=None):\n"
            "        oid = self._next_id; self._next_id += 1; self.refs[oid] = set()\n"
            "        if finalizer: self.finalizers[oid] = finalizer\n"
            "        return oid\n"
            "    def add_root(self, oid): self.roots.add(oid)\n"
            "    def remove_root(self, oid): self.roots.discard(oid)\n"
            "    def add_ref(self, src, dst): self.refs[src].add(dst)\n"
            "    def remove_ref(self, src, dst): self.refs[src].discard(dst)\n"
            "    def weak_ref(self, target): return lambda: target if target in self.refs else None\n"
            "    def is_alive(self, oid): return oid in self.refs\n"
            "    def collect(self): ...\n```\n"
            "Output ONLY fixed Python code defining `TracingGC`."
        ),
        hidden_tests=r'''
def test_gc_cycles_weakrefs_and_finalizer_resurrection():
    gc = TracingGC()
    # 1. Cyclic garbage collected cleanly + weak_ref cleared
    a = gc.alloc(); b = gc.alloc()
    gc.add_ref(a, b); gc.add_ref(b, a)
    wa = gc.weak_ref(a)
    assert wa() == a
    freed = gc.collect()
    assert freed == sorted([a, b])
    assert not gc.is_alive(a) and not gc.is_alive(b) and wa() is None

    # 2. Finalizer resurrection: phoenix resurrects itself + child on 1st GC, dies on 2nd GC without re-running finalizer!
    fin_runs = []
    anchor = gc.alloc()
    gc.add_root(anchor)
    child = gc.alloc()
    def phoenix_fin(g, oid):
        fin_runs.append(oid)
        g.add_ref(anchor, oid)  # resurrect!
    phoenix = gc.alloc(finalizer=phoenix_fin)
    gc.add_ref(phoenix, child)
    wp = gc.weak_ref(phoenix)

    assert gc.collect() == []  # Resurrected -> neither phoenix nor child freed!
    assert fin_runs == [phoenix]
    assert gc.is_alive(phoenix) and gc.is_alive(child) and wp() == phoenix

    # Detach again -> 2nd GC MUST collect both AND NOT run finalizer a second time!
    gc.remove_ref(anchor, phoenix)
    freed2 = gc.collect()
    assert freed2 == sorted([phoenix, child])
    assert fin_runs == [phoenix]
    assert wp() is None
''',
    ),
    Gauntlet2Problem(
        pid="H37_debug_circuit_breaker_fsm",
        track="debug_nm",
        forge_mode="code",
        title="Fix Half-Open Circuit Breaker FSM (Sliding Window, Probe Concurrency Cap, Cooldown)",
        buggy_starter='''class CircuitBreaker:
    def __init__(self, failure_threshold: int, window_size: int, cooldown_s: float, half_open_max: int = 1, success_threshold: int = 1, clock=None):
        self.failure_threshold = failure_threshold
        self.window_size = window_size
        self.cooldown_s = cooldown_s
        self.half_open_max = half_open_max
        self.success_threshold = success_threshold
        self.clock = clock or (lambda: 0.0)
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = None

    def allow_request(self) -> bool:
        if self.state == "OPEN":
            if self.clock() - self.opened_at >= self.cooldown_s:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.opened_at = self.clock()
''',
        prompt=(
            "Fix ALL bugs in `CircuitBreaker` state machine (`'CLOSED'`, `'OPEN'`, `'HALF_OPEN'`):\n"
            "1. Validate constructor params (`failure_threshold >= 1`, `window_size >= failure_threshold`, `cooldown_s >= 0`, `half_open_max >= 1`, `success_threshold >= 1`, else raise `ValueError`).\n"
            "2. **`CLOSED` state:** Track the last `window_size` completed outcomes (`record_success`/`record_failure`) in a bounded sliding window (FIFO deque of maxlen `window_size`). Transition to `'OPEN'` (recording `opened_at = clock()`) iff the count of failures in the current sliding window `>= failure_threshold`! (Old failures rolling out of the window MUST NOT trip the breaker!)\n"
            "3. **`OPEN` state:** Reject requests (`allow_request() -> False`) while `clock() - opened_at < cooldown_s`. Once `clock() - opened_at >= cooldown_s`, transition to `'HALF_OPEN'` (resetting half-open in-flight probe count and probe success streak to 0).\n"
            "4. **`HALF_OPEN` state:**\n"
            "   - Allow AT MOST `half_open_max` concurrent in-flight probes (`allow_request()` increments `in_flight` and returns `True` iff `in_flight < half_open_max`, else `False`!).\n"
            "   - `record_success()`: decrements `in_flight` (clamped >= 0), increments probe success streak; once streak `>= success_threshold`, transition back to `'CLOSED'` and CLEAR the old failure sliding window!\n"
            "   - `record_failure()`: decrements `in_flight`, IMMEDIATELY re-opens (`state = 'OPEN'`, `opened_at = clock()`, resetting in-flight & streak)!\n"
            "5. Property/method `current_state() -> str`: returns `'CLOSED'`, `'OPEN'`, or `'HALF_OPEN'` (reflecting cooldown expiry to `'HALF_OPEN'` without consuming a probe slot!).\n"
            "BUGGY CODE:\n```python\n"
            "class CircuitBreaker:\n"
            "    def __init__(self, failure_threshold, window_size, cooldown_s, half_open_max=1, success_threshold=1, clock=None):\n"
            "        self.failure_threshold, self.window_size, self.cooldown_s = failure_threshold, window_size, cooldown_s\n"
            "        self.half_open_max, self.success_threshold, self.clock = half_open_max, success_threshold, clock or (lambda: 0.0)\n"
            "        self.state, self.failures, self.opened_at = 'CLOSED', 0, None\n"
            "    def allow_request(self):\n"
            "        if self.state == 'OPEN':\n"
            "            if self.clock() - self.opened_at >= self.cooldown_s: self.state = 'HALF_OPEN'; return True\n"
            "            return False\n"
            "        return True\n"
            "    def record_success(self):\n"
            "        if self.state == 'HALF_OPEN': self.state = 'CLOSED'\n"
            "    def record_failure(self):\n"
            "        self.failures += 1\n"
            "        if self.failures >= self.failure_threshold: self.state = 'OPEN'; self.opened_at = self.clock()\n```\n"
            "Output ONLY fixed Python code defining `CircuitBreaker`."
        ),
        hidden_tests=r'''
def test_circuit_breaker_sliding_window_half_open_cap_and_reopen():
    t = [0.0]
    cb = CircuitBreaker(failure_threshold=2, window_size=4, cooldown_s=10.0, half_open_max=2, success_threshold=2, clock=lambda: t[0])
    # Sliding window: F, S, S, S -> window=[F,S,S,S]; next F evicts old F -> window=[S,S,S,F] -> MUST stay CLOSED!
    assert cb.allow_request() is True; cb.record_failure()
    for _ in range(3):
        assert cb.allow_request() is True; cb.record_success()
    assert cb.allow_request() is True; cb.record_failure()
    assert cb.current_state() == "CLOSED"
    # Second failure inside window -> OPEN!
    assert cb.allow_request() is True; cb.record_failure()
    assert cb.current_state() == "OPEN"
    assert cb.allow_request() is False

    # Advance past cooldown -> HALF_OPEN (max 2 concurrent probes!)
    t[0] = 10.0
    assert cb.current_state() == "HALF_OPEN"
    assert cb.allow_request() is True   # probe 1
    assert cb.allow_request() is True   # probe 2
    assert cb.allow_request() is False  # probe 3 REJECTED while 2 in flight!
    cb.record_failure()                 # probe failure -> IMMEDIATELY re-opens with fresh cooldown!
    assert cb.current_state() == "OPEN" and cb.allow_request() is False

    t[0] = 19.9; assert cb.allow_request() is False
    t[0] = 20.0
    assert cb.allow_request() is True; cb.record_success()
    assert cb.current_state() == "HALF_OPEN"  # needs 2 successes!
    assert cb.allow_request() is True; cb.record_success()
    assert cb.current_state() == "CLOSED"
    # Cleared window: 1 failure must NOT immediately re-trip!
    assert cb.allow_request() is True; cb.record_failure()
    assert cb.current_state() == "CLOSED"
''',
    ),
    Gauntlet2Problem(
        pid="H38_debug_tarjan_bridges_art_points",
        track="debug_nm",
        forge_mode="code",
        title="Fix Iterative Tarjan Bridges & Articulation Points (Multi-Edges, Self-Loops, N=4000)",
        buggy_starter='''def graph_critical_elements(n: int, edges: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int]]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    disc, low, timer = [-1]*n, [0]*n, [0]
    arts, bridges = set(), []
    def dfs(u, p):
        disc[u] = low[u] = timer[0]; timer[0] += 1
        for v in adj[u]:
            if v == p: continue  # BUG: breaks on parallel multi-edges & recurses deeply!
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]: arts.add(u)  # BUG: wrong for root!
                if low[v] > disc[u]: bridges.append((min(u,v), max(u,v)))
            else:
                low[u] = min(low[u], disc[v])
    if n: dfs(0, -1)  # BUG: misses disconnected components!
    return sorted(arts), sorted(bridges)
''',
        prompt=(
            "Fix ALL bugs in `graph_critical_elements(n: int, edges: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int]]]`:\n"
            "Given an undirected multigraph on vertices `0..n-1` with edges `(u, v)` (may have PARALLEL MULTI-EDGES, self-loops, and disconnected components):\n"
            "Return `(articulation_points, bridges)`:\n"
            "- `articulation_points`: sorted list of unique vertices whose removal strictly increases the number of connected components.\n"
            "- `bridges`: sorted list of unique canonical `(min(u, v), max(u, v))` edges whose removal increases connected components. "
            "**CRITICAL MULTI-EDGE RULE:** If TWO or more parallel edges connect `{u, v}`, removing ONE of them does NOT disconnect `u` from `v`, so `(min(u, v), max(u, v))` is **NOT** a bridge! (Self-loops `(u, u)` are NEVER bridges either!)\n"
            "- **CRITICAL:** Must be **ITERATIVE** (NO `RecursionError` on a 4,000-node path graph!) and handle `n == 0`.\n"
            "BUGGY CODE:\n```python\n"
            "def graph_critical_elements(n, edges):\n"
            "    adj = [[] for _ in range(n)]\n"
            "    for u, v in edges: adj[u].append(v); adj[v].append(u)\n"
            "    disc, low, timer, arts, bridges = [-1]*n, [0]*n, [0], set(), []\n"
            "    def dfs(u, p):\n"
            "        disc[u] = low[u] = timer[0]; timer[0] += 1\n"
            "        for v in adj[u]:\n"
            "            if v == p: continue\n"
            "            if disc[v] == -1:\n"
            "                dfs(v, u); low[u] = min(low[u], low[v])\n"
            "                if low[v] >= disc[u]: arts.add(u)\n"
            "                if low[v] > disc[u]: bridges.append((min(u,v), max(u,v)))\n"
            "            else: low[u] = min(low[u], disc[v])\n"
            "    if n: dfs(0, -1)\n"
            "    return sorted(arts), sorted(bridges)\n```\n"
            "Output ONLY fixed Python code defining `graph_critical_elements`."
        ),
        hidden_tests=r'''
import time

def test_multiedges_root_disconnected_and_deep_chain():
    assert graph_critical_elements(0, []) == ([], [])
    # Root with 1 child (NOT articulation!) + parallel edge (0,1) (NOT a bridge!) + bridge (1,2) + self-loop (2,2) + disconnected edge (3,4)
    edges = [(0, 1), (0, 1), (1, 2), (2, 2), (3, 4)]
    arts, bridges = graph_critical_elements(5, edges)
    assert arts == [1]
    assert bridges == [(1, 2), (3, 4)]

    # Triangle + bridge + star
    edges2 = [(0,1),(1,2),(2,0),(2,3),(3,4),(3,5)]
    assert graph_critical_elements(6, edges2) == ([2, 3], [(2, 3), (3, 4), (3, 5)])

    # Deep 4000-node path graph (MUST NOT hit RecursionError!)
    n = 4000
    chain = [(i, i + 1) for i in range(n - 1)]
    t0 = time.perf_counter()
    a_big, b_big = graph_critical_elements(n, chain)
    assert len(a_big) == n - 2 and len(b_big) == n - 1 and (time.perf_counter() - t0) < 0.8
''',
    ),
    Gauntlet2Problem(
        pid="H39_debug_decimal_orderbook",
        track="debug_nm",
        forge_mode="code",
        title="Fix Price-Time Priority Limit Order Book (Exact Decimal, Partial Fills, Self-Trade Prevention)",
        buggy_starter='''class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []

    def add_limit(self, order_id: str, trader: str, side: str, price: str, qty: int) -> list[dict]:
        # BUG: float prices, LIFO instead of FIFO, no matching or self-trade prevention!
        entry = {"id": order_id, "trader": trader, "price": float(price), "qty": qty}
        (self.bids if side == "BUY" else self.asks).append(entry)
        return []

    def cancel(self, order_id: str) -> bool:
        return False

    def top_of_book(self):
        return (None, None)
''',
        prompt=(
            "Fix/implement `OrderBook` — a deterministic Price-Time Priority Limit Order Book using EXACT `decimal.Decimal` prices:\n"
            "- `add_limit(order_id: str, trader: str, side: str, price: str, qty: int) -> list[dict]`:\n"
            "  * `side` is `'BUY'` or `'SELL'`, `price` parsed as `Decimal(price) > 0`, `qty > 0` (else `ValueError`; duplicate active `order_id` raises `ValueError`).\n"
            "  * Match incoming aggressive order against resting opposite-side orders in STRICT price-time priority:\n"
            "    - Incoming `'BUY'` matches lowest `'SELL'` orders with `ask.price <= buy.price` (FIFO by arrival order at equal price).\n"
            "    - Incoming `'SELL'` matches highest `'BUY'` orders with `bid.price >= sell.price` (FIFO at equal price).\n"
            "    - **Trade price** is ALWAYS the RESTING order's exact `Decimal` price! Each fill appends `{'maker_id': ..., 'taker_id': ..., 'price': Decimal, 'qty': int}`.\n"
            "    - **Self-Trade Prevention (`cancel_resting`):** If the next matchable resting order belongs to the SAME `trader` as the incoming order, DO NOT trade — instead CANCEL (remove) that resting order and continue matching remaining orders!\n"
            "  * Any remaining unfilled `qty > 0` rests on the book preserving arrival order.\n"
            "- `cancel(order_id: str) -> bool`: removes active resting order and returns `True`, or `False` if unknown or already completely filled/cancelled.\n"
            "- `top_of_book() -> tuple[tuple[Decimal, int] | None, tuple[Decimal, int] | None]`: returns `(best_bid, best_ask)` where each is `(price, total_qty_at_that_price)` or `None` if side is empty.\n"
            "BUGGY CODE:\n```python\n"
            "class OrderBook:\n"
            "    def __init__(self): self.bids, self.asks = [], []\n"
            "    def add_limit(self, order_id, trader, side, price, qty):\n"
            "        (self.bids if side == 'BUY' else self.asks).append({'id': order_id, 'trader': trader, 'price': float(price), 'qty': qty})\n"
            "        return []\n"
            "    def cancel(self, order_id): return False\n"
            "    def top_of_book(self): return (None, None)\n```\n"
            "Output ONLY fixed Python code defining `OrderBook`."
        ),
        hidden_tests=r'''
from decimal import Decimal

def test_price_time_fifo_decimal_and_self_trade_prevention():
    ob = OrderBook()
    assert ob.top_of_book() == (None, None)
    assert ob.add_limit("s1", "alice", "SELL", "100.10", 5) == []
    assert ob.add_limit("s2", "bob",   "SELL", "100.10", 7) == []
    assert ob.add_limit("s3", "carol", "SELL", "100.05", 4) == []
    assert ob.top_of_book() == (None, (Decimal("100.05"), 4))

    # Alice buys 10 @ 100.20:
    # 1) fills carol s3 (4 @ 100.05) -> 6 left
    # 2) encounters own order s1 (alice @ 100.10) -> SELF-TRADE CANCEL s1 (no trade!)
    # 3) fills bob s2 (6 of 7 @ 100.10) -> bob has 1 left resting!
    fills = ob.add_limit("b1", "alice", "BUY", "100.20", 10)
    assert fills == [
        {"maker_id": "s3", "taker_id": "b1", "price": Decimal("100.05"), "qty": 4},
        {"maker_id": "s2", "taker_id": "b1", "price": Decimal("100.10"), "qty": 6},
    ]
    assert ob.cancel("s1") is False  # already cancelled by STP!
    assert ob.cancel("s3") is False  # already filled!
    assert ob.top_of_book() == (None, (Decimal("100.10"), 1))
    assert ob.cancel("s2") is True
    assert ob.top_of_book() == (None, None)
''',
    ),
    Gauntlet2Problem(
        pid="H40_debug_event_loop_scheduler",
        track="debug_nm",
        forge_mode="code",
        title="Fix Deterministic Virtual-Clock Event Loop (Microtasks, FIFO Ties, Cancellation)",
        buggy_starter='''class VirtualEventLoop:
    def __init__(self):
        self.now = 0
        self.timers = []

    def call_soon(self, fn):
        self.timers.append((self.now, fn))

    def call_later(self, delay: int, fn):
        self.timers.append((self.now + delay, fn))

    def run_all(self) -> int:
        while self.timers:
            self.timers.sort(key=lambda x: x[0])
            t, fn = self.timers.pop(0)
            self.now = t
            fn()
        return self.now
''',
        prompt=(
            "Fix ALL bugs in `VirtualEventLoop` (a deterministic discrete-event scheduler with Microtasks, Timers, and Handles):\n"
            "- `now: int`: virtual clock starting at `0`.\n"
            "- `call_soon(fn) -> Handle`: queues `fn` in a FIFO **microtask queue** that executes at current `now` BEFORE the virtual clock EVER advances to ANY future timer (even if a microtask schedules another microtask!).\n"
            "- `call_later(delay: int, fn) -> Handle`: schedules `fn` at `self.now + delay` (`delay < 0` raises `ValueError`; if `delay == 0`, runs at `self.now`, AFTER already-queued microtasks or in FIFO order at `self.now` — specifically: microtasks ALWAYS drain completely before ANY timer fires, and timers at the same timestamp `t` fire in exact FIFO insertion order!).\n"
            "- `call_at(when: int, fn) -> Handle`: schedules `fn` at `max(self.now, when)` (raise `ValueError` if `when < 0`).\n"
            "- `Handle`: has `.cancel() -> None` and `.cancelled -> bool`. Cancelling a handle MUST prevent `fn` from running AND MUST NOT advance `self.now` to a cancelled timer's timestamp if no live timers remain at/after it!\n"
            "- `run_all() -> int`: drains microtasks and live timers until none remain, returning final `self.now`.\n"
            "BUGGY CODE:\n```python\n"
            "class VirtualEventLoop:\n"
            "    def __init__(self): self.now, self.timers = 0, []\n"
            "    def call_soon(self, fn): self.timers.append((self.now, fn))\n"
            "    def call_later(self, delay, fn): self.timers.append((self.now + delay, fn))\n"
            "    def run_all(self):\n"
            "        while self.timers:\n"
            "            self.timers.sort(key=lambda x: x[0]); t, fn = self.timers.pop(0); self.now = t; fn()\n"
            "        return self.now\n```\n"
            "Output ONLY fixed Python code defining `VirtualEventLoop`."
        ),
        hidden_tests=r'''
def test_microtasks_fifo_ties_nested_and_cancelled_clock_stop():
    loop = VirtualEventLoop()
    log = []
    # Cancelled far-future timer MUST NOT advance loop.now to 999!
    h_far = loop.call_later(999, lambda: log.append("BAD"))
    h_far.cancel()
    assert h_far.cancelled is True

    # Two timers at t=5 must run in FIFO insertion order; microtask scheduled by first MUST run BEFORE second timer at t=5!
    def t5_first():
        log.append(("t5_a", loop.now))
        loop.call_soon(lambda: log.append(("micro_from_t5_a", loop.now)))
    loop.call_later(5, t5_first)
    loop.call_later(5, lambda: log.append(("t5_b", loop.now)))
    loop.call_soon(lambda: log.append(("m1", loop.now)))
    loop.call_later(0, lambda: log.append(("zero_timer", loop.now)))
    loop.call_soon(lambda: log.append(("m2", loop.now)))

    end_t = loop.run_all()
    assert end_t == 5 and loop.now == 5
    assert log == [
        ("m1", 0),
        ("m2", 0),
        ("zero_timer", 0),
        ("t5_a", 5),
        ("micro_from_t5_a", 5),
        ("t5_b", 5),
    ]
''',
    ),

    # =========================================================================
    # TRACK 5 (PART B): NIGHTMARE SECURITY & CRYPTOGRAPHY ENGINEERING (H46-H50)
    # =========================================================================
    Gauntlet2Problem(
        pid="H46_sec_jwt_hs256_hardened",
        track="sys_sec_nm",
        forge_mode="code",
        title="Hardened RFC-7519 HS256 JWT Signer/Verifier (Blocks alg=none, Confusion, Skew, Replay)",
        prompt=(
            "Implement `jwt_sign(payload: dict, secret: bytes, kid: str | None = None) -> str` and "
            "`jwt_verify(token: str, keys: dict[str, bytes], *, now: int, issuer: str | None = None, audience: str | None = None, leeway_s: int = 0) -> dict` "
            "(standard library ONLY — NO `PyJWT`!).\n"
            "Security rules (raise `ValueError` on ANY violation!):\n"
            "1. Strict 3-part dot-separated `header.payload.signature` using unpadded URL-safe base64 (`[A-Za-z0-9_-]+`, reject `=` padding or whitespace!).\n"
            "2. **Algorithm Pinning:** Header MUST be a JSON object with EXACT `header.get('alg') == 'HS256'` (case-sensitive! Reject `'none'`, `'None'`, `'RS256'`, `'hs256'`, missing `alg`).\n"
            "3. **Key Selection:** Look up `kid = header.get('kid')` strictly in `keys` (`kid` must be a `str` present in `keys` — never allow path traversal or fallback default key if `kid` is unknown! If `kid` is omitted, only allow if `''` or `'default'`? Rule: require `kid` in `keys`!).\n"
            "4. **Constant-Time Signature Check:** Verify HMAC-SHA256 over ASCII `f'{h_b64}.{p_b64}'` using `hmac.compare_digest` BEFORE trusting any claims!\n"
            "5. **Time & Claim Validation:**\n"
            "   - If `'exp'` present: MUST be int/float (not bool!), reject if `now - leeway_s >= exp` (expired!).\n"
            "   - If `'nbf'` present: MUST be int/float (not bool!), reject if `now + leeway_s < nbf` (not yet valid!).\n"
            "   - If `issuer` specified: require `payload.get('iss') == issuer`.\n"
            "   - If `audience` specified: accept `payload.get('aud') == audience` OR (`isinstance(aud, list)` and `audience in aud`).\n"
            "Output ONLY Python code defining `jwt_sign` and `jwt_verify`."
        ),
        hidden_tests=r'''
import base64, inspect, json

def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

def test_jwt_hardening_and_attacks():
    src = inspect.getsource(solution)
    assert "compare_digest" in src
    keys = {"k1": b"super_secret_key_1", "k2": b"super_secret_key_2"}
    tok = jwt_sign({"sub": "u42", "iss": "auth.svc", "aud": ["api", "web"], "exp": 2000, "nbf": 1000}, keys["k1"], kid="k1")
    claims = jwt_verify(tok, keys, now=1500, issuer="auth.svc", audience="api")
    assert claims["sub"] == "u42"

    # Attack 1: alg=none bypass!
    h_none = _b64u(json.dumps({"alg": "none", "typ": "JWT", "kid": "k1"}).encode())
    p_evil = _b64u(json.dumps({"sub": "admin", "exp": 9999}).encode())
    # Attack 2: case-tricked alg=hs256 or RS256!
    h_rs = _b64u(json.dumps({"alg": "RS256", "kid": "k1"}).encode())
    # Attack 3: tampered signature / wrong kid / expired / nbf future / wrong aud / padded b64!
    parts = tok.split(".")
    bad_tokens = [
        f"{h_none}.{p_evil}.",
        f"{h_rs}.{p_evil}.{parts[2]}",
        f"{parts[0]}.{parts[1]}.{'A'*43}",
        f"{parts[0]}=.{parts[1]}.{parts[2]}",
        jwt_sign({"sub": "u", "exp": 1400}, keys["k1"], kid="k1"),
        jwt_sign({"sub": "u", "nbf": 1600, "exp": 2000}, keys["k1"], kid="k1"),
        jwt_sign({"sub": "u", "iss": "wrong", "aud": "api", "exp": 2000}, keys["k1"], kid="k1"),
        jwt_sign({"sub": "u", "iss": "auth.svc", "aud": "other", "exp": 2000}, keys["k1"], kid="k1"),
        jwt_sign({"sub": "u", "exp": 2000}, b"attacker", kid="../../etc/passwd"),
    ]
    for bad in bad_tokens:
        try:
            jwt_verify(bad, keys, now=1500, issuer="auth.svc", audience="api")
            raise AssertionError(f"SECURITY BYPASS accepted bad JWT: {bad[:40]}")
        except ValueError:
            pass
''',
    ),
    Gauntlet2Problem(
        pid="H47_sec_ssrf_ip_guard",
        track="sys_sec_nm",
        forge_mode="code",
        title="SSRF URL & DNS-Rebinding Guard (IPv4-Mapped IPv6, CGNAT, Hex/Int IP, Multi-A Rebind)",
        prompt=(
            "Implement `is_safe_outbound_url(url: str, resolve_fn) -> bool` using Python's `urllib.parse` & `ipaddress` to block ALL SSRF bypasses:\n"
            "Must return `False` (NEVER raise unhandled exceptions!) on ANY of:\n"
            "1. Non-`http`/`https` schemes (`file://`, `gopher://`, `ftp://`, `javascript:`), missing hostname, control chars/whitespace, or embedded `username`/`password` (`http://safe.com@127.0.0.1/`).\n"
            "2. Non-public literal IP hosts in ANY encoding (dotted decimal private/loopback/metadata, IPv6 `[::1]`, **IPv4-mapped IPv6** `[::ffff:127.0.0.1]` or `[::ffff:169.254.169.254]` — MUST unwrap `.ipv4_mapped`!, and obfuscated non-dotted IP encodings like decimal integer `http://2130706433/`, hex `http://0x7f000001/`, octal `http://0177.0.0.1/`). Clean public standard IPv4/IPv6 literals like `https://93.184.216.34/ping` MUST be ALLOWED without calling `resolve_fn`!\n"
            "3. DNS resolution via `resolve_fn(hostname) -> list[str]` returning an empty list, raising an exception, OR returning ANY non-public IP address (ALL resolved IPs must be globally routable public IPs — if even ONE IP in a multi-record response is private/internal, return `False`!).\n"
            "4. Blocked IP ranges (BOTH literal & DNS-resolved): loopback (`127.0.0.0/8`, `::1`), private (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), link-local/cloud metadata (`169.254.0.0/16`, `fe80::/10`), CGNAT (`100.64.0.0/10`), unspecified (`0.0.0.0/8`, `::`), multicast, reserved, documentation (`192.0.2.0/24`, `2001:db8::/32`). (Hint: check `ip.is_global` AND explicitly reject `100.64.0.0/10` / `0.0.0.0/8` / `192.0.2.0/24` / `::ffff:` mapped private!).\n"
            "Return `True` ONLY for clean `http`/`https` URLs whose resolved IP(s) are ALL strictly public globally routable addresses.\n"
            "Output ONLY Python code defining `is_safe_outbound_url`."
        ),
        hidden_tests=r'''
def test_ssrf_all_bypass_vectors():
    dns = {
        "api.example.com": ["93.184.216.34"],
        "dual.example.com": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"],
        "rebind.evil.com": ["93.184.216.34", "169.254.169.254"],
        "cgnat.evil.com": ["100.64.1.5"],
        "mapped6.evil.com": ["::ffff:10.0.0.1"],
        "empty.evil.com": [],
    }
    def fake_resolve(host):
        if host not in dns:
            raise OSError("NXDOMAIN")
        return dns[host]

    assert is_safe_outbound_url("https://api.example.com/v1/data", fake_resolve) is True
    assert is_safe_outbound_url("http://dual.example.com:8080/ok", fake_resolve) is True
    assert is_safe_outbound_url("https://93.184.216.34/ping", fake_resolve) is True

    blocked = [
        "file:///etc/passwd",
        "gopher://127.0.0.1:6379/_INFO",
        "http://api.example.com@127.0.0.1/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://2130706433/",       # decimal 127.0.0.1!
        "http://0x7f000001/",       # hex 127.0.0.1!
        "http://0177.0.0.1/",       # octal 127.0.0.1!
        "http://0.0.0.0/",
        "http://100.64.0.1/",       # CGNAT!
        "https://rebind.evil.com/", # 1 public + 1 metadata IP!
        "https://cgnat.evil.com/",
        "https://mapped6.evil.com/",
        "https://empty.evil.com/",
        "https://nxdomain.invalid/",
    ]
    for u in blocked:
        assert is_safe_outbound_url(u, fake_resolve) is False, f"SSRF BYPASS: {u}"
''',
    ),
    Gauntlet2Problem(
        pid="H48_sec_rfc6962_merkle_tree",
        track="sys_sec_nm",
        forge_mode="code",
        title="RFC-6962 Domain-Separated Merkle Tree & Audit Proof (Second-Preimage Resistant)",
        prompt=(
            "Implement `MerkleTree(leaves: list[bytes])` and `verify_merkle_proof(leaf: bytes, index: int, tree_size: int, proof: list[bytes], expected_root: bytes) -> bool` obeying **RFC 6962** (Certificate Transparency Merkle Hash Tree):\n"
            "- **Domain Separation (blocks Second-Preimage Attacks!):**\n"
            "  * Empty tree (`N == 0`): `root = sha256(b'').digest()`.\n"
            "  * Leaf hash: `H_leaf(d) = sha256(b'\\x00' + d).digest()`.\n"
            "  * Internal node hash: `H_node(l, r) = sha256(b'\\x01' + l + r).digest()`.\n"
            "- **RFC 6962 Exact Non-Power-of-Two Split:** For `n >= 2` leaves, let `k` be the largest power of two strictly `< n` (e.g., for `n=5`, `k=4`, left has `0..3`, right has `4`; for `n=6`, `k=4`, left `0..3`, right `4..5`). `MTH(D[0:n]) = H_node(MTH(D[0:k]), MTH(D[k:n]))`! (NEVER duplicate odd trailing leaves like Bitcoin — duplicating enables CVE-2012-2459 malleability!)\n"
            "- `tree.root() -> bytes`: 32-byte SHA-256 root.\n"
            "- `tree.proof(index: int) -> list[bytes]`: returns minimal RFC-6962 inclusion audit path (bottom-up sibling hashes) for `0 <= index < len(leaves)` (out-of-range raises `IndexError`).\n"
            "- `verify_merkle_proof(leaf, index, tree_size, proof, expected_root) -> bool`: verifies inclusion using `hmac.compare_digest` (returning `False` on invalid index/tree_size/proof length/tampered leaf or second-preimage attempt passing `left || right` as a leaf!).\n"
            "Output ONLY Python code defining `MerkleTree` and `verify_merkle_proof`."
        ),
        hidden_tests=r'''
import hashlib, hmac, inspect

def _h_leaf(d: bytes) -> bytes: return hashlib.sha256(b"\x00" + d).digest()
def _h_node(l: bytes, r: bytes) -> bytes: return hashlib.sha256(b"\x01" + l + r).digest()

def _mth(leaves: list[bytes]) -> bytes:
    n = len(leaves)
    if n == 0: return hashlib.sha256(b"").digest()
    if n == 1: return _h_leaf(leaves[0])
    k = 1 << ((n - 1).bit_length() - 1)
    return _h_node(_mth(leaves[:k]), _mth(leaves[k:]))

def test_rfc6962_roots_proofs_and_second_preimage_defense():
    src = inspect.getsource(solution)
    assert "compare_digest" in src
    assert MerkleTree([]).root() == hashlib.sha256(b"").digest()
    for n in (1, 2, 3, 5, 7, 8, 13):
        leaves = [f"leaf-{i}".encode() for i in range(n)]
        mt = MerkleTree(leaves)
        root = mt.root()
        assert root == _mth(leaves), f"Wrong RFC-6962 root for n={n}"
        for idx in range(n):
            pf = mt.proof(idx)
            assert verify_merkle_proof(leaves[idx], idx, n, pf, root) is True
            assert verify_merkle_proof(b"tampered", idx, n, pf, root) is False
            if n > 1:
                assert verify_merkle_proof(leaves[idx], (idx + 1) % n, n, pf, root) is False
    # Second-preimage attack: attacker submits internal node preimage (H_leaf(a) || H_leaf(b)) as a 1-leaf tree!
    l0, l1 = b"A", b"B"
    root2 = MerkleTree([l0, l1]).root()
    forged_leaf = _h_leaf(l0) + _h_leaf(l1)
    assert MerkleTree([forged_leaf]).root() != root2
    assert verify_merkle_proof(forged_leaf, 0, 1, [], root2) is False
''',
    ),
    Gauntlet2Problem(
        pid="H49_sec_rfc6238_totp_replay",
        track="sys_sec_nm",
        forge_mode="code",
        title="RFC-6238 TOTP Authenticator with Constant-Time Verify, Base32 & Anti-Replay Cache",
        prompt=(
            "Implement class `TOTPAuthenticator` obeying RFC-4226 (HOTP) & RFC-6238 (TOTP, HMAC-SHA1 default) with anti-replay:\n"
            "- `__init__(period: int = 30, digits: int = 6)`: validate `period > 0`, `6 <= digits <= 10` (else `ValueError`).\n"
            "- `generate(secret_b32: str, timestamp: int) -> str`:\n"
            "  * Decode Base32 `secret_b32` (ignore whitespace, case-insensitive, auto-pad missing `=` to multiple of 8; raise `ValueError` on empty or invalid base32 chars!).\n"
            "  * Counter `T = timestamp // period` encoded as 8-byte big-endian unsigned int (reject `timestamp < 0` with `ValueError`).\n"
            "  * Compute `mac = hmac.new(key, counter_bytes, hashlib.sha1).digest()`, dynamic offset `off = mac[-1] & 0x0F`, "
            "`code_int = int.from_bytes(mac[off:off+4], 'big') & 0x7FFFFFFF`, return zero-padded `f'{code_int % (10 ** digits):0{digits}d}'`.\n"
            "- `verify(user_id: str, secret_b32: str, code: str, now: int, window: int = 1) -> bool`:\n"
            "  * Check `code` is exact length `digits` ASCII digits and compare against counters `T + d` for `d in range(-window, window + 1)` (skipping negative counters) using `hmac.compare_digest`.\n"
            "  * **Anti-Replay Monotonic Watermark:** Track the highest accepted counter `last_counter[user_id]`. Reject (`False`) any counter `<= last_counter[user_id]` (blocks BOTH exact code reuse AND replaying an older counter from `-window` after a newer counter succeeded!). ONLY update `last_counter[user_id]` on a VALID match!\n"
            "Output ONLY Python code defining `TOTPAuthenticator`."
        ),
        hidden_tests=r'''
import inspect

def test_rfc6238_test_vectors_and_replay_protection():
    src = inspect.getsource(solution)
    assert "compare_digest" in src
    # Official RFC 6238 Appendix B SHA1 test vector: secret = b"12345678901234567890" -> base32 "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
    sec = "gezdgnbvgy3tqojqgezdgnbvgy3tqojq"  # lowercase, unpadded!
    auth8 = TOTPAuthenticator(period=30, digits=8)
    assert auth8.generate(sec, 59) == "94287082"
    assert auth8.generate(sec, 1111111109) == "07081804"
    assert auth8.generate(sec, 1234567890) == "89005924"

    # Drift window & anti-replay
    auth6 = TOTPAuthenticator(period=30, digits=6)
    c_prev = auth6.generate(sec, 970)   # T = 32
    c_now  = auth6.generate(sec, 1000)  # T = 33
    c_next = auth6.generate(sec, 1030)  # T = 34
    # Forged code must NOT poison replay watermark
    assert auth6.verify("alice", sec, "000000", now=1000, window=1) is False
    # Current code works once, then replay fails!
    assert auth6.verify("alice", sec, c_now, now=1000, window=1) is True
    assert auth6.verify("alice", sec, c_now, now=1005, window=1) is False
    # Older counter T=32 MUST ALSO fail after T=33 was accepted!
    assert auth6.verify("alice", sec, c_prev, now=1005, window=1) is False
    # Newer counter T=34 succeeds!
    assert auth6.verify("alice", sec, c_next, now=1005, window=1) is True
    try:
        auth6.generate("!!!NOT_BASE32!!!", 100)
        raise AssertionError("Expected ValueError for bad base32")
    except ValueError:
        pass
''',
    ),
    Gauntlet2Problem(
        pid="H50_sec_sql_ast_parameterizer",
        track="sys_sec_nm",
        forge_mode="code",
        title="SQLi-Proof Schema-Allowlisted Parameterized Query Builder (Live SQLite Verified)",
        prompt=(
            "Implement class `SafeQueryBuilder(schema: dict[str, set[str]])` mapping `{table_name: {allowed_columns}}`:\n"
            "- Validate every table and column name in `schema` against strict `^[A-Za-z_][A-Za-z0-9_]*$` (else `ValueError`).\n"
            "- `build_select(table: str, columns: list[str], filters: list[tuple[str, str, object]] | None = None, order_by: list[tuple[str, str]] | None = None, limit: int | None = None) -> tuple[str, list]`:\n"
            "  * `table` must exist in `schema`; `columns` must be non-empty and EVERY column in `schema[table]` (else `ValueError`). Quote identifiers safely with double quotes `\"col\"`.\n"
            "  * `filters`: list of `(col, op, val)` where `col in schema[table]` and `op` is strictly in `{'=', '!=', '<', '<=', '>', '>=', 'IN', 'IS NULL', 'IS NOT NULL'}` (else `ValueError`).\n"
            "    - ALL values MUST go into the `params` list using `?` placeholders — NEVER string-interpolated into SQL!\n"
            "    - For `'IN'`: `val` must be a list/tuple; if EMPTY `[]`, emit `1 = 0` (always false, valid SQLite syntax — NEVER broken `IN ()`!).\n"
            "    - For `'IS NULL'` / `'IS NOT NULL'`: emit `\"col\" IS NULL` with NO `?` parameter appended.\n"
            "  * `order_by`: list of `(col, direction)` where `col in schema[table]` and `direction.upper() in {'ASC', 'DESC'}` (else `ValueError` — blocks ORDER BY SQLi!).\n"
            "  * `limit`: if not `None`, MUST be `int` (NOT `bool`!) and `limit >= 0` (else `ValueError`), parameterized as `LIMIT ?` (or exact non-negative int check).\n"
            "Output ONLY Python code defining `SafeQueryBuilder`."
        ),
        hidden_tests=r'''
import sqlite3

def test_sqli_defense_empty_in_and_live_sqlite_execution():
    qb = SafeQueryBuilder({"users": {"id", "name", "role", "age"}})
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users(id INT, name TEXT, role TEXT, age INT)")
    conn.executemany("INSERT INTO users VALUES(?,?,?,?)", [
        (1, "alice", "admin", 30),
        (2, "bob", "user", 25),
        (3, "' OR 1=1 --", "user", 40),
    ])

    # Legit parameterized query with SQLi payload as literal data: matches ONLY row 3, NOT all rows!
    sql, params = qb.build_select("users", ["id", "name"], filters=[("name", "=", "' OR 1=1 --")])
    assert "OR 1=1" not in sql
    rows = conn.execute(sql, params).fetchall()
    assert rows == [(3, "' OR 1=1 --")]

    # Empty IN list -> valid SQL returning 0 rows!
    sql_empty, p_empty = qb.build_select("users", ["id"], filters=[("id", "IN", [])])
    assert conn.execute(sql_empty, p_empty).fetchall() == []

    # Non-empty IN + ORDER BY + LIMIT
    sql2, p2 = qb.build_select("users", ["id"], filters=[("id", "IN", [1, 2, 3]), ("age", ">=", 25)], order_by=[("age", "DESC")], limit=2)
    assert conn.execute(sql2, p2).fetchall() == [(3,), (1,)]

    # Injection attacks on table, column, op, order direction, limit MUST raise ValueError!
    for call in [
        lambda: qb.build_select("users; DROP TABLE users--", ["id"]),
        lambda: qb.build_select("users", ["id, (SELECT 1)"]),
        lambda: qb.build_select("users", ["id"], filters=[("id", "= 1 OR 1=1 --", 1)]),
        lambda: qb.build_select("users", ["id"], order_by=[("id", "ASC; DROP TABLE users")]),
        lambda: qb.build_select("users", ["id"], limit=True),
        lambda: qb.build_select("users", ["id"], limit=-1),
        lambda: qb.build_select("users", []),
    ]:
        try:
            res = call()
            raise AssertionError(f"SQLi BYPASS! Returned {res!r}")
        except ValueError:
            pass
''',
    ),
]

