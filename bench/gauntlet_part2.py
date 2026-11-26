"""50-Test Frontier Gauntlet — Part 2: 24 Brutal Algo, Multi-Bug Debug & Security Tests (G21-G40, G47-G50)."""

from __future__ import annotations

from .gauntlet_part1 import GauntletProblem


PART2_PROBLEMS: list[GauntletProblem] = [
    # =========================================================================
    # TRACK 3: HARD ALGORITHMIC & EDGE-CASE CODING (G21-G30)
    # =========================================================================
    GauntletProblem(
        pid="G21_convex_hull_twice_area",
        track="algo_code",
        forge_mode="code",
        title="Exact Integer 2*Area & Strict Vertex Count of 2D Convex Hull",
        prompt=(
            "Implement `convex_hull_stats(points: list[tuple[int, int]]) -> tuple[int, int]`.\n"
            "Given integer 2D points `(x, y)` (arbitrary coordinates up to +/- 10^9, may contain duplicates, "
            "collinear boundary points, all-collinear points, or <3 points):\n"
            "Return `(twice_area, num_extreme_vertices)`:\n"
            "- `twice_area`: exact non-negative integer equal to `2 * Area(ConvexHull(points))` (0 if <3 non-collinear points).\n"
            "- `num_extreme_vertices`: exact count of STRICT corner vertices of the convex hull (collinear points lying on "
            "the interior of an edge do NOT count! Specifically: 0 if empty; 1 if all points identical; 2 if all points are "
            "collinear with >=2 distinct points; else the number of strictly convex corners).\n"
            "- Do NOT mutate `points`. Use exact integer arithmetic only (NO floats!). Handle N=5000 fast (<0.5s).\n"
            "Output ONLY Python code defining `convex_hull_stats`."
        ),
        hidden_tests=r'''
import random, time

def test_degenerate_and_collinear():
    assert convex_hull_stats([]) == (0, 0)
    assert convex_hull_stats([(5, -5), (5, -5)]) == (0, 1)
    assert convex_hull_stats([(0, 0), (2, 2), (5, 5), (2, 2)]) == (0, 2)
    assert convex_hull_stats([(0, 0), (0, 10), (0, 5)]) == (0, 2)

def test_boundary_collinear_and_huge_coords():
    # 10x10 square with many collinear points on edges & interior points
    pts = [(0,0),(5,0),(10,0),(10,3),(10,10),(4,10),(0,10),(0,7),(5,5),(3,2),(0,0)]
    snap = list(pts)
    assert convex_hull_stats(pts) == (200, 4)
    assert pts == snap
    # Huge triangle: exact integer cross product (no float precision loss!)
    M = 10**9
    assert convex_hull_stats([(-M, -M), (M, -M), (0, M)]) == (4 * M * M, 3)

def test_scale():
    rng = random.Random(7)
    pts = [(rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6)) for _ in range(5000)]
    t0 = time.perf_counter()
    area2, nv = convex_hull_stats(pts)
    assert area2 > 0 and nv >= 3 and (time.perf_counter() - t0) < 0.8
''',
    ),
    GauntletProblem(
        pid="G22_interval_set_manager",
        track="algo_code",
        forge_mode="code",
        title="Half-Open IntervalSet (Add/Remove/Query/TotalLength)",
        prompt=(
            "Implement class `IntervalSet` managing disjoint half-open integer intervals `[l, r)` (coordinates up to +/- 10^12):\n"
            "- `add(l: int, r: int) -> None`: adds `[l, r)`, merging overlapping or touching intervals (`[1,3)` and `[3,5)` -> `[1,5)`). Ignore if `l >= r`.\n"
            "- `remove(l: int, r: int) -> None`: removes `[l, r)` (splitting existing intervals as needed). Ignore if `l >= r`.\n"
            "- `contains_range(l: int, r: int) -> bool`: returns `True` iff EVERY point in `[l, r)` is covered (return `True` for empty range `l >= r`).\n"
            "- `total_length() -> int`: exact integer sum of lengths of all stored intervals.\n"
            "- `intervals() -> list[tuple[int, int]]`: sorted list of minimal disjoint merged `(l, r)` tuples.\n"
            "Output ONLY Python code defining `IntervalSet`."
        ),
        hidden_tests=r'''
import random

def test_edges_and_huge():
    s = IntervalSet()
    assert s.total_length() == 0 and s.intervals() == []
    assert s.contains_range(5, 5) is True
    s.add(10, 5)
    assert s.total_length() == 0
    s.add(-10**12, 10**12)
    s.remove(0, 1)
    assert s.intervals() == [(-10**12, 0), (1, 10**12)]
    assert s.total_length() == 2 * 10**12 - 1
    assert s.contains_range(-100, 0) is True
    assert s.contains_range(-100, 1) is False
    s.add(0, 1)
    assert s.intervals() == [(-10**12, 10**12)]

def test_differential_fuzz():
    rng = random.Random(42)
    s = IntervalSet()
    ref = set()
    for _ in range(300):
        op = rng.choice(["add", "rem", "qry"])
        l = rng.randint(-30, 30)
        r = l + rng.randint(-2, 15)
        if op == "add":
            s.add(l, r)
            if l < r: ref.update(range(l, r))
        elif op == "rem":
            s.remove(l, r)
            if l < r: ref.difference_update(range(l, r))
        else:
            exp = all(x in ref for x in range(l, r)) if l < r else True
            assert s.contains_range(l, r) is exp
        assert s.total_length() == len(ref)
''',
    ),
    GauntletProblem(
        pid="G23_thompson_nfa_regex",
        track="algo_code",
        forge_mode="code",
        title="Polynomial-Time Full-String Regex Engine (No Backtracking Blowup)",
        prompt=(
            "Implement `regex_match(pattern: str, text: str) -> bool` performing FULL-STRING matching in polynomial time "
            "(Thompson NFA / state-set simulation — NO catastrophic exponential backtracking, and do NOT use Python's `re` module!).\n"
            "Supported syntax:\n"
            "- Literal characters, `.` (matches any single character including newline),\n"
            "- Grouping `(...)` (nested arbitrarily, empty group `()` matches empty string),\n"
            "- Alternation `|` (lowest precedence inside groups, e.g. `ab|cd`, `(a|)` matches `'a'` or `''`),\n"
            "- Quantifiers `*` (0+), `+` (1+), `?` (0 or 1) applying to the preceding atom/group,\n"
            "- Backslash escape `\\x` matching literal `x` (including `\\.`, `\\*`, `\\(`, `\\\\`).\n"
            "- Raise `ValueError` on invalid syntax (mismatched parens, trailing lone `\\`, quantifier with no atom like `*a`).\n"
            "- Pathological inputs like `regex_match('(a+)+b', 'a'*45 + 'c')` MUST finish in < 0.1s.\n"
            "Output ONLY Python code defining `regex_match` (do NOT import `re`)."
        ),
        hidden_tests=r'''
import ast, inspect, time

def test_no_re_module():
    tree = ast.parse(inspect.getsource(solution))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "re" for a in node.names), "Must not import re"
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "re", "Must not import from re"

def test_semantics():
    assert regex_match("", "") is True
    assert regex_match("a|b", "a") is True and regex_match("a|b", "b") is True and regex_match("a|b", "ab") is False
    assert regex_match("(ab|cd)+e?", "abcdab") is True
    assert regex_match("(ab|cd)+e?", "abcde") is True
    assert regex_match("(a|)b", "b") is True and regex_match("(a|)b", "ab") is True
    assert regex_match(r"a\.b\*\\c", "a.b*\\c") is True
    assert regex_match(r"a\.b", "axb") is False
    assert regex_match("((a|b)*c)?d", "ababacd") is True
    assert regex_match("((a|b)*c)?d", "d") is True
    assert regex_match("((a|b)*c)?d", "ababad") is False

def test_polynomial_time_and_errors():
    t0 = time.perf_counter()
    assert regex_match("(a+)+b", "a" * 45 + "c") is False
    assert regex_match("(a|aa)*b", "a" * 40 + "c") is False
    assert (time.perf_counter() - t0) < 0.25
    for bad in ["(abc", "abc)", "*a", "+a", "a\\"]:
        try:
            regex_match(bad, "a")
            raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
''',
    ),
    GauntletProblem(
        pid="G24_rfc6902_json_patch",
        track="algo_code",
        forge_mode="code",
        title="Atomic RFC 6902 JSON Patch with Escaping & Rollback",
        prompt=(
            "Implement `apply_json_patch(doc, ops: list[dict])` obeying RFC 6902 (JSON Patch) & RFC 6901 (JSON Pointer):\n"
            "- ATOMIC & NON-MUTATING: return a deep-copied patched document; NEVER mutate the input `doc`, even if a later op fails!\n"
            "- Support all 6 ops: `add`, `remove`, `replace`, `move`, `copy`, `test`.\n"
            "- JSON Pointer rules:\n"
            "  * `\"\"` refers to the whole root document.\n"
            "  * Non-empty paths MUST start with `/`.\n"
            "  * Unescape `~1` -> `/` and `~0` -> `~` IN THE RIGHT ORDER (`~01` -> `~1`, NOT `/`!).\n"
            "  * Array indices must be non-negative integers WITHOUT leading zeros (except `'0'`; `'01'` or `'-1'` raises `ValueError`!), "
            "or `'-'` for appending in `add` ONLY (`remove`/`replace` with `'-'` raises `ValueError`).\n"
            "  * Out-of-bounds array index, missing parent path, missing key on `remove`/`replace`, or failed `test` raises `ValueError`.\n"
            "  * For `test`, strict JSON equality: `1 != True` and `0 != False` (`bool` is NOT equal to `int`!).\n"
            "  * For `move`, moving a location into a proper child of itself raises `ValueError`.\n"
            "Output ONLY Python code defining `apply_json_patch`."
        ),
        hidden_tests=r'''
import copy

def test_ops_and_pointer_escaping():
    doc = {"a/b": [1, 2], "c~d": {"~1": "tilde_one"}, "x": 1}
    snap = copy.deepcopy(doc)
    out = apply_json_patch(doc, [
        {"op": "test", "path": "/c~0d/~01", "value": "tilde_one"},
        {"op": "add", "path": "/a~1b/-", "value": 3},
        {"op": "add", "path": "/a~1b/1", "value": 99},
        {"op": "copy", "from": "/a~1b/0", "path": "/copied"},
        {"op": "move", "from": "/x", "path": "/moved"},
        {"op": "replace", "path": "/c~0d/~01", "value": True},
        {"op": "remove", "path": "/a~1b/0"},
    ])
    assert doc == snap, "Input doc was mutated!"
    assert out == {"a/b": [99, 2, 3], "c~d": {"~1": True}, "copied": 1, "moved": 1}

def test_strict_bool_atomicity_and_errors():
    doc = {"arr": [10, 20], "flag": True, "num": 1, "tree": {"sub": {}}}
    snap = copy.deepcopy(doc)
    for bad_ops in [
        [{"op": "test", "path": "/flag", "value": 1}],     # True != 1!
        [{"op": "test", "path": "/num", "value": True}],   # 1 != True!
        [{"op": "add", "path": "/arr/01", "value": 5}],    # leading zero index!
        [{"op": "remove", "path": "/arr/-"}],              # '-' invalid on remove!
        [{"op": "move", "from": "/tree", "path": "/tree/sub/x"}],  # move into child!
        [{"op": "add", "path": "/ok", "value": 1}, {"op": "remove", "path": "/missing"}],
    ]:
        try:
            apply_json_patch(doc, bad_ops)
            raise AssertionError(f"Expected ValueError for {bad_ops}")
        except ValueError:
            pass
    assert doc == snap
''',
    ),
    GauntletProblem(
        pid="G25_iterative_tarjan_scc_dag",
        track="algo_code",
        forge_mode="code",
        title="Iterative Tarjan SCC + Condensation DAG Maximum Weight Path (N=5000)",
        prompt=(
            "Implement `max_scc_path_weight(n: int, weights: list[int], edges: list[tuple[int, int]]) -> int`.\n"
            "Given a directed graph on vertices `0..n-1` with non-negative integer vertex weights `weights[v] >= 0` "
            "and directed edges `(u, v)` (may contain cycles, self-loops, parallel edges, disconnected components):\n"
            "A walk may start at ANY vertex and end at ANY vertex, collecting each distinct visited vertex's weight AT MOST ONCE "
            "(i.e., equivalent to contracting every Strongly Connected Component into its weight sum and finding the maximum-weight "
            "path in the resulting condensation DAG!). Return 0 if `n == 0`.\n"
            "CRITICAL: Must be ITERATIVE (NO recursion limit crash on a 5,000-node chain or 5,000-node cycle!) and run in < 0.5s for N=5000.\n"
            "Output ONLY Python code defining `max_scc_path_weight`."
        ),
        hidden_tests=r'''
import time

def test_basic_and_cycles():
    assert max_scc_path_weight(0, [], []) == 0
    assert max_scc_path_weight(1, [7], [(0, 0)]) == 7
    # SCC {0,1} weight 10 -> SCC {2} weight 5; separate SCC {3,4} weight 20
    assert max_scc_path_weight(5, [4, 6, 5, 9, 11], [(0,1),(1,0),(1,2),(3,4),(4,3)]) == 20
    # Diamond over SCCs with parallel edges
    assert max_scc_path_weight(4, [3, 5, 2, 4], [(0,1),(0,1),(0,2),(1,3),(2,3)]) == 12

def test_deep_chain_no_recursion_error():
    n = 5000
    w = [1] * n
    edges = [(i, i + 1) for i in range(n - 1)] + [(2500, 2400)]
    t0 = time.perf_counter()
    assert max_scc_path_weight(n, w, edges) == 5000
    assert (time.perf_counter() - t0) < 0.8
''',
    ),
    GauntletProblem(
        pid="G26_knuth_plass_wrap",
        track="algo_code",
        forge_mode="code",
        title="Exact Minimum-Raggedness Word Wrap DP (Cube Penalty + Tie-Breaking)",
        prompt=(
            "Implement `min_raggedness_wrap(words: list[str], max_width: int) -> tuple[int, list[str]]`.\n"
            "Pack `words` (preserving order) into lines joined by single spaces `' '` such that every line has length `<= max_width`.\n"
            "- If `words` is empty, return `(0, [])`.\n"
            "- If any single word has `len(word) > max_width`, raise `ValueError`.\n"
            "- Penalty of each line EXCEPT the final line is `(max_width - len(line)) ** 3`; the final line has penalty `0`.\n"
            "- Minimize:\n"
            "  1. Total raggedness penalty,\n"
            "  2. Break ties by FEWEST number of lines,\n"
            "  3. Break remaining ties by placing MORE words on earlier lines (i.e., maximizing the number of words on line 1, then line 2, etc.).\n"
            "Return `(min_penalty, lines)` where each element of `lines` is `' '.join(line_words)`.\n"
            "Output ONLY Python code defining `min_raggedness_wrap`."
        ),
        hidden_tests=r'''
def test_basic_and_non_greedy_dp():
    assert min_raggedness_wrap([], 10) == (0, [])
    # Classic non-greedy case: width=6, ["a", "bb", "cc", "dddd"]
    # Greedy: ["a bb", "cc", "dddd"] -> (6-4)^3 + (6-2)^3 + 0 = 8 + 64 = 72
    # Optimal: ["a", "bb cc", "dddd"] -> (6-1)^3? Wait: 125 + 1 = 126 (worse!)
    # Better non-greedy: width=10, ["aaa", "bb", "cc", "ddddd"]
    # Greedy: ["aaa bb cc", "ddddd"] -> (10-9)^3 = 1!
    # Try width=12, ["aaaa", "bbb", "cc", "ddddd", "eeeee"]
    # Line1 "aaaa bbb cc" (11, pen 1), Line2 "ddddd eeeee" (11, last -> 0) -> pen 1
    # Classic Knuth-Plass beat greedy: width=10, ["aaa", "bbb", "cc", "dd", "eeeee"]
    # Greedy: "aaa bbb cc"(10,0), "dd eeeee"(8,last=0) -> 0.
    # Let's test width=10, ["aaaa", "bbb", "cccc", "dddd"]
    # Greedy: ["aaaa bbb"(8 -> 2^3=8), "cccc dddd"(9 -> 0)] = 8
    # Alt: ["aaaa"(4 -> 216), "bbb cccc"(8 -> 8), "dddd"(0)] = 224
    # True non-greedy: width=12, ["aaa", "bbb", "ccc", "ddddd", "eeeee"]
    # Greedy: ["aaa bbb ccc"(11->1), "ddddd eeeee"(11->0)] -> 1
    # True non-greedy: width=11, ["aaa", "bb", "ccc", "dddd", "eeee"]
    # Greedy: ["aaa bb ccc"(10->1), "dddd eeee"(9->0)] = 1
    # Check width=7, ["a", "b", "c", "dddd", "eeee"] -> greedy ["a b c"(5->8), "dddd"(4->27), "eeee"(0)] = 35
    # vs ["a b"(3->64), "c dddd"(6->1), "eeee"(0)] = 65.
    # Check width=9, ["aaa", "bbb", "cc", "dddddd"] -> greedy ["aaa bbb"(7->8), "cc dddddd"(9->0)] = 8
    # Check exact tie-break and brute-force agreement!
    words = ["aa", "bb", "cc", "dd", "ee", "ff"]
    def brute(ws, W):
        best = None
        n = len(ws)
        def rec(i, cur_lines, cur_pen):
            nonlocal best
            if i == n:
                # remove last line penalty
                key = (cur_pen, len(cur_lines), tuple(-len(l.split()) for l in cur_lines))
                if best is None or key < best[0]:
                    best = (key, cur_pen, list(cur_lines))
                return
            for j in range(i + 1, n + 1):
                ln = " ".join(ws[i:j])
                if len(ln) > W: break
                p = 0 if j == n else (W - len(ln)) ** 3
                rec(j, cur_lines + [ln], cur_pen + p)
        rec(0, [], 0)
        return best[1], best[2]
    for ws, W in [
        (["a", "bb", "ccc", "dddd", "ee", "ff"], 8),
        (["hello", "my", "dear", "world", "of", "code", "now"], 11),
        (["aa", "aa", "aa", "aa"], 5),
        (["one"], 3),
    ]:
        assert min_raggedness_wrap(ws, W) == brute(ws, W)
    try:
        min_raggedness_wrap(["toolong"], 5)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
''',
    ),
    GauntletProblem(
        pid="G27_semver_dep_resolver",
        track="algo_code",
        forge_mode="code",
        title="Backtracking SemVer Dependency Resolver with Diamond Conflicts",
        prompt=(
            "Implement `resolve_deps(root_reqs: dict[str, str], registry: dict[str, dict[str, dict[str, str]]]) -> dict[str, str] | None`.\n"
            "- Versions are `'MAJOR.MINOR.PATCH'` non-negative integers (e.g. `'1.2.10'`, where `1.2.10 > 1.2.9`).\n"
            "- `registry[pkg][ver]` maps each dependency name to a constraint string (comma-separated clauses ALL of which must hold):\n"
            "  * `==X.Y.Z`, `>=X.Y.Z`, `<=X.Y.Z`, `>X.Y.Z`, `<X.Y.Z`, `!=X.Y.Z`,\n"
            "  * Caret `^X.Y.Z`: compatible SemVer range: if `X > 0`: `>=X.Y.Z, <(X+1).0.0`; "
            "else if `Y > 0`: `>=0.Y.Z, <0.(Y+1).0`; else: `>=0.0.Z, <0.0.(Z+1)`.\n"
            "- Find an assignment mapping EVERY transitively required package (and ONLY required packages — no unused packages!) "
            "to a single version in `registry` satisfying ALL root and transitive constraints (handle cyclic deps safely!).\n"
            "- If multiple valid solutions exist, prefer HIGHER versions when choosing for packages in sorted package-name order "
            "(i.e., maximize version tuple of alphabetically first root/required package, then next, etc.).\n"
            "- Return `None` if unsatisfiable or a required package is missing from `registry`.\n"
            "Output ONLY Python code defining `resolve_deps`."
        ),
        hidden_tests=r'''
def test_diamond_conflict_backtracking_and_caret():
    reg = {
        "app": {"2.0.0": {"libA": "^1.0.0", "libB": ">=1.0.0"}, "1.0.0": {"libA": "^1.0.0"}},
        "libA": {"1.2.0": {"core": ">=2.0.0"}, "1.1.0": {"core": "^1.0.0"}},
        "libB": {"1.0.0": {"core": "<2.0.0"}},
        "core": {"2.0.0": {}, "1.9.0": {}, "1.10.0": {}},
    }
    # app 2.0.0 needs libA ^1.0.0 and libB >=1.0.0; libA 1.2.0 conflicts with libB on core -> MUST backtrack to libA 1.1.0 + core 1.10.0!
    res = resolve_deps({"app": ">=1.0.0"}, reg)
    assert res == {"app": "2.0.0", "libA": "1.1.0", "libB": "1.0.0", "core": "1.10.0"}

def test_caret_zero_major_and_cycles_and_unsat():
    reg = {
        "a": {"1.0.0": {"b": "^0.2.3"}},
        "b": {"0.3.0": {}, "0.2.9": {"a": "^1.0.0"}, "0.2.2": {}},
    }
    assert resolve_deps({"a": "==1.0.0"}, reg) == {"a": "1.0.0", "b": "0.2.9"}
    assert resolve_deps({"a": "==1.0.0", "b": ">=0.3.0"}, reg) is None
    assert resolve_deps({"missing": ">=0.0.0"}, reg) is None
''',
    ),
    GauntletProblem(
        pid="G28_undo_redo_snapshot_buffer",
        track="algo_code",
        forge_mode="code",
        title="Transactional TextBuffer with Undo/Redo & Independent Snapshots",
        prompt=(
            "Implement class `TextBuffer` supporting:\n"
            "- `__init__(initial: str = '')`\n"
            "- `text() -> str`\n"
            "- `insert(pos: int, s: str) -> None`: inserts `s` at `0 <= pos <= len(text)` (out-of-range raises `IndexError`). "
            "If `s == ''`, no-op (do not push undo entry or clear redo stack). Non-empty edit clears redo stack.\n"
            "- `delete(pos: int, length: int) -> str`: deletes `length >= 0` chars starting at `pos` (`0 <= pos` and `pos + length <= len(text)`, "
            "else `IndexError`/`ValueError`). Returns deleted substring. If `length == 0`, no-op.\n"
            "- `undo() -> bool`: undoes last edit (returning `True`), or `False` if nothing to undo.\n"
            "- `redo() -> bool`: redoes last undone edit (returning `True`), or `False` if nothing to redo.\n"
            "- `snapshot() -> int`: saves an IMMUTABLE checkpoint of current text AND undo/redo history, returning an integer `snap_id`.\n"
            "- `restore(snap_id: int) -> None`: restores exact text AND undo/redo stacks to `snap_id` (future edits after restore "
            "MUST NOT corrupt the saved snapshot — restoring the same `snap_id` twice must give identical state!). Unknown `snap_id` raises `KeyError`.\n"
            "Output ONLY Python code defining `TextBuffer`."
        ),
        hidden_tests=r'''
def test_undo_redo_and_immutable_snapshots():
    tb = TextBuffer("hello")
    tb.insert(5, " world")
    assert tb.text() == "hello world"
    s1 = tb.snapshot()
    assert tb.delete(5, 6) == " world"
    assert tb.text() == "hello"
    assert tb.undo() is True and tb.text() == "hello world"
    assert tb.redo() is True and tb.text() == "hello"
    tb.insert(0, ">>")
    assert tb.redo() is False  # new edit cleared redo!
    tb.restore(s1)
    assert tb.text() == "hello world"
    assert tb.undo() is True and tb.text() == "hello"
    # Second restore of s1 must NOT have been mutated by undo above!
    tb.restore(s1)
    assert tb.text() == "hello world"
    assert tb.redo() is False
    try:
        tb.insert(99, "x")
        raise AssertionError("Expected IndexError")
    except IndexError:
        pass
    try:
        tb.restore(999)
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass
''',
    ),
    GauntletProblem(
        pid="G29_streaming_sliding_median",
        track="algo_code",
        forge_mode="code",
        title="O(log k) Exact Rational Sliding-Window Median (Lazy Dual Heaps)",
        prompt=(
            "Implement class `SlidingMedian`:\n"
            "- `__init__(k: int)`: window size `k >= 1` (raise `ValueError` if `k <= 0`).\n"
            "- `add(x: int) -> Fraction`: appends integer `x` to the sliding window (evicting oldest if window already had `k` items) "
            "and returns the EXACT median of the current window (size `min(count, k)`) as a `fractions.Fraction` "
            "(for even window size `2m`, exact average `(a + b) / 2` as `Fraction`!).\n"
            "- Must run in `O(log k)` amortized per `add()` (handle 20,000 additions with `k = 500` in < 0.8s!).\n"
            "Output ONLY Python code defining `SlidingMedian`."
        ),
        hidden_tests=r'''
from fractions import Fraction
import random, time

def test_exact_fuzz_and_perf():
    try:
        SlidingMedian(0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
    rng = random.Random(101)
    for k in (1, 2, 3, 4, 7):
        sm = SlidingMedian(k)
        win = []
        for _ in range(120):
            x = rng.randint(-10, 10)
            got = sm.add(x)
            assert isinstance(got, Fraction)
            win.append(x)
            if len(win) > k: win.pop(0)
            sw = sorted(win)
            m = len(sw)
            exp = Fraction(sw[m//2]) if m % 2 == 1 else Fraction(sw[m//2 - 1] + sw[m//2], 2)
            assert got == exp
    sm_big = SlidingMedian(500)
    t0 = time.perf_counter()
    for i in range(20000):
        res = sm_big.add((i * 37) % 1000 - 500)
    assert isinstance(res, Fraction) and (time.perf_counter() - t0) < 1.0
''',
    ),
    GauntletProblem(
        pid="G30_decaying_topk_counter",
        track="algo_code",
        forge_mode="code",
        title="Sliding-Window Top-K Counter with Boundary Expiry & Lex Tie-Break",
        prompt=(
            "Implement class `WindowTopK`:\n"
            "- `hit(key: str, ts: int, count: int = 1) -> None`: records `count` hits for `key` at integer timestamp `ts` "
            "(raise `ValueError` if `count <= 0`; timestamps may arrive OUT OF ORDER!).\n"
            "- `top_k(k: int, now: int, window: int) -> list[tuple[str, int]]`: returns up to `k` `(key, total_count)` pairs "
            "counting ONLY hits with `now - window < ts <= now` (strictly greater than `now - window`, less than or equal to `now` — "
            "so hits in the future `ts > now` OR hits at exact expiry `ts == now - window` are EXCLUDED!).\n"
            "- Sort by `total_count` DESCENDING, breaking ties by `key` ASCENDING (lexicographically smallest first).\n"
            "- If `k <= 0` or `window <= 0`, return `[]`.\n"
            "Output ONLY Python code defining `WindowTopK`."
        ),
        hidden_tests=r'''
def test_boundaries_out_of_order_and_ties():
    c = WindowTopK()
    c.hit("b", 10, 2)
    c.hit("a", 12, 2)
    c.hit("c", 5, 10)   # at now=15, window=10 -> 15-10=5 -> EXCLUDED!
    c.hit("future", 20, 99)  # future ts > now=15 -> EXCLUDED!
    c.hit("a", 8, 1)    # out-of-order timestamp! total for 'a' in (5,15] = 3
    c.hit("d", 15, 2)   # exact ts == now -> INCLUDED!
    assert c.top_k(3, now=15, window=10) == [("a", 3), ("b", 2), ("d", 2)]
    assert c.top_k(2, now=15, window=10) == [("a", 3), ("b", 2)]
    assert c.top_k(0, now=15, window=10) == []
    assert c.top_k(5, now=15, window=0) == []
    try:
        c.hit("x", 1, 0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
''',
    ),

    # =========================================================================
    # TRACK 4: ADVERSARIAL MULTI-BUG DEBUGGING (G31-G40)
    # =========================================================================
    GauntletProblem(
        pid="G31_debug_token_bucket",
        track="debug_code",
        forge_mode="code",
        title="Fix 5 Bugs in Exact Fraction TokenBucket Rate Limiter",
        buggy_starter='''class TokenBucket:
    def __init__(self, rate: float, capacity: float, clock=None):
        self.rate = rate
        self.capacity = capacity
        self.clock = clock or (lambda: 0.0)
        self.tokens = capacity
        self.last = self.clock()

    def allow(self, cost: float = 1.0) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens > cost:
            self.tokens -= cost
            return True
        return False
''',
        prompt=(
            "Fix ALL bugs in `TokenBucket` below.\n"
            "SPEC:\n"
            "1. `TokenBucket(rate, capacity, clock=None)`: raise `ValueError` if `rate < 0` or `capacity < 0`. "
            "Use exact `fractions.Fraction(str(x))` internally so fractional refills like `10 * 0.1 == 1.0` NEVER lose tokens to binary float drift!\n"
            "2. `allow(cost=1)`: raise `ValueError` if `cost < 0`; if `cost == 0`, return `True` without consuming tokens. "
            "Refill uses `max(0, now - last)` so backward clock jumps NEVER subtract tokens or move `last` backwards! "
            "Allow if `tokens >= cost` (inclusive `>=`, NOT `>`!).\n"
            "3. Add `available() -> Fraction`: returns current refilled token count cleanly.\n"
            "BUGGY CODE:\n```python\n"
            "class TokenBucket:\n"
            "    def __init__(self, rate: float, capacity: float, clock=None):\n"
            "        self.rate = rate\n"
            "        self.capacity = capacity\n"
            "        self.clock = clock or (lambda: 0.0)\n"
            "        self.tokens = capacity\n"
            "        self.last = self.clock()\n"
            "    def allow(self, cost: float = 1.0) -> bool:\n"
            "        now = self.clock()\n"
            "        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)\n"
            "        self.last = now\n"
            "        if self.tokens > cost:\n"
            "            self.tokens -= cost\n"
            "            return True\n"
            "        return False\n```\n"
            "Output ONLY fixed Python code defining `TokenBucket`."
        ),
        hidden_tests=r'''
from fractions import Fraction

def test_exact_refill_boundary_and_clock_skew():
    t = [0]
    tb = TokenBucket(rate=1, capacity=1, clock=lambda: t[0])
    assert tb.allow(1) is True  # exact == boundary!
    assert tb.allow(1) is False
    # 10 increments of 0.1s (float clock!) -> exact 1.0 token, zero float drift!
    for i in range(1, 11):
        t[0] = round(i * 0.1, 1)
        tb.available()
    assert tb.available() == Fraction(1, 1)
    assert tb.allow(1) is True
    # Clock goes backwards -> must NOT lose tokens or corrupt last!
    t[0] = 0.5
    assert tb.available() == Fraction(0, 1)
    t[0] = 2.0  # +1.0s after real last=1.0, NOT +1.5s!
    assert tb.available() == Fraction(1, 1)
    assert tb.allow(0) is True
    for fn in [lambda: TokenBucket(-1, 5), lambda: TokenBucket(1, -1), lambda: tb.allow(-0.5)]:
        try:
            fn(); raise AssertionError("Expected ValueError")
        except ValueError:
            pass
''',
    ),
    GauntletProblem(
        pid="G32_debug_consistent_hash_ring",
        track="debug_code",
        forge_mode="code",
        title="Fix 5 Bugs in ConsistentHashRing (Wrap-Around, Distinct Replicas, Remove)",
        buggy_starter='''import bisect, hashlib

class ConsistentHashRing:
    def __init__(self, replicas: int = 3):
        self.replicas = replicas
        self._keys = []
        self._ring = {}

    def _hash(self, s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        for i in range(self.replicas):
            h = self._hash(f"{node}#{i}")
            self._ring[h] = node
            bisect.insort(self._keys, h)

    def remove_node(self, node: str) -> None:
        for i in range(self.replicas):
            h = self._hash(f"{node}#{i}")
            del self._ring[h]
            self._keys.remove(h)

    def get_node(self, key: str):
        idx = bisect.bisect(self._keys, self._hash(key))
        return self._ring[self._keys[idx]]

    def get_nodes(self, key: str, count: int) -> list[str]:
        idx = bisect.bisect(self._keys, self._hash(key))
        return [self._ring[self._keys[(idx + i) % len(self._keys)]] for i in range(count)]
''',
        prompt=(
            "Fix ALL bugs in `ConsistentHashRing`:\n"
            "1. `replicas <= 0` raises `ValueError`. Adding an already-present `node` must be a clean idempotent no-op (no duplicate ring entries!).\n"
            "2. Removing a node not in the ring must raise `KeyError`.\n"
            "3. `get_node(key)`: returns `None` if ring is empty; uses `bisect_left` with wrap-around `% len(keys)` (never `IndexError` when hash > max ring key!).\n"
            "4. `get_nodes(key, count)`: returns up to `min(count, num_physical_nodes)` DISTINCT physical nodes in clockwise ring order (NEVER returns the same physical node twice because of virtual replicas!). Return `[]` if ring is empty or `count <= 0`.\n"
            "5. Keep ring keys strictly unique (if a hash collision occurs, handle cleanly so `remove_node` never breaks another node).\n"
            "BUGGY CODE:\n```python\n"
            "import bisect, hashlib\n"
            "class ConsistentHashRing:\n"
            "    def __init__(self, replicas: int = 3):\n"
            "        self.replicas = replicas\n"
            "        self._keys = []\n"
            "        self._ring = {}\n"
            "    def _hash(self, s: str) -> int:\n"
            "        return int(hashlib.md5(s.encode()).hexdigest(), 16)\n"
            "    def add_node(self, node: str) -> None:\n"
            "        for i in range(self.replicas):\n"
            "            h = self._hash(f'{node}#{i}')\n"
            "            self._ring[h] = node\n"
            "            bisect.insort(self._keys, h)\n"
            "    def remove_node(self, node: str) -> None:\n"
            "        for i in range(self.replicas):\n"
            "            h = self._hash(f'{node}#{i}')\n"
            "            del self._ring[h]\n"
            "            self._keys.remove(h)\n"
            "    def get_node(self, key: str):\n"
            "        idx = bisect.bisect(self._keys, self._hash(key))\n"
            "        return self._ring[self._keys[idx]]\n"
            "    def get_nodes(self, key: str, count: int) -> list[str]:\n"
            "        idx = bisect.bisect(self._keys, self._hash(key))\n"
            "        return [self._ring[self._keys[(idx + i) % len(self._keys)]] for i in range(count)]\n```\n"
            "Output ONLY fixed Python code defining `ConsistentHashRing`."
        ),
        hidden_tests=r'''
def test_ring_all_bugs():
    try:
        ConsistentHashRing(0); raise AssertionError("Expected ValueError")
    except ValueError:
        pass
    r = ConsistentHashRing(replicas=50)
    assert r.get_node("k") is None and r.get_nodes("k", 3) == []
    r.add_node("n1"); r.add_node("n1")  # idempotent!
    r.add_node("n2"); r.add_node("n3")
    # Wrap-around stress + distinct physical nodes!
    for i in range(200):
        k = f"key_{i}"
        top1 = r.get_node(k)
        top3 = r.get_nodes(k, 3)
        top5 = r.get_nodes(k, 5)
        assert top1 == top3[0]
        assert len(top3) == 3 and len(set(top3)) == 3
        assert top5 == top3
    assert r.get_nodes("k", 0) == []
    r.remove_node("n2")
    assert set(r.get_nodes("anything", 5)) == {"n1", "n3"}
    try:
        r.remove_node("n2"); raise AssertionError("Expected KeyError")
    except KeyError:
        pass
''',
    ),
    GauntletProblem(
        pid="G33_debug_order_stat_bst",
        track="debug_code",
        forge_mode="code",
        title="Fix Order-Statistic MultiSet BST (Insert/Delete/Kth/Rank)",
        buggy_starter='''class OrderStatBST:
    class _Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None
            self.size = 1

    def __init__(self):
        self.root = None

    def insert(self, key: int) -> None:
        def _ins(n):
            if not n: return self._Node(key)
            if key < n.key: n.left = _ins(n.left)
            else: n.right = _ins(n.right)
            n.size += 1
            return n
        self.root = _ins(self.root)

    def delete(self, key: int) -> bool:
        # BUG: broken subtree sizes and missing return flag!
        def _del(n, k):
            if not n: return None
            if k < n.key: n.left = _del(n.left, k)
            elif k > n.key: n.right = _del(n.right, k)
            else:
                if not n.left: return n.right
                if not n.right: return n.left
                succ = n.right
                while succ.left: succ = succ.left
                n.key = succ.key
                n.right = _del(n.right, succ.key)
            return n
        self.root = _del(self.root, key)
        return True

    def kth(self, k: int) -> int:
        n = self.root
        while n:
            ls = n.left.size if n.left else 0
            if k == ls: return n.key
            if k < ls: n = n.left
            else: k -= ls + 1; n = n.right
        raise IndexError("k out of range")
''',
        prompt=(
            "Fix ALL bugs in `OrderStatBST` (a multiset supporting duplicate integer keys!):\n"
            "1. `insert(key: int) -> None`: inserts one occurrence of `key` (duplicates allowed!).\n"
            "2. `delete(key: int) -> bool`: removes ONE occurrence of `key` and returns `True`, or returns `False` if absent (NEVER corrupting `size`!).\n"
            "3. `kth(k: int) -> int`: returns the 0-indexed `k`-th smallest element (raise `IndexError` if `k < 0` or `k >= len(self)`).\n"
            "4. `count_less(x: int) -> int`: returns exact number of stored elements strictly `< x`.\n"
            "5. `__len__() -> int`: returns total number of elements (including duplicates).\n"
            "BUGGY CODE:\n```python\n"
            "class OrderStatBST:\n"
            "    class _Node:\n"
            "        def __init__(self, key): self.key, self.left, self.right, self.size = key, None, None, 1\n"
            "    def __init__(self): self.root = None\n"
            "    def insert(self, key: int) -> None:\n"
            "        def _ins(n):\n"
            "            if not n: return self._Node(key)\n"
            "            if key < n.key: n.left = _ins(n.left)\n"
            "            else: n.right = _ins(n.right)\n"
            "            n.size += 1; return n\n"
            "        self.root = _ins(self.root)\n"
            "    def delete(self, key: int) -> bool:\n"
            "        def _del(n, k):\n"
            "            if not n: return None\n"
            "            if k < n.key: n.left = _del(n.left, k)\n"
            "            elif k > n.key: n.right = _del(n.right, k)\n"
            "            else:\n"
            "                if not n.left: return n.right\n"
            "                if not n.right: return n.left\n"
            "                succ = n.right\n"
            "                while succ.left: succ = succ.left\n"
            "                n.key = succ.key; n.right = _del(n.right, succ.key)\n"
            "            return n\n"
            "        self.root = _del(self.root, key); return True\n"
            "    def kth(self, k: int) -> int:\n"
            "        n = self.root\n"
            "        while n:\n"
            "            ls = n.left.size if n.left else 0\n"
            "            if k == ls: return n.key\n"
            "            if k < ls: n = n.left\n"
            "            else: k -= ls + 1; n = n.right\n"
            "        raise IndexError('k out of range')\n```\n"
            "Output ONLY fixed Python code defining `OrderStatBST`."
        ),
        hidden_tests=r'''
import random

def test_differential_fuzz():
    rng = random.Random(33)
    bst = OrderStatBST()
    ref = []
    assert len(bst) == 0 and bst.delete(5) is False
    for _ in range(300):
        op = rng.choice(["ins", "ins", "del", "kth", "lt"])
        x = rng.randint(-15, 15)
        if op == "ins":
            bst.insert(x); ref.append(x); ref.sort()
        elif op == "del":
            got = bst.delete(x)
            if x in ref:
                ref.remove(x); assert got is True
            else:
                assert got is False
        elif op == "kth" and ref:
            k = rng.randint(0, len(ref) - 1)
            assert bst.kth(k) == ref[k]
        elif op == "lt":
            assert bst.count_less(x) == sum(1 for v in ref if v < x)
        assert len(bst) == len(ref)
    for bad_k in (-1, len(ref)):
        try:
            bst.kth(bad_k); raise AssertionError("Expected IndexError")
        except IndexError:
            pass
''',
    ),
    GauntletProblem(
        pid="G34_debug_rfc4180_csv",
        track="debug_code",
        forge_mode="code",
        title="Fix RFC-4180 CSV Parser (Escaped Quotes, Embedded Newlines, CRLF, Errors)",
        buggy_starter='''def parse_csv(text: str) -> list[list[str]]:
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        rows.append([f.strip('"') for f in line.split(",")])
    return rows
''',
        prompt=(
            "Fix ALL bugs in `parse_csv(text: str) -> list[list[str]]` (do NOT import Python's `csv` module — write the state-machine parser!):\n"
            "1. Support both `\\r\\n` and `\\n` record delimiters; an empty string `\"\"` returns `[]`.\n"
            "2. Quoted fields `\"...\"`: may contain commas `,`, embedded newlines (`\\n` or `\\r\\n` preserved verbatim inside the field!), "
            "and escaped double-quotes `\"\"` -> literal `\"`.\n"
            "3. Preserve exact leading/trailing spaces inside unquoted AND quoted fields (do NOT strip whitespace!).\n"
            "4. Empty trailing fields: `'a,b,'` -> `[['a', 'b', '']]`; trailing final newline `'a,b\\n'` -> `[['a', 'b']]` (do NOT append an extra empty row after final newline, while `'a\\n\\nb'` has an empty row `['']` between!).\n"
            "5. Strict syntax errors: raise `ValueError` on unclosed quote (`'\"abc'`), stray quote inside unquoted field (`'a\"b'`), "
            "or trailing junk after closing quote (`'\"abc\"def'`).\n"
            "BUGGY CODE:\n```python\n"
            "def parse_csv(text: str) -> list[list[str]]:\n"
            "    if not text: return []\n"
            "    return [[f.strip('\"') for f in line.split(',')] for line in text.splitlines()]\n```\n"
            "Output ONLY fixed Python code defining `parse_csv` (do NOT import `csv`)."
        ),
        hidden_tests=r'''
import ast, inspect

def test_no_csv_import():
    tree = ast.parse(inspect.getsource(solution))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "csv" for a in node.names), "Must not import csv"
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "csv", "Must not import from csv"

def test_rfc4180_cases_and_errors():
    assert parse_csv("") == []
    assert parse_csv('a,"b,c","d""e"\r\n1,"line1\nline2",') == [["a", "b,c", 'd"e'], ["1", "line1\nline2", ""]]
    assert parse_csv(' a ," b "\n') == [[" a ", " b "]]
    assert parse_csv(" a ,b,\n") == [[" a ", "b", ""]]
    assert parse_csv('""\n') == [[""]]
    assert parse_csv("a\n\nb") == [["a"], [""], ["b"]]
    for bad in ['"unclosed', 'a"b,c', '"ok"junk,x']:
        try:
            r = parse_csv(bad); raise AssertionError(f"Expected ValueError for {bad!r}, got {r}")
        except ValueError:
            pass
''',
    ),
    GauntletProblem(
        pid="G35_debug_currency_arbitrage",
        track="debug_code",
        forge_mode="code",
        title="Fix Exact Rational Currency Arbitrage Cycle Detector",
        buggy_starter='''import math

def has_arbitrage(rates: list[tuple[str, str, str]]) -> bool:
    # rates: (src, dst, rate_str) e.g. ("USD", "EUR", "0.9")
    nodes = sorted({u for u, _, _ in rates} | {v for _, v, _ in rates})
    dist = {u: float("inf") for u in nodes}
    dist[nodes[0]] = 0.0
    for _ in range(len(nodes) - 1):
        for u, v, r in rates:
            w = -math.log(float(r))
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, r in rates:
        if dist[u] - math.log(float(r)) < dist[v]:
            return True
    return False
''',
        prompt=(
            "Fix ALL bugs in `has_arbitrage(rates: list[tuple[str, str, str]]) -> bool`:\n"
            "Each edge `(src, dst, rate_str)` converts 1 unit of `src` into `Fraction(rate_str)` units of `dst` (raise `ValueError` if rate <= 0).\n"
            "Return `True` iff there exists ANY directed cycle in ANY connected component whose EXACT rational product of rates is STRICTLY `> 1`!\n"
            "Bugs to fix:\n"
            "1. Empty `rates == []` must return `False` (never `IndexError`).\n"
            "2. Disconnected components: initializing only `nodes[0]` misses arbitrage in unreachable components!\n"
            "3. Floating-point `math.log` precision trap: a cycle whose exact product is `1` (e.g., `3/7 * 7/3` or `(1/3) * 3` or `1.0000000000000002` vs `1`) "
            "MUST NEVER trigger a false positive or false negative — use EXACT `Fraction` multiplicative relaxation (Bellman-Ford over `Fraction`, initializing `best[u] = Fraction(1)` for all nodes, relaxing `best[u] * r > best[v]`, or verifying extracted cycle product `> 1` exactly!).\n"
            "4. Parallel edges between same `(u, v)` and self-loops `(u, u, '1.0001')` vs `(u, u, '1')` must be handled accurately.\n"
            "BUGGY CODE:\n```python\n"
            "import math\n"
            "def has_arbitrage(rates):\n"
            "    nodes = sorted({u for u,_,_ in rates}|{v for _,v,_ in rates})\n"
            "    dist = {u: float('inf') for u in nodes}; dist[nodes[0]] = 0.0\n"
            "    for _ in range(len(nodes)-1):\n"
            "        for u,v,r in rates:\n"
            "            w = -math.log(float(r))\n"
            "            if dist[u]+w < dist[v]: dist[v] = dist[u]+w\n"
            "    return any(dist[u]-math.log(float(r)) < dist[v] for u,v,r in rates)\n```\n"
            "Output ONLY fixed Python code defining `has_arbitrage`."
        ),
        hidden_tests=r'''
def test_exact_rational_disconnected_and_self_loops():
    assert has_arbitrage([]) is False
    assert has_arbitrage([("A", "A", "1")]) is False
    assert has_arbitrage([("A", "A", "10000000000000001/10000000000000000")]) is True
    # Exact product == 1 (float log rounding trap!)
    assert has_arbitrage([("A", "B", "1/3"), ("B", "C", "1/7"), ("C", "A", "21")]) is False
    # Tiny real arbitrage invisible to float64!
    assert has_arbitrage([("A", "B", "1/3"), ("B", "C", "1/7"), ("C", "A", "210000000000000001/10000000000000000")]) is True
    # Disconnected component arbitrage (component 1 has none, component 2 has arbitrage!)
    assert has_arbitrage([("A", "B", "0.5"), ("B", "A", "1.0"), ("X", "Y", "2"), ("Y", "X", "0.51")]) is True
    try:
        has_arbitrage([("A", "B", "0")]); raise AssertionError("Expected ValueError")
    except ValueError:
        pass
''',
    ),
    GauntletProblem(
        pid="G36_debug_wildcard_trie",
        track="debug_code",
        forge_mode="code",
        title="Fix Wildcard Trie (Prefix-Safe Delete, Accurate Prefix Count, Duplicate Insert)",
        buggy_starter='''class WildcardTrie:
    def __init__(self):
        self.root = {"children": {}, "end": False, "prefix_count": 0}

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node["prefix_count"] += 1
            node = node["children"].setdefault(ch, {"children": {}, "end": False, "prefix_count": 0})
        node["end"] = True

    def delete(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node["children"]: return False
            node["prefix_count"] -= 1
            node = node["children"][ch]
        node["children"].clear()  # BUG: wipes longer words!
        node["end"] = False
        return True

    def count_prefix(self, prefix: str) -> int:
        node = self.root
        for ch in prefix:
            if ch not in node["children"]: return 0
            node = node["children"][ch]
        return node["prefix_count"]

    def match(self, pattern: str) -> list[str]:
        return []
''',
        prompt=(
            "Fix ALL bugs in `WildcardTrie` (stores a SET of distinct strings — inserting an existing word twice does NOT double-count it!):\n"
            "1. `insert(word: str) -> None`: idempotent set insert (including empty string `\"\"`!).\n"
            "2. `delete(word: str) -> bool`: removes `word` iff present (returns `True`/`False`), NEVER deleting longer words sharing `word` as prefix and NEVER corrupting `count_prefix` when `word` wasn't in the trie!\n"
            "3. `count_prefix(prefix: str) -> int`: exact number of distinct stored words starting with `prefix` (so `count_prefix(\"\")` == total stored words!).\n"
            "4. `match(pattern: str) -> list[str]`: returns sorted list of all stored words matching `pattern` where `.` matches ANY single character (exact length match!).\n"
            "BUGGY CODE:\n```python\n"
            "class WildcardTrie:\n"
            "    def __init__(self): self.root = {'children': {}, 'end': False, 'prefix_count': 0}\n"
            "    def insert(self, word: str) -> None:\n"
            "        node = self.root\n"
            "        for ch in word:\n"
            "            node['prefix_count'] += 1\n"
            "            node = node['children'].setdefault(ch, {'children': {}, 'end': False, 'prefix_count': 0})\n"
            "        node['end'] = True\n"
            "    def delete(self, word: str) -> bool:\n"
            "        node = self.root\n"
            "        for ch in word:\n"
            "            if ch not in node['children']: return False\n"
            "            node['prefix_count'] -= 1; node = node['children'][ch]\n"
            "        node['children'].clear(); node['end'] = False; return True\n"
            "    def count_prefix(self, prefix: str) -> int:\n"
            "        node = self.root\n"
            "        for ch in prefix:\n"
            "            if ch not in node['children']: return 0\n"
            "            node = node['children'][ch]\n"
            "        return node['prefix_count']\n"
            "    def match(self, pattern: str) -> list[str]: return []\n```\n"
            "Output ONLY fixed Python code defining `WildcardTrie`."
        ),
        hidden_tests=r'''
def test_trie_all_semantics():
    t = WildcardTrie()
    for w in ["app", "apple", "apply", "apt", "bat", "cat", ""]:
        t.insert(w)
    t.insert("app")  # duplicate insert must not double-count!
    assert t.count_prefix("") == 7
    assert t.count_prefix("app") == 3
    assert t.count_prefix("ap") == 4
    assert t.match(".at") == ["bat", "cat"]
    assert t.match("app..") == ["apple", "apply"]
    assert t.match("") == [""]
    # Deleting non-existent prefix must NOT corrupt counts!
    assert t.delete("ap") is False
    assert t.count_prefix("ap") == 4
    # Deleting 'app' must NOT delete 'apple'/'apply'!
    assert t.delete("app") is True
    assert t.delete("app") is False
    assert t.count_prefix("app") == 2
    assert t.match("app") == []
    assert t.match("app..") == ["apple", "apply"]
    assert t.delete("") is True and t.count_prefix("") == 5
''',
    ),
    GauntletProblem(
        pid="G37_debug_rollback_dsu",
        track="debug_code",
        forge_mode="code",
        title="Fix Rollback Union-Find (No Path Compression Corruption + Edge Counts)",
        buggy_starter='''class RollbackDSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.edges = [0] * n
        self.components = n
        self.history = []

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # BUG: path compression corrupts rollback!
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.history.append((rb, self.parent[rb], ra, self.size[ra]))
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True

    def checkpoint(self) -> int:
        return len(self.history)

    def rollback(self, cp: int) -> None:
        while len(self.history) > cp:
            rb, old_p, ra, old_s = self.history.pop()
            self.parent[rb] = old_p
            self.size[ra] = old_s
            self.components += 1
''',
        prompt=(
            "Fix ALL bugs in `RollbackDSU(n)` (Union-Find with exact checkpoints, rollbacks, AND per-component edge count tracking!):\n"
            "1. NO path compression (use union-by-size so `find` is `O(log n)` and rollback is 100% exact!).\n"
            "2. `union(a, b) -> bool`: adds an undirected edge between `a` and `b`.\n"
            "   - If `a` and `b` are ALREADY in the same component, still increment that component's edge count by 1 (and record it in history so rollback restores the edge count!), returning `False`.\n"
            "   - If in different components, merge them, `new_edges = edges[ra] + edges[rb] + 1`, decrement `components`, return `True`.\n"
            "3. `component_info(x: int) -> tuple[int, int]`: returns `(num_nodes_in_component, num_edges_in_component)`.\n"
            "4. `checkpoint() -> int` and `rollback(cp: int) -> None`: restores EXACT parent/size/edges/components state to `cp` (raise `ValueError` if `cp < 0` or `cp > current history len`).\n"
            "BUGGY CODE:\n```python\n"
            "class RollbackDSU:\n"
            "    def __init__(self, n: int):\n"
            "        self.parent, self.size, self.edges, self.components, self.history = list(range(n)), [1]*n, [0]*n, n, []\n"
            "    def find(self, x):\n"
            "        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])\n"
            "        return self.parent[x]\n"
            "    def union(self, a, b):\n"
            "        ra, rb = self.find(a), self.find(b)\n"
            "        if ra == rb: return False\n"
            "        self.history.append((rb, self.parent[rb], ra, self.size[ra]))\n"
            "        self.parent[rb] = ra; self.size[ra] += self.size[rb]; self.components -= 1; return True\n"
            "    def checkpoint(self): return len(self.history)\n"
            "    def rollback(self, cp):\n"
            "        while len(self.history) > cp:\n"
            "            rb, old_p, ra, old_s = self.history.pop()\n"
            "            self.parent[rb] = old_p; self.size[ra] = old_s; self.components += 1\n```\n"
            "Output ONLY fixed Python code defining `RollbackDSU`."
        ),
        hidden_tests=r'''
def test_rollback_exact_edges_and_components():
    d = RollbackDSU(5)
    assert d.union(0, 1) is True
    assert d.union(1, 2) is True
    cp1 = d.checkpoint()
    assert d.component_info(0) == (3, 2) and d.components == 3
    assert d.union(0, 2) is False  # same component, edge count MUST become 3!
    assert d.component_info(1) == (3, 3)
    assert d.union(3, 4) is True
    assert d.union(2, 3) is True
    assert d.component_info(4) == (5, 5) and d.components == 1
    d.rollback(cp1)
    assert d.components == 3
    assert d.component_info(0) == (3, 2)
    assert d.component_info(3) == (1, 0)
    assert d.find(0) != d.find(3)
    try:
        d.rollback(cp1 + 5); raise AssertionError("Expected ValueError")
    except ValueError:
        pass
''',
    ),
    GauntletProblem(
        pid="G38_debug_sql_three_valued_logic",
        track="debug_code",
        forge_mode="code",
        title="Fix Mini SQL Filter & Sort Engine (SQL 3-Valued NULL Logic + Precedence)",
        buggy_starter='''def query_rows(rows: list[dict], where=None, order_by=None) -> list[dict]:
    # Buggy: uses Python == on None, wrong NOT/AND/OR precedence, crashes sorting None!
    return list(rows)
''',
        prompt=(
            "Implement/fix `query_rows(rows: list[dict], where: str | None = None, order_by: list[tuple[str, str]] | None = None) -> list[dict]`.\n"
            "SPEC:\n"
            "1. `where` supports expressions with:\n"
            "   - Comparisons: `col = val`, `col != val`, `col < val`, `col <= val`, `col > val`, `col >= val` "
            "(where `val` is int, float, or single-quoted string `'foo'`),\n"
            "   - Null checks: `col IS NULL`, `col IS NOT NULL`,\n"
            "   - Boolean ops with strict SQL precedence: `(...)` > `NOT` > `AND` > `OR` (case-insensitive keywords!).\n"
            "   - **SQL Three-Valued Logic (`TRUE`, `FALSE`, `UNKNOWN`)**:\n"
            "     * ANY comparison (`=`, `!=`, `<`, `<=`, `>`, `>=`) where `row.get(col)` is `None` evaluates to `UNKNOWN`!\n"
            "     * `NOT UNKNOWN == UNKNOWN` (CRITICAL TRAP: `NOT (x = 5)` when `x is None` is `UNKNOWN`, NOT `TRUE`!!).\n"
            "     * `TRUE AND UNKNOWN == UNKNOWN`; `FALSE AND UNKNOWN == FALSE`; `TRUE OR UNKNOWN == TRUE`; `FALSE OR UNKNOWN == UNKNOWN`.\n"
            "     * A row is included in output ONLY IF the top-level `where` evaluates to `TRUE` (NOT `UNKNOWN` or `FALSE`).\n"
            "2. `order_by`: list of `(col, direction)` where `direction` is `'ASC'` or `'DESC'`. `None` values ALWAYS sort LAST (`NULLS LAST`) "
            "for BOTH `'ASC'` and `'DESC'`! Preserve relative order on ties (stable sort). Never mutate `rows`.\n"
            "Output ONLY Python code defining `query_rows`."
        ),
        hidden_tests=r'''
def test_three_valued_null_logic_and_precedence():
    rows = [
        {"id": 1, "dept": "eng", "sal": 100},
        {"id": 2, "dept": "eng", "sal": None},
        {"id": 3, "dept": "ops", "sal": 80},
        {"id": 4, "dept": None,  "sal": 120},
    ]
    # NOT (sal = 100) MUST exclude id=2 because sal is NULL -> UNKNOWN -> NOT UNKNOWN = UNKNOWN!
    r1 = query_rows(rows, where="NOT sal = 100", order_by=[("id", "ASC")])
    assert [r["id"] for r in r1] == [3, 4]
    # FALSE AND UNKNOWN -> FALSE; TRUE OR UNKNOWN -> TRUE!
    r2 = query_rows(rows, where="dept = 'eng' OR sal > 110 AND dept = 'ops'", order_by=[("sal", "DESC")])
    # Precedence: AND binds tighter than OR -> dept='eng' (1,2) OR (sal>110 AND dept='ops' -> none)
    # Sort sal DESC with NULLS LAST -> [1, 2]!
    assert [r["id"] for r in r2] == [1, 2]
    r3 = query_rows(rows, where="sal IS NULL OR dept IS NULL", order_by=[("sal", "ASC")])
    # id=4 (sal=120), id=2 (sal=None -> NULLS LAST!)
    assert [r["id"] for r in r3] == [4, 2]
''',
    ),
    GauntletProblem(
        pid="G39_debug_incremental_build_cache",
        track="debug_code",
        forge_mode="code",
        title="Fix Incremental DAG Build Engine (Diamond Transitive Invalidation + Cycle Check)",
        buggy_starter='''class BuildEngine:
    def __init__(self):
        self.targets = {}
        self.cache = {}

    def register(self, name: str, deps: list[str], fn) -> None:
        self.targets[name] = (deps, fn)

    def build(self, name: str):
        if name in self.cache:
            return self.cache[name]
        deps, fn = self.targets[name]
        vals = [self.build(d) for d in deps]
        res = fn(*vals)
        self.cache[name] = res
        return res
''',
        prompt=(
            "Fix/implement `BuildEngine` — an incremental build system with transitive dependency invalidation:\n"
            "- `set_input(name: str, value) -> None`: sets or updates a leaf input value. Updating an input to a NEW value "
            "(`!=` old value) MUST invalidate all transitively dependent cached targets (updating to an EQUAL value must NOT invalidate!).\n"
            "- `register(name: str, deps: list[str], fn) -> None`: registers a computed target depending on `deps` (each dep is an input or target). "
            "Re-registering an existing target invalidates it and its transitive dependents.\n"
            "- `build(name: str)`: computes & caches `name` (calling `fn(*dep_values)` AT MOST ONCE per target in a diamond DAG!).\n"
            "  * Raise `KeyError` on unknown target/dep, and `ValueError` on ANY dependency cycle (including self-loops).\n"
            "  * **Cutoff / Early-Stop Optimization:** If an upstream target `B` recomputes because input `A` changed, BUT `B` produces the EXACT SAME output value as before (`==`), "
            "downstream target `C` (which depends only on `B`) MUST NOT re-execute its `fn`! (Still returns cached `C` cleanly!)\n"
            "BUGGY CODE:\n```python\n"
            "class BuildEngine:\n"
            "    def __init__(self): self.targets, self.cache = {}, {}\n"
            "    def register(self, name, deps, fn): self.targets[name] = (deps, fn)\n"
            "    def build(self, name):\n"
            "        if name in self.cache: return self.cache[name]\n"
            "        deps, fn = self.targets[name]\n"
            "        res = fn(*[self.build(d) for d in deps])\n"
            "        self.cache[name] = res; return res\n```\n"
            "Output ONLY Python code defining `BuildEngine`."
        ),
        hidden_tests=r'''
def test_diamond_invalidation_cutoff_and_cycles():
    eng = BuildEngine()
    calls = {"b": 0, "c": 0, "d": 0}
    eng.set_input("x", 4)
    eng.register("b", ["x"], lambda x: (calls.__setitem__("b", calls["b"]+1) or x % 2))
    eng.register("c", ["b"], lambda b: (calls.__setitem__("c", calls["c"]+1) or b + 10))
    eng.register("d", ["b", "c"], lambda b, c: (calls.__setitem__("d", calls["d"]+1) or b * 100 + c))

    assert eng.build("d") == 10
    assert calls == {"b": 1, "c": 1, "d": 1}  # diamond: b called ONCE!
    # Same input value -> zero recomputation!
    eng.set_input("x", 4)
    assert eng.build("d") == 10 and calls == {"b": 1, "c": 1, "d": 1}
    # x=6 -> b recomputes (6%2 == 0 == old b!) -> CUTOFF: c and d MUST NOT recompute!
    eng.set_input("x", 6)
    assert eng.build("d") == 10 and calls == {"b": 2, "c": 1, "d": 1}
    # x=7 -> b changes 0->1 -> c and d recompute!
    eng.set_input("x", 7)
    assert eng.build("d") == 111 and calls == {"b": 3, "c": 2, "d": 2}
    # Cycle detection
    eng.register("p", ["q"], lambda q: q)
    eng.register("q", ["p"], lambda p: p)
    try:
        eng.build("p"); raise AssertionError("Expected ValueError for cycle")
    except ValueError:
        pass
''',
    ),
    GauntletProblem(
        pid="G40_debug_http_chunked_decoder",
        track="debug_code",
        forge_mode="code",
        title="Fix Streaming HTTP/1.1 Chunked Transfer-Encoding Decoder",
        buggy_starter='''class ChunkedDecoder:
    def __init__(self):
        self.buf = b""
        self.out = bytearray()
        self.done = False

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        pos = 0
        while True:
            idx = self.buf.find(b"\\r\\n", pos)
            if idx == -1: break
            size = int(self.buf[pos:idx], 16)
            if size == 0:
                self.done = True
                break
            start = idx + 2
            self.out.extend(self.buf[start:start+size])
            pos = start + size + 2
        return bytes(self.out)
''',
        prompt=(
            "Fix ALL bugs in streaming `ChunkedDecoder` (RFC 9112 HTTP/1.1 chunked transfer-encoding):\n"
            "- `feed(data: bytes) -> bytes`: incrementally accepts arbitrary byte splits (even 1 byte at a time!) and returns ONLY the newly decoded body bytes from this `feed` call.\n"
            "- `done: bool`: becomes `True` ONLY after the terminating `0` chunk AND the final trailing CRLF (including any optional trailer headers `Key: Value\\r\\n...\\r\\n`) are completely received!\n"
            "- `trailers: dict[str, str]`: stores any parsed trailer headers (lowercased header names, stripped values).\n"
            "- Support optional chunk extensions (`1A;ext=foo\\r\\n`), uppercase/lowercase hex sizes, and arbitrary byte framing splits.\n"
            "- Strict validation: raise `ValueError` on invalid hex size, negative/signed size (`+5`, `-1`), missing `\\r\\n` after chunk data, or feeding non-empty data after `done == True`.\n"
            "BUGGY CODE:\n```python\n"
            "class ChunkedDecoder:\n"
            "    def __init__(self): self.buf, self.out, self.done = b'', bytearray(), False\n"
            "    def feed(self, data: bytes) -> bytes:\n"
            "        self.buf += data; pos = 0\n"
            "        while True:\n"
            "            idx = self.buf.find(b'\\r\\n', pos)\n"
            "            if idx == -1: break\n"
            "            size = int(self.buf[pos:idx], 16)\n"
            "            if size == 0: self.done = True; break\n"
            "            start = idx + 2; self.out.extend(self.buf[start:start+size]); pos = start + size + 2\n"
            "        return bytes(self.out)\n```\n"
            "Output ONLY fixed Python code defining `ChunkedDecoder`."
        ),
        hidden_tests=r'''
def test_byte_by_byte_extensions_trailers_and_errors():
    raw = b"5;ext=1\r\nHello\r\n6\r\n World\r\n0;final=yes\r\nX-Check: abc\r\n\r\n"
    dec = ChunkedDecoder()
    parts = []
    for i in range(len(raw)):
        parts.append(dec.feed(raw[i:i+1]))
    assert b"".join(parts) == b"Hello World"
    assert dec.done is True
    assert dec.trailers == {"x-check": "abc"}
    try:
        dec.feed(b"extra"); raise AssertionError("Expected ValueError after done")
    except ValueError:
        pass
    for bad in [b"-1\r\n\r\n", b"+2\r\nab\r\n0\r\n\r\n", b"2\r\nabXX", b"ZZ\r\n"]:
        d2 = ChunkedDecoder()
        try:
            d2.feed(bad); raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
''',
    ),

    # =========================================================================
    # TRACK 5 (PART B): REAL SECURITY ENGINEERING & AUDIT FIXES (G47-G50)
    # =========================================================================
    GauntletProblem(
        pid="G47_sec_path_traversal_guard",
        track="systems_sec",
        forge_mode="code",
        title="Harden safe_resolve() Against Path Traversal, Prefix Collision & Encoding Bypass",
        prompt=(
            "Implement `safe_resolve(root: str | Path, user_path: str) -> Path` that safely resolves a user-supplied relative "
            "path inside `root` (using `pathlib.Path`) and raises `ValueError` on ANY traversal or escape attempt:\n"
            "Must BLOCK (raise `ValueError`):\n"
            "1. Null bytes (`\\x00`), empty/whitespace-only path, or Windows backslashes (`\\\\`).\n"
            "2. Absolute paths (`/etc/passwd`) or URL-encoded traversal (`%2e%2e`, `%2f`, double-encoded `%252e%252e` — iteratively URL-unquote until stable!).\n"
            "3. `..` escaping outside `root` OR symlink inside `root` pointing outside `root`.\n"
            "4. Classic prefix-collision bypass: if `root = '/tmp/app'`, resolving to `'/tmp/app_secret/x'` MUST be blocked "
            "(use `Path.is_relative_to`, NEVER `str(resolved).startswith(str(root))`!).\n"
            "Allow legitimate nested relative paths like `'docs/v1/readme.md'` or `'a/b/../c.txt'` that stay strictly inside `root`.\n"
            "Output ONLY Python code defining `safe_resolve`."
        ),
        hidden_tests=r'''
from pathlib import Path
import tempfile

def test_security_bypasses_and_legit():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root = base / "app"
        secret = base / "app_secret"
        root.mkdir(); secret.mkdir()
        (secret / "key.txt").write_text("SECRET")
        (root / "docs").mkdir()
        (root / "docs" / "ok.txt").write_text("ok")
        link = root / "evil_link"
        link.symlink_to(secret / "key.txt")

        assert safe_resolve(root, "docs/ok.txt") == (root / "docs" / "ok.txt").resolve()
        assert safe_resolve(root, "docs/sub/../ok.txt") == (root / "docs" / "ok.txt").resolve()

        for bad in [
            "", "   ", "/etc/passwd", "../app_secret/key.txt",
            "%2e%2e/app_secret/key.txt", "%252e%252e%252fapp_secret/key.txt",
            "docs/ok.txt\x00.png", "..\\app_secret\\key.txt", "evil_link",
        ]:
            try:
                r = safe_resolve(root, bad)
                raise AssertionError(f"SECURITY BYPASS! {bad!r} resolved to {r}")
            except ValueError:
                pass
''',
    ),
    GauntletProblem(
        pid="G48_sec_webhook_hmac_replay",
        track="systems_sec",
        forge_mode="code",
        title="Constant-Time HMAC Webhook Verifier with Replay-Window & Key Rotation",
        prompt=(
            "Implement class `WebhookVerifier`:\n"
            "- `__init__(secrets: list[str], max_skew_s: int = 300)`: supports ZERO-DOWNTIME key rotation across multiple active `secrets` "
            "(raise `ValueError` if `secrets` is empty or `max_skew_s < 0`).\n"
            "- `sign(payload: bytes, timestamp: int, nonce: str, secret: str) -> str`: computes `'v1=' + hmac_sha256_hex(secret, f'{timestamp}.{nonce}.'.encode() + payload)`.\n"
            "- `verify(payload: bytes, timestamp: int, nonce: str, sig_header: str, now: int) -> bool`:\n"
            "  * Reject (`False`) if `abs(now - timestamp) > max_skew_s` (both stale AND too-far-future timestamps!).\n"
            "  * Reject (`False`) if `nonce` is empty or was ALREADY accepted within the active window (replay attack!), "
            "AND prune expired nonces older than `now - max_skew_s`.\n"
            "  * Must check signature against ALL active `secrets` using `hmac.compare_digest` (NEVER `==`!), "
            "requiring exact `'v1='` prefix and 64-char hex digest. ONLY record `nonce` as used if verification SUCCEEDS "
            "(an attacker sending a forged signature with a victim's nonce MUST NOT poison/DoS the real webhook!).\n"
            "Output ONLY Python code defining `WebhookVerifier`."
        ),
        hidden_tests=r'''
import inspect

def test_hmac_rotation_replay_and_no_nonce_poisoning():
    src = inspect.getsource(solution)
    assert "compare_digest" in src
    v = WebhookVerifier(["old_secret", "new_secret"], max_skew_s=60)
    payload = b'{"event":"paid"}'
    # Forged signature with nonce 'n1' MUST fail AND NOT poison 'n1'!
    assert v.verify(payload, 1000, "n1", "v1=" + "0"*64, now=1000) is False
    # Real signature with old_secret succeeds!
    sig_old = v.sign(payload, 1000, "n1", "old_secret")
    assert v.verify(payload, 1000, "n1", sig_old, now=1010) is True
    # Replay of same nonce fails!
    assert v.verify(payload, 1000, "n1", sig_old, now=1015) is False
    # Key rotation: new_secret also succeeds!
    sig_new = v.sign(payload, 1020, "n2", "new_secret")
    assert v.verify(payload, 1020, "n2", sig_new, now=1020) is True
    # Skew violations (too old & too far future)
    sig_old_ts = v.sign(payload, 900, "n3", "new_secret")
    assert v.verify(payload, 900, "n3", sig_old_ts, now=1000) is False
    sig_fut = v.sign(payload, 1200, "n4", "new_secret")
    assert v.verify(payload, 1200, "n4", sig_fut, now=1000) is False
    # Wrong prefix / empty nonce
    assert v.verify(payload, 1020, "", v.sign(payload, 1020, "", "new_secret"), now=1020) is False
    assert v.verify(payload, 1020, "n5", sig_new[3:], now=1020) is False
''',
    ),
    GauntletProblem(
        pid="G49_sec_rbac_explicit_deny",
        track="systems_sec",
        forge_mode="code",
        title="IAM/RBAC Policy Evaluator (Explicit DENY Precedence + Safe Wildcards)",
        prompt=(
            "Implement `is_allowed(policies: list[dict], principal: str, action: str, resource: str, context: dict | None = None) -> bool`.\n"
            "Each policy dict has:\n"
            "- `effect`: `'ALLOW'` or `'DENY'` (invalid effect raises `ValueError`)\n"
            "- `principals`: list[str] (glob patterns using `*` and `?`)\n"
            "- `actions`: list[str] (glob patterns using `*` and `?`)\n"
            "- `resources`: list[str] (glob patterns using `*` and `?`)\n"
            "- optional `conditions`: dict[str, dict] where keys are condition ops (`'eq'`, `'ne'`, `'in'`) mapping `{ctx_key: expected}` "
            "(ALL conditions must hold against `context or {}`; missing context key fails `'eq'`/`'in'` and passes `'ne'` ONLY if strict? "
            "Rule: missing context key ALWAYS fails ANY condition check!).\n"
            "Evaluation rules (AWS IAM semantics):\n"
            "1. Default DENY: return `False` unless at least one matching policy has `effect == 'ALLOW'`.\n"
            "2. Explicit DENY ALWAYS wins: if ANY matching policy has `effect == 'DENY'`, return `False` regardless of `ALLOW` policies!\n"
            "3. Glob matching `*` (0+ chars) and `?` (1 char) MUST be polynomial-time (no ReDoS on `'*a*a*a*a*a*a*b'`!).\n"
            "Output ONLY Python code defining `is_allowed`."
        ),
        hidden_tests=r'''
import time

def test_explicit_deny_conditions_and_no_redos():
    policies = [
        {"effect": "ALLOW", "principals": ["user:*"], "actions": ["s3:Get*", "s3:List*"], "resources": ["arn:bucket/*"]},
        {"effect": "DENY",  "principals": ["user:bob"], "actions": ["*"], "resources": ["arn:bucket/secret/*"]},
        {"effect": "ALLOW", "principals": ["svc:*"], "actions": ["db:Write"], "resources": ["arn:db/prod"],
         "conditions": {"eq": {"mfa": True}, "in": {"env": ["prod", "stage"]}}},
    ]
    assert is_allowed(policies, "user:alice", "s3:GetObject", "arn:bucket/secret/a") is True
    assert is_allowed(policies, "user:bob", "s3:GetObject", "arn:bucket/secret/a") is False  # explicit DENY wins!
    assert is_allowed(policies, "user:bob", "s3:GetObject", "arn:bucket/public/a") is True
    assert is_allowed(policies, "user:alice", "s3:DeleteObject", "arn:bucket/a") is False   # default DENY!
    assert is_allowed(policies, "svc:pay", "db:Write", "arn:db/prod", {"mfa": True, "env": "prod"}) is True
    assert is_allowed(policies, "svc:pay", "db:Write", "arn:db/prod", {"mfa": False, "env": "prod"}) is False
    assert is_allowed(policies, "svc:pay", "db:Write", "arn:db/prod", {"env": "prod"}) is False  # missing ctx key!
    # ReDoS resistance
    t0 = time.perf_counter()
    evil = [{"effect": "ALLOW", "principals": ["*"], "actions": ["*"], "resources": ["*a*a*a*a*a*a*a*b"]}]
    assert is_allowed(evil, "u", "a", "a" * 50 + "c") is False
    assert (time.perf_counter() - t0) < 0.2
''',
    ),
    GauntletProblem(
        pid="G50_sec_sandboxed_template",
        track="systems_sec",
        forge_mode="code",
        title="SSTI-Hardened Sandboxed Template Renderer (Blocks Dunder/Format Injection)",
        prompt=(
            "Implement `render_template(template: str, context: dict) -> str` safely WITHOUT `eval()`, `exec()`, or `str.format()`:\n"
            "Supported syntax:\n"
            "- Literal text and placeholders `{{ expr }}` (with optional whitespace inside `{{ ... }}`).\n"
            "- `expr` is a dotted path `a.b.c` optionally followed by pipe filters `| upper`, `| lower`, `| trim`, `| default('fallback')` "
            "(or double-quoted `default(\"fallback\")`), chainable left-to-right (e.g. `{{ user.name | trim | upper }}`).\n"
            "- Dotted traversal resolves dict keys (preferred if mapping) OR public attributes on objects.\n"
            "- If a path segment is missing or resolves to `None`: if a subsequent `default(...)` filter is present, replace with the default; "
            "otherwise render empty string `''`.\n"
            "- **HARD SECURITY RULES (SSTI Defense):**\n"
            "  * ANY path segment starting with `_` (including `__class__`, `__globals__`, `__subclasses__`, `__init__`, `_private`) MUST raise `ValueError`!\n"
            "  * Unknown filter names, function call syntax `()`, brackets `[]`, or unclosed `{{` MUST raise `ValueError`!\n"
            "  * Values rendered from `context` MUST NOT be re-parsed as templates (e.g., if `context['x'] == '{{ secret }}'`, output literal `'{{ secret }}'`!).\n"
            "Output ONLY Python code defining `render_template`."
        ),
        hidden_tests=r'''
from types import SimpleNamespace

def test_rendering_filters_and_ssti_defense():
    ctx = {
        "user": SimpleNamespace(name="  alice  ", role=None, _secret="NOPE"),
        "nested": {"a": {"b": "Hello"}},
        "inject": "{{ secret }}",
        "secret": "TOP_SECRET",
    }
    assert render_template("Hi {{ user.name | trim | upper }}!", ctx) == "Hi ALICE!"
    assert render_template("Role={{ user.role | default('guest') | upper }}", ctx) == "Role=GUEST"
    assert render_template("Miss={{ user.missing | default(\"none\") }}", ctx) == "Miss=none"
    assert render_template("Raw={{ inject }}", ctx) == "Raw={{ secret }}"  # NO double-evaluation!
    assert render_template("Path={{ nested.a.b | lower }}", ctx) == "Path=hello"

    for evil in [
        "{{ user.__class__.__mro__ }}",
        "{{ user._secret }}",
        "{{ nested.__init__.__globals__ }}",
        "{{ user.name | eval }}",
        "{{ user.name() }}",
        "{{ nested['a'] }}",
        "Hello {{ unclosed",
    ]:
        try:
            out = render_template(evil, ctx)
            raise AssertionError(f"SSTI BYPASS! {evil!r} -> {out!r}")
        except ValueError:
            pass
''',
    ),
]
