# Local-Model-Killer Challenge Prompts

Copy-paste any prompt below into **Vanilla** (single-pass local model) vs **`/cortheon`** (or `cortheon solve`) on the **exact same frozen weights** and watch the difference live.

Every ground-truth answer below is mechanically verified in Python / SymPy / Z3. Keep the `<details>` answer collapsed while testing!

---

## 1. 25-Digit Exact Rational Markov Cover Time (`ReasonForge`)

**Why Vanilla fails:** A 112-state asymmetric Markov chain with a teleport reset produces a reduced rational with a **25-digit numerator and 24-digit denominator** — completely impossible to derive mentally. Cortheon v2's `B=4` PAL engine locks exact `2/4` consensus and atomically aborts the 4th GPU stream in **~36 seconds**.

```text
/cortheon 7 servers sit in a ring (0..6). A token starts at server 1 (server 1 counts as already visited). Each tick it moves clockwise (+1 mod 7) with probability 2/3 or counterclockwise (-1 mod 7) with probability 1/3, except whenever it is at server 0 it instead teleports uniformly at random to one of the 7 servers (including 0). Find the exact rational expected number of ticks E[T] until all 7 servers have been visited at least once.
```

<details>
<summary>Verified Exact Ground Truth</summary>

- **Exact rational:** `2630600114653531854763423 / 183488738276457059168800`
- **Decimal:** `14.336575308998444...`
- **`p mod 1000`:** `423` · **`q mod 1000`:** `800` · **`(p + q) mod 1000`:** `223`
- **Reachable non-absorbing states:** `112`

</details>

---

## 2. Length-14 Constrained Automaton Combinatorics (`ReasonForge`)

**Why Vanilla fails:** Coupled forbidden substrings (`AA`, `BB`, `ABC`) plus a global imbalance inequality (`#A > #B`) defeat closed-form inclusion-exclusion — single-pass local models hallucinate a recurrence or guess a 5-digit number.

```text
/cortheon How many strings of length 14 over the alphabet {A, B, C} satisfy ALL of the following? 1. No substring 'AA'. 2. No substring 'BB'. 3. No substring 'ABC'. 4. The total count of 'A' characters is STRICTLY GREATER than the total count of 'B' characters. Give the exact integer.
```

<details>
<summary>Verified Exact Ground Truth</summary>

- **Exact integer:** `54662`

</details>

---

## 3. Weighted Constrained 4-Coloring Optimization (`ReasonForge` / Z3)

**Why Vanilla fails:** Local models routinely miss either the odd-endpoint color-gap prohibition (`|c[u] - c[v]| != 3`) or the global color-cardinality cap (`<= 2` per color) while trying to minimize a weighted sum by hand.

```text
/cortheon Consider an undirected graph on 8 vertices labeled 0..7 with these 15 edges: (0,1), (0,2), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,6), (5,7), (6,7), (0,7), (2,6). Assign each vertex v in {0..7} an integer color c[v] in {1, 2, 3, 4} subject to ALL constraints: 1. Adjacent vertices must have different colors. 2. c[0] == 1. 3. For every edge (u, v) where BOTH u and v are odd, |c[u] - c[v]| != 3. 4. No color may be used more than 2 times across the 8 vertices. Minimize sum((v + 1) * c[v] for v in range(8)).
```

<details>
<summary>Verified Exact Ground Truth</summary>

- **Exact minimum:** `88`

</details>

---

## 4. Non-Backtracking Closed Walks on the Triangular Prism $C_3 \square K_2$ (`ReasonForge`)

**Why Vanilla fails:** Standard adjacency-matrix powers $A^{12}$ fail because of BOTH the non-backtracking constraint ($v_{i+2} \neq v_i$) AND the subset coverage constraint ($\ge 4$ distinct vertices visited).

```text
/cortheon Consider the 6-vertex triangular prism graph with vertices 0..5 and edges (0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5). How many 12-step closed walks start at vertex 0, end at vertex 0, never immediately backtrack (v[i+2] != v[i] for all i), and visit at least 4 distinct vertices? Give the exact integer and its remainder mod 1000.
```

<details>
<summary>Verified Exact Ground Truth</summary>

- **Exact integer:** `1018`
- **Mod 1000:** `18`

</details>

---

## 5. Constrained 4×4 Magic Square SMT Optimization (`ReasonForge` / Z3)

**Why Vanilla fails:** Searching $16! \approx 2.09 \times 10^{13}$ permutations with magic-sum, strict corner-ordering, and fixed diagonal entries is impossible without an SMT solver (`z3.Optimize`) or exact backtracking.

```text
/cortheon Assign distinct integers from {1..16} to a 4x4 grid g[r][c] (0-indexed) so all 4 rows, 4 columns, and 2 main diagonals sum to 34, the four corners satisfy g[0][0] < g[0][3] < g[3][0] < g[3][3], and g[0][0] = 1, g[1][1] = 11, g[2][2] = 6. Find the maximum possible value of 100 * g[0][1] + g[1][0].
```

<details>
<summary>Verified Exact Ground Truth</summary>

- **Exact maximum:** `1408`
- **Witness grid:** `[[1, 14, 15, 4], [8, 11, 10, 5], [12, 7, 6, 9], [13, 2, 3, 16]]`

</details>

---

## 6. Exact Rational Expression Parser (`CodeForge v5`, Oracle-Free)

**Why Vanilla fails:** Single-pass local models almost always break at least one of: floating-point drift on `0.1 + 0.2`, right-associative exponentiation `2^3^2 == 512`, unary precedence below exponentiation (`-2^2 == -4` AND `2^-2^2 == 1/16`), chained unary exponents (`2^--3 == 8`), or strict `ValueError` on `4^(1/2)` and `"1 2"`.

```text
/cortheon code Implement eval_expr(expr: str) -> Fraction (returning fractions.Fraction) for exact rational arithmetic: exact integers & decimals (0.1 + 0.2 == Fraction(3, 10), .5 == Fraction(1, 2)); operators +, -, *, /, ^ (right-associative: 2^3^2 == 512) and parentheses; unary +/- has LOWER precedence than ^ (-2^2 == -4, 2^-2^2 == Fraction(1, 16), 2^--3 == 8); 0^0 == Fraction(1, 1); 0 to a negative power, non-integer exponent, division by zero, or syntax error raises ValueError.
```

<details>
<summary>Verification Assertions</summary>

```python
from fractions import Fraction
assert eval_expr("0.1 + 0.2") == Fraction(3, 10)
assert eval_expr(".5 + .25") == Fraction(3, 4)
assert eval_expr("-2^2") == Fraction(-4, 1)
assert eval_expr("2^3^2") == Fraction(512, 1)
assert eval_expr("2^-2^2") == Fraction(1, 16)
assert eval_expr("2^--3") == Fraction(8, 1)
assert eval_expr("0^0") == Fraction(1, 1)
for bad in ["1/0", "0^-1", "4^(1/2)", "", "1 2", "."]:
    try:
        eval_expr(bad)
        raise AssertionError(f"Expected ValueError for {bad!r}")
    except ValueError:
        pass
```

</details>

---

## 7. Speed-of-Light Distributed Systems Physics Audit (`ThinkForge`, Oracle-Free)

**Why Vanilla fails:** Single-pass local models happily propose synchronous 3-region Raft/Paxos or global 2PC without checking the physics: `2 × 35ms = 70ms` already exceeds the `<40ms` p99 write budget, and unbudgeted replication/WAL amplification saturates disk/NIC limits. Cortheon's `ThinkForge` writes & runs a Python sizing calculator to prove the physical boundary first.

```text
/cortheon think Design a multi-region active-active 50,000 TPS payment ledger across 3 regions with 35ms inter-region RTT, strict <40ms p99 write commit latency, zero lost committed transactions (RPO=0) on single-AZ failure, and 4 KB average transaction payload with 3x synchronous WAL replication. Prove mathematically what is physically possible vs impossible under the RTT and NIC/NVMe budgets, and specify the exact commit protocol and failover behavior.
```

<details>
<summary>Key Mechanical Checks to Verify in Output</summary>

- Proves **cross-region synchronous consensus cannot meet `<40ms` p99** when inter-region RTT is `35ms` (`1 RTT = 35ms` leaves `<5ms` for serialization + fsync + tail jitter; `2 RTT = 70ms` strictly violates SLA).
- Separates **intra-region multi-AZ quorum** (e.g. 3 AZs within 1–2ms RTT for RPO=0 on single-AZ failure, meeting `<40ms` p99) from **asynchronous or partitioned-home-region cross-region replication** (explicitly stating the RPO/latency tradeoff on full-region loss).
- Computes exact write bandwidth: `50,000 × 4 KB = 200 MB/s` raw logical ingress → `600 MB/s` (`~4.8 Gbps` payload + protocol/WAL overhead) across 3× replication.

</details>

---

## 8. Coding-Agent Boss Fight: MVCC Serializable Snapshot Isolation (`OMP` / `Pi` `/cortheon build`)

**Why Vanilla fails:** Even inside a full coding harness with `bash` + `pytest`, single-pass local models routinely miss the **3-transaction Read-Only Transaction Anomaly (Fekete et al., 2004)** or drop `SIREAD` locks at commit time instead of retaining them until all overlapping concurrent transactions finish — failing randomized 4-tx serializability fuzzing.

```text
/cortheon build Build ssi_kv.py and test_ssi_kv.py: an in-memory MVCC key-value store implementing true Serializable Snapshot Isolation (SSI) with begin(), get(tx, k), scan(tx, lo, hi), put(tx, k, v), delete(tx, k), commit(tx), and abort(tx) using SIREAD key+range predicate locks and rw-antidependency dangerous-structure detection (Cahill/Fekete — NOT global locking and NOT blocking readers/writers). It MUST: (1) allow non-conflicting concurrent writes and non-dangerous rw overlaps to commit (prove liveness in tests), (2) catch 2-tx Write Skew, range-predicate Phantoms, AND the 3-tx Read-Only Transaction Anomaly (including when the pivot or third transaction commits before the others, requiring post-commit SIREAD retention), and (3) pass an adversarial fuzzer of 1,000 random interleaved 3- and 4-transaction schedules checked against an exact itertools.permutations serializability oracle verifying that every committed history's observed reads/scans and final DB state equal some true serial execution of the committed transactions with 0 violations.
```
