"""Real Multi-File Software Engineering Benchmark for DevForge vs Vanilla Flash-Next.

Creates an isolated 6-file git repository (`ledgerflow/`) and tests TWO real-world dev tasks:
  R1_cross_file_bugs:  Fix 4 interlocking cross-file bugs across accounts.py, ledger.py, escrow.py, reports.py
  R2_multifile_feature: Implement a brand-new cross-file Atomic Batch Settlement + Fee-Split feature

Both Vanilla Flash-Next (budgeted single-stream) and DevForge run against fresh git clones
and are graded by comprehensive HIDDEN pytest suites they never see!
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.dev_agent import run_dev_forge
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.repo_index import apply_patch_ops, build_repo_map, parse_patch_ops, read_numbered_slice


BASE_REPO_FILES: dict[str, str] = {
    "ledgerflow/__init__.py": "",
    "ledgerflow/models.py": '''from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


@dataclass
class Account:
    account_id: str
    currency: str
    balance: float = 0.0
    frozen: bool = False


@dataclass
class Entry:
    entry_id: str
    from_account: str
    to_account: str
    amount: float
    currency: str
    memo: str = ""


@dataclass
class EscrowHold:
    hold_id: str
    payer: str
    payee: str
    amount: float
    currency: str
    status: str = "OPEN"  # OPEN, RELEASED, REFUNDED
''',
    "ledgerflow/accounts.py": '''from decimal import Decimal
from .models import Account


class AccountStore:
    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}

    def create(self, account_id: str, currency: str, initial_balance=0) -> Account:
        if account_id in self._accounts:
            raise ValueError(f"Duplicate account: {account_id}")
        bal = float(initial_balance)
        if bal < 0:
            raise ValueError("Initial balance cannot be negative")
        acc = Account(account_id=account_id, currency=currency, balance=bal)
        self._accounts[account_id] = acc
        return acc

    def get(self, account_id: str) -> Account:
        if account_id not in self._accounts:
            raise KeyError(f"Unknown account: {account_id}")
        return self._accounts[account_id]

    def all_accounts(self) -> list[Account]:
        return list(self._accounts.values())
''',
    "ledgerflow/ledger.py": '''from .accounts import AccountStore
from .models import Entry


class LedgerEngine:
    def __init__(self, store: AccountStore, escrow_mgr=None) -> None:
        self.store = store
        self.escrow_mgr = escrow_mgr
        self.entries: list[Entry] = []
        self._seq = 0

    def transfer(self, from_id: str, to_id: str, amount, memo: str = "") -> Entry:
        if from_id == to_id:
            raise ValueError("Cannot transfer to self")
        amt = float(amount)
        if amt <= 0:
            raise ValueError("Amount must be positive")
        src = self.store.get(from_id)
        dst = self.store.get(to_id)
        if src.currency != dst.currency:
            raise ValueError("Currency mismatch")
        if src.frozen or dst.frozen:
            raise ValueError("Frozen account")
        # BUG: ignores active escrow holds on src, uses binary float arithmetic!
        if src.balance < amt:
            raise ValueError("Insufficient funds")
        src.balance -= amt
        dst.balance += amt
        self._seq += 1
        entry = Entry(
            entry_id=f"E{self._seq}",
            from_account=from_id,
            to_account=to_id,
            amount=amt,
            currency=src.currency,
            memo=memo,
        )
        self.entries.append(entry)
        return entry
''',
    "ledgerflow/escrow.py": '''from .accounts import AccountStore
from .models import EscrowHold


class EscrowManager:
    def __init__(self, store: AccountStore) -> None:
        self.store = store
        self.holds: dict[str, EscrowHold] = {}

    def create_hold(self, hold_id: str, payer: str, payee: str, amount) -> EscrowHold:
        if hold_id in self.holds:
            raise ValueError(f"Duplicate hold: {hold_id}")
        amt = float(amount)
        if amt <= 0:
            raise ValueError("Hold amount must be positive")
        src = self.store.get(payer)
        dst = self.store.get(payee)
        if src.currency != dst.currency:
            raise ValueError("Currency mismatch")
        if src.balance < amt:
            raise ValueError("Insufficient funds for hold")
        hold = EscrowHold(hold_id=hold_id, payer=payer, payee=payee, amount=amt, currency=src.currency)
        self.holds[hold_id] = hold
        return hold

    def held_amount(self, account_id: str) -> float:
        return sum(h.amount for h in self.holds.values() if h.payer == account_id and h.status == "OPEN")

    def release(self, hold_id: str, ledger) -> None:
        hold = self.holds[hold_id]
        # BUG: does not check if already RELEASED/REFUNDED (double payout!), and transfer may fail if balance held!
        hold.status = "RELEASED"
        ledger.transfer(hold.payer, hold.payee, hold.amount, memo=f"escrow:{hold_id}")

    def refund(self, hold_id: str) -> None:
        hold = self.holds[hold_id]
        hold.status = "REFUNDED"
''',
    "ledgerflow/reports.py": '''from .accounts import AccountStore
from .ledger import LedgerEngine


def currency_totals(store: AccountStore) -> dict[str, float]:
    totals: dict[str, float] = {}
    for acc in store.all_accounts():
        totals[acc.currency] = totals.get(acc.currency, 0.0) + acc.balance
    return totals


def account_statement(ledger: LedgerEngine, account_id: str) -> list:
    return [e for e in ledger.entries if e.from_account == account_id or e.to_account == account_id]
''',
    "tests/test_basic.py": '''from ledgerflow.accounts import AccountStore
from ledgerflow.escrow import EscrowManager
from ledgerflow.ledger import LedgerEngine
from ledgerflow.reports import currency_totals


def test_basic_transfer():
    store = AccountStore()
    store.create("alice", "USD", "100.00")
    store.create("bob", "USD", "50.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)
    e = eng.transfer("alice", "bob", "25.00")
    assert e.entry_id == "E1"
    assert float(store.get("alice").balance) == 75.0
    assert float(store.get("bob").balance) == 75.0
    assert float(currency_totals(store)["USD"]) == 150.0
''',
}


TASK_R1 = """Fix ALL cross-file accounting, Decimal precision, and escrow bugs in the `ledgerflow` package so it obeys these invariants:
1. Exact Decimal money everywhere (`models.py`, `accounts.py`, `ledger.py`, `escrow.py`, `reports.py`):
   - All balances, entry amounts, hold amounts, `held_amount()`, and `currency_totals()` MUST use exact `decimal.Decimal` (accepting `Decimal`, `str`, or `int` inputs cleanly — convert via `Decimal(str(x))`, NEVER binary `float`!).
   - Reject `NaN`/`Infinity` and non-positive transfer/hold amounts with `ValueError`.
2. Available balance vs held balance (`accounts.py`, `ledger.py`, `escrow.py`):
   - `AccountStore` / `EscrowManager`: an `OPEN` escrow hold reserves funds from the payer's available balance (`available = balance - held_amount(account_id)`).
   - `create_hold()` MUST check `available_balance >= amount` (NOT raw `balance >= amount`!), reject `payer == payee`, and reject frozen payer/payee with `ValueError`.
   - `LedgerEngine.transfer()` MUST check `available_balance >= amount` (accounting for `self.escrow_mgr.held_amount(from_id)` when `self.escrow_mgr` is present!).
3. Escrow state machine & atomicity (`escrow.py`, `ledger.py`):
   - `release(hold_id, ledger)` MUST raise `KeyError` for unknown `hold_id` and `ValueError` if `hold.status != "OPEN"` (preventing double-release or releasing a refunded hold!).
   - Crucially, releasing an `OPEN` hold MUST be able to spend the EXACT funds reserved by THAT hold (even if `available_balance == 0` because the hold reserved all funds!), AND if the underlying transfer fails (e.g., payee account became frozen after hold creation!), the hold MUST remain `"OPEN"` (atomic rollback — never leave hold marked `RELEASED` when transfer failed!).
   - `refund(hold_id)` MUST raise `KeyError` for unknown `hold_id` and `ValueError` if `hold.status != "OPEN"`.
4. Add `available_balance(account_id: str) -> Decimal` helper on `LedgerEngine` (returns `balance - held_amount` if `escrow_mgr` else `balance`), and keep all existing tests passing!
"""

HIDDEN_TESTS_R1 = '''from decimal import Decimal
import pytest
from ledgerflow.accounts import AccountStore
from ledgerflow.escrow import EscrowManager
from ledgerflow.ledger import LedgerEngine
from ledgerflow.reports import currency_totals


def test_exact_decimal_no_float_drift():
    store = AccountStore()
    store.create("a", "USD", "0.30")
    store.create("b", "USD", "0.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)
    eng.transfer("a", "b", "0.10")
    eng.transfer("a", "b", "0.20")
    assert isinstance(store.get("a").balance, Decimal)
    assert isinstance(store.get("b").balance, Decimal)
    assert store.get("a").balance == Decimal("0.00")
    assert store.get("b").balance == Decimal("0.30")
    totals = currency_totals(store)
    assert isinstance(totals["USD"], Decimal)
    assert totals["USD"] == Decimal("0.30")


def test_escrow_hold_reserves_available_balance_and_releases_cleanly():
    store = AccountStore()
    store.create("alice", "USD", "100.00")
    store.create("bob", "USD", "0.00")
    store.create("charlie", "USD", "0.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)

    esc.create_hold("h1", "alice", "bob", "80.00")
    assert eng.available_balance("alice") == Decimal("20.00")
    assert esc.held_amount("alice") == Decimal("80.00")

    # Cannot create second hold exceeding available (20.00)!
    with pytest.raises(ValueError):
        esc.create_hold("h2", "alice", "charlie", "30.00")

    # Cannot transfer more than available (20.00)!
    with pytest.raises(ValueError):
        eng.transfer("alice", "charlie", "25.00")

    # Can transfer exact remaining available 20.00 -> available is now 0.00
    eng.transfer("alice", "charlie", "20.00")
    assert eng.available_balance("alice") == Decimal("0.00")

    # Now releasing h1 MUST succeed using the 80.00 reserved by h1!
    esc.release("h1", eng)
    assert store.get("alice").balance == Decimal("0.00")
    assert store.get("bob").balance == Decimal("80.00")
    assert store.get("charlie").balance == Decimal("20.00")
    assert esc.held_amount("alice") == Decimal("0.00")

    # Double release or refund after release MUST raise ValueError!
    with pytest.raises(ValueError):
        esc.release("h1", eng)
    with pytest.raises(ValueError):
        esc.refund("h1")


def test_escrow_atomic_rollback_when_payee_frozen():
    store = AccountStore()
    store.create("alice", "USD", "50.00")
    store.create("bob", "USD", "0.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)

    esc.create_hold("h1", "alice", "bob", "50.00")
    store.get("bob").frozen = True
    with pytest.raises(ValueError):
        esc.release("h1", eng)
    # Hold MUST still be OPEN and funds untouched!
    assert esc.holds["h1"].status == "OPEN"
    assert store.get("alice").balance == Decimal("50.00")
    assert store.get("bob").balance == Decimal("0.00")
    assert esc.held_amount("alice") == Decimal("50.00")

    # Refunding now succeeds and frees available balance
    esc.refund("h1")
    assert esc.holds["h1"].status == "REFUNDED"
    assert eng.available_balance("alice") == Decimal("50.00")
    with pytest.raises(ValueError):
        esc.refund("h1")


def test_validation_edges():
    store = AccountStore()
    store.create("alice", "USD", "10.00")
    store.create("bob", "USD", "10.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)
    with pytest.raises(ValueError):
        esc.create_hold("h_self", "alice", "alice", "1.00")
    with pytest.raises(ValueError):
        eng.transfer("alice", "bob", Decimal("NaN"))
    with pytest.raises(KeyError):
        esc.release("missing", eng)
'''


TASK_R2 = """Implement a new cross-file feature: **Atomic Batch Settlements with Fee-Splitting & Idempotency** across `ledgerflow/models.py`, `ledgerflow/ledger.py`, and `ledgerflow/reports.py` (while also preserving Decimal precision and escrow hold checks):

1. In `ledgerflow/models.py`, add:
   - `@dataclass class SettlementLeg`: `from_account: str`, `to_account: str`, `amount: Decimal`, `memo: str = ""`
   - `@dataclass class BatchResult`: `batch_id: str`, `entries: list[Entry]`, `total_volume: Decimal`, `total_fees: Decimal`, `deduplicated: bool = False`
2. In `ledgerflow/ledger.py`, add `execute_batch(self, batch_id: str, legs: list[SettlementLeg], fee_rate=Decimal("0"), fee_account: str | None = None) -> BatchResult`:
   - **Idempotency:** If `batch_id` was already executed successfully before, DO NOT re-execute or mutate balances; return a `BatchResult` with the original `entries`, `total_volume`, `total_fees`, and `deduplicated=True`! (Failed/rolled-back batches MUST NOT poison `batch_id` — retrying the same `batch_id` after fixing balances must succeed!)
   - **Validation & Fee calculation:**
     * `legs` must be non-empty (empty raises `ValueError`).
     * `fee_rate` converted via `Decimal(str(fee_rate))` must satisfy `0 <= fee_rate < 1`; if `fee_rate > 0`, `fee_account` must be provided and exist (else raise `ValueError`/`KeyError`).
     * For each leg with positive Decimal `amount`: compute `fee = (amount * fee_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)` and `net_to_recipient = amount - fee` (exact conservation: payer pays `amount`, recipient gets `net_to_recipient`, `fee_account` gets `fee` if `fee > 0`!).
     * Check currency match across `from_account`, `to_account`, and `fee_account` (when `fee > 0`), and reject frozen accounts or `from_account == to_account`.
     * All debits must respect available balance (taking `escrow_mgr.held_amount` into account!) — NOTE: an earlier leg in the SAME batch that credits account X *can* fund a later leg from account X (sequential execution within the batch!), while any failure on leg k MUST **atomically roll back ALL balance changes, entries, and sequence numbers** from legs 0..k-1!
   - **Recorded entries:** For each leg, append the net transfer entry (`from_account -> to_account` for `net_to_recipient`, or skip if `net_to_recipient == 0`), and if `fee > 0`, append a fee entry (`from_account -> fee_account` for `fee`, memo `f"fee:{batch_id}"`). `total_volume` = sum of gross leg `amount`s; `total_fees` = sum of `fee`s.
3. In `ledgerflow/reports.py`, add `settlement_audit(ledger: LedgerEngine) -> dict`:
   - Returns `{"batch_count": int, "total_volume": Decimal, "total_fees": Decimal, "entry_count": int}` across all successfully committed (non-deduplicated) batches and total ledger entries.
4. Ensure `AccountStore`, `LedgerEngine`, `EscrowManager`, and `currency_totals` use exact `Decimal` and respect escrow holds so all existing tests AND batch settlement invariants pass!
"""

HIDDEN_TESTS_R2 = '''from decimal import Decimal
import pytest
from ledgerflow.accounts import AccountStore
from ledgerflow.escrow import EscrowManager
from ledgerflow.ledger import LedgerEngine
from ledgerflow.models import SettlementLeg
from ledgerflow.reports import currency_totals, settlement_audit


def test_atomic_batch_with_fees_and_chained_liquidity():
    store = AccountStore()
    store.create("alice", "USD", "100.00")
    store.create("bob", "USD", "0.00")
    store.create("carol", "USD", "0.00")
    store.create("treasury", "USD", "0.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)

    # Leg 1: alice -> bob 100.00 (2.5% fee = 2.50 -> bob gets 97.50, treasury gets 2.50)
    # Leg 2: bob -> carol 50.00 funded by leg 1! (2.5% fee = 1.25 -> carol gets 48.75, treasury gets 1.25)
    legs = [
        SettlementLeg("alice", "bob", Decimal("100.00"), "pay1"),
        SettlementLeg("bob", "carol", Decimal("50.00"), "pay2"),
    ]
    res = eng.execute_batch("B1", legs, fee_rate=Decimal("0.025"), fee_account="treasury")
    assert res.deduplicated is False
    assert res.total_volume == Decimal("150.00")
    assert res.total_fees == Decimal("3.75")
    assert store.get("alice").balance == Decimal("0.00")
    assert store.get("bob").balance == Decimal("47.50")
    assert store.get("carol").balance == Decimal("48.75")
    assert store.get("treasury").balance == Decimal("3.75")
    assert currency_totals(store)["USD"] == Decimal("100.00")  # exact money conservation!

    # Idempotency: replaying B1 returns deduplicated=True without double-charging!
    res2 = eng.execute_batch("B1", legs, fee_rate=Decimal("0.025"), fee_account="treasury")
    assert res2.deduplicated is True
    assert res2.total_volume == Decimal("150.00")
    assert store.get("treasury").balance == Decimal("3.75")

    audit = settlement_audit(eng)
    assert audit["batch_count"] == 1
    assert audit["total_volume"] == Decimal("150.00")
    assert audit["total_fees"] == Decimal("3.75")


def test_atomic_rollback_respecting_escrow_and_retry_same_batch_id():
    store = AccountStore()
    store.create("alice", "USD", "100.00")
    store.create("bob", "USD", "50.00")
    store.create("carol", "USD", "0.00")
    store.create("treasury", "USD", "0.00")
    esc = EscrowManager(store)
    eng = LedgerEngine(store, escrow_mgr=esc)

    # Hold 40.00 on bob -> bob available is only 10.00!
    esc.create_hold("h1", "bob", "carol", "40.00")

    bad_legs = [
        SettlementLeg("alice", "carol", Decimal("30.00")),  # would succeed
        SettlementLeg("bob", "carol", Decimal("20.00")),    # MUST fail due to escrow hold!
    ]
    orig_entries = len(eng.entries)
    with pytest.raises(ValueError):
        eng.execute_batch("B_FAIL", bad_legs, fee_rate=Decimal("0.01"), fee_account="treasury")

    # COMPLETE atomic rollback!
    assert store.get("alice").balance == Decimal("100.00")
    assert store.get("bob").balance == Decimal("50.00")
    assert store.get("carol").balance == Decimal("0.00")
    assert store.get("treasury").balance == Decimal("0.00")
    assert len(eng.entries) == orig_entries

    # Same batch_id must NOT be poisoned — retrying after refunding hold succeeds!
    esc.refund("h1")
    ok_res = eng.execute_batch("B_FAIL", bad_legs, fee_rate=Decimal("0.01"), fee_account="treasury")
    assert ok_res.deduplicated is False
    assert ok_res.total_volume == Decimal("50.00")
    assert ok_res.total_fees == Decimal("0.50")
    assert currency_totals(store)["USD"] == Decimal("150.00")
'''


def _create_temp_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="ff_ledgerflow_"))
    for rel, content in BASE_REPO_FILES.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bench@flashforge.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "FlashForge"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial ledgerflow"], cwd=repo, check=True, capture_output=True)
    return repo


def _grade_repo(repo: Path, hidden_test_src: str) -> tuple[bool, str]:
    ht = repo / "tests" / "test_hidden_grade.py"
    ht.write_text(hidden_test_src, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_basic.py", "tests/test_hidden_grade.py"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=20,
    )
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    summary = out.splitlines()[-1] if out else f"exit={proc.returncode}"
    return (proc.returncode == 0), summary


async def _run_vanilla_repo(client: TensorFoldClient, repo: Path, task: str, thinking_budget: int | None) -> tuple[bool, str, float, dict[str, int]]:
    client.reset_stats()
    t0 = time.perf_counter()
    rmap = build_repo_map(repo)
    slices = [read_numbered_slice(repo, f) for f in sorted(BASE_REPO_FILES.keys())]
    prompt = (
        f"TASK:\n{task}\n\n{rmap}\n\n" + "\n\n".join(slices) + "\n\n"
        "Output ONLY multi-file patch blocks using:\n"
        "<<<FILE: relative/path.py>>>\n<complete file>\n<<<END_FILE>>>\n"
        "or <<<EDIT: relative/path.py>>>\n<<<SEARCH>>>\n...\n<<<REPLACE>>>\n...\n<<<END_EDIT>>>"
    )
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": prompt}],
            tag="vanilla_repo",
            max_tokens=8192,
            temperature=0.6,
            seed=42,
            enable_thinking=True,
            thinking_budget=thinking_budget,
            loop_guard=True,
        )
    )
    wall = time.perf_counter() - t0
    ops = parse_patch_ops(res.content)
    ok_apply, err, _ = apply_patch_ops(repo, ops)
    if not ok_apply:
        return False, f"PATCH_ERR: {err[:180]}", wall, client.stats_dict()
    return True, "applied", wall, client.stats_dict()


async def main() -> None:
    base_url = "http://127.0.0.1:8080/v1"
    tasks = [
        ("R1_cross_file_ledger_bugs", TASK_R1, HIDDEN_TESTS_R1),
        ("R2_atomic_batch_settlement", TASK_R2, HIDDEN_TESTS_R2),
    ]
    verify_cmd = f"{sys.executable} -m pytest -q tests/test_basic.py"

    # Planted control check: base repo must pass basic tests and FAIL both hidden suites!
    ctrl_repo = _create_temp_repo()
    try:
        base_ok = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_basic.py"], cwd=ctrl_repo, capture_output=True).returncode == 0
        assert base_ok, "Base repo basic test should pass!"
        for tid, _, htests in tasks:
            ok_h, msg_h = _grade_repo(ctrl_repo, htests)
            assert not ok_h, f"Planted control FAILED for {tid}: unpatched repo passed hidden suite! ({msg_h})"
            print(f"[REPO CONTROL OK] {tid} unpatched repo caught -> {msg_h}")
    finally:
        shutil.rmtree(ctrl_repo, ignore_errors=True)

    results: list[dict[str, Any]] = []
    async with TensorFoldClient(base_url=base_url) as vc, FlashForge(ForgeConfig(base_url=base_url, num_streams=4)) as ff:
        for tid, task_text, hidden_tests in tasks:
            print(f"\n=== MULTI-FILE REPO TASK: {tid} ===")
            # 1. Vanilla (budgeted 1500 so it doesn't just spiral empty)
            r_van = _create_temp_repo()
            try:
                ok_a, msg_a, wall_v, stats_v = await _run_vanilla_repo(vc, r_van, task_text, thinking_budget=1500)
                if ok_a:
                    v_pass, v_detail = _grade_repo(r_van, hidden_tests)
                else:
                    v_pass, v_detail = False, msg_a
                print(f"  Vanilla(B=1500): {'PASS' if v_pass else 'FAIL':4s} | {wall_v:5.1f}s | tok={stats_v['completion_tokens']:5d} | {v_detail}")
            finally:
                shutil.rmtree(r_van, ignore_errors=True)

            # 2. DevForge (full multi-file worktree racing + diff red-teamer + graveyard Wave 2)
            r_dev = _create_temp_repo()
            try:
                d_res = await run_dev_forge(
                    client=ff.client,
                    skills=ff.skills,
                    memory=ff.memory,
                    repo_dir=r_dev,
                    task=task_text,
                    verify_cmd=verify_cmd,
                    num_streams=4,
                    apply_winner=True,
                )
                if d_res.ok:
                    d_pass, d_detail = _grade_repo(r_dev, hidden_tests)
                else:
                    d_pass, d_detail = False, f"DevForge verify failed: {d_res.verify_output[:160]}"
                print(
                    f"  DevForge       : {'PASS' if d_pass else 'FAIL':4s} | {d_res.wall_s:5.1f}s | "
                    f"tok={d_res.token_stats['completion_tokens']:5d} (calls={d_res.token_stats['calls']}) | "
                    f"files={d_res.touched_files} rt_kills={len(d_res.redteam_kills)} w2={d_res.wave2_triggered} | {d_detail}"
                )
            finally:
                shutil.rmtree(r_dev, ignore_errors=True)

            results.append({
                "task": tid,
                "vanilla_budgeted": {"passed": v_pass, "wall_s": round(wall_v, 2), "detail": v_detail, "stats": stats_v},
                "dev_forge": {
                    "passed": d_pass,
                    "wall_s": round(d_res.wall_s, 2),
                    "detail": d_detail,
                    "winner": d_res.winner_name,
                    "files": d_res.touched_files,
                    "redteam_kills": len(d_res.redteam_kills),
                    "wave2": d_res.wave2_triggered,
                    "stats": d_res.token_stats,
                },
            })

    out = Path("bench/results/repo_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved multi-file repo benchmark to {out}")


if __name__ == "__main__":
    asyncio.run(main())
