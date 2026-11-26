"""Frontier-Replica Proving Ground (PG1, PG2, PG3) — Live on Qwen3.8-Flash-Next-TFQ3 (local server).

Compares a standard single-stream Vanilla ReAct agent (growing raw chat history, no B=3 forks,
no BrainState compaction, no Independent Adversarial Referee) against `ReplicaAgent` on the
EXACT SAME frozen 3-bit `Qwen3.8-Flash-Next-TFQ3` weights across 3 real-world frontier challenges:

  PG1: Greenfield Crash-Safe CRC32 WAL Transactional KV Database (`minidb/`) from an EMPTY workspace!
  PG2: Vague Production Incident Investigation & 3-Root-Cause Fix in an 8-file Async Job Queue repo (`pulsequeue/`)!
  PG3: Deep Read-Only Systems Investigation on the REAL `tf-pr-flashnext-tfq3` Zig/Metal TensorFold codebase!
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.referee import verify_citations_on_disk
from flash_forge.replica_agent import AGENT_STEP_SCHEMA, ReplicaAgent, ReplicaConfig, _AGENT_SYSTEM_PROMPT, _summarize_action
from flash_forge.tools import AgentToolbelt


# =============================================================================
# PG1: GREENFIELD CRASH-SAFE CRC32 WAL TRANSACTIONAL KV DATABASE (EMPTY DIR!)
# =============================================================================

PG1_MISSION = """You are starting in an EMPTY workspace. Design, architect, build, and thoroughly test a clean, modular Python package `minidb/` implementing a **persistent, crash-safe, CRC32-checksummed Write-Ahead-Logged (WAL) Transactional Key-Value Database**.

REQUIRED PUBLIC API (importable as `from minidb import MiniDB`):
1. `MiniDB(db_dir: str | Path)`:
   - Creates `db_dir` if needed; opens/recovers existing WAL file(s) inside `db_dir`.
   - Supports `close()` and context manager (`with MiniDB(db_dir) as db:`).
2. Core Operations (auto-committed durably to WAL with `flush` + `os.fsync`):
   - `put(key: str, value: Any) -> None` (values are JSON-serializable: int, float, str, bool, list, dict, None; falsy values like `0`, `False`, `""`, `[]` MUST be stored and returned accurately!)
   - `get(key: str, default: Any = None) -> Any` (returns stored value if key exists, else `default`)
   - `delete(key: str) -> bool` (returns `True` if key existed and was deleted, `False` if absent; durably logs tombstone if deleted)
   - `scan(start: str | None = None, end: str | None = None) -> list[tuple[str, Any]]` (returns live `(key, value)` pairs sorted ascending by key, matching `start <= key < end` where `None` means unbounded)
3. Atomic Transactions (`tx = db.begin()`):
   - `tx.put(key, value)`, `tx.get(key, default=None)`, `tx.delete(key) -> bool`
   - Read-your-own-writes inside `tx` (including reading a key deleted in `tx` returns `default`), while `db.get(key)` does NOT see uncommitted writes until `tx.commit()`!
   - `tx.commit() -> None`: appends ONE atomic CRC32-checksummed multi-op commit frame to the WAL (fsynced) and applies all ops to in-memory state.
   - `tx.rollback() -> None`: discards buffered writes cleanly.
   - Any operation (`put`, `get`, `delete`, `commit`, `rollback`) on an already committed or rolled-back transaction MUST raise `RuntimeError` or `ValueError`.
4. Crash Safety & Tail Corruption Tolerance:
   - Every WAL record MUST be framed with length + `zlib.crc32` checksum!
   - On reopen (`MiniDB(db_dir)`), replay all valid committed WAL records in order.
   - If a simulated power-loss crash leaves a **truncated, partially written, or bit-flipped corrupt record / trailing garbage bytes** at the tail of the WAL file, `MiniDB(db_dir)` MUST recover EVERY valid committed record before the corrupt tail, truncate away the torn tail cleanly, and accept new writes that survive subsequent reopens!
5. Compaction:
   - `db.compact() -> None`: atomically rewrites the WAL (temp file + fsync + `os.replace`) containing ONLY the current live key-value entries so deleted/overwritten history genuinely shrinks disk usage!

DELIVERABLES:
- Clean multi-file package (`minidb/__init__.py`, `minidb/wal.py`, `minidb/db.py`, etc.).
- Comprehensive pytest suite in `tests/test_minidb.py` testing CRUD, falsy values, scans, tx isolation/rollback/commit, simulated torn-tail crash recovery, and compaction size reduction.
- Run `pytest -q` and confirm 100% pass before finishing.
"""

PG1_HIDDEN_GRADER = '''import os
from pathlib import Path
import pytest
from minidb import MiniDB


def test_hidden_crud_falsy_scan_and_persistence(tmp_path: Path):
    db_dir = tmp_path / "db1"
    with MiniDB(db_dir) as db:
        db.put("z_zero", 0)
        db.put("z_false", False)
        db.put("z_empty", "")
        db.put("z_list", [])
        db.put("alpha", {"nested": [1, 2, 3]})
        db.put("beta", "b")
        db.put("gamma", "g")
        assert db.get("z_zero", -1) == 0
        assert db.get("z_false", True) is False
        assert db.get("z_empty", "x") == ""
        assert db.get("z_list", None) == []
        assert db.get("missing", "DEF") == "DEF"
        assert db.scan("alpha", "gamma") == [("alpha", {"nested": [1, 2, 3]}), ("beta", "b")]
        assert db.delete("beta") is True
        assert db.delete("beta") is False
        assert db.get("beta", "GONE") == "GONE"

    # Reopen from disk in fresh instance
    with MiniDB(db_dir) as db2:
        assert db2.get("z_zero", -1) == 0
        assert db2.get("z_false", True) is False
        assert db2.get("beta", "GONE") == "GONE"
        assert db2.scan("alpha", "gamma") == [("alpha", {"nested": [1, 2, 3]})]


def test_hidden_tx_isolation_rollback_and_atomicity(tmp_path: Path):
    db_dir = tmp_path / "db_tx"
    with MiniDB(db_dir) as db:
        db.put("k1", "orig")
        db.put("k2", "stay")

        tx1 = db.begin()
        tx1.put("k1", "changed")
        tx1.delete("k2")
        tx1.put("k3", "new")
        # Read-your-own-writes inside tx1, invisible outside!
        assert tx1.get("k1") == "changed"
        assert tx1.get("k2", "MISSING") == "MISSING"
        assert db.get("k1") == "orig"
        assert db.get("k2") == "stay"
        assert db.get("k3") is None
        tx1.rollback()
        assert db.get("k1") == "orig"
        assert db.get("k2") == "stay"
        with pytest.raises((RuntimeError, ValueError)):
            tx1.put("k1", "after_rollback")

        tx2 = db.begin()
        tx2.put("k1", "final")
        tx2.delete("k2")
        tx2.put("k3", 42)
        tx2.commit()
        with pytest.raises((RuntimeError, ValueError)):
            tx2.commit()

    with MiniDB(db_dir) as db2:
        assert db2.get("k1") == "final"
        assert db2.get("k2") is None
        assert db2.get("k3") == 42


def test_hidden_torn_wal_crash_recovery_and_post_recovery_writes(tmp_path: Path):
    db_dir = tmp_path / "db_crash"
    with MiniDB(db_dir) as db:
        for i in range(10):
            db.put(f"k{i:02d}", f"val_{i}")
        tx = db.begin()
        tx.put("tx_a", 100)
        tx.put("tx_b", 200)
        tx.commit()

    # Find WAL file(s) in db_dir and simulate torn write / bit-flipped crash tail!
    files = [p for p in db_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    assert files, "No persisted WAL file found in db_dir!"
    wal_file = max(files, key=lambda p: p.stat().st_size)
    with open(wal_file, "ab") as f:
        f.write(b"\\x00\\x00\\x00\\x99CORRUPT_TORN_TAIL_FRAME_POWER_LOSS")

    # Reopen MUST recover all 12 committed keys cleanly and survive a new write!
    with MiniDB(db_dir) as db2:
        for i in range(10):
            assert db2.get(f"k{i:02d}") == f"val_{i}"
        assert db2.get("tx_a") == 100
        assert db2.get("tx_b") == 200
        db2.put("after_crash", "recovered_ok")

    with MiniDB(db_dir) as db3:
        assert db3.get("after_crash") == "recovered_ok"
        assert db3.get("k09") == "val_9"


def test_hidden_compaction_shrinks_disk_usage(tmp_path: Path):
    db_dir = tmp_path / "db_compact"
    with MiniDB(db_dir) as db:
        for i in range(150):
            db.put("hot_key", "X" * 200 + str(i))
            db.put(f"temp_{i}", "Y" * 200)
            db.delete(f"temp_{i}")
        db.put("survivor", "alive")

        def total_bytes():
            return sum(p.stat().st_size for p in db_dir.rglob("*") if p.is_file())

        before = total_bytes()
        db.compact()
        after = total_bytes()
        assert after < before * 0.35, f"Compaction did not shrink disk usage enough: {before} -> {after}"

    with MiniDB(db_dir) as db2:
        assert db2.get("hot_key") == "X" * 200 + "149"
        assert db2.get("survivor") == "alive"
        assert db2.get("temp_0") is None
'''


# =============================================================================
# PG2: VAGUE PRODUCTION INCIDENT IN AN 8-FILE JOB QUEUE + RATE LIMITER SERVICE
# =============================================================================

PG2_REPO_FILES: dict[str, str] = {
    "pulsequeue/__init__.py": "",
    "pulsequeue/models.py": '''from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Job:
    job_id: str
    tenant_id: str
    payload: Any
    max_retries: int = 2
    attempts: int = 0
    status: str = "QUEUED"  # QUEUED, RUNNING, SUCCEEDED, FAILED, DEAD
    available_at: float = 0.0
    last_error: Optional[str] = None
''',
    "pulsequeue/limiter.py": '''from collections import defaultdict, deque


class TenantRateLimiter:
    """Sliding-window per-tenant rate limiter (max_per_window calls per window_s)."""

    def __init__(self, max_per_window: int, window_s: float) -> None:
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, tenant_id: str, now: float) -> None:
        q = self._hits[tenant_id]
        cutoff = now - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()

    def allow(self, tenant_id: str, now: float) -> bool:
        self._prune(tenant_id, now)
        if len(self._hits[tenant_id]) >= self.max_per_window:
            return False
        self._hits[tenant_id].append(now)
        return True

    def can_run(self, tenant_id: str, now: float) -> bool:
        self._prune(tenant_id, now)
        return len(self._hits[tenant_id]) < self.max_per_window
''',
    "pulsequeue/storage.py": '''from .models import Job


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.persisted_pending: list[str] = []

    def put(self, job: Job) -> None:
        self.jobs[job.job_id] = job

    def get(self, job_id: str) -> Job:
        return self.jobs[job_id]

    def save_pending_snapshot(self, job_ids: list[str]) -> None:
        self.persisted_pending = list(job_ids)
''',
    "pulsequeue/scheduler.py": '''from collections import deque
from typing import Optional
from .limiter import TenantRateLimiter
from .models import Job


class FairScheduler:
    """Schedules ready jobs while respecting per-tenant rate limits."""

    def __init__(self, limiter: TenantRateLimiter) -> None:
        self.limiter = limiter
        self._queue: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        self._queue.append(job)

    def pending_jobs(self) -> list[Job]:
        return list(self._queue)

    def pop_next(self, now: float) -> Optional[Job]:
        if not self._queue:
            return None
        # BUG 2: Head-of-line blocking! Only checks the front job — if front tenant is rate-limited
        # or delayed (available_at > now), it returns None and STARVES all other tenants behind it!
        head = self._queue[0]
        if head.available_at > now:
            return None
        if not self.limiter.allow(head.tenant_id, now):
            return None
        return self._queue.popleft()
''',
    "pulsequeue/worker.py": '''from typing import Callable
from .models import Job


class JobWorker:
    def __init__(self, handler: Callable[[Job], None], retry_delay_s: float = 5.0) -> None:
        self.handler = handler
        self.retry_delay_s = retry_delay_s

    def execute(self, job: Job, now: float) -> tuple[bool, bool]:
        """Run handler on job. Returns (succeeded, should_requeue)."""
        # BUG 1a: Never guards against already-SUCCEEDED or DEAD jobs being executed twice!
        job.status = "RUNNING"
        job.attempts += 1
        try:
            self.handler(job)
            job.status = "SUCCEEDED"
            return True, False
        except Exception as exc:
            job.last_error = str(exc)
            # BUG 1b: Off-by-one retry check! `attempts <= max_retries + 1` runs max_retries + 2 times total
            # (should be at most `1 initial attempt + max_retries retries` -> requeue iff `attempts <= max_retries`!).
            if job.attempts <= job.max_retries + 1:
                job.status = "QUEUED"
                job.available_at = now + self.retry_delay_s
                return False, True
            job.status = "DEAD"
            return False, False
''',
    "pulsequeue/service.py": '''from typing import Any, Callable
from .limiter import TenantRateLimiter
from .models import Job
from .scheduler import FairScheduler
from .storage import JobStore
from .worker import JobWorker


class PulseQueueService:
    def __init__(
        self,
        handler: Callable[[Job], None],
        max_per_window: int = 2,
        window_s: float = 10.0,
        retry_delay_s: float = 5.0,
    ) -> None:
        self.store = JobStore()
        self.limiter = TenantRateLimiter(max_per_window=max_per_window, window_s=window_s)
        self.scheduler = FairScheduler(self.limiter)
        self.worker = JobWorker(handler=handler, retry_delay_s=retry_delay_s)
        self.in_flight: set[str] = set()
        self.shutdown_requested: bool = False
        self._enqueued_ids: set[str] = set()

    def submit(self, job_id: str, tenant_id: str, payload: Any, max_retries: int = 2, now: float = 0.0) -> Job:
        if self.shutdown_requested:
            raise RuntimeError("Service is shutting down")
        # BUG 1c: Duplicate submit of same job_id overwrites & re-enqueues an already SUCCEEDED/QUEUED job!
        job = Job(job_id=job_id, tenant_id=tenant_id, payload=payload, max_retries=max_retries, available_at=now)
        self.store.put(job)
        self.scheduler.enqueue(job)
        return job

    def tick(self, now: float) -> int:
        """Process all currently eligible (non-delayed, non-rate-limited) jobs at timestamp `now`."""
        ran = 0
        while True:
            job = self.scheduler.pop_next(now)
            if job is None:
                break
            self.in_flight.add(job.job_id)
            ok, requeue = self.worker.execute(job, now)
            self.in_flight.discard(job.job_id)
            ran += 1
            if requeue:
                self.scheduler.enqueue(job)
        return ran

    def drain_and_shutdown(self, now: float) -> dict[str, Any]:
        """Gracefully stop accepting new jobs, run immediately eligible jobs, and persist remaining delayed/rate-limited jobs."""
        self.shutdown_requested = True
        # BUG 3: Clears scheduler queue without draining eligible jobs or saving remaining pending/delayed jobs to store!
        self.scheduler._queue.clear()
        return {
            "drained_count": 0,
            "persisted_pending": [],
            "in_flight": list(self.in_flight),
        }
''',
    "tests/test_smoke.py": '''from pulsequeue.service import PulseQueueService


def test_smoke_happy_path():
    ran = []
    svc = PulseQueueService(handler=lambda j: ran.append(j.job_id), max_per_window=5, window_s=10.0)
    svc.submit("j1", "t1", {"x": 1}, now=0.0)
    svc.submit("j2", "t1", {"x": 2}, now=0.0)
    assert svc.tick(0.0) == 2
    assert ran == ["j1", "j2"]
    assert svc.store.get("j1").status == "SUCCEEDED"
''',
}

PG2_MISSION = """Production Incident Ticket on `pulsequeue/`:
Existing smoke tests pass (`pytest -q`), but production is hitting THREE real incidents across the codebase:

1. **Duplicate Execution & Retry Overrun:**
   - Resubmitting an existing `job_id` (duplicate webhook/delivery) causes already-succeeded or already-queued jobs to execute TWICE instead of being idempotent (returning the existing `Job` without re-enqueueing or re-running!).
   - Calling `worker.execute()` on an already `SUCCEEDED` or `DEAD` job still re-runs the handler instead of refusing/skipping.
   - Failing jobs with `max_retries=N` are executing `N + 2` times instead of at most `1 initial attempt + N retries` (`N + 1` total attempts).

2. **Cross-Tenant Starvation (Head-of-Line Blocking):**
   - When Tenant A floods the queue and hits its per-tenant rate limit (or has delayed retry jobs with `available_at > now`), Tenant B's ready, non-rate-limited jobs sitting behind Tenant A get completely starved instead of running immediately!
   - Also make sure `scheduler.pop_next(now)` preserves FIFO fairness among eligible jobs and does not lose skipped delayed/rate-limited jobs from the queue.

3. **Graceful Shutdown Drops Queued & Delayed Jobs:**
   - Calling `service.drain_and_shutdown(now)` must:
     (a) set `shutdown_requested = True` so new `submit()` calls raise `RuntimeError`,
     (b) execute (drain) all currently eligible queued jobs at timestamp `now` (`drained_count`),
     (c) persist the IDs of ALL remaining pending jobs (delayed retry jobs or rate-limited jobs that couldn't run at `now`) via `self.store.save_pending_snapshot(remaining_ids)` WITHOUT dropping them silently,
     (d) return `{"drained_count": int, "persisted_pending": list[str], "in_flight": list[str]}` accurately.

Investigate the repo, reproduce each incident with probes/tests, fix ALL root causes cleanly across all affected modules, add regression tests covering all 3 incidents, and confirm `pytest -q` passes 100%.
"""

PG2_HIDDEN_GRADER = '''import pytest
from pulsequeue.service import PulseQueueService


def test_hidden_incident1_idempotency_and_exact_retry_count():
    calls = []

    def flaky(job):
        calls.append(job.job_id)
        if job.job_id == "always_fail":
            raise ValueError("boom")

    svc = PulseQueueService(handler=flaky, max_per_window=10, window_s=10.0, retry_delay_s=5.0)
    j1 = svc.submit("ok1", "t1", {}, max_retries=2, now=0.0)
    # Duplicate submit while queued MUST NOT enqueue twice!
    j1_dup = svc.submit("ok1", "t1", {}, max_retries=2, now=0.0)
    assert j1_dup.job_id == j1.job_id
    assert svc.tick(0.0) == 1
    assert calls == ["ok1"]

    # Duplicate submit AFTER success MUST NOT re-run!
    svc.submit("ok1", "t1", {}, max_retries=2, now=1.0)
    assert svc.tick(1.0) == 0
    assert calls == ["ok1"]

    # Direct worker guard on already-SUCCEEDED job
    ok, req = svc.worker.execute(svc.store.get("ok1"), now=2.0)
    assert calls == ["ok1"] and not req

    # Exact retry count: max_retries=2 -> 1 initial + 2 retries = 3 total attempts, then DEAD!
    svc.submit("always_fail", "t1", {}, max_retries=2, now=0.0)
    svc.tick(0.0)   # attempt 1 -> requeue at t=5
    svc.tick(5.0)   # attempt 2 -> requeue at t=10
    svc.tick(10.0)  # attempt 3 -> DEAD!
    svc.tick(15.0)  # nothing left!
    svc.tick(20.0)
    j_fail = svc.store.get("always_fail")
    assert j_fail.attempts == 3, f"Expected exactly 3 attempts (1 + 2 retries), got {j_fail.attempts}"
    assert j_fail.status == "DEAD"
    assert calls.count("always_fail") == 3


def test_hidden_incident2_no_cross_tenant_starvation():
    ran = []
    svc = PulseQueueService(handler=lambda j: ran.append((j.tenant_id, j.job_id)), max_per_window=1, window_s=10.0)
    # Tenant A submits 3 jobs (limit 1/10s), Tenant B submits 1 job behind them!
    svc.submit("a1", "tenantA", {}, now=0.0)
    svc.submit("a2", "tenantA", {}, now=0.0)
    svc.submit("a3", "tenantA", {}, now=0.0)
    svc.submit("b1", "tenantB", {}, now=0.0)

    count = svc.tick(0.0)
    # MUST run a1 AND b1 immediately (2 jobs), leaving a2 and a3 queued for later!
    assert count == 2, f"Tenant B starved! ran={ran}"
    assert ran == [("tenantA", "a1"), ("tenantB", "b1")]

    # After window passes, a2 runs (not lost!)
    assert svc.tick(11.0) == 1
    assert ran[-1] == ("tenantA", "a2")

    # Also test delayed job in front does NOT block ready job behind it!
    svc.submit("delayed_c", "tenantC", {}, now=50.0)
    svc.submit("ready_d", "tenantD", {}, now=11.0)
    assert svc.tick(11.0) == 1
    assert ran[-1] == ("tenantD", "ready_d")


def test_hidden_incident3_graceful_drain_and_persist_pending():
    ran = []
    def handler(job):
        ran.append(job.job_id)
        if job.job_id == "will_retry":
            raise RuntimeError("transient")

    svc = PulseQueueService(handler=handler, max_per_window=1, window_s=10.0, retry_delay_s=5.0)
    svc.submit("t1_a", "t1", {}, max_retries=1, now=0.0)
    svc.submit("t1_b", "t1", {}, max_retries=1, now=0.0)        # rate-limited at t=0
    svc.submit("will_retry", "t2", {}, max_retries=2, now=0.0) # fails at t=0 -> delayed retry at t=5
    svc.submit("future_job", "t3", {}, max_retries=1, now=9.0) # delayed until t=9

    report = svc.drain_and_shutdown(now=0.0)
    # t1_a and will_retry were eligible at t=0 -> drained_count >= 2!
    assert report["drained_count"] == 2
    assert "t1_a" in ran and "will_retry" in ran
    # Remaining pending jobs (t1_b, will_retry retry, future_job) MUST be persisted!
    assert set(report["persisted_pending"]) == {"t1_b", "will_retry", "future_job"}
    assert set(svc.store.persisted_pending) == {"t1_b", "will_retry", "future_job"}
    assert report["in_flight"] == []
    with pytest.raises(RuntimeError):
        svc.submit("late", "t1", {}, now=1.0)
'''


# =============================================================================
# PG3: REAL READ-ONLY DEEP SYSTEMS INVESTIGATION ON ACTUAL TENSORFOLD ZIG REPO
# =============================================================================

TENSORFOLD_REPO = Path(os.environ.get("TENSORFOLD_REPO", "./tensorfold"))

PG3_MISSION = """You are conducting a READ-ONLY systems engineering investigation of this actual TensorFold Zig/Metal repository.
Using ONLY verified `search_code` and `read_file` calls on real `.zig` source files (do NOT just paraphrase `CHANGELOG.md` — verify everything directly in `zig/src/`!), answer ALL THREE questions with exact `path:line` citations and real symbol/function names:

1. **TFQ3 Sidecar Detection & Zero-Copy Loading (Flash-Next on Metal):**
   - What exact sidecar filenames (`pw_v3_tables.bin`, `layer_XX.tffpw`, `mtp_00.tffpw`, etc.) are expected, and in which exact `.zig` file(s), line(s), and function(s) are they detected and loaded?
2. **Decode Dispatch (`B=1` vs Deduplicated `B=2..16` Routed MoE on Metal):**
   - In which exact `.zig` file(s), line(s), and function(s) does Flash-Next Metal dispatch `B=1` TFQ3 routed MoE decode versus deduplicated multi-row (`B=2..16`) decode (and name the actual Metal kernel / pipeline function names)?
3. **Flag Interaction (`--ssd-experts` vs `--ple-on-ssd` with TFQ3):**
   - Where in the `.zig` code (`file:line`) is `--ssd-experts` refused when TFQ3 sidecars are present, and how is `--ple-on-ssd` handled alongside TFQ3?

Every single factual claim MUST cite an exact verified `zig/src/...:line` location from the files you actually read!
"""


def grade_pg3_report(report_text: str) -> tuple[bool, str]:
    """Mechanically verify PG3 report: 0 hallucinated file:line citations, real zig/src files cited, key factual hits!"""
    n_cited, cite_errs = verify_citations_on_disk(TENSORFOLD_REPO, report_text)
    if cite_errs:
        return False, f"HALLUCINATED CITATIONS ({len(cite_errs)}): {cite_errs[:3]}"
    zig_cites = [m for m in re.findall(r"([A-Za-z0-9_./-]+\.zig):(\d+)", report_text) if "zig/src/" in m[0]]
    if len(zig_cites) < 4:
        return False, f"Too few real zig/src/*.zig citations ({len(zig_cites)} found, need >=4)"

    low = report_text.lower()
    checks = {
        "sidecars": ("pw_v3_tables.bin" in low and "tffpw" in low),
        "batch_dispatch": ("dedup" in low or "b=2" in low or "batch" in low or "2..16" in low) and ("tfq3" in low or "pw_v3" in low or "half4" in low or "coset" in low),
        "ssd_vs_ple": ("ssd" in low and "ple" in low and ("refus" in low or "error" in low or "incompat" in low or "reject" in low)),
    }
    missing = [k for k, ok in checks.items() if not ok]
    if missing:
        return False, f"Missing required factual coverage: {missing} (valid zig cites={len(zig_cites)})"

    # Spot-check that cited zig lines actually exist near real flashnext/tfq3 code!
    matched_code = 0
    for rel, ln_s in zig_cites[:12]:
        p = TENSORFOLD_REPO / rel
        if p.is_file():
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            ln = int(ln_s)
            window = "\n".join(lines[max(0, ln - 10) : min(len(lines), ln + 10)]).lower()
            if any(tok in window for tok in ("tffpw", "pw_v3", "tfq3", "ssd", "ple", "moe", "expert", "flash", "metal", "batch", "dedup")):
                matched_code += 1
    if matched_code < 3:
        return False, f"Cited zig lines did not match TFQ3/FlashNext symbols (matched={matched_code}/{len(zig_cites)})"
    return True, f"PASS (0 hallucinated citations, {len(zig_cites)} verified zig/src cites, {matched_code} symbol-confirmed)"


# =============================================================================
# PG4: LOW-LEVEL ARM64 NEON 4-BIT AFFINE QUANTIZED GEMV KERNEL OPTIMIZATION
# =============================================================================

PG4_KERNEL_H = r'''#ifndef AFFINE4_KERNEL_H
#define AFFINE4_KERNEL_H
#include <stddef.h>
#include <stdint.h>

// Group size G = 64.
// For each row r in [0, M):
//   - num_groups = (K + 63) / 64
//   - packed_bytes_per_row = (K + 1) / 2
//   - packed_w[r * packed_bytes_per_row + (k >> 1)]:
//       even k -> low nibble (byte & 0x0F), odd k -> high nibble (byte >> 4)
//   - group g = k / 64:
//       w(r, k) = (float)q * scales[r * num_groups + g] + biases[r * num_groups + g]
//   - y[r] = sum_{k=0..K-1} w(r, k) * x[k]
void affine4_gemv_ref(
    const uint8_t *packed_w,
    const float *scales,
    const float *biases,
    const float *x,
    float *y,
    int M,
    int K
);

void affine4_gemv_fast(
    const uint8_t *packed_w,
    const float *scales,
    const float *biases,
    const float *x,
    float *y,
    int M,
    int K
);

#endif
'''

PG4_KERNEL_C = r'''#include "kernel.h"

void affine4_gemv_ref(
    const uint8_t *packed_w,
    const float *scales,
    const float *biases,
    const float *x,
    float *y,
    int M,
    int K
) {
    int num_groups = (K + 63) / 64;
    int row_bytes = (K + 1) / 2;
    for (int r = 0; r < M; r++) {
        const uint8_t *wrow = packed_w + (size_t)r * (size_t)row_bytes;
        const float *srow = scales + (size_t)r * (size_t)num_groups;
        const float *brow = biases + (size_t)r * (size_t)num_groups;
        double acc = 0.0;
        for (int k = 0; k < K; k++) {
            uint8_t byte = wrow[k >> 1];
            uint8_t q = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            int g = k / 64;
            float w = (float)q * srow[g] + brow[g];
            acc += (double)w * (double)x[k];
        }
        y[r] = (float)acc;
    }
}

// Baseline unoptimized implementation (edit ONLY this function / helpers in kernel.c!)
void affine4_gemv_fast(
    const uint8_t *packed_w,
    const float *scales,
    const float *biases,
    const float *x,
    float *y,
    int M,
    int K
) {
    int num_groups = (K + 63) / 64;
    int row_bytes = (K + 1) / 2;
    for (int r = 0; r < M; r++) {
        const uint8_t *wrow = packed_w + (size_t)r * (size_t)row_bytes;
        const float *srow = scales + (size_t)r * (size_t)num_groups;
        const float *brow = biases + (size_t)r * (size_t)num_groups;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            uint8_t byte = wrow[k >> 1];
            uint8_t q = (k & 1) ? (byte >> 4) : (byte & 0x0F);
            int g = k / 64;
            float w = (float)q * srow[g] + brow[g];
            acc += w * x[k];
        }
        y[r] = acc;
    }
}
'''

PG4_BENCH_C = r'''#include "kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint32_t rng_state = 0x12345678u;
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xorshift32() & 0xFFFF) / 65535.0f);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static int check_shape(int M, int K) {
    int ng = (K + 63) / 64;
    int rb = (K + 1) / 2;
    uint8_t *w = (uint8_t *)malloc((size_t)M * rb + 64);
    float *s = (float *)malloc((size_t)M * ng * sizeof(float) + 64);
    float *b = (float *)malloc((size_t)M * ng * sizeof(float) + 64);
    float *x = (float *)malloc((size_t)K * sizeof(float) + 64);
    float *y_ref = (float *)calloc((size_t)M, sizeof(float));
    float *y_fast = (float *)calloc((size_t)M, sizeof(float));
    for (size_t i = 0; i < (size_t)M * rb; i++) w[i] = (uint8_t)(xorshift32() & 0xFF);
    for (size_t i = 0; i < (size_t)M * ng; i++) { s[i] = randf(-0.5f, 0.5f); b[i] = randf(-1.0f, 1.0f); }
    for (int k = 0; k < K; k++) x[k] = randf(-2.0f, 2.0f);

    affine4_gemv_ref(w, s, b, x, y_ref, M, K);
    affine4_gemv_fast(w, s, b, x, y_fast, M, K);

    float max_err = 0.0f;
    for (int r = 0; r < M; r++) {
        float err = fabsf(y_ref[r] - y_fast[r]);
        if (isnan(y_fast[r]) || isinf(y_fast[r]) || err > max_err) max_err = err;
    }
    int ok = (max_err < 2e-3f);
    printf("SHAPE (%d x %d): max_err=%.6g [%s]\n", M, K, max_err, ok ? "PASS" : "FAIL");
    free(w); free(s); free(b); free(x); free(y_ref); free(y_fast);
    return ok;
}

int main(void) {
    int shapes[][2] = {{1, 1}, {1, 64}, {5, 65}, {7, 130}, {13, 255}, {64, 512}, {256, 2048}};
    for (int i = 0; i < 7; i++) {
        if (!check_shape(shapes[i][0], shapes[i][1])) return 1;
    }
    int M = 1024, K = 4096, ng = K / 64, rb = K / 2;
    uint8_t *w = (uint8_t *)malloc((size_t)M * rb);
    float *s = (float *)malloc((size_t)M * ng * sizeof(float));
    float *b = (float *)malloc((size_t)M * ng * sizeof(float));
    float *x = (float *)malloc((size_t)K * sizeof(float));
    float *y_ref = (float *)calloc((size_t)M, sizeof(float));
    float *y_fast = (float *)calloc((size_t)M, sizeof(float));
    for (size_t i = 0; i < (size_t)M * rb; i++) w[i] = (uint8_t)(xorshift32() & 0xFF);
    for (size_t i = 0; i < (size_t)M * ng; i++) { s[i] = randf(-0.25f, 0.25f); b[i] = randf(-0.5f, 0.5f); }
    for (int k = 0; k < K; k++) x[k] = randf(-1.5f, 1.5f);

    // Warmup + exactness on benchmark shape
    affine4_gemv_ref(w, s, b, x, y_ref, M, K);
    affine4_gemv_fast(w, s, b, x, y_fast, M, K);
    float max_err = 0.0f;
    for (int r = 0; r < M; r++) {
        float err = fabsf(y_ref[r] - y_fast[r]);
        if (isnan(y_fast[r]) || err > max_err) max_err = err;
    }
    if (max_err >= 5e-3f) {
        printf("BENCH EXACTNESS FAIL: max_err=%.6g\n", max_err);
        return 1;
    }

    int iters = 40;
    double best_ref = 1e9, best_fast = 1e9;
    for (int round = 0; round < 5; round++) {
        double t0 = now_ms();
        for (int it = 0; it < iters; it++) affine4_gemv_ref(w, s, b, x, y_ref, M, K);
        double d_ref = (now_ms() - t0) / iters;
        double t1 = now_ms();
        for (int it = 0; it < iters; it++) affine4_gemv_fast(w, s, b, x, y_fast, M, K);
        double d_fast = (now_ms() - t1) / iters;
        if (d_ref < best_ref) best_ref = d_ref;
        if (d_fast < best_fast) best_fast = d_fast;
    }
    double speedup = best_ref / best_fast;
    printf("BENCH (1024x4096): ref=%.3f ms  fast=%.3f ms  speedup=%.2fx  max_err=%.6g\n",
           best_ref, best_fast, speedup, max_err);
    free(w); free(s); free(b); free(x); free(y_ref); free(y_fast);
    return (speedup >= 3.0) ? 0 : 2;
}
'''

PG4_MISSION = """Optimize the low-level 4-bit block-quantized affine GEMV kernel `affine4_gemv_fast` in `kernel.c` on this Apple Silicon ARM64 machine!

STRICT RULES (Referee-verified):
1. Edit ONLY `affine4_gemv_fast` (and any static helper functions you add) inside `kernel.c`. NEVER modify `affine4_gemv_ref`, `kernel.h`, or `bench.c`!
2. Compile and run with: `clang -O3 -Wall -Wextra bench.c kernel.c -o bench_runner && ./bench_runner`
3. Must pass ALL exactness checks (`max_err < 2e-3` on aligned AND unaligned edge shapes like `(1,1)`, `(5,65)`, `(7,130)`, `(13,255)`, `(256,2048)`, `(1024,4096)` — NO out-of-bounds memory reads!).
4. Must achieve **>= 3.0x verified speedup** over `affine4_gemv_ref` on `(1024 x 4096)`!
   Key kernel optimizations to consider on Apple Silicon ARM64 (`#include <arm_neon.h>`):
   - Algebraic factorization per 64-element group: `sum_{k in g} (q_k * s + b) * x_k = s * sum(q_k * x_k) + b * sum(x_k)`! Precompute `group_x_sum[g]` ONCE outside the `M` loop so you NEVER add `b * x_k` per row!
   - Process full 64-element groups (32 packed bytes!) with ARM NEON SIMD (`vld1q_u8`, nibble unpack `vandq_u8`/`vshrq_n_u8`, `vmovl_u8` -> `vmovl_u16` -> `vcvtq_f32_u32`, `vfmaq_f32`) with clean scalar tail handling for `K % 64 != 0`!
   - Tile 2 or 4 rows simultaneously to reuse loaded `x[k]` NEON registers across multiple rows and maximize memory bandwidth + FMA ILP!
5. Run the compile + `./bench_runner` check AFTER your final edit and verify `exit=0` and `speedup >= 3.0x` before calling finish().
"""


def _setup_pg4_workspace() -> Path:
    ws = Path(tempfile.mkdtemp(prefix="ff_pg4_"))
    (ws / "kernel.h").write_text(PG4_KERNEL_H, encoding="utf-8")
    (ws / "kernel.c").write_text(PG4_KERNEL_C, encoding="utf-8")
    (ws / "bench.c").write_text(PG4_BENCH_C, encoding="utf-8")
    return ws


def _grade_pg4_hidden(ws: Path) -> tuple[bool, str]:
    """Compile candidate kernel.c against an IMMUTABLE hidden bench harness (anti-cheat!)."""
    hidden_dir = Path(tempfile.mkdtemp(prefix="ff_pg4_grade_"))
    try:
        (hidden_dir / "kernel.h").write_text(PG4_KERNEL_H, encoding="utf-8")
        (hidden_dir / "bench.c").write_text(PG4_BENCH_C, encoding="utf-8")
        shutil.copy2(ws / "kernel.c", hidden_dir / "kernel.c")
        comp = subprocess.run(
            ["clang", "-O3", "-Wall", "-Wextra", "bench.c", "kernel.c", "-o", "bench_runner"],
            cwd=str(hidden_dir),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if comp.returncode != 0:
            return False, f"COMPILE FAIL: {comp.stderr[:240]}"
        run = subprocess.run(
            ["./bench_runner"],
            cwd=str(hidden_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out_lines = [ln.strip() for ln in run.stdout.splitlines() if ln.strip()]
        summary = out_lines[-1] if out_lines else f"exit={run.returncode}"
        return (run.returncode == 0), summary
    finally:
        shutil.rmtree(hidden_dir, ignore_errors=True)


# =============================================================================
# FAIR BASELINE: STANDARD SINGLE-STREAM VANILLA REACT AGENT (SAME TOOLS + SCHEMA)
# =============================================================================

async def run_vanilla_react(
    client: TensorFoldClient,
    workspace: Path,
    goal: str,
    *,
    read_only: bool = False,
    max_turns: int = 18,
) -> dict[str, Any]:
    """Standard single-stream ReAct loop with raw growing chat history (NO BrainState compaction,
    NO B=3 strategic forks, NO forge_* escalation, NO Independent Referee Gate).
    """
    t0 = time.perf_counter()
    client.reset_stats()
    await client.probe()
    tools = AgentToolbelt(workspace, read_only=read_only)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _AGENT_SYSTEM_PROMPT + ("\nMODE: READ-ONLY." if read_only else "")},
        {"role": "user", "content": f"MISSION:\n{goal}\n\nWorkspace: `{workspace}`\nInitial listing:\n{tools.list_dir('.', 2).output}"},
    ]
    summary = ""
    turns = 0
    for turn in range(1, max_turns + 1):
        turns = turn
        res = await client.complete(
            CallSpec(
                messages=messages,
                tag=f"vreact_t{turn}",
                max_tokens=4096,
                temperature=0.3,
                seed=42 + turn,
                enable_thinking=True,
                thinking_budget=650,
                json_schema=AGENT_STEP_SCHEMA,
                schema_name="agent_step",
                grammar_budget=True,
                loop_guard=True,
            )
        )
        obj = res.json_payload()
        if not obj or not obj.get("tool"):
            messages.append({"role": "assistant", "content": res.content[:400] or "{}"})
            messages.append({"role": "user", "content": "Invalid action. Emit valid JSON matching schema."})
            continue
        tname, arg_s = _summarize_action(obj)
        print(f"  [VanillaReAct T{turn:02d}/{max_turns}] {tname}({arg_s})", flush=True)
        if tname == "finish":
            summary = str(obj.get("final_summary", ""))
            break

        # Execute tool (no forge_code/forge_reason in Vanilla baseline)
        if tname == "list_dir":
            tr = tools.list_dir(str(obj.get("path") or "."), int(obj.get("end_line") or 2))
        elif tname == "repo_map":
            tr = tools.repo_map(str(obj.get("path") or "."))
        elif tname == "search_code":
            tr = tools.search_code(str(obj.get("pattern") or ""), str(obj.get("path") or "."))
        elif tname == "read_file":
            tr = tools.read_file(str(obj.get("path") or ""), int(obj.get("start_line") or 1), int(obj.get("end_line") or 0) or None)
        elif tname == "write_file":
            tr = tools.write_file(str(obj.get("path") or ""), str(obj.get("code_or_content") or ""))
        elif tname == "edit_file":
            tr = tools.edit_file(str(obj.get("path") or ""), str(obj.get("edit_target") or ""), str(obj.get("code_or_content") or ""))
        elif tname == "run_command":
            tr = tools.run_command(str(obj.get("cmd") or ""))
        elif tname == "python_probe":
            tr = tools.python_probe(str(obj.get("code_or_content") or ""))
        else:
            tr = tools.list_dir(".")

        messages.append({"role": "assistant", "content": res.content[:2000]})
        messages.append({"role": "user", "content": f"Tool {tname} ({'OK' if tr.ok else 'ERROR'}):\n{tr.output[:2400]}"})

    return {
        "summary": summary,
        "turns": turns,
        "wall_s": round(time.perf_counter() - t0, 2),
        "stats": client.stats_dict(),
    }


def _run_hidden_pytest(workspace: Path, hidden_code: str) -> tuple[bool, str]:
    ht = workspace / "_hidden_proving_grade_test.py"
    ht.write_text(hidden_code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", str(ht)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=25,
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode == 0:
            return True, (out.splitlines()[-1] if out else "ok")
        # Return failing test names + concise tail so diagnostics are crystal clear
        fails = [ln for ln in out.splitlines() if ln.startswith("FAILED ") or ln.startswith("E   ")]
        summary = " | ".join(fails[:4]) if fails else (out.splitlines()[-1] if out else f"exit={proc.returncode}")
        return False, summary[:320]
    except Exception as exc:
        return False, f"GRADER ERROR: {exc}"
    finally:
        ht.unlink(missing_ok=True)


def _setup_pg2_workspace() -> Path:
    ws = Path(tempfile.mkdtemp(prefix="ff_pg2_"))
    for rel, content in PG2_REPO_FILES.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return ws


async def run_proving_ground(
    selected: list[str],
    base_url: str,
    run_vanilla: bool,
    referee_url: str | None = None,
    out_file: str = "bench/results/replica_proving_ground.json",
) -> None:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mem_db = out_path.parent / "pg_memory.db"
    if mem_db.exists():
        mem_db.unlink()

    # Planted control for PG2: unpatched pulsequeue MUST pass smoke test and FAIL hidden incident grader!
    if "PG2" in selected:
        ctrl = _setup_pg2_workspace()
        try:
            sm = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_smoke.py"], cwd=ctrl, capture_output=True)
            assert sm.returncode == 0, "PG2 base smoke test must pass!"
            ok_h, msg_h = _run_hidden_pytest(ctrl, PG2_HIDDEN_GRADER)
            assert not ok_h, f"PG2 planted control failed: unpatched repo passed hidden suite ({msg_h})"
            print(f"[PG2 CONTROL OK] Unpatched pulsequeue passes smoke test and FAILS hidden incidents -> {msg_h}")
        finally:
            shutil.rmtree(ctrl, ignore_errors=True)

    report: dict[str, Any] = {}
    ref_client = TensorFoldClient(base_url=referee_url) if referee_url else None
    if ref_client:
        await ref_client.probe()
        print(f"[CROSS-MODEL REFEREE] Active -> {ref_client.caps.model_id} @ {referee_url}")
    try:
      async with TensorFoldClient(base_url=base_url) as vc, FlashForge(
          ForgeConfig(base_url=base_url, enable_memory=True, memory_db=mem_db)
      ) as ff:

        # --- PG1: Greenfield WAL DB ---
        if "PG1" in selected:
            print("\n" + "=" * 76)
            print("PG1: GREENFIELD CRASH-SAFE CRC32 WAL TRANSACTIONAL DB (EMPTY WORKSPACE)")
            print("=" * 76)
            pg1_entry: dict[str, Any] = {}
            if run_vanilla:
                ws_v = Path(tempfile.mkdtemp(prefix="ff_pg1_van_"))
                try:
                    vr = await run_vanilla_react(vc, ws_v, PG1_MISSION, max_turns=18)
                    v_ok, v_msg = _run_hidden_pytest(ws_v, PG1_HIDDEN_GRADER)
                    print(f"  [PG1 VanillaReAct] PASS={v_ok} turns={vr['turns']} wall={vr['wall_s']}s | {v_msg}")
                    pg1_entry["vanilla_react"] = {**vr, "passed": v_ok, "grade": v_msg}
                finally:
                    shutil.rmtree(ws_v, ignore_errors=True)

            ws_r = Path(tempfile.mkdtemp(prefix="ff_pg1_rep_"))
            try:
                agent = ReplicaAgent(ff.client, ff.skills, ff.memory, ws_r, ReplicaConfig(max_turns=22, verbose=True), referee_client=ref_client)
                rr = await agent.run(PG1_MISSION)
                r_ok, r_msg = _run_hidden_pytest(ws_r, PG1_HIDDEN_GRADER)
                print(
                    f"  [PG1 ReplicaAgent] PASS={r_ok} referee_ok={rr.confirmed_by_referee} "
                    f"turns={rr.turns_used} forks={rr.strategic_forks} ref_rejects={rr.referee_rejections} "
                    f"wall={rr.wall_s:.1f}s files={rr.modified_files} | {r_msg}"
                )
                pg1_entry["replica_agent"] = {
                    "passed": r_ok,
                    "grade": r_msg,
                    "confirmed_by_referee": rr.confirmed_by_referee,
                    "turns": rr.turns_used,
                    "forks": rr.strategic_forks,
                    "referee_rejections": rr.referee_rejections,
                    "files": rr.modified_files,
                    "wall_s": round(rr.wall_s, 2),
                    "stats": rr.token_stats,
                    "referee_log": rr.referee_log,
                }
            finally:
                shutil.rmtree(ws_r, ignore_errors=True)
            report["PG1_greenfield_wal_db"] = pg1_entry
            out_path.write_text(json.dumps(report, indent=2))

        # --- PG2: Vague Production Incident across 8-File Service ---
        if "PG2" in selected:
            print("\n" + "=" * 76)
            print("PG2: VAGUE 3-INCIDENT PRODUCTION DEBUGGING (8-FILE ASYNC QUEUE SERVICE)")
            print("=" * 76)
            pg2_entry: dict[str, Any] = {}
            if run_vanilla:
                ws_v = _setup_pg2_workspace()
                try:
                    vr = await run_vanilla_react(vc, ws_v, PG2_MISSION, max_turns=18)
                    v_ok, v_msg = _run_hidden_pytest(ws_v, PG2_HIDDEN_GRADER)
                    print(f"  [PG2 VanillaReAct] PASS={v_ok} turns={vr['turns']} wall={vr['wall_s']}s | {v_msg}")
                    pg2_entry["vanilla_react"] = {**vr, "passed": v_ok, "grade": v_msg}
                finally:
                    shutil.rmtree(ws_v, ignore_errors=True)

            ws_r = _setup_pg2_workspace()
            try:
                agent = ReplicaAgent(ff.client, ff.skills, ff.memory, ws_r, ReplicaConfig(max_turns=22, verbose=True), referee_client=ref_client)
                rr = await agent.run(PG2_MISSION)
                r_ok, r_msg = _run_hidden_pytest(ws_r, PG2_HIDDEN_GRADER)
                print(
                    f"  [PG2 ReplicaAgent] PASS={r_ok} referee_ok={rr.confirmed_by_referee} "
                    f"turns={rr.turns_used} forks={rr.strategic_forks} ref_rejects={rr.referee_rejections} "
                    f"wall={rr.wall_s:.1f}s files={rr.modified_files} | {r_msg}"
                )
                pg2_entry["replica_agent"] = {
                    "passed": r_ok,
                    "grade": r_msg,
                    "confirmed_by_referee": rr.confirmed_by_referee,
                    "turns": rr.turns_used,
                    "forks": rr.strategic_forks,
                    "referee_rejections": rr.referee_rejections,
                    "files": rr.modified_files,
                    "wall_s": round(rr.wall_s, 2),
                    "stats": rr.token_stats,
                    "referee_log": rr.referee_log,
                }
            finally:
                shutil.rmtree(ws_r, ignore_errors=True)
            report["PG2_vague_incident_debug"] = pg2_entry
            out_path.write_text(json.dumps(report, indent=2))

        # --- PG3: Real Read-Only Investigation on ACTUAL TensorFold Zig/Metal Repo ---
        if "PG3" in selected:
            print("\n" + "=" * 76)
            print("PG3: REAL READ-ONLY INVESTIGATION ON ACTUAL TENSORFOLD ZIG/METAL CODEBASE")
            print("=" * 76)
            pg3_entry: dict[str, Any] = {}
            if run_vanilla:
                vr = await run_vanilla_react(vc, TENSORFOLD_REPO, PG3_MISSION, read_only=True, max_turns=16)
                v_ok, v_msg = grade_pg3_report(vr["summary"])
                print(f"  [PG3 VanillaReAct] PASS={v_ok} turns={vr['turns']} wall={vr['wall_s']}s | {v_msg}")
                pg3_entry["vanilla_react"] = {**vr, "passed": v_ok, "grade": v_msg}

            agent = ReplicaAgent(
                ff.client, ff.skills, ff.memory, TENSORFOLD_REPO,
                ReplicaConfig(max_turns=18, read_only=True, verbose=True),
                referee_client=ref_client,
            )
            rr = await agent.run(PG3_MISSION)
            r_ok, r_msg = grade_pg3_report(rr.summary)
            print(
                f"  [PG3 ReplicaAgent] PASS={r_ok} referee_ok={rr.confirmed_by_referee} "
                f"turns={rr.turns_used} forks={rr.strategic_forks} ref_rejects={rr.referee_rejections} "
                f"wall={rr.wall_s:.1f}s | {r_msg}"
            )
            pg3_entry["replica_agent"] = {
                "passed": r_ok,
                "grade": r_msg,
                "confirmed_by_referee": rr.confirmed_by_referee,
                "turns": rr.turns_used,
                "forks": rr.strategic_forks,
                "referee_rejections": rr.referee_rejections,
                "wall_s": round(rr.wall_s, 2),
                "stats": rr.token_stats,
                "referee_log": rr.referee_log,
                "summary": rr.summary,
            }
            report["PG3_tensorfold_zig_investigation"] = pg3_entry
            out_path.write_text(json.dumps(report, indent=2))

        # --- PG4: Low-Level ARM64 NEON 4-bit Affine Quantized GEMV Kernel Optimization ---
        if "PG4" in selected:
            print("\n" + "=" * 76)
            print("PG4: LOW-LEVEL ARM64 NEON 4-BIT AFFINE GEMV KERNEL OPTIMIZATION (>=3x EXACT)")
            print("=" * 76)
            # Planted control: unoptimized starter kernel MUST compile cleanly, pass exactness, and fail speedup (< 3.0x)!
            ctrl4 = _setup_pg4_workspace()
            try:
                c4_ok, c4_msg = _grade_pg4_hidden(ctrl4)
                assert not c4_ok and "COMPILE FAIL" not in c4_msg and "BENCH (1024x4096):" in c4_msg, (
                    f"PG4 planted control unexpected: ok={c4_ok} msg={c4_msg}"
                )
                print(f"[PG4 CONTROL OK] Unoptimized kernel caught -> {c4_msg}")
            finally:
                shutil.rmtree(ctrl4, ignore_errors=True)

            pg4_entry: dict[str, Any] = {}
            if run_vanilla:
                ws_v = _setup_pg4_workspace()
                try:
                    vr = await run_vanilla_react(vc, ws_v, PG4_MISSION, max_turns=16)
                    v_ok, v_msg = _grade_pg4_hidden(ws_v)
                    print(f"  [PG4 VanillaReAct] PASS={v_ok} turns={vr['turns']} wall={vr['wall_s']}s | {v_msg}")
                    pg4_entry["vanilla_react"] = {**vr, "passed": v_ok, "grade": v_msg}
                finally:
                    shutil.rmtree(ws_v, ignore_errors=True)

            ws_r = _setup_pg4_workspace()
            try:
                agent = ReplicaAgent(ff.client, ff.skills, ff.memory, ws_r, ReplicaConfig(max_turns=18, verbose=True), referee_client=ref_client)
                rr = await agent.run(PG4_MISSION)
                r_ok, r_msg = _grade_pg4_hidden(ws_r)
                print(
                    f"  [PG4 ReplicaAgent] PASS={r_ok} referee_ok={rr.confirmed_by_referee} "
                    f"turns={rr.turns_used} forks={rr.strategic_forks} ref_rejects={rr.referee_rejections} "
                    f"wall={rr.wall_s:.1f}s | {r_msg}"
                )
                pg4_entry["replica_agent"] = {
                    "passed": r_ok,
                    "grade": r_msg,
                    "confirmed_by_referee": rr.confirmed_by_referee,
                    "turns": rr.turns_used,
                    "forks": rr.strategic_forks,
                    "referee_rejections": rr.referee_rejections,
                    "wall_s": round(rr.wall_s, 2),
                    "stats": rr.token_stats,
                    "referee_log": rr.referee_log,
                }
            finally:
                shutil.rmtree(ws_r, ignore_errors=True)
            report["PG4_neon_affine4_gemv_kernel"] = pg4_entry
            out_path.write_text(json.dumps(report, indent=2))

    finally:
      if ref_client:
          await ref_client.close()
    print(f"\nSaved Proving Ground report to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", nargs="*", default=["PG1", "PG2", "PG3", "PG4"])
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--referee-url", default=None)
    p.add_argument("--out", default="bench/results/replica_proving_ground.json")
    p.add_argument("--no-vanilla", action="store_true")
    args = p.parse_args()
    asyncio.run(run_proving_ground(args.challenges, args.url, run_vanilla=not args.no_vanilla, referee_url=args.referee_url, out_file=args.out))


if __name__ == "__main__":
    main()
