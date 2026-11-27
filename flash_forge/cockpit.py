"""Cortheon Live Web Cockpit & Real-Time Event Bus.

Zero-dependency, 100%-local live telemetry dashboard and thread-safe / async-friendly
event bus designed to mount straight onto ``cortheon proxy`` at ``/ui``.

Key components:
  - :class:`CockpitBus`: bounded ring buffers for live slot trace events and completed
    run summaries, real-time KPI counters, and non-blocking drop-oldest fan-out to
    async SSE subscribers (so a slow or paused browser tab never stalls inference).
  - :func:`create_cockpit_router`: FastAPI ``APIRouter`` exposing:
      * ``GET /ui`` — self-contained, zero-CDN, XSS-safe dark scientific cockpit.
      * ``GET /v1/cortheon/snapshot`` — JSON state snapshot.
      * ``GET /v1/cortheon/events`` — live SSE stream (``snapshot`` + ``trace`` + keepalives).
      * ``GET /v1/cortheon/flywheel/export`` — downloadable deduplicated, PII-scrubbed DPO JSONL.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Optional
import uuid

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .flywheel import DEFAULT_FLYWHEEL_LOG, FlywheelPair, harvest_from_results_dir
from .sandbox import sandbox_backend

__all__ = [
    "COCKPIT_VERSION",
    "CockpitBus",
    "CockpitSubscription",
    "build_dpo_ndjson",
    "create_cockpit_router",
    "render_cockpit_html",
]

COCKPIT_VERSION = "2.2.0"

_DEFAULT_TOKENS_PER_ABORTED_SLOT = 600
_DEFAULT_TOKENS_PER_VERIFIER_CALL = 1200
_RANGE_RE = re.compile(r"^\s*(?:slots?\s+)?(\d+)\s*\.\.\s*(\d+)\s*$", re.IGNORECASE)

_COUNTER_KEYS: tuple[str, ...] = (
    "total_runs",
    "verified_runs",
    "early_aborted_slots",
    "rescues_triggered",
    "spec_drafter_accepts",
    "spec_escalations",
    "bugs_proven",
    "flywheel_pairs_recorded",
    "total_tokens",
    "tokens_saved_est",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_nonneg_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, num)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _to_json_safe(value: Any, _depth: int = 0) -> Any:
    """Convert arbitrary Python values into strictly JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _depth >= 8:
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item, _depth + 1) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item, _depth + 1) for item in sorted(value, key=str)]
    return str(value)


def _extract_tokens(ev: dict[str, Any]) -> int:
    for key in ("tokens", "total_tokens"):
        val = _coerce_nonneg_int(ev.get(key))
        if val is not None:
            return val
    prompt_tok = _coerce_nonneg_int(ev.get("prompt_tokens"))
    comp_tok = _coerce_nonneg_int(ev.get("completion_tokens"))
    if prompt_tok is not None or comp_tok is not None:
        return (prompt_tok or 0) + (comp_tok or 0)
    stats = ev.get("token_stats")
    if isinstance(stats, dict):
        total = _coerce_nonneg_int(stats.get("total_tokens"))
        if total is not None:
            return total
        p = _coerce_nonneg_int(stats.get("prompt_tokens")) or 0
        c = _coerce_nonneg_int(stats.get("completion_tokens")) or 0
        if p or c:
            return p + c
    return 0


def _extract_aborted_count(ev: dict[str, Any]) -> int:
    for key in ("early_aborted_slots", "aborted_slots", "count"):
        val = _coerce_nonneg_int(ev.get(key))
        if val is not None:
            return val
    for key in ("cancelled", "cancelled_slots"):
        if key not in ev or ev[key] is None:
            continue
        raw = ev[key]
        int_val = _coerce_nonneg_int(raw)
        if int_val is not None:
            return int_val
        if isinstance(raw, (list, tuple, set)):
            return len(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return 0
            m = _RANGE_RE.match(text)
            if m:
                start_i, end_i = int(m.group(1)), int(m.group(2))
                return max(0, end_i - start_i + 1)
            parts = [p.strip() for p in text.split(",") if p.strip()]
            return len(parts) if parts else 0
    return 1


def _extract_explicit_tokens_saved(ev: dict[str, Any]) -> Optional[int]:
    for key in ("tokens_saved_est", "tokens_saved"):
        val = _coerce_nonneg_int(ev.get(key))
        if val is not None:
            return val
    return None


def _enqueue_drop_oldest(
    queue: asyncio.Queue[dict[str, Any]],
    item: dict[str, Any],
    sub: Optional["CockpitSubscription"] = None,
) -> None:
    """Push an item into ``queue`` without blocking, dropping the oldest item if full."""
    while queue.full():
        try:
            queue.get_nowait()
            if sub is not None:
                sub.dropped_events += 1
        except asyncio.QueueEmpty:
            break
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:  # pragma: no cover - defensive for maxsize==0
        if sub is not None:
            sub.dropped_events += 1


class CockpitSubscription(AsyncIterator[dict[str, Any]]):
    """Live event subscription supporting both ``async for`` and ``async with``.

    Registers its ``asyncio.Queue`` immediately when created inside a running event loop
    so events published right after ``bus.subscribe()`` are never lost to startup races.
    """

    def __init__(self, bus: "CockpitBus", maxsize: int = 256) -> None:
        self._bus = bus
        self._maxsize = max(1, int(maxsize))
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue[dict[str, Any]]] = None
        self._registered = False
        self._closed = False
        self.dropped_events: int = 0
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            self._register(running)

    @property
    def queue(self) -> Optional[asyncio.Queue[dict[str, Any]]]:
        return self._queue

    @property
    def closed(self) -> bool:
        return self._closed

    def _register(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._registered or self._closed:
            return
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=self._maxsize)
        self._bus._register_subscription(self)
        self._registered = True

    def _ensure_registered(self) -> asyncio.Queue[dict[str, Any]]:
        if not self._registered:
            self._register(asyncio.get_running_loop())
        assert self._queue is not None
        return self._queue

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._registered:
            self._bus._unregister_subscription(self)
            self._registered = False

    async def aclose(self) -> None:
        self.close()

    async def next_event(self, timeout: Optional[float] = None) -> dict[str, Any]:
        """Await the next event without closing the subscription on timeout."""
        if self._closed:
            raise StopAsyncIteration
        q = self._ensure_registered()
        if timeout is None:
            return await q.get()
        return await asyncio.wait_for(q.get(), timeout=timeout)

    def __aiter__(self) -> "CockpitSubscription":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        q = self._ensure_registered()
        try:
            return await q.get()
        except asyncio.CancelledError:
            self.close()
            raise

    async def __aenter__(self) -> "CockpitSubscription":
        if not self._closed:
            self._ensure_registered()
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown guard
            pass


class CockpitBus:
    """Thread-safe, async-friendly telemetry bus for Cortheon's Live Web Cockpit."""

    def __init__(
        self,
        *,
        max_events: int = 500,
        max_runs: int = 100,
        subscriber_queue_size: int = 256,
        version: str = COCKPIT_VERSION,
        sandbox: Optional[str] = None,
    ) -> None:
        self.max_events = max(0, int(max_events))
        self.max_runs = max(0, int(max_runs))
        self.subscriber_queue_size = max(1, int(subscriber_queue_size))
        self._version = str(version or COCKPIT_VERSION)
        self._sandbox_override = sandbox

        self._lock = threading.Lock()
        self._seq: int = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._runs: deque[dict[str, Any]] = deque(maxlen=self.max_runs)
        self._counters: dict[str, int] = {k: 0 for k in _COUNTER_KEYS}
        self._subscribers: list[CockpitSubscription] = []

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def _register_subscription(self, sub: CockpitSubscription) -> None:
        with self._lock:
            if sub not in self._subscribers:
                self._subscribers.append(sub)

    def _unregister_subscription(self, sub: CockpitSubscription) -> None:
        with self._lock:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass

    def _apply_counters_locked(self, etype: str, ev: dict[str, Any]) -> None:
        tokens = _extract_tokens(ev)
        explicit_saved = _extract_explicit_tokens_saved(ev)

        if etype == "slot_done":
            if tokens:
                self._counters["total_tokens"] += tokens
        elif etype == "early_abort":
            aborted = _extract_aborted_count(ev)
            self._counters["early_aborted_slots"] += aborted
            if explicit_saved is not None:
                self._counters["tokens_saved_est"] += explicit_saved
            elif aborted > 0:
                self._counters["tokens_saved_est"] += aborted * _DEFAULT_TOKENS_PER_ABORTED_SLOT
        elif etype == "wave2_start":
            if tokens:
                self._counters["total_tokens"] += tokens
        elif etype == "drafter_accepted":
            self._counters["spec_drafter_accepts"] += 1
            if tokens:
                self._counters["total_tokens"] += tokens
            if explicit_saved is not None:
                self._counters["tokens_saved_est"] += explicit_saved
            else:
                saved_calls = _coerce_nonneg_int(ev.get("verifier_calls_saved"))
                calls_mult = saved_calls if (saved_calls is not None and saved_calls > 0) else 1
                self._counters["tokens_saved_est"] += calls_mult * _DEFAULT_TOKENS_PER_VERIFIER_CALL
        elif etype in ("escalated", "spec_escalated"):
            self._counters["spec_escalations"] += 1
            if tokens:
                self._counters["total_tokens"] += tokens
        elif etype == "bug_proven":
            count = _coerce_nonneg_int(ev.get("count"))
            self._counters["bugs_proven"] += count if (count is not None and count > 0) else 1
        elif etype in ("rescue_triggered", "death_spiral_rescue", "rescue"):
            self._counters["rescues_triggered"] += 1
            if tokens:
                self._counters["total_tokens"] += tokens
        elif etype == "flywheel_recorded":
            pairs = _coerce_nonneg_int(ev.get("pairs"))
            if pairs is None:
                pairs = _coerce_nonneg_int(ev.get("count"))
            self._counters["flywheel_pairs_recorded"] += pairs if (pairs is not None and pairs > 0) else 1
        elif etype == "done":
            self._counters["total_runs"] += 1
            if bool(ev.get("verified")):
                self._counters["verified_runs"] += 1
            if tokens:
                self._counters["total_tokens"] += tokens
            if explicit_saved is not None:
                self._counters["tokens_saved_est"] += explicit_saved
        else:
            if tokens:
                self._counters["total_tokens"] += tokens
            if explicit_saved is not None:
                self._counters["tokens_saved_est"] += explicit_saved

    def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        """Stamp ``seq`` + UTC ``ts``, update counters, record in ring buffer, and broadcast."""
        if isinstance(event, dict):
            normalized: dict[str, Any] = _to_json_safe(dict(event))
        else:
            normalized = {"type": "event", "message": _to_json_safe(event)}

        raw_type = normalized.get("type") or normalized.get("event") or normalized.get("kind") or "event"
        etype = str(raw_type).strip().lower() or "event"
        normalized["type"] = etype

        try:
            current_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        with self._lock:
            self._seq += 1
            normalized["seq"] = self._seq
            normalized["ts"] = _utc_now_iso()

            self._apply_counters_locked(etype, normalized)
            if self.max_events > 0:
                self._events.append(dict(normalized))

            stale: list[CockpitSubscription] = []
            for sub in list(self._subscribers):
                if sub.closed or sub._loop is None or sub._queue is None:
                    stale.append(sub)
                    continue
                if sub._loop.is_closed():
                    stale.append(sub)
                    continue
                payload = dict(normalized)
                if current_loop is sub._loop:
                    _enqueue_drop_oldest(sub._queue, payload, sub)
                else:
                    try:
                        sub._loop.call_soon_threadsafe(_enqueue_drop_oldest, sub._queue, payload, sub)
                    except RuntimeError:
                        stale.append(sub)

            for sub in stale:
                try:
                    self._subscribers.remove(sub)
                except ValueError:
                    pass

        return dict(normalized)

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        """Allow passing ``bus`` directly as an ``on_event=`` callback."""
        return self.publish(event)

    def record_run(self, summary: dict[str, Any]) -> None:
        """Record completed run metadata into the bounded ``runs`` ring buffer."""
        raw = _to_json_safe(dict(summary)) if isinstance(summary, dict) else {"summary": _to_json_safe(summary)}
        run_record: dict[str, Any] = dict(raw)
        run_record["id"] = str(raw.get("id") or f"run-{uuid.uuid4().hex[:8]}")
        run_record["mode"] = str(raw.get("mode") or "auto")
        run_record["wall_s"] = round(_coerce_float(raw.get("wall_s"), 0.0), 3)
        run_record["verified"] = bool(raw.get("verified", False))
        run_record["calls"] = _coerce_nonneg_int(raw.get("calls")) or 0
        run_record["tokens"] = _coerce_nonneg_int(raw.get("tokens")) or 0
        aborted = _coerce_nonneg_int(raw.get("aborted_slots"))
        if aborted is None:
            aborted = _coerce_nonneg_int(raw.get("early_aborted_slots")) or 0
        run_record["aborted_slots"] = aborted
        run_record["summary"] = str(raw.get("summary") or "")
        run_record.setdefault("ts", _utc_now_iso())

        with self._lock:
            if self.max_runs > 0:
                self._runs.append(run_record)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of live counters, recent events, and recent runs."""
        active_sandbox = self._sandbox_override or sandbox_backend()
        with self._lock:
            return {
                "version": self._version,
                "sandbox_backend": active_sandbox,
                "counters": dict(self._counters),
                "recent_events": [dict(ev) for ev in self._events],
                "recent_runs": [dict(run) for run in self._runs],
            }

    def subscribe(self, *, maxsize: Optional[int] = None) -> CockpitSubscription:
        """Subscribe to live events (supports both ``async for`` and ``async with``)."""
        qsize = self.subscriber_queue_size if maxsize is None else maxsize
        return CockpitSubscription(self, maxsize=qsize)


# ---------------------------------------------------------------------------
# Flywheel DPO NDJSON Export Helper
# ---------------------------------------------------------------------------


def _parse_bool_flag(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    text = str(raw or "").strip().lower()
    return text in ("1", "true", "yes", "on")


def _resolve_results_dir(results_dir: Optional[Path]) -> Optional[Path]:
    if results_dir is not None:
        return Path(results_dir).expanduser()
    cwd_results = Path.cwd() / "results"
    if cwd_results.is_dir():
        return cwd_results
    pkg_results = Path(__file__).resolve().parents[1] / "results"
    if pkg_results.is_dir():
        return pkg_results
    return None


def build_dpo_ndjson(
    *,
    flywheel_path: Optional[Path] = None,
    include_results: bool = False,
    results_dir: Optional[Path] = None,
    verified_only: bool = True,
) -> tuple[str, int]:
    """Harvest, PII-scrub, and exact-deduplicate verified DPO preference pairs as NDJSON."""
    pairs: list[FlywheelPair] = []

    fw_target = Path(flywheel_path).expanduser() if flywheel_path is not None else DEFAULT_FLYWHEEL_LOG
    if fw_target.exists():
        pairs.extend(harvest_from_results_dir(fw_target))

    if include_results:
        res_target = _resolve_results_dir(results_dir)
        if res_target is not None and res_target.exists():
            pairs.extend(harvest_from_results_dir(res_target))

    seen: set[tuple[str, str, str]] = set()
    lines: list[str] = []
    for raw_pair in pairs:
        if verified_only and not raw_pair.chosen_verified:
            continue
        clean = raw_pair.scrubbed()
        if not clean.prompt or not clean.chosen or not clean.rejected or clean.chosen == clean.rejected:
            continue
        key = (clean.prompt, clean.chosen, clean.rejected)
        if key in seen:
            continue
        seen.add(key)
        lines.append(json.dumps(clean.to_dpo_dict(), ensure_ascii=False))

    payload = ("\n".join(lines) + "\n") if lines else ""
    return payload, len(lines)


# ---------------------------------------------------------------------------
# Self-Contained, Zero-CDN, XSS-Safe Dark Scientific Cockpit HTML
# ---------------------------------------------------------------------------

_COCKPIT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CORTHEON // Live Verification Cockpit</title>
<style>
  :root {
    --bg-void: #060910;
    --bg-panel: #0c121e;
    --bg-card: #111a2b;
    --bg-card-hover: #162238;
    --border-subtle: #1c2a42;
    --border-bright: #2a3f63;
    --text-primary: #e6eefb;
    --text-secondary: #8ea3c4;
    --text-muted: #5a7194;
    --cyan: #22d3ee;
    --cyan-soft: rgba(34, 211, 238, 0.12);
    --emerald: #10b981;
    --emerald-soft: rgba(16, 185, 129, 0.13);
    --amber: #f59e0b;
    --amber-soft: rgba(245, 158, 11, 0.13);
    --violet: #a78bfa;
    --violet-soft: rgba(167, 139, 250, 0.14);
    --red: #f43f5e;
    --red-soft: rgba(244, 63, 94, 0.14);
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    padding: 0;
    min-height: 100%;
    background-color: var(--bg-void);
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(34, 211, 238, 0.08), transparent),
      linear-gradient(rgba(34, 211, 238, 0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(34, 211, 238, 0.025) 1px, transparent 1px);
    background-size: 100% 100%, 36px 36px, 36px 36px;
    color: var(--text-primary);
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
  }

  .shell {
    max-width: 1520px;
    margin: 0 auto;
    padding: 18px 22px 28px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Header */
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 18px;
    background: linear-gradient(180deg, rgba(17, 26, 43, 0.94), rgba(12, 18, 30, 0.96));
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .brand-mark {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(167, 139, 250, 0.16));
    border: 1px solid rgba(34, 211, 238, 0.45);
    display: grid;
    place-items: center;
    flex-shrink: 0;
  }

  .brand-title {
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: var(--text-primary);
  }

  .brand-sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .status-cluster {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid var(--border-subtle);
    background: rgba(11, 17, 29, 0.9);
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }

  .pill-sse-connected {
    border-color: rgba(16, 185, 129, 0.45);
    background: var(--emerald-soft);
    color: #34d399;
  }

  .pill-sse-reconnecting {
    border-color: rgba(245, 158, 11, 0.45);
    background: var(--amber-soft);
    color: #fbbf24;
  }

  .pill-sandbox {
    border-color: rgba(34, 211, 238, 0.35);
    background: var(--cyan-soft);
    color: #67e8f9;
  }

  .pill-sandbox-unsafe {
    border-color: rgba(244, 63, 94, 0.5);
    background: var(--red-soft);
    color: #fda4af;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 8px currentColor;
  }

  .dot-pulse {
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.82); }
  }

  /* KPI Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
  }

  @media (max-width: 1280px) {
    .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  }

  @media (max-width: 700px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }

  .kpi-card {
    position: relative;
    overflow: hidden;
    padding: 14px 15px 13px;
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 11px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: border-color 0.15s ease, background-color 0.15s ease;
  }

  .kpi-card:hover {
    border-color: var(--border-bright);
    background: var(--bg-card);
  }

  .kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent, var(--cyan));
    opacity: 0.85;
  }

  .kpi-emerald { --accent: var(--emerald); }
  .kpi-amber   { --accent: var(--amber); }
  .kpi-violet  { --accent: var(--violet); }
  .kpi-cyan    { --accent: var(--cyan); }
  .kpi-red     { --accent: var(--red); }

  .kpi-label {
    font-family: var(--mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
  }

  .kpi-value-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }

  .kpi-value {
    font-family: var(--mono);
    font-size: 27px;
    font-weight: 700;
    line-height: 1.1;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
  }

  .kpi-sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .export-btn {
    margin-top: 2px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 7px;
    border: 1px solid rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.14);
    color: #6ee7b7;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
  }

  .export-btn:hover {
    background: rgba(16, 185, 129, 0.25);
    border-color: rgba(16, 185, 129, 0.8);
    color: #ecfdf5;
  }

  .export-btn:active {
    transform: translateY(1px);
  }

  /* Main Split Stage */
  .stage {
    display: grid;
    grid-template-columns: minmax(0, 1.38fr) minmax(0, 1fr);
    gap: 16px;
    align-items: stretch;
  }

  @media (max-width: 1080px) {
    .stage { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    min-height: 540px;
    max-height: 70vh;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 12px 15px;
    border-bottom: 1px solid var(--border-subtle);
    background: rgba(17, 26, 43, 0.72);
  }

  .panel-title-group {
    display: flex;
    align-items: center;
    gap: 9px;
  }

  .panel-title {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-primary);
  }

  .count-chip {
    padding: 2px 7px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 10.5px;
    background: rgba(34, 211, 238, 0.12);
    color: var(--cyan);
    border: 1px solid rgba(34, 211, 238, 0.25);
    font-variant-numeric: tabular-nums;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }

  .filter-bar {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px;
    border-radius: 8px;
    background: rgba(6, 9, 16, 0.75);
    border: 1px solid var(--border-subtle);
  }

  .filter-pill {
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-secondary);
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.12s ease;
  }

  .filter-pill:hover {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.05);
  }

  .filter-pill.active {
    color: #06131a;
    background: var(--cyan);
    font-weight: 700;
  }

  .toggle-btn {
    border: 1px solid var(--border-bright);
    background: rgba(17, 26, 43, 0.9);
    color: var(--text-secondary);
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 7px;
    cursor: pointer;
    transition: all 0.12s ease;
  }

  .toggle-btn[aria-pressed="true"] {
    border-color: rgba(16, 185, 129, 0.5);
    background: var(--emerald-soft);
    color: #6ee7b7;
  }

  /* Live Trace Stream */
  .stream-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    scrollbar-width: thin;
    scrollbar-color: var(--border-bright) transparent;
  }

  .trace-card {
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--tone, var(--cyan));
    background: rgba(17, 26, 43, 0.65);
    padding: 9px 11px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    animation: slideIn 0.16s ease-out;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .tone-cyan    { --tone: var(--cyan); background: linear-gradient(90deg, rgba(34, 211, 238, 0.08), rgba(17, 26, 43, 0.68) 40%); }
  .tone-emerald { --tone: var(--emerald); background: linear-gradient(90deg, rgba(16, 185, 129, 0.09), rgba(17, 26, 43, 0.68) 40%); }
  .tone-amber   { --tone: var(--amber); background: linear-gradient(90deg, rgba(245, 158, 11, 0.1), rgba(17, 26, 43, 0.68) 40%); }
  .tone-violet  { --tone: var(--violet); background: linear-gradient(90deg, rgba(167, 139, 250, 0.11), rgba(17, 26, 43, 0.68) 40%); }
  .tone-red     { --tone: var(--red); background: linear-gradient(90deg, rgba(244, 63, 94, 0.12), rgba(17, 26, 43, 0.68) 40%); }

  .trace-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    font-family: var(--mono);
    font-size: 11px;
  }

  .trace-badges {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }

  .seq-tag {
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
  }

  .type-badge {
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--tone, var(--cyan));
    border: 1px solid var(--tone, var(--cyan));
    background: rgba(6, 9, 16, 0.55);
  }

  .engine-tag {
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 10px;
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
  }

  .ts-tag {
    color: var(--text-muted);
    font-size: 10.5px;
    font-variant-numeric: tabular-nums;
  }

  .trace-summary {
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-primary);
    word-break: break-word;
  }

  .trace-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }

  /* Runs Table */
  .table-wrap {
    flex: 1;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border-bright) transparent;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    text-align: left;
    padding: 10px 12px;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    background: #0f1727;
    border-bottom: 1px solid var(--border-subtle);
  }

  thead th.num, tbody td.num {
    text-align: right;
  }

  tbody tr {
    border-bottom: 1px solid rgba(28, 42, 66, 0.6);
    transition: background-color 0.1s ease;
  }

  tbody tr:hover {
    background: rgba(34, 211, 238, 0.045);
  }

  tbody td {
    padding: 9px 12px;
    vertical-align: middle;
    color: var(--text-primary);
  }

  .status-pill {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  .status-verified {
    background: var(--emerald-soft);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.45);
  }

  .status-unverified {
    background: var(--amber-soft);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.45);
  }

  .run-summary-sub {
    display: block;
    margin-top: 3px;
    font-size: 10.5px;
    color: var(--text-muted);
    max-width: 220px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .aborted-hi {
    color: #fbbf24;
    font-weight: 700;
  }

  .empty-state {
    padding: 42px 20px;
    text-align: center;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-muted);
  }
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <polygon points="12,2 21,7 21,17 12,22 3,17 3,7" stroke="#22d3ee" stroke-width="1.8" fill="rgba(34,211,238,0.12)"/>
          <circle cx="12" cy="12" r="3.2" fill="#10b981"/>
        </svg>
      </div>
      <div>
        <div class="brand-title">CORTHEON // LIVE VERIFICATION COCKPIT</div>
        <div class="brand-sub">LOCAL EXECUTION-VERIFIED TELEMETRY &middot; <span id="version-tag">v__VERSION__</span></div>
      </div>
    </div>
    <div class="status-cluster">
      <div id="sse-pill" class="pill pill-sse-reconnecting" role="status" aria-live="polite">
        <span class="dot dot-pulse"></span>
        <span id="sse-label">CONNECTING</span>
      </div>
      <div id="sandbox-pill" class="pill __SANDBOX_CLASS__" title="Active OS Execution Sandbox">
        <span id="sandbox-label">__SANDBOX__</span>
      </div>
      <div class="pill" title="Cumulative processed and estimated saved tokens">
        <span id="token-summary">0 tok &middot; ~0 saved</span>
      </div>
    </div>
  </header>

  <section class="kpi-grid" aria-label="Verification KPIs">
    <article class="kpi-card kpi-emerald">
      <div class="kpi-label">Verified Runs</div>
      <div class="kpi-value" id="kpi-verified-runs">0</div>
      <div class="kpi-sub" id="kpi-verified-sub">0 total runs</div>
    </article>

    <article class="kpi-card kpi-amber">
      <div class="kpi-label">Early-Aborted GPU Slots</div>
      <div class="kpi-value" id="kpi-early-aborts">0</div>
      <div class="kpi-sub" id="kpi-tokens-saved">~0 tok saved est</div>
    </article>

    <article class="kpi-card kpi-violet">
      <div class="kpi-label">Death-Spiral Rescues</div>
      <div class="kpi-value" id="kpi-rescues">0</div>
      <div class="kpi-sub">clean-context B=3</div>
    </article>

    <article class="kpi-card kpi-cyan">
      <div class="kpi-label">SpecVerify Drafter Saves</div>
      <div class="kpi-value" id="kpi-spec-accepts">0</div>
      <div class="kpi-sub" id="kpi-spec-sub">0 escalations</div>
    </article>

    <article class="kpi-card kpi-red">
      <div class="kpi-label">Proven Diff Bugs</div>
      <div class="kpi-value" id="kpi-bugs-proven">0</div>
      <div class="kpi-sub">sandbox reproducers</div>
    </article>

    <article class="kpi-card kpi-emerald">
      <div class="kpi-label">Flywheel DPO Pairs</div>
      <div class="kpi-value-row">
        <div class="kpi-value" id="kpi-flywheel-pairs">0</div>
      </div>
      <a id="export-dpo-btn" class="export-btn" href="/v1/cortheon/flywheel/export" download="cortheon_dpo.jsonl">
        Export DPO JSONL
      </a>
    </article>
  </section>

  <main class="stage">
    <section class="panel" aria-label="Live Slot Race Stream">
      <div class="panel-header">
        <div class="panel-title-group">
          <span class="panel-title">LIVE SLOT RACE STREAM</span>
          <span class="count-chip" id="stream-count">0</span>
        </div>
        <div class="controls">
          <div class="filter-bar" role="group" aria-label="Stream category filter">
            <button type="button" class="filter-pill active" data-filter="all">all</button>
            <button type="button" class="filter-pill" data-filter="reason">reason</button>
            <button type="button" class="filter-pill" data-filter="code">code</button>
            <button type="button" class="filter-pill" data-filter="spec">spec</button>
            <button type="button" class="filter-pill" data-filter="guard">guard</button>
            <button type="button" class="filter-pill" data-filter="rescue">rescue</button>
          </div>
          <button type="button" id="autoscroll-toggle" class="toggle-btn" aria-pressed="true">Auto-scroll: ON</button>
        </div>
      </div>
      <div id="trace-stream" class="stream-body">
        <div class="empty-state" id="stream-empty">Awaiting live verification traces from Cortheon proxy...</div>
      </div>
    </section>

    <section class="panel" aria-label="Recent Verified Runs">
      <div class="panel-header">
        <div class="panel-title-group">
          <span class="panel-title">RECENT VERIFIED RUNS</span>
          <span class="count-chip" id="runs-count">0</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>mode</th>
              <th>status</th>
              <th class="num">wall_s</th>
              <th class="num">calls</th>
              <th class="num">tokens</th>
              <th class="num">aborted</th>
            </tr>
          </thead>
          <tbody id="runs-tbody">
            <tr id="runs-empty-row">
              <td colspan="6" class="empty-state">No completed runs recorded yet.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
</div>

<script>
(function() {
  "use strict";

  const MAX_UI_EVENTS = 500;
  const state = {
    counters: {
      total_runs: 0,
      verified_runs: 0,
      early_aborted_slots: 0,
      rescues_triggered: 0,
      spec_drafter_accepts: 0,
      spec_escalations: 0,
      bugs_proven: 0,
      flywheel_pairs_recorded: 0,
      total_tokens: 0,
      tokens_saved_est: 0
    },
    events: [],
    runs: [],
    seenSeq: 0,
    activeFilter: "all",
    autoScroll: true,
    refreshTimer: null
  };

  const els = {
    versionTag: document.getElementById("version-tag"),
    ssePill: document.getElementById("sse-pill"),
    sseLabel: document.getElementById("sse-label"),
    sandboxPill: document.getElementById("sandbox-pill"),
    sandboxLabel: document.getElementById("sandbox-label"),
    tokenSummary: document.getElementById("token-summary"),
    kpiVerifiedRuns: document.getElementById("kpi-verified-runs"),
    kpiVerifiedSub: document.getElementById("kpi-verified-sub"),
    kpiEarlyAborts: document.getElementById("kpi-early-aborts"),
    kpiTokensSaved: document.getElementById("kpi-tokens-saved"),
    kpiRescues: document.getElementById("kpi-rescues"),
    kpiSpecAccepts: document.getElementById("kpi-spec-accepts"),
    kpiSpecSub: document.getElementById("kpi-spec-sub"),
    kpiBugsProven: document.getElementById("kpi-bugs-proven"),
    kpiFlywheelPairs: document.getElementById("kpi-flywheel-pairs"),
    streamCount: document.getElementById("stream-count"),
    traceStream: document.getElementById("trace-stream"),
    runsCount: document.getElementById("runs-count"),
    runsTbody: document.getElementById("runs-tbody"),
    autoscrollToggle: document.getElementById("autoscroll-toggle")
  };

  function fmtInt(n) {
    const val = Number(n) || 0;
    return val.toLocaleString();
  }

  function fmtTime(isoStr) {
    if (!isoStr || typeof isoStr !== "string") return "--:--:--";
    const d = new Date(isoStr);
    if (Number.isNaN(d.getTime())) return String(isoStr).slice(11, 23) || "--:--:--";
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mm = String(d.getUTCMinutes()).padStart(2, "0");
    const ss = String(d.getUTCSeconds()).padStart(2, "0");
    const ms = String(d.getUTCMilliseconds()).padStart(3, "0");
    return hh + ":" + mm + ":" + ss + "." + ms + "Z";
  }

  function setSseState(connected) {
    if (connected) {
      els.ssePill.className = "pill pill-sse-connected";
      els.sseLabel.textContent = "CONNECTED";
    } else {
      els.ssePill.className = "pill pill-sse-reconnecting";
      els.sseLabel.textContent = "RECONNECTING";
    }
  }

  function setSandbox(backend) {
    const label = String(backend || "unknown");
    els.sandboxLabel.textContent = label;
    els.sandboxPill.className = label === "unsafe-disabled"
      ? "pill pill-sandbox-unsafe"
      : "pill pill-sandbox";
  }

  function renderCounters() {
    const c = state.counters;
    els.kpiVerifiedRuns.textContent = fmtInt(c.verified_runs);
    const total = Number(c.total_runs) || 0;
    const rate = total > 0 ? Math.round(((Number(c.verified_runs) || 0) / total) * 100) + "% pass" : "awaiting runs";
    els.kpiVerifiedSub.textContent = fmtInt(total) + " total · " + rate;

    els.kpiEarlyAborts.textContent = fmtInt(c.early_aborted_slots);
    els.kpiTokensSaved.textContent = "~" + fmtInt(c.tokens_saved_est) + " tok saved est";

    els.kpiRescues.textContent = fmtInt(c.rescues_triggered);

    els.kpiSpecAccepts.textContent = fmtInt(c.spec_drafter_accepts);
    els.kpiSpecSub.textContent = fmtInt(c.spec_escalations) + " escalations";

    els.kpiBugsProven.textContent = fmtInt(c.bugs_proven);
    els.kpiFlywheelPairs.textContent = fmtInt(c.flywheel_pairs_recorded);

    els.tokenSummary.textContent = fmtInt(c.total_tokens) + " tok · ~" + fmtInt(c.tokens_saved_est) + " saved";
  }

  function eventCategories(ev) {
    const tags = new Set(["all"]);
    const eng = String(ev.engine || ev.mode || "").toLowerCase();
    const etype = String(ev.type || ev.event || "").toLowerCase();

    if (eng.includes("reason") || eng === "pal" || eng === "math") tags.add("reason");
    if (eng.includes("code")) tags.add("code");
    if (eng.includes("spec")) tags.add("spec");
    if (eng.includes("guard") || eng.includes("diff")) tags.add("guard");
    if (eng.includes("rescue")) tags.add("rescue");

    if (etype === "drafter_accepted" || etype === "escalated" || etype === "spec_escalated") tags.add("spec");
    if (etype === "bug_proven") tags.add("guard");
    if (etype === "rescue_triggered" || etype === "rescue" || etype === "death_spiral_rescue" || etype === "wave2_start") {
      tags.add("rescue");
    }
    return tags;
  }

  function eventTone(ev) {
    const etype = String(ev.type || ev.event || "").toLowerCase();
    if (etype === "start") return "tone-cyan";
    if (etype === "early_abort" || etype === "escalated" || etype === "spec_escalated") return "tone-amber";
    if (etype === "wave2_start" || etype === "rescue_triggered" || etype === "rescue" || etype === "death_spiral_rescue") {
      return "tone-violet";
    }
    if (etype === "bug_proven") return "tone-red";
    if (etype === "drafter_accepted" || etype === "flywheel_recorded") return "tone-emerald";
    if (etype === "slot_done") {
      const ok = ev.ok !== undefined ? Boolean(ev.ok) : (ev.perfect !== undefined ? Boolean(ev.perfect) : true);
      return ok ? "tone-emerald" : "tone-amber";
    }
    if (etype === "done") {
      return ev.verified ? "tone-emerald" : "tone-amber";
    }
    return "tone-cyan";
  }

  function eventLabel(ev) {
    const etype = String(ev.type || ev.event || "event").toLowerCase();
    if (etype === "slot_done") {
      const ok = ev.ok !== undefined ? Boolean(ev.ok) : (ev.perfect !== undefined ? Boolean(ev.perfect) : true);
      return ok ? "slot_done [verified]" : "slot_done [fail]";
    }
    if (etype === "done") return ev.verified ? "done [verified]" : "done";
    return etype;
  }

  function buildEventSummary(ev) {
    const etype = String(ev.type || ev.event || "event").toLowerCase();
    if (etype === "start") {
      const head = ev.engine || ev.mode || "solve";
      const streams = ev.streams ? " · " + ev.streams + " parallel streams" : "";
      return "Launching " + head + streams;
    }
    if (etype === "slot_done") {
      const slot = ev.slot !== undefined ? "slot " + ev.slot : "slot";
      const role = ev.role ? " (" + ev.role + ")" : "";
      const tests = (ev.passed !== undefined && ev.total !== undefined) ? " · " + ev.passed + "/" + ev.total + " checks" : "";
      const summary = ev.summary ? " · " + ev.summary : "";
      const ans = ev.answer !== undefined && ev.answer !== "" ? " -> " + ev.answer : "";
      return slot + role + tests + summary + ans;
    }
    if (etype === "early_abort") {
      const c = ev.cancelled !== undefined ? ev.cancelled : ev.cancelled_slots;
      const cText = Array.isArray(c) ? "slots [" + c.join(", ") + "]" : (c !== undefined ? String(c) : "lagging slots");
      const reason = ev.reason ? " (" + ev.reason + ")" : "";
      const cons = ev.consensus_answer ? " · consensus -> " + ev.consensus_answer : "";
      return "Aborted " + cText + reason + cons;
    }
    if (etype === "wave2_start") {
      const s = ev.streams ? ev.streams + " falsification streams" : "Wave-2 falsification";
      return ev.reason ? s + " · " + ev.reason : s;
    }
    if (etype === "drafter_accepted") {
      const dm = ev.drafter_model || "drafter";
      const ans = ev.answer ? " -> " + ev.answer : "";
      const saved = ev.verifier_calls_saved ? " (saved " + ev.verifier_calls_saved + " verifier calls)" : "";
      return "Accepted by " + dm + ans + saved;
    }
    if (etype === "escalated" || etype === "spec_escalated") {
      const vm = ev.verifier_model || "verifier";
      return "Escalated -> " + vm + (ev.reason ? " (" + ev.reason + ")" : "");
    }
    if (etype === "bug_proven") {
      const cls = ev.passed_on_base === true ? "REGRESSION: " : (ev.passed_on_base === false ? "PRE-EXISTING: " : "");
      return cls + (ev.title || ev.summary || "Execution reproducer failed");
    }
    if (etype === "rescue_triggered" || etype === "rescue" || etype === "death_spiral_rescue") {
      const fails = ev.consecutive_failures ? ev.consecutive_failures + " consecutive failures · " : "";
      return fails + (ev.summary || ev.reason || "Clean-context B=3 graveyard backtracking injected");
    }
    if (etype === "flywheel_recorded") {
      const dom = ev.domain ? "[" + ev.domain + "] " : "";
      return dom + (ev.summary || "Verified DPO rescue pair persisted to flywheel.jsonl");
    }
    if (etype === "done") {
      const winner = ev.winner_name ? "winner=" + ev.winner_name + " · " : "";
      return winner + (ev.summary || (ev.verified ? "Verified completion" : "Completed"));
    }
    return ev.summary || ev.message || JSON.stringify(ev);
  }

  function createTraceCard(ev) {
    const card = document.createElement("article");
    card.className = "trace-card " + eventTone(ev);

    const head = document.createElement("div");
    head.className = "trace-head";

    const badges = document.createElement("div");
    badges.className = "trace-badges";

    const seqEl = document.createElement("span");
    seqEl.className = "seq-tag";
    seqEl.textContent = "#" + String(ev.seq || 0);
    badges.appendChild(seqEl);

    const typeEl = document.createElement("span");
    typeEl.className = "type-badge";
    typeEl.textContent = eventLabel(ev);
    badges.appendChild(typeEl);

    const engVal = ev.engine || ev.mode;
    if (engVal) {
      const engEl = document.createElement("span");
      engEl.className = "engine-tag";
      engEl.textContent = String(engVal);
      badges.appendChild(engEl);
    }

    const tsEl = document.createElement("span");
    tsEl.className = "ts-tag";
    tsEl.textContent = fmtTime(ev.ts);

    head.appendChild(badges);
    head.appendChild(tsEl);
    card.appendChild(head);

    const summaryEl = document.createElement("div");
    summaryEl.className = "trace-summary";
    summaryEl.textContent = buildEventSummary(ev);
    card.appendChild(summaryEl);

    const metaParts = [];
    const wall = ev.elapsed_s !== undefined ? ev.elapsed_s : (ev.latency_s !== undefined ? ev.latency_s : ev.wall_s);
    if (wall !== undefined && wall !== null) {
      const num = Number(wall);
      if (!Number.isNaN(num)) metaParts.push(num.toFixed(2) + "s");
    }
    if (ev.tokens !== undefined && ev.tokens !== null) {
      metaParts.push(fmtInt(ev.tokens) + " tok");
    }
    if (ev.wave !== undefined && ev.wave !== null) {
      metaParts.push("wave " + String(ev.wave));
    }
    if (ev.role) {
      metaParts.push("role=" + String(ev.role));
    }
    if (metaParts.length > 0) {
      const metaEl = document.createElement("div");
      metaEl.className = "trace-meta";
      for (let i = 0; i < metaParts.length; i++) {
        const chip = document.createElement("span");
        chip.textContent = metaParts[i];
        metaEl.appendChild(chip);
      }
      card.appendChild(metaEl);
    }
    return card;
  }

  function renderStream() {
    while (els.traceStream.firstChild) {
      els.traceStream.removeChild(els.traceStream.firstChild);
    }
    const visible = [];
    for (let i = 0; i < state.events.length; i++) {
      const ev = state.events[i];
      if (eventCategories(ev).has(state.activeFilter)) {
        visible.push(ev);
      }
    }
    els.streamCount.textContent = fmtInt(visible.length);
    if (visible.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = state.activeFilter === "all"
        ? "Awaiting live verification traces from Cortheon proxy..."
        : "No events matching filter '" + state.activeFilter + "'.";
      els.traceStream.appendChild(empty);
      return;
    }
    const frag = document.createDocumentFragment();
    for (let i = 0; i < visible.length; i++) {
      frag.appendChild(createTraceCard(visible[i]));
    }
    els.traceStream.appendChild(frag);
    if (state.autoScroll) {
      els.traceStream.scrollTop = els.traceStream.scrollHeight;
    }
  }

  function renderRuns() {
    while (els.runsTbody.firstChild) {
      els.runsTbody.removeChild(els.runsTbody.firstChild);
    }
    els.runsCount.textContent = fmtInt(state.runs.length);
    if (state.runs.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.className = "empty-state";
      td.textContent = "No completed runs recorded yet.";
      tr.appendChild(td);
      els.runsTbody.appendChild(tr);
      return;
    }

    const frag = document.createDocumentFragment();
    for (let i = state.runs.length - 1; i >= 0; i--) {
      const run = state.runs[i];
      const tr = document.createElement("tr");

      const tdMode = document.createElement("td");
      const modeSpan = document.createElement("span");
      modeSpan.className = "engine-tag";
      modeSpan.textContent = String(run.mode || "auto");
      tdMode.appendChild(modeSpan);
      if (run.summary) {
        const sub = document.createElement("span");
        sub.className = "run-summary-sub";
        sub.textContent = String(run.summary);
        sub.setAttribute("title", String(run.summary));
        tdMode.appendChild(sub);
      }
      tr.appendChild(tdMode);

      const tdStatus = document.createElement("td");
      const stPill = document.createElement("span");
      stPill.className = run.verified ? "status-pill status-verified" : "status-pill status-unverified";
      stPill.textContent = run.verified ? "VERIFIED" : "UNVERIFIED";
      tdStatus.appendChild(stPill);
      tr.appendChild(tdStatus);

      const tdWall = document.createElement("td");
      tdWall.className = "num";
      tdWall.textContent = (Number(run.wall_s) || 0).toFixed(2) + "s";
      tr.appendChild(tdWall);

      const tdCalls = document.createElement("td");
      tdCalls.className = "num";
      tdCalls.textContent = fmtInt(run.calls);
      tr.appendChild(tdCalls);

      const tdTokens = document.createElement("td");
      tdTokens.className = "num";
      tdTokens.textContent = fmtInt(run.tokens);
      tr.appendChild(tdTokens);

      const tdAborted = document.createElement("td");
      tdAborted.className = Number(run.aborted_slots) > 0 ? "num aborted-hi" : "num";
      tdAborted.textContent = fmtInt(run.aborted_slots);
      tr.appendChild(tdAborted);

      frag.appendChild(tr);
    }
    els.runsTbody.appendChild(frag);
  }

  function applySnapshot(snap) {
    if (!snap || typeof snap !== "object") return;
    if (snap.version) els.versionTag.textContent = "v" + String(snap.version);
    if (snap.sandbox_backend) setSandbox(snap.sandbox_backend);
    if (snap.counters && typeof snap.counters === "object") {
      state.counters = Object.assign({}, state.counters, snap.counters);
      renderCounters();
    }
    if (Array.isArray(snap.recent_events)) {
      state.events = snap.recent_events.slice(-MAX_UI_EVENTS);
      let maxSeq = state.seenSeq;
      for (let i = 0; i < state.events.length; i++) {
        const s = Number(state.events[i].seq) || 0;
        if (s > maxSeq) maxSeq = s;
      }
      state.seenSeq = maxSeq;
      renderStream();
    }
    if (Array.isArray(snap.recent_runs)) {
      state.runs = snap.recent_runs.slice();
      renderRuns();
    }
  }

  function applyLiveCounters(ev) {
    const c = state.counters;
    const etype = String(ev.type || ev.event || "").toLowerCase();
    const tok = Number(ev.tokens || ev.total_tokens || 0) || 0;
    const expSaved = ev.tokens_saved_est !== undefined ? Number(ev.tokens_saved_est)
      : (ev.tokens_saved !== undefined ? Number(ev.tokens_saved) : null);

    if (etype === "slot_done") {
      c.total_tokens += tok;
    } else if (etype === "early_abort") {
      let aborted = 1;
      if (typeof ev.early_aborted_slots === "number") aborted = ev.early_aborted_slots;
      else if (typeof ev.aborted_slots === "number") aborted = ev.aborted_slots;
      else if (Array.isArray(ev.cancelled)) aborted = ev.cancelled.length;
      else if (Array.isArray(ev.cancelled_slots)) aborted = ev.cancelled_slots.length;
      else if (typeof ev.cancelled === "number") aborted = ev.cancelled;
      c.early_aborted_slots += Math.max(0, aborted);
      c.tokens_saved_est += expSaved !== null ? Math.max(0, expSaved) : Math.max(0, aborted) * 600;
    } else if (etype === "drafter_accepted") {
      c.spec_drafter_accepts += 1;
      c.total_tokens += tok;
      const callsSaved = Number(ev.verifier_calls_saved) || 1;
      c.tokens_saved_est += expSaved !== null ? Math.max(0, expSaved) : callsSaved * 1200;
    } else if (etype === "escalated" || etype === "spec_escalated") {
      c.spec_escalations += 1;
      c.total_tokens += tok;
    } else if (etype === "bug_proven") {
      c.bugs_proven += Number(ev.count) > 0 ? Number(ev.count) : 1;
    } else if (etype === "rescue_triggered" || etype === "rescue" || etype === "death_spiral_rescue") {
      c.rescues_triggered += 1;
      c.total_tokens += tok;
    } else if (etype === "flywheel_recorded") {
      const pairs = Number(ev.pairs || ev.count || 1) || 1;
      c.flywheel_pairs_recorded += Math.max(1, pairs);
    } else if (etype === "done") {
      c.total_runs += 1;
      if (ev.verified) c.verified_runs += 1;
      c.total_tokens += tok;
    } else if (tok > 0) {
      c.total_tokens += tok;
    }
    renderCounters();
  }

  function scheduleSnapshotRefresh() {
    if (state.refreshTimer !== null) return;
    state.refreshTimer = window.setTimeout(function() {
      state.refreshTimer = null;
      fetchSnapshot();
    }, 250);
  }

  function handleTraceEvent(ev) {
    if (!ev || typeof ev !== "object") return;
    const seq = Number(ev.seq) || 0;
    if (seq > 0 && seq <= state.seenSeq) return;
    if (seq > state.seenSeq) state.seenSeq = seq;

    state.events.push(ev);
    if (state.events.length > MAX_UI_EVENTS) {
      state.events.splice(0, state.events.length - MAX_UI_EVENTS);
    }
    applyLiveCounters(ev);

    if (eventCategories(ev).has(state.activeFilter)) {
      const emptyEl = els.traceStream.querySelector(".empty-state");
      if (emptyEl) emptyEl.remove();
      els.traceStream.appendChild(createTraceCard(ev));
      while (els.traceStream.children.length > MAX_UI_EVENTS) {
        els.traceStream.removeChild(els.traceStream.firstChild);
      }
      els.streamCount.textContent = fmtInt(els.traceStream.children.length);
      if (state.autoScroll) {
        els.traceStream.scrollTop = els.traceStream.scrollHeight;
      }
    }

    const etype = String(ev.type || ev.event || "").toLowerCase();
    if (etype === "done") {
      scheduleSnapshotRefresh();
    }
  }

  function fetchSnapshot() {
    fetch("/v1/cortheon/snapshot", { cache: "no-store" })
      .then(function(res) { return res.ok ? res.json() : null; })
      .then(function(data) { if (data) applySnapshot(data); })
      .catch(function() {});
  }

  function connectEvents() {
    const es = new EventSource('/v1/cortheon/events');
    es.onopen = function() { setSseState(true); };
    es.addEventListener("snapshot", function(msg) {
      setSseState(true);
      try { applySnapshot(JSON.parse(msg.data)); } catch (_e) {}
    });
    es.addEventListener("trace", function(msg) {
      setSseState(true);
      try { handleTraceEvent(JSON.parse(msg.data)); } catch (_e) {}
    });
    es.onerror = function() { setSseState(false); };
  }

  // Filter buttons
  const filterBtns = document.querySelectorAll(".filter-pill");
  for (let i = 0; i < filterBtns.length; i++) {
    filterBtns[i].addEventListener("click", function() {
      const next = this.getAttribute("data-filter") || "all";
      state.activeFilter = next;
      for (let j = 0; j < filterBtns.length; j++) {
        filterBtns[j].classList.toggle("active", filterBtns[j] === this);
      }
      renderStream();
    });
  }

  // Auto-scroll toggle
  els.autoscrollToggle.addEventListener("click", function() {
    state.autoScroll = !state.autoScroll;
    this.setAttribute("aria-pressed", state.autoScroll ? "true" : "false");
    this.textContent = state.autoScroll ? "Auto-scroll: ON" : "Auto-scroll: OFF";
    if (state.autoScroll) {
      els.traceStream.scrollTop = els.traceStream.scrollHeight;
    }
  });

  fetchSnapshot();
  connectEvents();
})();
</script>
</body>
</html>
"""


def render_cockpit_html(bus: CockpitBus) -> str:
    """Render the self-contained, zero-CDN, XSS-safe cockpit HTML page."""
    snap = bus.snapshot()
    version_safe = html.escape(str(snap.get("version") or COCKPIT_VERSION), quote=True)
    sandbox_raw = str(snap.get("sandbox_backend") or sandbox_backend())
    sandbox_safe = html.escape(sandbox_raw, quote=True)
    sandbox_cls = "pill-sandbox-unsafe" if sandbox_raw == "unsafe-disabled" else "pill-sandbox"
    return (
        _COCKPIT_HTML_TEMPLATE.replace("__VERSION__", version_safe)
        .replace("__SANDBOX__", sandbox_safe)
        .replace("__SANDBOX_CLASS__", sandbox_cls)
    )


# ---------------------------------------------------------------------------
# FastAPI Router Factory
# ---------------------------------------------------------------------------


def create_cockpit_router(
    bus: CockpitBus,
    *,
    flywheel_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    keepalive_s: float = 15.0,
) -> APIRouter:
    """Create the FastAPI ``APIRouter`` mounting Cortheon's Live Web Cockpit & SSE endpoints."""
    router = APIRouter(tags=["cortheon-cockpit"])
    ka_interval = max(0.05, float(keepalive_s))

    @router.get("/ui", response_class=HTMLResponse)
    async def cockpit_ui() -> HTMLResponse:
        return HTMLResponse(
            content=render_cockpit_html(bus),
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": (
                    "default-src 'none'; "
                    "style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; "
                    "connect-src 'self'; "
                    "img-src 'self' data:"
                ),
            },
        )

    @router.get("/v1/cortheon/snapshot")
    async def cockpit_snapshot() -> JSONResponse:
        return JSONResponse(
            content=bus.snapshot(),
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/v1/cortheon/events")
    async def cockpit_events(request: Request) -> StreamingResponse:
        async def _sse_stream() -> AsyncIterator[str]:
            async with bus.subscribe() as sub:
                snap = bus.snapshot()
                recent = snap.get("recent_events") or []
                last_seq = int(recent[-1].get("seq", 0)) if recent else 0
                yield f"event: snapshot\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n"

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await sub.next_event(timeout=ka_interval)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    except StopAsyncIteration:
                        break
                    except asyncio.CancelledError:
                        break

                    ev_seq = int(ev.get("seq", 0) or 0)
                    if ev_seq and ev_seq <= last_seq:
                        continue
                    if ev_seq > last_seq:
                        last_seq = ev_seq

                    payload = json.dumps(ev, ensure_ascii=False)
                    if ev_seq:
                        yield f"id: {ev_seq}\nevent: trace\ndata: {payload}\n\n"
                    else:
                        yield f"event: trace\ndata: {payload}\n\n"

        return StreamingResponse(
            _sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/v1/cortheon/flywheel/export")
    async def cockpit_flywheel_export(
        include_results: str = Query(default="0", description="Include harvested results/ benchmarks (0 or 1)."),
    ) -> Response:
        include_res = _parse_bool_flag(include_results)
        ndjson_body, count = build_dpo_ndjson(
            flywheel_path=flywheel_path,
            include_results=include_res,
            results_dir=results_dir,
        )
        return Response(
            content=ndjson_body,
            status_code=200,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": 'attachment; filename="cortheon_dpo.jsonl"',
                "Cache-Control": "no-store",
                "X-Cortheon-Exported-Pairs": str(count),
            },
        )

    return router
