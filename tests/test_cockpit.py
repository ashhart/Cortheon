"""Offline unit and FastAPI integration tests for Cortheon's Live Web Cockpit & Event Bus."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from flash_forge.cockpit import (
    COCKPIT_VERSION,
    CockpitBus,
    build_dpo_ndjson,
    create_cockpit_router,
    render_cockpit_html,
)
from flash_forge.flywheel import record_live_trace


# ---------------------------------------------------------------------------
# CockpitBus unit tests: publishing, seq, ring buffers, counters, backpressure
# ---------------------------------------------------------------------------


def test_bus_publish_monotonic_seq_utc_ts_and_json_safety():
    bus = CockpitBus(max_events=20, sandbox="macos-seatbelt")

    e1 = bus.publish({"type": "start", "engine": "reason", "streams": 4, "path": Path("/tmp/demo")})
    e2 = bus({"event": "slot_done", "slot": 1, "ok": True, "tokens": 350, "bad_float": float("nan")})
    e3 = bus.publish({"type": "done", "verified": True, "tags": {"z3", "pal"}})

    assert [e1["seq"], e2["seq"], e3["seq"]] == [1, 2, 3]
    for ev in (e1, e2, e3):
        assert isinstance(ev["ts"], str)
        assert ev["ts"].endswith("Z")
        assert "T" in ev["ts"]

    assert e1["path"] == "/tmp/demo"
    assert e2["type"] == "slot_done"
    assert e2["bad_float"] is None
    assert e3["tags"] == ["pal", "z3"]

    snap = bus.snapshot()
    # Must be strictly JSON-serializable
    encoded = json.dumps(snap)
    decoded = json.loads(encoded)
    assert decoded["version"] == COCKPIT_VERSION
    assert decoded["sandbox_backend"] == "macos-seatbelt"
    assert [ev["seq"] for ev in decoded["recent_events"]] == [1, 2, 3]


def test_bus_ring_buffer_caps_events_and_runs():
    bus = CockpitBus(max_events=5, max_runs=3)

    for i in range(12):
        bus.publish({"type": "slot_done", "slot": i + 1, "tokens": 100})

    for i in range(7):
        bus.record_run(
            {
                "id": f"run-{i + 1}",
                "mode": "code",
                "wall_s": 0.5 + i,
                "verified": i % 2 == 0,
                "calls": 4,
                "tokens": 1000 + i,
                "aborted_slots": 2,
                "summary": f"summary {i + 1}",
            }
        )

    snap = bus.snapshot()
    assert len(snap["recent_events"]) == 5
    assert [ev["seq"] for ev in snap["recent_events"]] == [8, 9, 10, 11, 12]
    assert snap["counters"]["total_tokens"] == 1200

    assert len(snap["recent_runs"]) == 3
    assert [r["id"] for r in snap["recent_runs"]] == ["run-5", "run-6", "run-7"]
    assert snap["recent_runs"][-1]["mode"] == "code"
    assert snap["recent_runs"][-1]["aborted_slots"] == 2
    assert snap["recent_runs"][-1]["summary"] == "summary 7"


def test_bus_counters_across_all_event_types():
    bus = CockpitBus()

    # slot_done -> total_tokens
    bus.publish({"type": "slot_done", "slot": 1, "tokens": 420, "ok": True})
    bus.publish({"event": "slot_done", "slot": 2, "prompt_tokens": 100, "completion_tokens": 280})

    # early_abort variants: list, int, code_forge field, range str, fallback 1
    bus.publish({"type": "early_abort", "cancelled": [3, 4], "elapsed_s": 1.8})  # +2 slots, +1200 saved
    bus.publish({"event": "early_abort", "early_aborted_slots": 3, "tokens_saved": 1500})  # +3 slots, +1500 saved
    bus.publish({"type": "early_abort", "cancelled": "2..4"})  # +3 slots, +1800 saved
    bus.publish({"type": "early_abort"})  # +1 slot, +600 saved
    bus.publish({"type": "early_abort", "cancelled": []})  # +0 slots, +0 saved

    # wave2_start -> recorded without inflating rescue/spec counters
    bus.publish({"type": "wave2_start", "engine": "reason", "streams": 4, "reason": "Wave-1 split"})

    # drafter_accepted -> spec_drafter_accepts + tokens_saved_est
    bus.publish({"type": "drafter_accepted", "drafter_model": "qwen-4b", "verifier_calls_saved": 2, "tokens": 150})
    bus.publish({"type": "drafter_accepted", "drafter_model": "qwen-4b", "tokens_saved_est": 900})

    # escalated -> spec_escalations
    bus.publish({"type": "escalated", "verifier_model": "glm-5", "reason": "drafter timeout", "tokens": 50})

    # bug_proven -> bugs_proven
    bus.publish({"type": "bug_proven", "title": "Off-by-one boundary", "passed_on_base": True})
    bus.publish({"type": "bug_proven", "title": "Batch bugs", "count": 2})

    # rescue_triggered -> rescues_triggered
    bus.publish({"type": "rescue_triggered", "consecutive_failures": 2, "tokens": 300})

    # flywheel_recorded -> flywheel_pairs_recorded
    bus.publish({"type": "flywheel_recorded", "domain": "code"})
    bus.publish({"type": "flywheel_recorded", "domain": "math", "pairs": 2})

    # done -> total_runs + verified_runs (note: early_aborted_slots on done must NOT double-count!)
    bus.publish({"type": "done", "verified": True, "tokens": 500, "early_aborted_slots": 99})
    bus.publish({"type": "done", "verified": False, "tokens": 200})

    c = bus.snapshot()["counters"]
    assert c["total_runs"] == 2
    assert c["verified_runs"] == 1
    assert c["early_aborted_slots"] == 2 + 3 + 3 + 1 + 0
    assert c["rescues_triggered"] == 1
    assert c["spec_drafter_accepts"] == 2
    assert c["spec_escalations"] == 1
    assert c["bugs_proven"] == 3
    assert c["flywheel_pairs_recorded"] == 3
    assert c["total_tokens"] == 420 + 380 + 150 + 50 + 300 + 500 + 200
    assert c["tokens_saved_est"] == (1200 + 1500 + 1800 + 600) + (2400 + 900)


def test_bus_multithreaded_publish_preserves_strict_monotonic_seq():
    bus = CockpitBus(max_events=500)

    def _worker(tid: int) -> list[int]:
        seqs: list[int] = []
        for i in range(40):
            ev = bus.publish({"type": "slot_done", "slot": f"{tid}-{i}", "tokens": 10})
            seqs.append(ev["seq"])
        return seqs

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_worker, tid) for tid in range(8)]
        all_seqs = [s for f in futures for s in f.result()]

    assert len(all_seqs) == 320
    assert sorted(all_seqs) == list(range(1, 321))
    snap = bus.snapshot()
    assert snap["counters"]["total_tokens"] == 3200
    assert [ev["seq"] for ev in snap["recent_events"]] == list(range(1, 321))


def test_bus_slow_subscriber_nonblocking_drop_oldest_backpressure():
    async def _run() -> None:
        bus = CockpitBus(subscriber_queue_size=4)
        async with bus.subscribe() as slow_sub:
            assert bus.subscriber_count == 1
            t0 = time.perf_counter()
            for i in range(1, 51):
                bus.publish({"type": "slot_done", "slot": i, "tokens": 10})
            elapsed = time.perf_counter() - t0
            # Publishing 50 events into a size-4 queue must never stall
            assert elapsed < 0.5
            assert slow_sub.dropped_events == 46

            received: list[int] = []
            for _ in range(4):
                ev = await slow_sub.next_event(timeout=0.5)
                received.append(ev["seq"])
            assert received == [47, 48, 49, 50]

        assert bus.subscriber_count == 0

    asyncio.run(_run())


def test_bus_subscribe_async_delivery_order_and_cleanup():
    async def _run() -> None:
        bus = CockpitBus()

        # 1. Context-manager subscription + cross-thread publisher
        received_a: list[int] = []
        received_b: list[int] = []

        async with bus.subscribe() as sub_a, bus.subscribe() as sub_b:
            assert bus.subscriber_count == 2

            def _bg_publish() -> None:
                for idx in range(1, 6):
                    bus.publish({"type": "slot_done", "slot": idx, "tokens": idx * 10})

            thread = threading.Thread(target=_bg_publish)
            thread.start()

            for _ in range(5):
                ev_a = await asyncio.wait_for(sub_a.__anext__(), timeout=2.0)
                ev_b = await asyncio.wait_for(sub_b.__anext__(), timeout=2.0)
                received_a.append(ev_a["seq"])
                received_b.append(ev_b["seq"])

            thread.join(timeout=2.0)

        assert received_a == [1, 2, 3, 4, 5]
        assert received_b == [1, 2, 3, 4, 5]
        assert bus.subscriber_count == 0

        # 2. Direct `async for` subscription cleans up immediately on task cancel
        got_live: list[int] = []
        ready = asyncio.Event()

        async def _consumer() -> None:
            sub = bus.subscribe()
            ready.set()
            async for ev in sub:
                got_live.append(ev["seq"])

        task = asyncio.create_task(_consumer())
        await ready.wait()
        assert bus.subscriber_count == 1
        bus.publish({"type": "start", "engine": "code"})
        bus.publish({"type": "done", "verified": True})
        await asyncio.sleep(0.05)
        assert got_live == [6, 7]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert bus.subscriber_count == 0

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# FastAPI mounted router tests: /ui, /snapshot, /events, /flywheel/export
# ---------------------------------------------------------------------------


def _build_test_app(
    bus: CockpitBus,
    *,
    flywheel_path: Path | None = None,
    results_dir: Path | None = None,
    keepalive_s: float = 15.0,
) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_cockpit_router(
            bus,
            flywheel_path=flywheel_path,
            results_dir=results_dir,
            keepalive_s=keepalive_s,
        )
    )
    return app


def test_ui_route_zero_cdn_xss_safe_and_required_elements(tmp_path: Path):
    bus = CockpitBus(sandbox="macos-seatbelt")
    app = _build_test_app(bus, flywheel_path=tmp_path / "missing.jsonl")
    client = TestClient(app)

    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert "default-src 'none'" in resp.headers.get("content-security-policy", "")

    body = resp.text
    # Zero external CDN or third-party URLs
    assert "https://cdn" not in body.lower()
    assert "http://" not in body.lower()
    assert "https://" not in body.lower()

    # Core header badges & live endpoints
    assert "CORTHEON // LIVE VERIFICATION COCKPIT" in body
    assert "macos-seatbelt" in body
    assert "EventSource('/v1/cortheon/events')" in body
    assert "/v1/cortheon/snapshot" in body
    assert "/v1/cortheon/flywheel/export" in body

    # All 6 KPI cards + Export button
    for label in (
        "Verified Runs",
        "Early-Aborted GPU Slots",
        "Death-Spiral Rescues",
        "SpecVerify Drafter Saves",
        "Proven Diff Bugs",
        "Flywheel DPO Pairs",
        "Export DPO JSONL",
    ):
        assert label in body

    # Filter pills & auto-scroll toggle & runs table columns
    for filt in ("all", "reason", "code", "spec", "guard", "rescue"):
        assert f'data-filter="{filt}"' in body
    assert "Auto-scroll: ON" in body
    for col in ("mode", "status", "wall_s", "calls", "tokens", "aborted"):
        assert col in body

    # Strict XSS hygiene: textContent used, no dangerous DOM sinks
    assert ".textContent" in body
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in body


def test_ui_escapes_untrusted_server_metadata():
    evil_bus = CockpitBus(
        version='2.2.0"><script>alert(1)</script>',
        sandbox='<img src=x onerror=alert(2)>',
    )
    rendered = render_cockpit_html(evil_bus)
    assert "<script>alert(1)</script>" not in rendered
    assert "<img src=x onerror=alert(2)>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(2)&gt;" in rendered


def test_snapshot_endpoint_returns_exact_counters_events_and_runs(tmp_path: Path):
    bus = CockpitBus(sandbox="linux-bwrap")
    bus.publish({"type": "start", "engine": "code", "streams": 4})
    bus.publish({"type": "slot_done", "slot": 1, "role": "algo_fast", "ok": True, "tokens": 510})
    bus.publish({"type": "early_abort", "cancelled": [3, 4], "elapsed_s": 1.2})
    bus.publish({"type": "bug_proven", "title": "Null dereference on empty list"})
    bus.publish({"type": "done", "verified": True, "tokens": 200})
    bus.record_run(
        {
            "id": "run-abc",
            "mode": "code",
            "wall_s": 1.84,
            "verified": True,
            "calls": 3,
            "tokens": 710,
            "aborted_slots": 2,
            "summary": "algo_fast passed 12/12",
        }
    )

    app = _build_test_app(bus, flywheel_path=tmp_path / "missing.jsonl")
    client = TestClient(app)

    resp = client.get("/v1/cortheon/snapshot")
    assert resp.status_code == 200
    data = resp.json()

    assert data["version"] == COCKPIT_VERSION
    assert data["sandbox_backend"] == "linux-bwrap"
    assert data["counters"]["total_runs"] == 1
    assert data["counters"]["verified_runs"] == 1
    assert data["counters"]["early_aborted_slots"] == 2
    assert data["counters"]["bugs_proven"] == 1
    assert data["counters"]["total_tokens"] == 710
    assert data["counters"]["tokens_saved_est"] == 1200
    assert len(data["recent_events"]) == 5
    assert len(data["recent_runs"]) == 1
    assert data["recent_runs"][0]["id"] == "run-abc"
    assert data["recent_runs"][0]["aborted_slots"] == 2


def test_flywheel_export_empty_when_missing_and_valid_ndjson_when_present(tmp_path: Path):
    bus = CockpitBus()
    missing_fw = tmp_path / "does_not_exist.jsonl"
    app_empty = _build_test_app(bus, flywheel_path=missing_fw)
    client_empty = TestClient(app_empty)

    resp_empty = client_empty.get("/v1/cortheon/flywheel/export")
    assert resp_empty.status_code == 200
    assert "application/x-ndjson" in resp_empty.headers.get("content-type", "")
    assert 'filename="cortheon_dpo.jsonl"' in resp_empty.headers.get("content-disposition", "")
    assert resp_empty.headers.get("x-cortheon-exported-pairs") == "0"
    assert resp_empty.text == ""

    # Populate a tmp_path flywheel.jsonl with duplicates, unverified entries, and PII
    fw_file = tmp_path / "flywheel.jsonl"
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1L2"
    record_live_trace(
        fw_file,
        prompt=f"Fix loader at /Users/alice/repo/app.py using {secret}",
        chosen="def load(): return 42",
        rejected="def load(): return 192.168.1.99",
        domain="code",
        chosen_verified=True,
        source="live_rescue",
    )
    # Exact duplicate -> should be deduplicated
    record_live_trace(
        fw_file,
        prompt=f"Fix loader at /Users/alice/repo/app.py using {secret}",
        chosen="def load(): return 42",
        rejected="def load(): return 192.168.1.99",
        domain="code",
        chosen_verified=True,
        source="live_rescue",
    )
    # Distinct second valid pair
    record_live_trace(
        fw_file,
        prompt="Compute 6 * 7",
        chosen="ANSWER: 42",
        rejected="ANSWER: 41",
        domain="math",
        chosen_verified=True,
        source="live_rescue",
    )
    # Unverified record -> should be excluded from DPO export
    record_live_trace(
        fw_file,
        prompt="Unverified question",
        chosen="maybe 1",
        rejected="maybe 2",
        domain="math",
        chosen_verified=False,
    )
    # Missing rejected -> should be excluded from DPO export
    record_live_trace(
        fw_file,
        prompt="SFT only trace",
        chosen="def ok(): return True",
        rejected="",
        domain="code",
        chosen_verified=True,
    )

    app_fw = _build_test_app(bus, flywheel_path=fw_file)
    client_fw = TestClient(app_fw)
    resp = client_fw.get("/v1/cortheon/flywheel/export")
    assert resp.status_code == 200
    assert "application/x-ndjson" in resp.headers.get("content-type", "")
    assert 'filename="cortheon_dpo.jsonl"' in resp.headers.get("content-disposition", "")
    assert resp.headers.get("x-cortheon-exported-pairs") == "2"

    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    assert len(lines) == 2
    rows = [json.loads(ln) for ln in lines]
    for row in rows:
        assert set(row.keys()) == {"prompt", "chosen", "rejected", "domain", "source"}

    # Confirm PII was scrubbed
    full_text = resp.text
    assert "/Users/alice" not in full_text
    assert secret not in full_text
    assert "192.168.1.99" not in full_text
    assert "<HOME>" in full_text
    assert "<REDACTED_SECRET>" in full_text
    assert "<PRIVATE_IP>" in full_text


def test_flywheel_export_include_results_toggle(tmp_path: Path):
    bus = CockpitBus()
    fw_file = tmp_path / "flywheel.jsonl"
    record_live_trace(
        fw_file,
        prompt="Live rescue prompt",
        chosen="def f(): return 1",
        rejected="def f(): return 0",
        domain="code",
        chosen_verified=True,
    )

    res_dir = tmp_path / "results"
    res_dir.mkdir()
    (res_dir / "mini_vanilla.json").write_text(
        json.dumps(
            {
                "dataset": "mini_math",
                "model": "qwen-test",
                "results": [
                    {"id": "P1", "prompt": "Find 2+2", "passed": False, "predicted": "5", "expected": "4"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (res_dir / "mini_cortheon.json").write_text(
        json.dumps(
            {
                "dataset": "mini_math",
                "model": "qwen-test",
                "arm": "cortheon",
                "results": [
                    {"id": "P1", "prompt": "Find 2+2", "passed": True, "predicted": "4", "expected": "4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    app = _build_test_app(bus, flywheel_path=fw_file, results_dir=res_dir)
    client = TestClient(app)

    r0 = client.get("/v1/cortheon/flywheel/export?include_results=0")
    assert r0.status_code == 200
    assert len(r0.text.splitlines()) == 1

    r1 = client.get("/v1/cortheon/flywheel/export?include_results=1")
    assert r1.status_code == 200
    rows = [json.loads(ln) for ln in r1.text.splitlines() if ln.strip()]
    assert len(rows) == 2
    prompts = {r["prompt"] for r in rows}
    assert prompts == {"Live rescue prompt", "Find 2+2"}


def test_sse_events_endpoint_streams_snapshot_keepalive_and_live_traces(tmp_path: Path):
    async def _run() -> None:
        bus = CockpitBus(sandbox="macos-seatbelt")
        bus.publish({"type": "start", "engine": "reason", "streams": 4})

        app = _build_test_app(bus, flywheel_path=tmp_path / "missing.jsonl", keepalive_s=0.06)

        disconnect_ev = asyncio.Event()
        sent_status = 0
        sent_headers: dict[str, str] = {}
        chunks: asyncio.Queue[str] = asyncio.Queue()

        async def receive() -> dict[str, Any]:
            await disconnect_ev.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            nonlocal sent_status
            if message["type"] == "http.response.start":
                sent_status = int(message["status"])
                for k, v in message.get("headers", []):
                    sent_headers[k.decode("latin-1").lower()] = v.decode("latin-1")
            elif message["type"] == "http.response.body":
                raw = message.get("body", b"")
                if raw:
                    await chunks.put(raw.decode("utf-8"))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/cortheon/events",
            "raw_path": b"/v1/cortheon/events",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 18652),
        }

        asgi_task = asyncio.create_task(app(scope, receive, send))
        try:
            # 1. Initial snapshot frame
            first_chunk = await asyncio.wait_for(chunks.get(), timeout=2.0)
            assert sent_status == 200
            assert "text/event-stream" in sent_headers.get("content-type", "")
            assert sent_headers.get("cache-control") == "no-cache"
            assert first_chunk.startswith("event: snapshot\ndata: ")
            snap_json = first_chunk.split("data: ", 1)[1].strip()
            snap_obj = json.loads(snap_json)
            assert snap_obj["sandbox_backend"] == "macos-seatbelt"
            assert len(snap_obj["recent_events"]) == 1
            assert bus.subscriber_count == 1

            # 2. Periodic keepalive comment when idle
            ka_chunk = await asyncio.wait_for(chunks.get(), timeout=2.0)
            assert ka_chunk == ": keepalive\n\n"

            # 3. Live trace events arrive in strict seq order
            bus.publish({"type": "slot_done", "slot": 1, "role": "pal_exact", "ok": True, "tokens": 300})
            bus.publish({"type": "early_abort", "cancelled": [3, 4], "elapsed_s": 1.1})

            traces: list[dict[str, Any]] = []
            while len(traces) < 2:
                chunk = await asyncio.wait_for(chunks.get(), timeout=2.0)
                if chunk.startswith(": keepalive"):
                    continue
                assert "event: trace\n" in chunk
                data_line = [ln for ln in chunk.splitlines() if ln.startswith("data:")][0]
                traces.append(json.loads(data_line.split("data:", 1)[1].strip()))

            assert [t["seq"] for t in traces] == [2, 3]
            assert [t["type"] for t in traces] == ["slot_done", "early_abort"]
        finally:
            disconnect_ev.set()
            await asyncio.wait_for(asgi_task, timeout=2.0)

        assert bus.subscriber_count == 0

    asyncio.run(_run())


def test_build_dpo_ndjson_direct_helper(tmp_path: Path):
    empty_text, empty_n = build_dpo_ndjson(flywheel_path=tmp_path / "none.jsonl")
    assert empty_text == ""
    assert empty_n == 0


def test_bus_handles_malformed_events_and_minimal_record_run():
    bus = CockpitBus(max_events=10, max_runs=10)
    # Non-dict input and minimal record_run should never raise
    ev = bus.publish("raw status string")  # type: ignore[arg-type]
    assert ev["type"] == "event"
    assert ev["message"] == "raw status string"
    assert ev["seq"] == 1

    bus.record_run({})
    runs = bus.snapshot()["recent_runs"]
    assert len(runs) == 1
    assert runs[0]["id"].startswith("run-")
    assert runs[0]["mode"] == "auto"
    assert runs[0]["verified"] is False
    assert runs[0]["calls"] == 0
    assert runs[0]["tokens"] == 0
    assert runs[0]["aborted_slots"] == 0

