from __future__ import annotations

from fastapi.testclient import TestClient

from flash_forge.client import CompletionResult, ModelCapabilities
from flash_forge.duel import DUEL_PRESETS
from flash_forge.proxy import create_app
from flash_forge.rescue import detect_death_spiral, inject_rescue_note


def test_detect_death_spiral_counts_only_trailing_failures_since_latest_user():
    msgs = [
        {"role": "user", "content": "Old task"},
        {"role": "tool", "content": "AssertionError: old fail"},
        {"role": "tool", "content": "AssertionError: old fail 2"},
        {"role": "user", "content": "Fix SSI read-only anomaly"},
        {"role": "assistant", "content": "Running tests"},
        {"role": "tool", "content": "FAILED test_ssi.py::test_ro - AssertionError"},
    ]
    # Only 1 failure since latest user -> no spiral yet
    assert detect_death_spiral(msgs) is None

    # Second consecutive failure -> triggers!
    msgs.append({"role": "assistant", "content": "Trying patch 2"})
    msgs.append({"role": "tool", "content": "Traceback (most recent call last):\nAssertionError: cycle missed"})
    sp = detect_death_spiral(msgs)
    assert sp is not None
    assert sp.consecutive_failures == 2
    assert "Fix SSI" in sp.latest_user_goal
    assert len(sp.failure_snippets) == 2

    # Passing tool output resets spiral!
    msgs.append({"role": "tool", "content": "17 passed in 0.42s"})
    assert detect_death_spiral(msgs) is None


def test_inject_rescue_note_merges_into_system_without_late_system_message():
    msgs = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix bug"},
    ]
    out = inject_rescue_note(msgs, "[Cortheon Auto-Rescue] Fix invariant X")
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "You are a coding assistant." in out[0]["content"]
    assert "[Cortheon Auto-Rescue]" in out[0]["content"]

    no_sys = [{"role": "user", "content": "Fix bug"}]
    out2 = inject_rescue_note(no_sys, "[Cortheon Auto-Rescue] Fix invariant X")
    assert len(out2) == 2 and out2[0]["role"] == "system"


def test_proxy_auto_rescue_injects_diagnosis_once_on_death_spiral(monkeypatch):
    import httpx

    app = create_app(upstream_url="http://127.0.0.1:9999/v1", auto_rescue=True)
    forwarded: list[dict] = []
    rescue_batches: list[int] = []

    async def _fake_probe(self):
        return ModelCapabilities(model_id="glm-5.3-flash-tfq3")

    async def _fake_batch(self, specs):
        rescue_batches.append(len(specs))
        return [
            CompletionResult(
                content="Retain SIREAD locks past commit until overlapping txns finish.",
                reasoning="",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=30,
                reasoning_tokens=10,
                cached_tokens=0,
                latency_s=0.1,
                tag=s.tag,
            )
            for s in specs
        ]

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"id":"chatcmpl-r","choices":[{"index":0,"message":{"role":"assistant","content":"Fixed"},"finish_reason":"stop"}]}'

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def post(self, url, json=None):
            forwarded.append(json)
            return _FakeResp()

    monkeypatch.setattr("flash_forge.client.TensorFoldClient.probe", _fake_probe)
    monkeypatch.setattr("flash_forge.client.TensorFoldClient.batch", _fake_batch)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    tc = TestClient(app)
    payload = {
        "model": "cortheon",
        "tools": [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "system", "content": "Coding agent"},
            {"role": "user", "content": "Implement SSI"},
            {"role": "tool", "content": "FAILED test_ssi.py::test_ro - AssertionError"},
            {"role": "tool", "content": "FAILED test_ssi.py::test_ro - AssertionError"},
        ],
    }
    r1 = tc.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    assert rescue_batches == [3]
    assert "[Cortheon Auto-Rescue" in forwarded[0]["messages"][0]["content"]

    # Second identical turn must NOT re-trigger rescue (cooldown fingerprint)
    r2 = tc.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200
    assert rescue_batches == [3]


def test_duel_presets_defined():
    assert {"ring", "automaton", "coloring"} <= set(DUEL_PRESETS.keys())
    assert "2630600114653531854763423/183488738276457059168800" == DUEL_PRESETS["ring"][2]
