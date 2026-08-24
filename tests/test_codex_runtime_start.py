"""Cold-start and contention guarantees for the Codex runtime launcher."""

from __future__ import annotations

from cortheon.codex_plugins.cortheon.hooks import hook_transport


def test_runtime_lock_contender_waits_through_cold_start(monkeypatch) -> None:
    expected = {"protocol_version": "1.0.0"}
    attempts = 0

    def delayed_health(_url: str, _expected: dict[str, str]) -> bool:
        nonlocal attempts
        attempts += 1
        return attempts == 21

    monkeypatch.setattr(hook_transport, "_runtime_start_lock", lambda _port: None)
    monkeypatch.setattr(hook_transport, "_runtime_healthy_at", delayed_health)
    monkeypatch.setattr(hook_transport.time, "sleep", lambda _seconds: None)

    assert hook_transport._start_runtime(
        "http://127.0.0.1:19001",
        expected,
        ["/opt/cortheon-runtime"],
    )
    assert attempts == 21
