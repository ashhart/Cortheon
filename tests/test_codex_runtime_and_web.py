"""Codex runtime ownership and attributable current-web evidence."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import urllib.error
from pathlib import Path
from typing import Any

from cortheon.codex_plugins.cortheon.hooks import cortheon_hook, hook_transport
from cortheon.cognitive_hooks import CognitiveHookTracker, _host_observations
from cortheon.cognitive_install import install_codex
from cortheon.cognitive_runtime import CognitiveRuntime


def _request(capability: str = "search") -> dict[str, Any]:
    return {
        "request_id": "web-1",
        "capability": capability,
        "query": "current release",
        "parameters": {"purpose": "verify the current release"},
    }


def _receipt(observation: dict[str, Any]) -> dict[str, Any]:
    prefix = "[CORTHEON_HOST_EVIDENCE] "
    first = observation["content"].splitlines()[0]
    assert first.startswith(prefix)
    return json.loads(first[len(prefix) :])


def test_codex_search_uses_only_direct_structured_attribution() -> None:
    response = {
        "content": [{"type": "text", "text": "host envelope"}],
        "structuredContent": {
            "results": [
                {
                    "url": "HTTPS://Example.COM:443/release#fragment",
                    "title": "Release note",
                    "snippet": "Version 4 shipped.",
                    "publishedAt": "2026-08-22",
                    "provider": "host-index",
                    "sourceType": "release-note",
                    "authority": "primary",
                }
            ]
        },
    }
    metadata = cortheon_hook._tool_metadata(response)
    observations = _host_observations(
        _request(),
        "web__run",
        cortheon_hook._tool_output(response),
        succeeded=True,
        host_input={"search_query": [{"q": "current release"}]},
        tool_metadata=metadata,
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation["kind"] == "web"
    assert observation["url"] == "https://example.com/release"
    assert observation["source"] == observation["url"]
    assert observation["published_at"] == "2026-08-22"
    assert observation["retrieved_at"].endswith("Z")
    receipt = _receipt(observation)
    assert receipt["lineage"] == {
        "origin": "https://example.com",
        "provider": "host-index",
        "source_type": "release-note",
    }
    assert receipt["authority"] == "primary"


def test_codex_web_never_promotes_nested_or_text_urls() -> None:
    response = {
        "output": "https://text-spoof.invalid",
        "details": {
            "nested": {
                "url": "https://nested-spoof.invalid",
                "snippet": "not directly attributable",
            }
        },
    }
    assert cortheon_hook._tool_metadata(response) == {}
    observations = _host_observations(
        _request(),
        "websearch",
        cortheon_hook._tool_output(response),
        succeeded=True,
        host_input={"query": "current release"},
        tool_metadata={},
    )
    assert observations[0]["status"] == "failed"
    assert "url" not in observations[0]


def test_codex_outer_exec_nested_web_trace_is_not_promoted_to_web_evidence() -> None:
    response = {
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Script completed\nOutput:\n"
                    "Release note (https://source.example/release) Version 4 shipped."
                ),
            }
        ]
    }
    metadata = cortheon_hook._tool_metadata(response)
    assert metadata == {}
    observations = _host_observations(
        _request(),
        "exec",
        cortheon_hook._tool_output(response),
        succeeded=True,
        host_input={
            "code": ("const r = await tools.web__run({search_query:[{q:'current release'}]});")
        },
        tool_metadata=metadata,
    )
    assert observations[0]["kind"] == "web"
    assert observations[0]["status"] == "failed"
    assert "url" not in observations[0]
    assert _receipt(observations[0])["tool"] == "websearch"


def test_codex_fetch_rejects_mixed_origins_and_invalid_dates() -> None:
    mixed = _host_observations(
        _request("fetch"),
        "webfetch",
        "host page",
        succeeded=True,
        host_input={"url": "https://source.example/page"},
        tool_metadata={"finalUrl": "https://attacker.invalid/page"},
    )
    assert mixed[0]["status"] == "failed" and "url" not in mixed[0]
    invalid_date = _host_observations(
        _request("fetch"),
        "webfetch",
        "host page",
        succeeded=True,
        host_input={"url": "https://source.example/page"},
        tool_metadata={"publishedAt": "not-a-date"},
    )
    assert invalid_date[0]["status"] == "failed"


def test_codex_tracker_accepts_web_and_replans_to_a_new_origin() -> None:
    runtime = CognitiveRuntime(require_host_receipts=True)
    tracker = CognitiveHookTracker(runtime=runtime)
    registered = tracker.register(
        "codex",
        "session",
        "turn",
        goal="Research the current Cortheon release and cite fresh sources.",
        effort="quick",
    )
    assert registered["next_action"]["request"]["capability"] == "search"
    tracker.pre_tool(
        "codex",
        "session",
        "turn",
        "websearch",
        tool_input={"query": "current Cortheon release"},
    )
    observed = tracker.post_tool(
        "codex",
        "session",
        "turn",
        "websearch",
        succeeded=True,
        tool_output="host search envelope",
        tool_metadata={
            "results": [
                {
                    "url": "https://example.com/release",
                    "snippet": "Version 4 shipped.",
                    "publishedAt": "2026-08-22",
                    "provider": "host-index",
                }
            ]
        },
    )
    assert observed["observed"] is True
    assert "observation_error" not in observed
    assert observed["next_action"]["request"]["parameters"]["purpose"] == "corroboration"
    assert runtime.metrics["observations_accepted"] == 1


def test_post_tool_sends_only_bounded_structured_metadata(monkeypatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/v1/hooks/post-tool"
        captured.append(payload)
        return {"tracked": True}

    monkeypatch.setattr(cortheon_hook, "_post", fake_post)
    cortheon_hook._post_tool_use(
        {
            "session_id": "session",
            "turn_id": "turn",
            "tool_name": "websearch",
            "tool_response": {
                "structuredContent": {
                    "results": [
                        {
                            "url": "https://example.org/release",
                            "snippet": "released",
                        }
                    ],
                    "secret": "must not cross the hook boundary",
                }
            },
        }
    )
    assert captured[0]["tool_metadata"] == {
        "results": [{"url": "https://example.org/release", "snippet": "released"}]
    }


def test_research_prompt_starts_one_automatic_hook_session(monkeypatch, capsys) -> None:
    registrations: list[dict[str, Any]] = []

    def fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/v1/hooks/register"
        registrations.append(payload)
        return {"automatic": True}

    monkeypatch.setattr(cortheon_hook, "_ensure_runtime", lambda: True)
    monkeypatch.setattr(cortheon_hook, "_post", fake_post)
    prompt = "Research the current release from fresh web sources and cite it."
    cortheon_hook._user_prompt_submit(
        {
            "session_id": "session",
            "turn_id": "turn",
            "prompt": prompt,
        }
    )
    assert registrations == [
        {
            "host": "codex",
            "host_session_id": "session",
            "turn_id": "turn",
            "goal": prompt,
            "effort": "quick",
        }
    ]
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "AUTOMATIC SESSION IS ACTIVE" in context
    assert "lifecycle tools" in context


def test_default_runtime_autostart_is_bounded(monkeypatch, tmp_path: Path) -> None:
    expected = {"protocol_version": "1.0.0"}
    launched: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.delenv("CORTHEON_RUNTIME_URL", raising=False)
    monkeypatch.setattr(hook_transport, "_ACTIVE_RUNTIME_URL", None)
    monkeypatch.setattr(hook_transport, "_expected_runtime_identity", lambda: expected)
    monkeypatch.setattr(
        hook_transport,
        "_runtime_health_payload",
        lambda _url: (
            {
                "ok": True,
                "service": "cortheon-cognitive",
                "storage": "memory_only",
                **expected,
            }
            if launched
            else None
        ),
    )
    monkeypatch.setattr(hook_transport, "_runtime_command", lambda: ["/opt/cortheon-runtime"])
    monkeypatch.setattr(
        hook_transport,
        "_runtime_start_lock",
        lambda _port: (
            tmp_path / "runtime.lock",
            os.open(tmp_path / "runtime.lock", os.O_CREAT | os.O_WRONLY, 0o600),
        ),
    )
    monkeypatch.setattr(hook_transport.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hook_transport.subprocess,
        "Popen",
        lambda command, **options: launched.append((command, options)),
    )
    assert hook_transport._ensure_runtime()
    assert launched[0][0] == ["/opt/cortheon-runtime"]
    assert launched[0][1]["start_new_session"] is True
    assert launched[0][1]["env"]["CORTHEON_COGNITIVE_PORT"] == "8743"


def test_unavailable_runtime_health_is_absent_not_stale(monkeypatch) -> None:
    monkeypatch.setattr(
        hook_transport.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("connection refused")
        ),
    )
    assert hook_transport._runtime_health_payload("http://127.0.0.1:8743") is None


def test_custom_runtime_is_never_spawned_without_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("CORTHEON_RUNTIME_URL", "http://127.0.0.1:19001")
    monkeypatch.delenv("CORTHEON_RUNTIME_AUTOSTART", raising=False)
    monkeypatch.setattr(hook_transport, "_runtime_healthy", lambda: False)
    monkeypatch.setattr(
        hook_transport.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("spawned")),
    )
    assert not hook_transport._ensure_runtime()


def test_stale_runtime_identity_is_rejected(monkeypatch) -> None:
    response = io.BytesIO(
        json.dumps(
            {
                "ok": True,
                "service": "cortheon-cognitive",
                "storage": "memory_only",
                "version": "0.1.0",
                "protocol_version": "1.0.0",
                "source_fingerprint": "stale",
            }
        ).encode()
    )
    monkeypatch.setattr(
        hook_transport,
        "_expected_runtime_identity",
        lambda: {
            "version": "0.1.0",
            "protocol_version": "1.0.0",
            "source_fingerprint": "current",
        },
    )
    monkeypatch.setattr(hook_transport.urllib.request, "urlopen", lambda *_a, **_k: response)

    assert not hook_transport._runtime_healthy()


def test_stale_default_runtime_recovers_on_identity_scoped_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    expected = {
        "version": "0.1.0",
        "protocol_version": "1.0.0",
        "source_fingerprint": "current-source",
    }
    matching = {
        "ok": True,
        "service": "cortheon-cognitive",
        "storage": "memory_only",
        **expected,
    }
    stale = {**matching, "source_fingerprint": "stale-source"}
    launched: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.delenv("CORTHEON_RUNTIME_URL", raising=False)
    monkeypatch.setattr(hook_transport, "_ACTIVE_RUNTIME_URL", None)
    monkeypatch.setattr(hook_transport.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(hook_transport, "_expected_runtime_identity", lambda: expected)
    fallback = hook_transport._fallback_runtime_url(expected)

    def health(url: str) -> dict[str, Any] | None:
        if url == "http://127.0.0.1:8743":
            return stale
        if url == fallback and launched:
            return matching
        return None

    monkeypatch.setattr(hook_transport, "_runtime_health_payload", health)
    monkeypatch.setattr(hook_transport, "_runtime_command", lambda: ["/opt/cortheon-runtime"])
    monkeypatch.setattr(hook_transport.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hook_transport.subprocess,
        "Popen",
        lambda command, **options: launched.append((command, options)),
    )

    assert hook_transport._ensure_runtime()
    assert hook_transport._runtime_url() == fallback
    assert launched[0][1]["env"]["CORTHEON_COGNITIVE_PORT"] == fallback.rsplit(":", 1)[1]


def test_concurrent_runtime_start_has_one_process_owner(monkeypatch, tmp_path: Path) -> None:
    expected = {"protocol_version": "1.0.0"}
    matching = {
        "ok": True,
        "service": "cortheon-cognitive",
        "storage": "memory_only",
        **expected,
    }
    launched: list[list[str]] = []
    entered = threading.Event()
    release = threading.Event()
    results: list[bool] = []
    monkeypatch.setattr(hook_transport.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(hook_transport.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hook_transport,
        "_runtime_health_payload",
        lambda _url: matching if launched else None,
    )

    def popen(command: list[str], **_options: Any) -> None:
        launched.append(command)
        entered.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(hook_transport.subprocess, "Popen", popen)

    first = threading.Thread(
        target=lambda: results.append(
            hook_transport._start_runtime(
                "http://127.0.0.1:18991", expected, ["/opt/cortheon-runtime"]
            )
        )
    )
    first.start()
    assert entered.wait(timeout=2)
    second = threading.Thread(
        target=lambda: results.append(
            hook_transport._start_runtime(
                "http://127.0.0.1:18991", expected, ["/opt/cortheon-runtime"]
            )
        )
    )
    second.start()
    second.join(timeout=2)
    release.set()
    first.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert results == [True, True]
    assert launched == [["/opt/cortheon-runtime"]]
    assert not list(tmp_path.glob("*.lock"))


def test_installer_owns_exact_mcp_and_runtime_interpreters(tmp_path: Path) -> None:
    installed = install_codex(
        dry_run=False,
        run_cli=False,
        install_root=tmp_path / "marketplace",
    )
    plugin = Path(installed.details["plugin"])
    mcp = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["cortheon"] == {
        "command": "./scripts/cortheon-mcp",
        "args": [],
        "cwd": ".",
    }
    python = sys.executable
    assert python in (plugin / "scripts/cortheon-mcp").read_text(encoding="utf-8")
    runtime = plugin / "scripts/cortheon-runtime"
    assert python in runtime.read_text(encoding="utf-8")
    assert "-m cortheon.cognitive_cli serve" in runtime.read_text(encoding="utf-8")
    assert os.access(runtime, os.X_OK)
    identity = json.loads((plugin / "scripts/cortheon-runtime.json").read_text())
    assert identity["protocol_version"] == "1.0.0"
    assert len(identity["source_fingerprint"]) == 16


def test_reinstall_removes_obsolete_owned_plugin_files(tmp_path: Path) -> None:
    root = tmp_path / "marketplace"
    first = install_codex(dry_run=False, run_cli=False, install_root=root)
    obsolete = Path(first.details["plugin"]) / "hooks" / "obsolete_hook.py"
    obsolete.write_text("raise RuntimeError('stale')\n", encoding="utf-8")

    install_codex(dry_run=False, run_cli=False, install_root=root)

    assert not obsolete.exists()
