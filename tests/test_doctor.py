"""Offline unit tests for `cortheon doctor`, CLI flags, and `proxy.py` lifespan."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from flash_forge.cli import main as cli_main
from flash_forge.client import ModelCapabilities, TensorFoldClient
from flash_forge.doctor import (
    _recommend_streams,
    check_harnesses,
    check_sandbox,
    check_solver_stack,
    format_doctor_report,
    run_doctor,
)
from flash_forge.integrations import install_commands
from flash_forge.proxy import create_app, format_proxy_startup_banner
from flash_forge.sandbox import ExecResult, sandbox_backend

_IS_MACOS_SEATBELT = sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec")


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any], url: str = "http://127.0.0.1:9999/v1/models") -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)
        self._request = httpx.Request("GET", url)

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self._request,
                response=httpx.Response(self.status_code, request=self._request, text=self.text),
            )


class _ScriptedHttp:
    def __init__(self, *, get_result: Any, post_results: list[Any] | None = None) -> None:
        self._get_result = get_result
        self._post_results = list(post_results or [])
        self.posted_payloads: list[dict[str, Any]] = []
        self.closed = False

    async def get(self, url: str) -> Any:
        if isinstance(self._get_result, Exception):
            raise self._get_result
        return self._get_result

    async def post(self, url: str, json: dict[str, Any] | None = None) -> Any:
        self.posted_payloads.append(dict(json or {}))
        nxt = self._post_results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def aclose(self) -> None:
        self.closed = True


def _make_client_factory(fake_http: _ScriptedHttp):
    def _factory(**kwargs: Any) -> TensorFoldClient:
        c = TensorFoldClient(**kwargs)
        c._http = fake_http  # type: ignore[assignment]
        return c

    return _factory


# ---------------------------------------------------------------------------
# 1. OS Sandbox Containment
# ---------------------------------------------------------------------------


def test_sandbox_check_live_containment(monkeypatch):
    monkeypatch.delenv("CORTHEON_UNSAFE_NO_SANDBOX", raising=False)
    probe_file = Path.home() / ".cortheon_sandbox_probe_never_created"
    assert not probe_file.exists()

    res = check_sandbox()
    assert res.id == "sandbox"
    assert res.details["exec_ok"] is True
    assert res.details["probe_elapsed_ms"] < 2000.0
    assert not probe_file.exists()

    if _IS_MACOS_SEATBELT:
        assert res.status == "pass"
        assert res.details["backend"] == "macos-seatbelt"
        assert res.details["network_blocked"] is True
        assert res.details["outside_write_blocked"] is True
    else:
        assert res.status in ("pass", "warn")


def test_sandbox_check_linux_bwrap_and_rlimit_fallback(monkeypatch, tmp_path: Path):
    # Case A: Linux bwrap active -> full probe passes
    monkeypatch.setattr("flash_forge.doctor.sandbox_backend", lambda: "linux-bwrap")
    monkeypatch.setattr(
        "flash_forge.doctor.run_python",
        lambda code, timeout_s=4.0: ExecResult(
            ok=True,
            returncode=0,
            stdout="EXEC=OK\nNET=BLOCKED:OSError:101\nWRITE=BLOCKED:OSError:30\n",
            stderr="",
            elapsed_s=0.04,
        ),
    )
    res_bwrap = check_sandbox()
    assert res_bwrap.status == "pass"
    assert res_bwrap.details["backend"] == "linux-bwrap"
    assert res_bwrap.details["network_blocked"] is True
    assert res_bwrap.details["outside_write_blocked"] is True

    # Case B: Linux bwrap containment breach -> fails honestly
    monkeypatch.setattr(
        "flash_forge.doctor.run_python",
        lambda code, timeout_s=4.0: ExecResult(
            ok=True,
            returncode=0,
            stdout="EXEC=OK\nNET=ALLOWED\nWRITE=ESCAPED\n",
            stderr="",
            elapsed_s=0.03,
        ),
    )
    res_breach = check_sandbox()
    assert res_breach.status == "fail"
    assert res_breach.details["network_blocked"] is False
    assert res_breach.details["outside_write_blocked"] is False

    # Case C: subprocess-rlimit fallback -> NEVER creates probe file in $HOME!
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("flash_forge.doctor.sandbox_backend", lambda: "subprocess-rlimit")

    seen_scripts: list[str] = []

    def _capture_exec(code: str, timeout_s: float = 4.0) -> ExecResult:
        seen_scripts.append(code)
        return ExecResult(ok=True, returncode=0, stdout="EXEC=OK\n", stderr="", elapsed_s=0.02)

    monkeypatch.setattr("flash_forge.doctor.run_python", _capture_exec)
    res_fallback = check_sandbox()
    assert res_fallback.status == "warn"
    assert res_fallback.details["backend"] == "subprocess-rlimit"
    assert res_fallback.details["exec_ok"] is True
    assert res_fallback.details["network_blocked"] is None
    assert res_fallback.details["outside_write_blocked"] is None
    assert "bwrap" in res_fallback.hint
    assert ".cortheon_sandbox_probe_never_created" not in seen_scripts[0]
    assert not (fake_home / ".cortheon_sandbox_probe_never_created").exists()

    # Case D: unsafe-disabled -> warns with actionable hint
    monkeypatch.setattr("flash_forge.doctor.sandbox_backend", lambda: "unsafe-disabled")
    res_unsafe = check_sandbox()
    assert res_unsafe.status == "warn"
    assert "CORTHEON_UNSAFE_NO_SANDBOX" in res_unsafe.hint


# ---------------------------------------------------------------------------
# 2. Local Solver Stack
# ---------------------------------------------------------------------------


def test_solver_stack_check_live_and_missing_packages(monkeypatch):
    res = check_solver_stack()
    assert res.status == "pass"
    assert res.details["python_ok"] is True
    for pkg in ("sympy", "z3", "numpy"):
        assert res.details["packages"][pkg]["installed"] is True
        assert res.details["packages"][pkg]["version"]

    # Optional pytest absent -> still PASS
    orig_probe = __import__("flash_forge.doctor", fromlist=["_probe_module"])._probe_module

    def _missing_pytest(mod_name: str, dist_name: str):
        if mod_name == "pytest":
            return False, None, "ModuleNotFoundError: No module named 'pytest'"
        return orig_probe(mod_name, dist_name)

    monkeypatch.setattr("flash_forge.doctor._probe_module", _missing_pytest)
    res_no_pytest = check_solver_stack()
    assert res_no_pytest.status == "pass"
    assert res_no_pytest.details["missing_optional"] == ["pytest"]
    assert res_no_pytest.details["missing_required"] == []

    # Required z3 absent -> FAIL with pip install hint
    def _missing_z3(mod_name: str, dist_name: str):
        if mod_name == "z3":
            return False, None, "ModuleNotFoundError: No module named 'z3'"
        return orig_probe(mod_name, dist_name)

    monkeypatch.setattr("flash_forge.doctor._probe_module", _missing_z3)
    res_no_z3 = check_solver_stack()
    assert res_no_z3.status == "fail"
    assert res_no_z3.details["missing_required"] == ["z3"]
    assert "z3-solver" in res_no_z3.hint


# ---------------------------------------------------------------------------
# 3. Harness Integrations (non-destructive)
# ---------------------------------------------------------------------------


def test_harnesses_check_non_destructive(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("flash_forge.doctor.shutil.which", lambda _name: None)
    home = tmp_path / "clean_home"
    home.mkdir()

    # Empty home -> PASS (standalone mode), creates nothing
    res_empty = check_harnesses(home_dir=home)
    assert res_empty.status == "pass"
    assert res_empty.details["detected"] == []
    assert res_empty.details["installed"] == []
    assert list(home.iterdir()) == []

    # Detected harnesses without /cortheon installed -> WARN
    (home / ".claude").mkdir()
    (home / ".cursor").mkdir()
    res_warn = check_harnesses(home_dir=home)
    assert res_warn.status == "warn"
    assert res_warn.details["detected"] == ["claude", "cursor"]
    assert res_warn.details["installed"] == []
    assert "cortheon install-commands --harness all" in res_warn.hint

    # After installing into claude + omp profile -> PASS, flags remaining cursor
    install_commands(["claude", "omp"], home_dir=home, omp_profile="dev")
    res_inst = check_harnesses(home_dir=home)
    assert res_inst.status == "pass"
    assert set(res_inst.details["installed"]) == {"omp", "claude"}
    assert res_inst.details["missing_on_detected"] == ["cursor"]


# ---------------------------------------------------------------------------
# 4 & 5. Backend Probe + Live Fast-Route Probe (TensorFold & Generic OpenAI)
# ---------------------------------------------------------------------------


def test_doctor_tensorfold_native_backend_and_live_probe(tmp_path: Path):
    fake_http = _ScriptedHttp(
        get_result=_FakeResponse(
            200,
            {
                "data": [
                    {
                        "id": "glm-5.3-flash-tfq3",
                        "owned_by": "tensorfold",
                        "context_length": 131072,
                        "capabilities": {
                            "structures": True,
                            "thinking_budget": True,
                            "reasoning": True,
                            "thinking_switch": True,
                            "loop_guard": True,
                            "prompt_cache": True,
                        },
                        "reasoning": {"supported_efforts": []},
                    }
                ]
            },
        ),
        post_results=[
            _FakeResponse(
                200,
                {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 14, "completion_tokens": 2},
                },
            )
        ],
    )

    report = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:8080/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_http),
        )
    )
    assert report.ok is True
    assert fake_http.closed is True

    backend = report.check_by_id("backend")
    live = report.check_by_id("live_probe")
    assert backend is not None and live is not None

    assert backend.status == "pass"
    assert backend.details["backend_flavor"] == "TensorFold Native"
    assert backend.details["model_id"] == "glm-5.3-flash-tfq3"
    assert backend.details["context_length"] == 131072
    assert backend.details["thinking_budget"] is True
    assert backend.details["supported_efforts"] == []
    assert backend.details["reasoning_headroom_active"] is False

    assert live.status == "pass"
    assert live.details["non_empty_content"] is True
    assert live.details["thinking_starvation_failures"] == 0
    assert live.details["tok_per_s"] > 0
    assert report.recommended_streams in (4, 8)

    # Confirm the live probe sent enable_thinking=False and max_tokens=16
    assert len(fake_http.posted_payloads) == 1
    sent = fake_http.posted_payloads[0]
    assert sent["max_tokens"] == 16
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["cortheon"] is False


def test_doctor_generic_openai_omlx_reasoning_and_recovery(tmp_path: Path):
    # Subcase A: oMLX / Ollama serving a reasoning model that honors enable_thinking=False
    fake_omlx = _ScriptedHttp(
        get_result=_FakeResponse(
            200,
            {"data": [{"id": "ds41-flash", "owned_by": "omlx", "max_model_len": 1048576}]},
        ),
        post_results=[
            _FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok", "reasoning_content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 2,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )
        ],
    )
    report_a = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:8000/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_omlx),
        )
    )
    assert report_a.ok is True
    b_a = report_a.check_by_id("backend")
    l_a = report_a.check_by_id("live_probe")
    assert b_a is not None and l_a is not None
    assert "Generic OpenAI" in b_a.details["backend_flavor"]
    assert b_a.details["model_id"] == "ds41-flash"
    assert b_a.details["context_length"] == 1048576
    assert b_a.details["thinking_budget"] is False
    assert b_a.details["reasoning_headroom_active"] is True
    assert l_a.status == "pass"
    assert l_a.details["thinking_starvation_failures"] == 0
    assert "cortheon" not in fake_omlx.posted_payloads[0]
    assert fake_omlx.posted_payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}

    # Subcase B: Custom alias on Ollama/oMLX ignores enable_thinking=False on attempt 1,
    # starves max_tokens=16 into reasoning_content, and recovers via PR #2 adaptive retry!
    fake_recover = _ScriptedHttp(
        get_result=_FakeResponse(200, {"data": [{"id": "my-custom-alias", "context_length": 65536}]}),
        post_results=[
            _FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "", "reasoning_content": "thinking..."},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 16,
                        "completion_tokens_details": {"reasoning_tokens": 16},
                    },
                },
            ),
            _FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok", "reasoning_content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 22,
                        "completion_tokens_details": {"reasoning_tokens": 20},
                    },
                },
            ),
        ],
    )
    report_b = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:11434/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_recover),
        )
    )
    assert report_b.ok is True
    b_b = report_b.check_by_id("backend")
    l_b = report_b.check_by_id("live_probe")
    assert b_b is not None and l_b is not None
    assert l_b.status == "pass"
    assert l_b.details["thinking_starvation_failures"] == 0
    assert l_b.details["adaptive_recovery_triggered"] is True
    assert l_b.details["reasoning_observed"] is True
    assert b_b.details["reasoning_headroom_active"] is True
    assert len(fake_recover.posted_payloads) == 2
    assert fake_recover.posted_payloads[1]["max_tokens"] >= 1024

    # Subcase C: Unrecoverable thinking starvation -> FAIL with thinking_starvation_failures=1
    starve_resp = _FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "", "reasoning_content": "endless thinking..."},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 1024,
                "completion_tokens_details": {"reasoning_tokens": 1024},
            },
        },
    )
    fake_starve = _ScriptedHttp(
        get_result=_FakeResponse(200, {"data": [{"id": "ds41-flash", "context_length": 65536}]}),
        post_results=[starve_resp, starve_resp],
    )
    report_c = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:8000/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_starve),
        )
    )
    assert report_c.ok is False
    l_c = report_c.check_by_id("live_probe")
    assert l_c is not None
    assert l_c.status == "fail"
    assert l_c.details["thinking_starvation_failures"] == 1


def test_doctor_unreachable_backend_and_auth_error_graceful(tmp_path: Path):
    fake_down = _ScriptedHttp(get_result=httpx.ConnectError("Connection refused"))
    report = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:19999/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_down),
        )
    )
    assert report.ok is False
    assert fake_down.closed is True

    backend = report.check_by_id("backend")
    live = report.check_by_id("live_probe")
    assert backend is not None and live is not None
    assert backend.status == "fail"
    assert backend.details["reachable"] is False
    assert "Start a local OpenAI-compatible server" in backend.hint
    assert live.status == "skip"

    rendered = format_doctor_report(report, use_color=False)
    assert "Traceback" not in rendered
    assert "[FAIL] Backend Probe" in rendered
    assert "Hint:" in rendered

    # HTTP 401 -> actionable --api-key hint
    fake_401 = _ScriptedHttp(get_result=_FakeResponse(401, {"error": "unauthorized"}))
    report_401 = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:8080/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_401),
        )
    )
    assert report_401.ok is False
    b_401 = report_401.check_by_id("backend")
    assert b_401 is not None and "--api-key" in b_401.hint


def test_doctor_skip_live_and_streams_recommendation(tmp_path: Path):
    fake_http = _ScriptedHttp(
        get_result=_FakeResponse(
            200,
            {"data": [{"id": "qwen3-30b", "context_length": 32768}]},
        ),
    )
    report = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:8080/v1",
            skip_live=True,
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_http),
        )
    )
    assert report.ok is True
    assert fake_http.posted_payloads == []
    live = report.check_by_id("live_probe")
    assert live is not None and live.status == "skip"
    assert report.recommended_streams == 4

    # Direct recommendation bands: 2 (slow), 4 (balanced), 8 (fast TensorFold + prompt cache)
    s2, _ = _recommend_streams(is_tensorfold=False, prompt_cache=False, latency_ms=2600.0, tok_per_s=12.0)
    s4, _ = _recommend_streams(is_tensorfold=False, prompt_cache=False, latency_ms=250.0, tok_per_s=65.0)
    s8, _ = _recommend_streams(is_tensorfold=True, prompt_cache=True, latency_ms=95.0, tok_per_s=140.0)
    assert (s2, s4, s8) == (2, 4, 8)


# ---------------------------------------------------------------------------
# 6. Formatter (`NO_COLOR`) + CLI (`--json`, `--help`, `--streams` docs)
# ---------------------------------------------------------------------------


def test_doctor_formatter_respects_no_color_and_prints_quickstart(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NO_COLOR", "1")
    fake_http = _ScriptedHttp(
        get_result=_FakeResponse(
            200,
            {"data": [{"id": "ds41-flash", "context_length": 65536}]},
        ),
        post_results=[
            _FakeResponse(
                200,
                {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )
        ],
    )
    report = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:8080/v1",
            home_dir=tmp_path,
            client_factory=_make_client_factory(fake_http),
        )
    )
    out = format_doctor_report(report)
    assert "\033[" not in out
    assert "[PASS] OS Sandbox Containment" in out or "[WARN] OS Sandbox Containment" in out
    assert "[PASS] Backend Probe" in out
    assert "reasoning_headroom=True" in out
    assert "cortheon proxy --upstream http://127.0.0.1:8080/v1 --streams" in out
    assert "cortheon duel --url http://127.0.0.1:8080/v1 --streams" in out
    assert "cortheon install-commands --harness all" in out


def test_doctor_cli_json_and_help_flags(monkeypatch, capsys, tmp_path: Path):
    fake_http = _ScriptedHttp(
        get_result=_FakeResponse(
            200,
            {"data": [{"id": "glm-5.3-flash-tfq3", "owned_by": "tensorfold", "context_length": 131072, "capabilities": {"thinking_budget": True}}]},
        ),
        post_results=[
            _FakeResponse(
                200,
                {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )
        ],
    )
    monkeypatch.setattr("flash_forge.doctor.TensorFoldClient", _make_client_factory(fake_http))

    cli_main(["doctor", "--url", "http://127.0.0.1:8080/v1", "--json"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["ok"] is True
    assert payload["url"] == "http://127.0.0.1:8080/v1"
    assert payload["recommended_streams"] in (2, 4, 8)
    assert [c["id"] for c in payload["checks"]] == [
        "sandbox",
        "solver_stack",
        "harnesses",
        "backend",
        "live_probe",
    ]

    # Verify `cortheon doctor --help`, `proxy --help`, and `solve --help`
    for subcmd, expected_snippets in (
        ("doctor", ["--url", "--api-key", "--model", "--skip-live", "--json"]),
        ("proxy", ["--streams", "2=fast sanity check ~2-3 calls", "4=balanced default ~4-7 calls", "8=max diversity"]),
        ("solve", ["--model", "--streams", "2=fast sanity check ~2-3 calls", "4=balanced default ~4-7 calls", "8=max diversity"]),
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli_main([subcmd, "--help"])
        assert exc_info.value.code == 0
        help_text = " ".join(capsys.readouterr().out.split())
        for snippet in expected_snippets:
            assert snippet in help_text


# ---------------------------------------------------------------------------
# 7. Proxy Lifespan & Startup Banner (ZERO FastAPI deprecation warnings)
# ---------------------------------------------------------------------------


def test_proxy_lifespan_banner_and_zero_fastapi_deprecation_warnings(monkeypatch, capsys):
    closed_count = 0

    async def _fake_probe(self):
        return ModelCapabilities(model_id="glm-5.3-flash-tfq3", context_length=131072)

    async def _fake_close(self):
        nonlocal closed_count
        closed_count += 1

    monkeypatch.setattr("flash_forge.client.TensorFoldClient.probe", _fake_probe)
    monkeypatch.setattr("flash_forge.client.TensorFoldClient.close", _fake_close)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = create_app(upstream_url="http://127.0.0.1:9999/v1", num_streams=4, auto_rescue=True)
        with TestClient(app) as tc:
            resp = tc.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["upstream_model"] == "glm-5.3-flash-tfq3"
            assert body["sandbox_backend"] == sandbox_backend()
            assert body["streams"] == 4
            assert body["auto_rescue"] is True

    fastapi_deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and ("on_event" in str(w.message) or "proxy.py" in str(w.filename))
    ]
    assert fastapi_deprecations == []
    assert closed_count == 2  # both forge_default and forge_deep closed cleanly

    out = capsys.readouterr().out
    assert "Cortheon Universal Proxy" in out
    assert f"OS Sandbox  : {sandbox_backend()}" in out
    assert "--streams 4 (balanced default, ~4-7 calls)" in out
    assert "Auto-Rescue : ON" in out

    # Also verify banner variants for --streams 2 / 8 and Auto-Rescue OFF
    b2 = format_proxy_startup_banner(
        upstream_url="http://127.0.0.1:8080/v1",
        upstream_model="qwen3",
        num_streams=2,
        auto_rescue=False,
        sandbox="linux-bwrap",
    )
    assert "--streams 2 (fast sanity check, ~2-3 calls)" in b2
    assert "OS Sandbox  : linux-bwrap" in b2
    assert "Auto-Rescue : OFF" in b2

    b8 = format_proxy_startup_banner(
        upstream_url="http://127.0.0.1:8080/v1",
        num_streams=8,
        auto_rescue=True,
    )
    assert "--streams 8 (max diversity" in b8


def test_always_on_reasoning_model_fast_route_recovery():
    """Models advertising reasoning_optional=False (e.g. Inkling-2-Mini) cap thinking on fast-route
    calls and retry cleanly if max_tokens=16 still starves inside reasoning_content."""
    models_resp = _FakeResponse(
        200,
        {
            "object": "list",
            "data": [
                {
                    "id": "inkling-2-mini",
                    "owned_by": "tensorfold",
                    "context_length": 32768,
                    "capabilities": {
                        "thinking_budget": True,
                        "reasoning": True,
                        "reasoning_optional": False,
                        "thinking_switch": True,
                        "loop_guard": True,
                        "prompt_cache": True,
                    },
                    "reasoning": {
                        "default_enabled": True,
                        "supported_efforts": ["max", "high", "medium", "low", "minimal"],
                    },
                }
            ],
        },
    )
    starved_resp = _FakeResponse(
        200,
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "thinking..."},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 16,
                "completion_tokens_details": {"reasoning_tokens": 16},
            },
        },
    )
    ok_resp = _FakeResponse(
        200,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "ok", "reasoning_content": "brief"},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 22,
                "completion_tokens_details": {"reasoning_tokens": 18},
            },
        },
    )
    fake_http = _ScriptedHttp(get_result=models_resp, post_results=[starved_resp, ok_resp])
    report = asyncio.run(
        run_doctor(
            url="http://127.0.0.1:18654/v1",
            skip_live=False,
            client_factory=_make_client_factory(fake_http),
        )
    )
    assert len(fake_http.posted_payloads) == 2
    assert fake_http.posted_payloads[0]["thinking_budget"] == 0
    assert fake_http.posted_payloads[0]["reasoning_effort"] == "minimal"
    assert fake_http.posted_payloads[1]["max_tokens"] >= 1024
    live_check = next(c for c in report.checks if c.id == "live_probe")
    assert live_check.status == "pass"
    assert live_check.details["thinking_starvation_failures"] == 0
    assert live_check.details["adaptive_recovery_triggered"] is True


def test_solve_forwards_model_override_to_forge_config(monkeypatch):
    """`solve --model` must reach ForgeConfig.

    Without the flag, ``TensorFoldClient.probe()`` falls back to ``data[0]`` from
    ``/v1/models``, which on a multi-model backend (LM Studio, vLLM) is an
    arbitrary entry -- in practice an embedding or OCR model -- and every stream
    then talks to the wrong model with no error at all.
    """
    captured = {}

    class _StubForge:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def solve(self, prompt, force_mode=None, on_event=None):
            return SimpleNamespace(
                mode="fast",
                wall_s=0.0,
                token_stats={},
                raw_trace=None,
                answer="2",
                flywheel_recorded=False,
                lesson_learned=None,
            )

    monkeypatch.setattr("flash_forge.cli.FlashForge", _StubForge)
    cli_main(["solve", "what is 1+1", "--model", "my-model"])
    assert captured["cfg"].model == "my-model"


def test_solve_without_model_flag_leaves_it_unset(monkeypatch):
    captured = {}

    class _StubForge:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def solve(self, prompt, force_mode=None, on_event=None):
            return SimpleNamespace(
                mode="fast", wall_s=0.0, token_stats={}, raw_trace=None,
                answer="2", flywheel_recorded=False, lesson_learned=None,
            )

    monkeypatch.setattr("flash_forge.cli.FlashForge", _StubForge)
    cli_main(["solve", "what is 1+1"])
    assert captured["cfg"].model is None
