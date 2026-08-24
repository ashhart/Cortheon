from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cortheon import __version__
from cortheon.cognitive_cli import build_parser, doctor, host_conformance, main, runtime_results
from cortheon.cognitive_http import _SOURCE_FINGERPRINT
from cortheon.cognitive_install import generic_mcp_config
from cortheon.cognitive_protocol import CORTHEON_PROTOCOL_VERSION


def _runtime_health(*, fingerprint: str = _SOURCE_FINGERPRINT) -> dict[str, object]:
    return {
        "ok": True,
        "service": "cortheon-cognitive",
        "version": __version__,
        "protocol_version": CORTHEON_PROTOCOL_VERSION,
        "source_fingerprint": fingerprint,
        "storage": "memory_only",
        "active_sessions": 0,
        "active_hook_turns": 0,
    }


def test_doctor_rejects_a_reachable_stale_runtime() -> None:
    with patch(
        "cortheon.cognitive_cli._runtime_health", return_value=_runtime_health(fingerprint="0" * 16)
    ):
        report = doctor()

    runtime = next(item for item in report["checks"] if item["name"] == "runtime")
    assert report["ok"] is False
    assert runtime["required"] is True
    assert runtime["identity_matches"] is False


def test_conformance_rejects_a_reachable_stale_runtime(tmp_path: Path) -> None:
    plugin = tmp_path / "opencode_plugin.js"
    plugin.write_text("export default {};\n", encoding="utf-8")
    completed = subprocess.CompletedProcess(["node"], 0, stdout="", stderr="")
    statuses = {"opencode": {"configured": True, "valid": True}}
    with (
        patch("cortheon.cognitive_install.host_installation_status", return_value=statuses),
        patch(
            "cortheon.cognitive_cli._asset_paths",
            return_value={"opencode_plugin": str(plugin)},
        ),
        patch("shutil.which", return_value="/fake/node"),
        patch("subprocess.run", return_value=completed),
        patch(
            "cortheon.cognitive_cli._runtime_health",
            return_value=_runtime_health(fingerprint="0" * 16),
        ),
    ):
        report = host_conformance(hosts=["opencode"])

    assert report["ok"] is False
    assert report["runtime"]["ok"] is True
    assert report["runtime_identity_matches"] is False


def test_doctor_requires_an_explicitly_selected_host(tmp_path: Path) -> None:
    with patch("cortheon.cognitive_cli._runtime_health", return_value={"ok": False}):
        report = doctor(hosts=["pi"], scope="project", project_dir=str(tmp_path))

    assert report["ok"] is False
    assert "integration:pi" in report["required_failures"]
    assert report["selected_hosts"] == ["pi"]


def test_generic_conformance_does_not_probe_http_runtime(tmp_path: Path) -> None:
    initialized = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "cortheon"}}}
    )
    completed = subprocess.CompletedProcess(
        ["cortheon-mcp"], 0, stdout=initialized + "\n", stderr=""
    )
    statuses = {"generic": {"configured": True, "valid": True}}
    with (
        patch("cortheon.cognitive_install.host_installation_status", return_value=statuses),
        patch("shutil.which", return_value="/fake/cortheon-mcp"),
        patch("subprocess.run", return_value=completed),
        patch("cortheon.cognitive_cli._runtime_health") as health,
    ):
        report = host_conformance(hosts=["generic"])

    assert report["ok"] is True
    assert report["runtime_required"] is False
    health.assert_not_called()


def test_generic_configuration_has_a_dedicated_read_only_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["configure", "--host", "generic"]) == 0

    payload = json.loads(capsys.readouterr().out)
    result = payload["results"][0]
    assert result["status"] == "configuration"
    assert result["details"]["writes_files"] is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["install", "--host", "generic"])


def test_generic_configuration_uses_the_installed_mcp_command(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.touch()
    mcp = executable.with_name("cortheon-mcp")
    mcp.touch()

    with (
        patch("shutil.which", return_value=None),
        patch.object(sys, "executable", str(executable)),
    ):
        command = generic_mcp_config().details["mcpServers"]["cortheon"]["command"]
    assert command == str(mcp.resolve())


def test_results_report_content_free_runtime_outcomes() -> None:
    metrics = {
        "ok": True,
        "sessions_started": 3,
        "sessions_completed": 2,
        "completion_withheld": 1,
        "sessions_abandoned": 0,
        "observations_accepted": 7,
        "hypotheses_originated": 2,
        "sessions_reframed": 1,
        "controller_zero_gain_stops": 1,
        "hook_turns_certified": 2,
        "hook_uncertified_releases": 0,
        "completion_latency_ms_mean": 125.5,
    }
    with (
        patch("cortheon.cognitive_cli._runtime_health", return_value=_runtime_health()),
        patch("cortheon.cognitive_cli_core.diagnostics.runtime_metrics", return_value=metrics),
    ):
        report = runtime_results()

    assert report == {
        "ok": True,
        "sessions_started": 3,
        "sessions_completed": 2,
        "completion_withheld": 1,
        "sessions_abandoned": 0,
        "observations_accepted": 7,
        "hypotheses_originated": 2,
        "sessions_reframed": 1,
        "controller_zero_gain_stops": 1,
        "hook_turns_certified": 2,
        "hook_uncertified_releases": 0,
        "completion_latency_ms_mean": 125.5,
        "runtime_identity_matches": True,
        "scope": "since_runtime_start",
        "runtime_identity": {
            "service": "cortheon-cognitive",
            "version": __version__,
            "protocol_version": CORTHEON_PROTOCOL_VERSION,
            "source_fingerprint": _SOURCE_FINGERPRINT,
            "storage": "memory_only",
        },
    }


def test_results_reject_a_stale_runtime_without_reading_metrics() -> None:
    with (
        patch(
            "cortheon.cognitive_cli._runtime_health",
            return_value=_runtime_health(fingerprint="0" * 16),
        ),
        patch("cortheon.cognitive_cli_core.diagnostics.runtime_metrics") as metrics,
    ):
        report = runtime_results()

    assert report["ok"] is False
    assert report["runtime_identity_matches"] is False
    assert report["error"] == "runtime identity mismatch"
    metrics.assert_not_called()


def test_results_distinguish_an_unavailable_runtime_from_a_stale_one() -> None:
    with (
        patch(
            "cortheon.cognitive_cli._runtime_health",
            return_value={"ok": False, "error": "connection refused"},
        ),
        patch("cortheon.cognitive_cli_core.diagnostics.runtime_metrics") as metrics,
    ):
        report = runtime_results()

    assert report["ok"] is False
    assert report["runtime_identity_matches"] is False
    assert report["error"] == "runtime unavailable"
    metrics.assert_not_called()
