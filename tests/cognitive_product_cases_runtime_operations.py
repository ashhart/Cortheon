from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from cortheon.cognitive_cli import doctor, host_conformance
from cortheon.cognitive_protocol import CORTHEON_PROTOCOL_VERSION
from cortheon.cognitive_runtime import CognitiveRuntime


def test_runtime_metrics_are_content_free_and_versioned():
    runtime = CognitiveRuntime()
    started = runtime.start("A uniquely secret project goal")
    runtime.finish(started["session"]["session_id"], mode="abandon")

    metrics = runtime.metrics
    assert metrics["protocol_version"] == CORTHEON_PROTOCOL_VERSION
    assert metrics["sessions_started"] == 1
    assert metrics["sessions_abandoned"] == 1
    assert "secret" not in json.dumps(metrics).lower()


def test_host_conformance_executes_one_probe_per_host():
    root = Path(__file__).parents[1] / "src" / "cortheon"
    statuses = {
        host: {"configured": True, "valid": True} for host in ("opencode", "pi", "codex", "generic")
    }

    def run(command, **_kwargs):
        executable = Path(command[0]).name
        stdout = ""
        if executable == "codex":
            stdout = "cortheon@cortheon-local installed, enabled"
        elif executable == "cortheon-mcp":
            stdout = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"serverInfo": {"name": "cortheon"}},
                }
            )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    with (
        patch(
            "cortheon.cognitive_cli._runtime_health",
            return_value={"ok": True, "storage": "memory_only"},
        ),
        patch(
            "cortheon.cognitive_install.host_installation_status",
            return_value=statuses,
        ),
        patch(
            "cortheon.cognitive_cli._asset_paths",
            return_value={
                "opencode_plugin": str(root / "opencode_plugin.js"),
                "pi_extension": str(root / "pi_extension.ts"),
                "codex_plugin": str(root / "codex_plugins" / "cortheon"),
            },
        ),
        patch("shutil.which", side_effect=lambda name: f"/fake/{name}"),
        patch("subprocess.run", side_effect=run),
    ):
        report = host_conformance()

    assert report["ok"] is True
    assert report["cross_host_contract_consistent"] is True
    assert set(report["hosts"]) == {"opencode", "pi", "codex", "generic"}
    assert all(item["ok"] for item in report["hosts"].values())
    assert report["hosts"]["generic"]["adapter_load"]["mcp_initialized"] is True
    assert report["hosts"]["codex"]["adapter_load"]["plugin_assets_valid"] is True


def test_doctor_runtime_is_optional_unless_required():
    optional = doctor("http://127.0.0.1:1", require_runtime=False)
    required = doctor("http://127.0.0.1:1", require_runtime=True)

    assert optional["ok"] is True
    assert required["ok"] is False
    assert required["required_failures"] == ["runtime"]
