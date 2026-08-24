from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cortheon import __version__
from cortheon.cognitive_cli import build_parser, doctor, host_conformance, main
from cortheon.cognitive_http import _SOURCE_FINGERPRINT
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


def test_legacy_slash_command_templates_fail_closed() -> None:
    root = Path(__file__).parents[1] / "integrations" / "slash-commands"
    templates = [root / "README.md", *sorted((root / "generated").glob("*.md"))]

    assert templates
    for template in templates:
        text = template.read_text(encoding="utf-8")
        assert "python3 -m cortheon.slash" not in text
        assert "Do not install" in text or "not shipped" in text
