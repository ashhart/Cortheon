from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cortheon.cognitive_http import _SOURCE_FINGERPRINT
from cortheon.cognitive_install import (
    InstallError,
    generic_mcp_config,
    host_installation_status,
    install_codex,
    install_hosts,
    install_opencode,
    install_pi,
    uninstall_hosts,
)


def test_codex_install_materializes_valid_marketplace_without_cli_mutation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "marketplace"

        result = install_codex(
            dry_run=False,
            run_cli=False,
            install_root=root,
        )

        manifest = root / "plugins" / "cortheon" / ".codex-plugin" / "plugin.json"
        marketplace = root / ".agents" / "plugins" / "marketplace.json"
        launcher = root / "plugins" / "cortheon" / "scripts" / "cortheon-mcp"
        manifest_payload = json.loads(manifest.read_text())
        assert manifest_payload["name"] == "cortheon"
        assert manifest_payload["version"] == f"0.1.0+codex.{_SOURCE_FINGERPRINT}"
        assert json.loads(marketplace.read_text())["name"] == "cortheon-local"
        assert stat.S_IMODE(launcher.stat().st_mode) == 0o755
        assert not list((root / "plugins" / "cortheon").rglob("__pycache__"))
        assert not list((root / "plugins" / "cortheon").rglob("*.pyc"))
        assert result.status == "installed"


def test_project_scope_preflight_prevents_partial_all_host_install():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        with pytest.raises(InstallError, match="user-installed"):
            install_hosts(
                ["all"],
                scope="project",
                project_dir=root,
                dry_run=False,
            )

        assert not (root / "opencode.json").exists()
        assert not (root / ".pi" / "settings.json").exists()


def test_preflight_rejects_later_malformed_config_before_any_host_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pi_settings = root / ".pi" / "settings.json"
        pi_settings.parent.mkdir(parents=True)
        pi_settings.write_text('{"extensions":"not-an-array"}\n')

        with pytest.raises(InstallError, match="array of strings"):
            install_hosts(
                ["opencode", "pi"],
                scope="project",
                project_dir=root,
                dry_run=False,
            )

        assert not (root / "opencode.json").exists()
        assert json.loads(pi_settings.read_text()) == {"extensions": "not-an-array"}


def test_generic_mcp_configuration_is_truthful_about_assurance():
    result = generic_mcp_config().public()

    assert result["details"]["mcpServers"]["cortheon"]["command"]
    assert result["details"]["assurance"] == "cooperative"
    assert result["details"]["writes_files"] is False
    assert result["status"] == "configuration"


def test_host_installation_status_reports_configuration_without_contents():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(
            os.environ,
            {
                "XDG_CONFIG_HOME": str(root / "config"),
                "XDG_DATA_HOME": str(root / "data"),
                "PI_CODING_AGENT_DIR": str(root / "pi"),
                "HOME": str(root / "home"),
            },
        ):
            install_opencode(scope="user", project_dir=root, dry_run=False)
            install_pi(scope="user", project_dir=root, dry_run=False)
            opencode_config = root / "config" / "opencode" / "opencode.json"
            payload = json.loads(opencode_config.read_text())
            payload["api_token"] = "topsecret"
            opencode_config.write_text(json.dumps(payload))
            statuses = host_installation_status()

        assert statuses["opencode"]["configured"] is True
        assert statuses["pi"]["configured"] is True
        assert "topsecret" not in json.dumps(statuses)


def test_uninstall_removes_only_cortheon_adapter_references():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        opencode = root / "opencode.json"
        pi = root / ".pi" / "settings.json"
        pi.parent.mkdir(parents=True)
        opencode.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "plugin": ["file:///existing.js", "file:///wheel/cortheon/opencode_plugin.js"],
                }
            )
        )
        pi.write_text(
            json.dumps(
                {
                    "theme": "light",
                    "extensions": ["./existing.ts", "/wheel/cortheon/pi_extension.ts"],
                }
            )
        )

        results = uninstall_hosts(
            ["opencode", "pi"], scope="project", project_dir=root, dry_run=False
        )

        assert [result.status for result in results] == ["removed", "removed"]
        assert json.loads(opencode.read_text()) == {
            "theme": "dark",
            "plugin": ["file:///existing.js"],
        }
        assert json.loads(pi.read_text()) == {"theme": "light", "extensions": ["./existing.ts"]}
        assert (root / "opencode.json.cortheon.bak").is_file()
        assert (root / ".pi" / "settings.json.cortheon.bak").is_file()


def test_generic_install_is_rejected_before_any_host_change():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with pytest.raises(InstallError, match="configuration-only"):
            install_hosts(["opencode", "generic"], scope="project", project_dir=root)
        assert not (root / "opencode.json").exists()


def test_omp_install_writes_user_mcp_and_skill():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(os.environ, {"HOME": str(root)}):
            first = install_hosts(["omp"], scope="user", dry_run=False)[0]
            second = install_hosts(["omp"], scope="user", dry_run=False)[0]

        mcp = root / ".omp" / "agent" / "mcp.json"
        skill = root / ".omp" / "agent" / "skills" / "cortheon-runtime" / "SKILL.md"
        assert first.status == "installed"
        assert second.status == "present"
        servers = json.loads(mcp.read_text())["mcpServers"]
        assert Path(servers["cortheon"]["command"]).name == "cortheon-mcp"
        assert servers["cortheon"]["args"] == []
        assert skill.is_file()
        assert "cooperative" in skill.read_text()


def test_omp_install_project_scope_preserves_other_servers():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mcp = root / ".omp" / "mcp.json"
        mcp.parent.mkdir(parents=True)
        mcp.write_text(
            json.dumps({"mcpServers": {"Jira": {"type": "http", "url": "https://mcp.jira/"}}})
        )

        result = install_hosts(["omp"], scope="project", project_dir=root, dry_run=False)[0]

        assert result.status == "installed"
        payload = json.loads(mcp.read_text())
        assert set(payload["mcpServers"]) == {"Jira", "cortheon"}
        assert payload["mcpServers"]["Jira"]["url"] == "https://mcp.jira/"
        assert (root / ".omp" / "skills" / "cortheon-runtime" / "SKILL.md").is_file()


def test_omp_status_reports_server_and_skill():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(os.environ, {"HOME": str(root)}):
            install_hosts(["omp"], scope="user", dry_run=False)
            status = host_installation_status()["omp"]

        assert status["configured"] is True
        assert status["valid"] is True
        assert status["skill_present"] is True
        assert status["scope"] == "user"
        assert "topsecret" not in json.dumps(status)


def test_omp_status_rejects_a_missing_managed_skill():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        install_hosts(["omp"], scope="project", project_dir=root)
        skill = root / ".omp" / "skills" / "cortheon-runtime" / "SKILL.md"
        skill.unlink()

        status = host_installation_status(scope="project", project_dir=root)["omp"]

        assert status["configured"] is True
        assert status["skill_present"] is False
        assert status["valid"] is False


def test_omp_profile_rejects_path_traversal():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            patch.dict(os.environ, {"HOME": str(root), "OMP_PROFILE": "../../escape"}),
            pytest.raises(InstallError, match="OMP profile names"),
        ):
            install_hosts(["omp"], scope="user")

        assert not (root.parent / "escape" / "agent" / "mcp.json").exists()


def test_omp_uninstall_removes_managed_install_and_preserves_other_servers():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        install_hosts(["omp"], scope="project", project_dir=root)
        mcp = root / ".omp" / "mcp.json"
        payload = json.loads(mcp.read_text())
        payload["mcpServers"]["Jira"] = {"type": "http", "url": "https://mcp.jira/"}
        mcp.write_text(json.dumps(payload))

        result = uninstall_hosts(["omp"], scope="project", project_dir=root, dry_run=False)[0]

        assert result.status == "removed"
        payload = json.loads(mcp.read_text())
        assert payload["mcpServers"] == {"Jira": {"type": "http", "url": "https://mcp.jira/"}}
        assert mcp.with_name("mcp.json.cortheon.bak").is_file()
        assert not (root / ".omp" / "skills" / "cortheon-runtime").exists()


def test_omp_install_preflight_rejects_malformed_config_before_other_host_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mcp = root / ".omp" / "mcp.json"
        mcp.parent.mkdir(parents=True)
        mcp.write_text('{"mcpServers": "not-an-object"}')

        with pytest.raises(InstallError, match="mcpServers"):
            install_hosts(["opencode", "omp"], scope="project", project_dir=root)

        assert not (root / "opencode.json").exists()
        assert json.loads(mcp.read_text()) == {"mcpServers": "not-an-object"}


def test_omp_install_preflights_skill_symlink_before_writing_config():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        external = root / "external-skill"
        external.mkdir()
        skill = root / ".omp" / "skills" / "cortheon-runtime"
        skill.parent.mkdir(parents=True)
        skill.symlink_to(external, target_is_directory=True)

        with pytest.raises(InstallError, match="symlinked OMP skill"):
            install_hosts(["omp"], scope="project", project_dir=root)

        assert not (root / ".omp" / "mcp.json").exists()
        assert not (external / "SKILL.md").exists()


def test_omp_install_rolls_back_skill_when_config_write_fails():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        skill = root / ".omp" / "skills" / "cortheon-runtime" / "SKILL.md"

        with (
            patch("cortheon.cognitive_install._atomic_json", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            install_hosts(["omp"], scope="project", project_dir=root)

        assert not (root / ".omp" / "mcp.json").exists()
        assert not skill.exists()


def test_omp_uninstall_removes_only_the_bundled_skill_file():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        install_hosts(["omp"], scope="project", project_dir=root)
        mcp = root / ".omp" / "mcp.json"
        payload = json.loads(mcp.read_text())
        payload["mcpServers"]["Jira"] = {"type": "http", "url": "https://mcp.jira/"}
        mcp.write_text(json.dumps(payload))
        skill = root / ".omp" / "skills" / "cortheon-runtime"
        operator_file = skill / "NOTES.md"
        operator_file.write_text("keep me\n")

        result = uninstall_hosts(["omp"], scope="project", project_dir=root)[0]

        assert result.status == "removed"
        assert operator_file.read_text() == "keep me\n"
        assert not (skill / "SKILL.md").exists()
        assert json.loads(mcp.read_text())["mcpServers"] == {
            "Jira": {"type": "http", "url": "https://mcp.jira/"}
        }


def test_omp_uninstall_restores_skill_when_config_write_fails():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        install_hosts(["omp"], scope="project", project_dir=root)
        mcp = root / ".omp" / "mcp.json"
        skill = root / ".omp" / "skills" / "cortheon-runtime" / "SKILL.md"

        with (
            patch("cortheon.cognitive_install._atomic_json", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            uninstall_hosts(["omp"], scope="project", project_dir=root)

        assert skill.is_file()
        assert "cortheon" in json.loads(mcp.read_text())["mcpServers"]


def test_omp_uninstall_dry_run_reports_a_skill_only_change():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        install_hosts(["omp"], scope="project", project_dir=root)
        mcp = root / ".omp" / "mcp.json"
        mcp.write_text("{}\n")
        skill = root / ".omp" / "skills" / "cortheon-runtime" / "SKILL.md"

        result = uninstall_hosts(["omp"], scope="project", project_dir=root, dry_run=True)[0]

        assert result.status == "planned"
        assert result.details["removed_skill"] is True
        assert skill.is_file()


def test_omp_uninstall_refuses_an_unrecognized_server_entry():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mcp = root / ".omp" / "mcp.json"
        mcp.parent.mkdir(parents=True)
        mcp.write_text(json.dumps({"mcpServers": {"cortheon": {"command": "someone-else"}}}))

        with pytest.raises(InstallError, match="unrecognized OMP server"):
            uninstall_hosts(["omp"], scope="project", project_dir=root)

        assert json.loads(mcp.read_text())["mcpServers"]["cortheon"] == {"command": "someone-else"}


def test_omp_uninstall_reports_absent_without_a_config():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        result = uninstall_hosts(["omp"], scope="project", project_dir=root, dry_run=False)[0]

        assert result.status == "absent"
        assert result.details["removed_server"] is False


def test_codex_uninstall_removes_only_verified_owned_marketplace():
    with tempfile.TemporaryDirectory() as directory:
        data_home = Path(directory) / "data"
        marketplace = data_home / "cortheon" / "codex-marketplace"
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}):
            install_codex(dry_run=False, run_cli=False, install_root=marketplace)
            result = uninstall_hosts(["codex"], run_codex_cli=False)[0]

        assert result.status == "removed"
        assert not marketplace.exists()


def test_codex_uninstall_refuses_an_unverified_directory():
    with tempfile.TemporaryDirectory() as directory:
        data_home = Path(directory) / "data"
        marketplace = data_home / "cortheon" / "codex-marketplace"
        marketplace.mkdir(parents=True)
        (marketplace / "unrelated.txt").write_text("owned by someone else\n")
        with (
            patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}),
            pytest.raises(InstallError, match="unverified"),
        ):
            uninstall_hosts(["codex"], run_codex_cli=False)

        assert (marketplace / "unrelated.txt").is_file()


def test_uninstall_preflights_codex_ownership_before_adapter_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_home = root / "config"
        data_home = root / "data"
        opencode = config_home / "opencode" / "opencode.json"
        opencode.parent.mkdir(parents=True)
        opencode.write_text(json.dumps({"plugin": ["file:///wheel/cortheon/opencode_plugin.js"]}))
        marketplace = data_home / "cortheon" / "codex-marketplace"
        marketplace.mkdir(parents=True)
        (marketplace / "unrelated.txt").write_text("owned by someone else\n")
        with (
            patch.dict(
                os.environ,
                {"XDG_CONFIG_HOME": str(config_home), "XDG_DATA_HOME": str(data_home)},
            ),
            pytest.raises(InstallError, match="unverified"),
        ):
            uninstall_hosts(["opencode", "codex"], run_codex_cli=False)

        assert json.loads(opencode.read_text())["plugin"]


def test_codex_status_comes_from_plugin_list_not_copied_source():
    with tempfile.TemporaryDirectory() as directory:
        data_home = Path(directory) / "data"
        marketplace = data_home / "cortheon" / "codex-marketplace"
        install_codex(dry_run=False, run_cli=False, install_root=marketplace)
        listed = {
            "installed": [
                {
                    "pluginId": "cortheon@cortheon-local",
                    "version": f"0.1.0+codex.{_SOURCE_FINGERPRINT}",
                    "installed": True,
                    "enabled": True,
                }
            ]
        }
        completed = __import__("subprocess").CompletedProcess(
            ["codex", "plugin", "list", "--json"],
            0,
            stdout=json.dumps(listed),
            stderr="",
        )
        with (
            patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}),
            patch("cortheon.cognitive_install_core.config.shutil.which", return_value="/bin/codex"),
            patch("cortheon.cognitive_install_core.config.subprocess.run", return_value=completed),
        ):
            status = host_installation_status()["codex"]

        assert status["configured"] is True
        assert status["plugin_listed"] is True
        assert status["version_matches"] is True


def test_codex_status_rejects_a_stale_cached_plugin_version():
    with tempfile.TemporaryDirectory() as directory:
        data_home = Path(directory) / "data"
        marketplace = data_home / "cortheon" / "codex-marketplace"
        install_codex(dry_run=False, run_cli=False, install_root=marketplace)
        listed = {
            "installed": [
                {
                    "pluginId": "cortheon@cortheon-local",
                    "version": "0.1.0+codex.stale",
                    "installed": True,
                    "enabled": True,
                }
            ]
        }
        completed = __import__("subprocess").CompletedProcess(
            ["codex", "plugin", "list", "--json"],
            0,
            stdout=json.dumps(listed),
            stderr="",
        )
        with (
            patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}),
            patch("cortheon.cognitive_install_core.config.shutil.which", return_value="/bin/codex"),
            patch("cortheon.cognitive_install_core.config.subprocess.run", return_value=completed),
        ):
            status = host_installation_status()["codex"]

        assert status["configured"] is True
        assert status["valid"] is False
        assert status["version_matches"] is False
