"""Offline unit tests for `install_mcp` and `FlashForge` live flywheel auto-logging.

100% isolated under `tmp_path` — never reads or writes the real `$HOME`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

import cortheon
from flash_forge.client import ModelCapabilities
from flash_forge.code_forge import CodeCandidate, CodeForgeTrace
from flash_forge.engine import FlashForge, ForgeConfig, ForgeResult, SolveResult
from flash_forge.flywheel import export_dpo_jsonl, harvest_from_results_dir
from flash_forge.integrations import MCP_HARNESSES, install_mcp
from flash_forge.reason_forge import BranchOutcome, ReasonForgeTrace
from flash_forge.sandbox import ExecResult


# ===========================================================================
# Part 1: One-Command MCP Auto-Installer (`flash_forge.integrations.install_mcp`)
# ===========================================================================


def test_install_mcp_all_harnesses_home_and_project_modes(tmp_path: Path) -> None:
    assert MCP_HARNESSES == ("claude", "cursor", "opencode", "codex", "windsurf")

    url = "http://127.0.0.1:18651/v1"
    cmd = "/opt/local/bin/cortheon"

    # --- Home mode ---
    home = tmp_path / "home"
    home.mkdir()
    written_home = install_mcp(
        MCP_HARNESSES,
        url=url,
        command=cmd,
        home_dir=home,
    )
    expected_home = [
        home / ".claude.json",
        home / ".cursor" / "mcp.json",
        home / ".config" / "opencode" / "opencode.json",
        home / ".codex" / "config.toml",
        home / ".codeium" / "windsurf" / "mcp_config.json",
    ]
    assert written_home == [p.resolve() for p in expected_home]
    for path in written_home:
        assert path.is_file()
        # Ensure no leftover .tmp files
        assert list(path.parent.glob("*.tmp")) == []

    # Verify claude, cursor, windsurf JSON schema (no api_key / no None)
    for json_path in (expected_home[0], expected_home[1], expected_home[4]):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        entry = data["mcpServers"]["cortheon"]
        assert entry["command"] == cmd
        assert entry["args"] == ["mcp", "--url", url]
        assert entry["env"] == {"CORTHEON_URL": url}
        assert "CORTHEON_API_KEY" not in entry["env"]
        assert "--api-key" not in entry["args"]
        assert "null" not in json_path.read_text(encoding="utf-8")

    # Verify opencode official local MCP schema
    oc_data = json.loads(expected_home[2].read_text(encoding="utf-8"))
    oc_entry = oc_data["mcp"]["cortheon"]
    assert oc_entry["type"] == "local"
    assert oc_entry["command"] == [cmd, "mcp", "--url", url]
    assert oc_entry["enabled"] is True
    assert oc_entry["environment"] == {"CORTHEON_URL": url}
    assert "CORTHEON_API_KEY" not in oc_entry["environment"]

    # Verify codex TOML schema
    codex_parsed = tomllib.loads(expected_home[3].read_text(encoding="utf-8"))
    codex_entry = codex_parsed["mcp_servers"]["cortheon"]
    assert codex_entry["command"] == cmd
    assert codex_entry["args"] == ["mcp", "--url", url]
    assert codex_entry["env"] == {"CORTHEON_URL": url}

    # --- Project mode ---
    proj = tmp_path / "my_project"
    proj.mkdir()
    home_unused = tmp_path / "home_unused"
    home_unused.mkdir()
    written_proj = install_mcp(
        "all",
        url=url,
        command=cmd,
        project_dir=proj,
        home_dir=home_unused,
    )
    expected_proj = [
        proj / ".mcp.json",
        proj / ".cursor" / "mcp.json",
        proj / "opencode.json",
        proj / ".codex" / "config.toml",
        proj / ".windsurf" / "mcp_config.json",
    ]
    assert written_proj == [p.resolve() for p in expected_proj]
    for path in written_proj:
        assert path.is_file()
    # Home directory must remain untouched in project_dir mode
    assert list(home_unused.iterdir()) == []


def test_install_mcp_executable_resolution_and_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    url = "http://127.0.0.1:9000/v1"

    # 1. Fallback to sys.executable -m flash_forge.cli when `cortheon` is NOT on PATH
    monkeypatch.setattr("flash_forge.integrations.shutil.which", lambda name: None)
    install_mcp(["claude", "opencode", "codex"], url=url, home_dir=home)

    claude_cfg = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    assert claude_cfg["mcpServers"]["cortheon"]["command"] == sys.executable
    assert claude_cfg["mcpServers"]["cortheon"]["args"] == [
        "-m",
        "flash_forge.cli",
        "mcp",
        "--url",
        url,
    ]
    assert claude_cfg["mcpServers"]["cortheon"]["env"] == {"CORTHEON_URL": url}

    oc_cfg = json.loads((home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert oc_cfg["mcp"]["cortheon"]["command"] == [
        sys.executable,
        "-m",
        "flash_forge.cli",
        "mcp",
        "--url",
        url,
    ]

    # 2. shutil.which("cortheon") found + explicit api_key
    monkeypatch.setattr(
        "flash_forge.integrations.shutil.which",
        lambda name: "/usr/local/bin/cortheon" if name == "cortheon" else None,
    )
    install_mcp(
        ["claude", "opencode", "codex"],
        url=url,
        api_key="my-secret-key",
        home_dir=home,
    )

    claude_cfg2 = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    c_ent = claude_cfg2["mcpServers"]["cortheon"]
    assert c_ent["command"] == "/usr/local/bin/cortheon"
    assert c_ent["args"] == ["mcp", "--url", url, "--api-key", "my-secret-key"]
    assert c_ent["env"] == {"CORTHEON_URL": url, "CORTHEON_API_KEY": "my-secret-key"}

    oc_cfg2 = json.loads((home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert oc_cfg2["mcp"]["cortheon"]["environment"]["CORTHEON_API_KEY"] == "my-secret-key"
    assert "--api-key" in oc_cfg2["mcp"]["cortheon"]["command"]

    codex_cfg2 = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    assert codex_cfg2["mcp_servers"]["cortheon"]["env"]["CORTHEON_API_KEY"] == "my-secret-key"
    assert codex_cfg2["mcp_servers"]["cortheon"]["args"] == [
        "mcp",
        "--url",
        url,
        "--api-key",
        "my-secret-key",
    ]

    # 3. Re-running without api_key cleanly removes old secret
    install_mcp(["claude", "codex"], url=url, api_key=None, home_dir=home)
    claude_cfg3 = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    assert "CORTHEON_API_KEY" not in claude_cfg3["mcpServers"]["cortheon"]["env"]
    assert "--api-key" not in claude_cfg3["mcpServers"]["cortheon"]["args"]
    codex_cfg3 = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    assert "CORTHEON_API_KEY" not in codex_cfg3["mcp_servers"]["cortheon"]["env"]
    assert "--api-key" not in codex_cfg3["mcp_servers"]["cortheon"]["args"]


def test_install_mcp_preserves_existing_json_servers_and_is_idempotent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()

    # Pre-populate home configs with existing servers and unrelated settings
    (home / ".claude.json").write_text(
        json.dumps(
            {
                "theme": "dark",
                "hasCompletedOnboarding": True,
                "mcpServers": {
                    "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
                    "cortheon": {"command": "stale-old", "args": ["old"]},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (home / ".cursor").mkdir(parents=True)
    (home / ".cursor" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"postgres": {"command": "pg-mcp", "args": []}}, "customFlag": 7}),
        encoding="utf-8",
    )
    (home / ".config" / "opencode").mkdir(parents=True)
    (home / ".config" / "opencode" / "opencode.json").write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "mcp": {
                    "context7": {"type": "remote", "url": "https://mcp.context7.com/mcp", "enabled": True}
                },
            }
        ),
        encoding="utf-8",
    )
    (home / ".codeium" / "windsurf").mkdir(parents=True)
    (home / ".codeium" / "windsurf" / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"linear": {"command": "linear-mcp", "args": ["--stdio"]}}}),
        encoding="utf-8",
    )

    json_harnesses = ("claude", "cursor", "opencode", "windsurf")
    first_paths = install_mcp(json_harnesses, command="/bin/cortheon", home_dir=home)
    first_contents = {p: p.read_text(encoding="utf-8") for p in first_paths}

    # Verify all existing servers & unrelated settings remain intact
    c_data = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    assert c_data["theme"] == "dark"
    assert c_data["hasCompletedOnboarding"] is True
    assert c_data["mcpServers"]["github"] == {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
    }
    assert c_data["mcpServers"]["cortheon"]["command"] == "/bin/cortheon"

    cur_data = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert cur_data["customFlag"] == 7
    assert "postgres" in cur_data["mcpServers"]
    assert "cortheon" in cur_data["mcpServers"]

    oc_data = json.loads((home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert oc_data["$schema"] == "https://opencode.ai/config.json"
    assert oc_data["mcp"]["context7"]["type"] == "remote"
    assert oc_data["mcp"]["cortheon"]["type"] == "local"

    ws_data = json.loads((home / ".codeium" / "windsurf" / "mcp_config.json").read_text(encoding="utf-8"))
    assert "linear" in ws_data["mcpServers"]
    assert "cortheon" in ws_data["mcpServers"]

    # Second run is 100% idempotent (byte-for-byte identical)
    second_paths = install_mcp(json_harnesses, command="/bin/cortheon", home_dir=home)
    assert second_paths == first_paths
    for p in second_paths:
        assert p.read_text(encoding="utf-8") == first_contents[p]

    # Also verify pre-populated preservation in project_dir mode across all 4 JSON harnesses
    (proj / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"sqlite": {"command": "sqlite-mcp", "args": []}}, "version": 1}),
        encoding="utf-8",
    )
    (proj / ".cursor").mkdir(parents=True)
    (proj / ".cursor" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"figma": {"command": "figma-mcp", "args": []}}}),
        encoding="utf-8",
    )
    (proj / "opencode.json").write_text(
        json.dumps({"mcp": {"sentry": {"type": "remote", "url": "https://mcp.sentry.dev", "enabled": True}}}),
        encoding="utf-8",
    )
    (proj / ".windsurf").mkdir(parents=True)
    (proj / ".windsurf" / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"slack": {"command": "slack-mcp", "args": []}}}),
        encoding="utf-8",
    )

    proj_first = install_mcp(json_harnesses, command="/bin/cortheon", project_dir=proj, home_dir=home)
    proj_first_text = {p: p.read_text(encoding="utf-8") for p in proj_first}
    assert set(json.loads((proj / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]) == {"sqlite", "cortheon"}
    assert set(json.loads((proj / ".cursor" / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]) == {
        "figma",
        "cortheon",
    }
    assert set(json.loads((proj / "opencode.json").read_text(encoding="utf-8"))["mcp"]) == {"sentry", "cortheon"}
    assert set(json.loads((proj / ".windsurf" / "mcp_config.json").read_text(encoding="utf-8"))["mcpServers"]) == {
        "slack",
        "cortheon",
    }
    proj_second = install_mcp(json_harnesses, command="/bin/cortheon", project_dir=proj, home_dir=home)
    assert proj_second == proj_first
    for p in proj_second:
        assert p.read_text(encoding="utf-8") == proj_first_text[p]


def test_install_mcp_codex_toml_preserves_comments_tables_and_idempotency(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_dir = home / ".codex"
    codex_dir.mkdir(parents=True)
    toml_path = codex_dir / "config.toml"

    # Case 1: Existing TOML without cortheon (comments, top-level keys, multiline string trap, existing server)
    original_toml = (
        "# Global Codex configuration\n"
        'model = "o3"\n'
        'notes = """\n'
        "[mcp_servers.cortheon]\n"
        'this line is inside a multiline string and must NOT be touched\n'
        '"""\n'
        "\n"
        "# Filesystem MCP server\n"
        "[mcp_servers.filesystem]\n"
        'command = "npx"\n'
        'args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]\n'
    )
    toml_path.write_text(original_toml, encoding="utf-8")

    install_mcp(["codex"], url="http://127.0.0.1:8080/v1", command="/bin/cortheon", home_dir=home)
    after_first = toml_path.read_text(encoding="utf-8")

    parsed = tomllib.loads(after_first)
    assert parsed["model"] == "o3"
    assert "[mcp_servers.cortheon]" in parsed["notes"]
    assert parsed["mcp_servers"]["filesystem"]["command"] == "npx"
    assert parsed["mcp_servers"]["cortheon"] == {
        "command": "/bin/cortheon",
        "args": ["mcp", "--url", "http://127.0.0.1:8080/v1"],
        "env": {"CORTHEON_URL": "http://127.0.0.1:8080/v1"},
    }
    assert "# Global Codex configuration" in after_first
    assert "# Filesystem MCP server" in after_first

    # Re-running twice must be byte-for-byte idempotent with ZERO duplicate blocks
    install_mcp(["codex"], url="http://127.0.0.1:8080/v1", command="/bin/cortheon", home_dir=home)
    after_second = toml_path.read_text(encoding="utf-8")
    assert after_second == after_first
    # Exactly 2 occurrences: 1 inside multiline string + 1 real table header
    assert after_second.count("[mcp_servers.cortheon]") == 2

    # Case 2: Updating an existing [mcp_servers.cortheon] + legacy [mcp_servers.cortheon.env] in the middle of a file
    proj = tmp_path / "proj"
    proj_toml = proj / ".codex" / "config.toml"
    proj_toml.parent.mkdir(parents=True)
    proj_toml.write_text(
        "# Project Codex config\n"
        "[mcp_servers.cortheon]\n"
        'command = "old-cortheon"\n'
        'args = ["mcp"]\n'
        "\n"
        "[mcp_servers.cortheon.env]\n"
        'CORTHEON_URL = "http://old:1234/v1"\n'
        "\n"
        "# Keep this comment above github!\n"
        "[mcp_servers.github]\n"
        'command = "gh-mcp"\n'
        'args = ["--stdio"]\n',
        encoding="utf-8",
    )

    install_mcp(
        ["codex"],
        url="http://127.0.0.1:18651/v1",
        command="/usr/local/bin/cortheon",
        project_dir=proj,
        home_dir=home,
    )
    updated_proj = proj_toml.read_text(encoding="utf-8")
    parsed_proj = tomllib.loads(updated_proj)
    assert parsed_proj["mcp_servers"]["cortheon"]["command"] == "/usr/local/bin/cortheon"
    assert parsed_proj["mcp_servers"]["cortheon"]["env"] == {"CORTHEON_URL": "http://127.0.0.1:18651/v1"}
    assert parsed_proj["mcp_servers"]["github"] == {"command": "gh-mcp", "args": ["--stdio"]}
    assert "# Project Codex config" in updated_proj
    assert "# Keep this comment above github!" in updated_proj
    assert updated_proj.count("[mcp_servers.cortheon]") == 1
    assert "[mcp_servers.cortheon.env]" not in updated_proj


def test_install_mcp_malformed_configs_raise_value_error_without_clobbering(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    # 1. Malformed JSON syntax
    bad_claude = home / ".claude.json"
    bad_raw = '{ "mcpServers": { broken json '
    bad_claude.write_text(bad_raw, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        install_mcp(["claude"], command="/bin/cortheon", home_dir=home)
    assert str(bad_claude.resolve()) in str(exc_info.value)
    assert bad_claude.read_text(encoding="utf-8") == bad_raw

    # 2. Valid JSON but non-object root
    bad_cursor = home / ".cursor" / "mcp.json"
    bad_cursor.parent.mkdir(parents=True)
    bad_cursor.write_text('["not", "an", "object"]', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info2:
        install_mcp(["cursor"], command="/bin/cortheon", home_dir=home)
    assert str(bad_cursor.resolve()) in str(exc_info2.value)
    assert bad_cursor.read_text(encoding="utf-8") == '["not", "an", "object"]'

    # 3. Valid JSON object but non-object mcpServers value
    bad_ws = home / ".codeium" / "windsurf" / "mcp_config.json"
    bad_ws.parent.mkdir(parents=True)
    bad_ws.write_text('{"mcpServers": ["bad"]}', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info3:
        install_mcp(["windsurf"], command="/bin/cortheon", home_dir=home)
    assert str(bad_ws.resolve()) in str(exc_info3.value)
    assert bad_ws.read_text(encoding="utf-8") == '{"mcpServers": ["bad"]}'

    # 4. Malformed TOML
    bad_toml = home / ".codex" / "config.toml"
    bad_toml.parent.mkdir(parents=True)
    bad_toml_raw = "[mcp_servers.broken\n= 123\n"
    bad_toml.write_text(bad_toml_raw, encoding="utf-8")
    with pytest.raises(ValueError) as exc_info4:
        install_mcp(["codex"], command="/bin/cortheon", home_dir=home)
    assert str(bad_toml.resolve()) in str(exc_info4.value)
    assert bad_toml.read_text(encoding="utf-8") == bad_toml_raw

    # 5. Unknown harness name
    with pytest.raises(ValueError, match="Unknown MCP harness"):
        install_mcp(["nonexistent_ide"], command="/bin/cortheon", home_dir=home)


# ===========================================================================
# Part 2: Live Self-Play Flywheel Auto-Logging (`flash_forge.engine.FlashForge`)
# ===========================================================================


def _make_mock_forge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    record_flywheel: bool,
    flywheel_path: Path | None,
    model_id: str = "qwen3-local-test",
) -> FlashForge:
    cfg = ForgeConfig(
        base_url="http://127.0.0.1:8080/v1",
        enable_memory=False,
        record_flywheel=record_flywheel,
        flywheel_path=flywheel_path,
    )
    ff = FlashForge(cfg)

    async def _fake_probe() -> ModelCapabilities:
        caps = ModelCapabilities(model_id=model_id)
        ff.client._caps = caps
        return caps

    monkeypatch.setattr(ff.client, "probe", _fake_probe)
    return ff


def test_engine_reason_wave2_rescue_records_pii_scrubbed_flywheel_and_exports_dpo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fw_path = tmp_path / "fw.jsonl"
    ff = _make_mock_forge(tmp_path, monkeypatch, record_flywheel=True, flywheel_path=fw_path)

    wrong_w1 = BranchOutcome(
        name="pal_exact",
        wave=1,
        kind="pal",
        raw_content="Loaded /Users/alice/bad_attempt.py via 192.168.1.42 -> ANSWER: 420",
        reasoning="",
        code="print('ANSWER: 420')",
        raw_answer="420",
        norm_answer="420",
    )
    good_w2 = BranchOutcome(
        name="w2_arbiter",
        wave=2,
        kind="pal",
        raw_content="print('ANSWER: 588')",
        reasoning="",
        code="print('ANSWER: 588')",
        exec_result=ExecResult(ok=True, stdout="ANSWER: 588\n", stderr="", returncode=0, elapsed_s=0.02),
        raw_answer="588",
        norm_answer="588",
        verified_constraints=True,
    )
    mock_trace = ReasonForgeTrace(
        wave1_outcomes=[wrong_w1],
        wave2_outcomes=[good_w2],
        wave2_triggered=True,
        wave2_reason="disagreement",
        graveyard=["[pal_exact] wrong answer 420"],
        consensus_type="wave2_verified (2 PAL)",
        final_answer="588",
        final_explanation="Verified Answer: **588**\n\nChecked via /Users/alice/verify.py",
        recovered_from_disagreement=True,
        verified=True,
    )

    async def _fake_solve_reasoning(**kwargs: Any) -> ReasonForgeTrace:
        return mock_trace

    monkeypatch.setattr("flash_forge.engine.solve_reasoning", _fake_solve_reasoning)

    prompt_with_pii = (
        "Compute the area from /Users/alice/problem.txt using token sk-abcdefghijklmnopqrstuvwxyz123456"
    )

    async def _run() -> ForgeResult:
        try:
            return await ff.solve(prompt_with_pii, force_mode="reason", learn=False)
        finally:
            await ff.close()

    res = asyncio.run(_run())
    assert isinstance(res, ForgeResult)
    assert isinstance(res, SolveResult)
    assert res.answer == "588"
    assert res.verified is True
    assert res.flywheel_recorded is True
    assert fw_path.is_file()

    lines = fw_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["domain"] == "math"
    assert rec["model"] == "qwen3-local-test"
    assert rec["chosen_verified"] is True
    assert "588" in rec["chosen"]
    assert "420" in rec["rejected"]
    assert rec["metadata"]["mode"] == "reason"
    assert rec["metadata"]["rejected_branch"] == "pal_exact"

    # PII must be thoroughly scrubbed
    dumped = json.dumps(rec)
    assert "alice" not in dumped
    assert "/Users/" not in dumped
    assert "192.168.1.42" not in dumped
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in dumped
    assert "<HOME>" in dumped
    assert "<PRIVATE_IP>" in dumped
    assert "<REDACTED_SECRET>" in dumped

    # End-to-end compatibility with harvest_from_results_dir + export_dpo_jsonl
    pairs = harvest_from_results_dir(fw_path)
    assert len(pairs) == 1
    dpo_out = tmp_path / "dpo.jsonl"
    stats = export_dpo_jsonl(pairs, dpo_out)
    assert stats.exported == 1
    assert stats.domain_counts == {"math": 1}
    dpo_row = json.loads(dpo_out.read_text(encoding="utf-8").strip())
    assert dpo_row["domain"] == "math"
    assert dpo_row["chosen"] == rec["chosen"]
    assert dpo_row["rejected"] == rec["rejected"]


def test_engine_reason_wave2_consensus_and_non_rescue_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fw_path = tmp_path / "fw_edge.jsonl"
    ff = _make_mock_forge(tmp_path, monkeypatch, record_flywheel=True, flywheel_path=fw_path)

    # 1. Wave-1 unanimous (wave2_triggered=False) -> must NOT record
    w1_unanimous = ReasonForgeTrace(
        wave1_outcomes=[
            BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="42", reasoning="", norm_answer="42")
        ],
        wave2_triggered=False,
        consensus_type="wave1_consensus (2 PAL)",
        final_answer="42",
        final_explanation="Verified Answer: **42**",
        verified=True,
    )

    # 2. Wave-2 fallback extract (unverified, no consensus) -> must NOT record
    w2_fallback = ReasonForgeTrace(
        wave1_outcomes=[
            BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="err", reasoning="", error="Timeout")
        ],
        wave2_triggered=True,
        consensus_type="fallback_extract",
        final_answer="7",
        final_explanation="guess",
        verified=False,
    )

    # 3. Wave-2 consensus (verified=False, consensus_type="wave2_verified (2 PAL)") with crashed Wave-1 -> records!
    w2_consensus = ReasonForgeTrace(
        wave1_outcomes=[
            BranchOutcome(
                name="pal_exact",
                wave=1,
                kind="pal",
                raw_content="x = 1 / 0",
                reasoning="",
                code="x = 1 / 0",
                exec_result=ExecResult(ok=False, stdout="", stderr="ZeroDivisionError", returncode=1, elapsed_s=0.01),
                error="ZeroDivisionError",
            )
        ],
        wave2_triggered=True,
        consensus_type="wave2_verified (2 PAL)",
        final_answer="99",
        final_explanation="Verified Answer: **99**",
        verified=False,
    )

    traces = iter([w1_unanimous, w2_fallback, w2_consensus])

    async def _fake_solve_reasoning(**kwargs: Any) -> ReasonForgeTrace:
        return next(traces)

    monkeypatch.setattr("flash_forge.engine.solve_reasoning", _fake_solve_reasoning)

    async def _run() -> list[ForgeResult]:
        try:
            r1 = await ff.solve("P1", force_mode="reason", learn=False)
            r2 = await ff.solve("P2", force_mode="reason", learn=False)
            r3 = await ff.solve("P3", force_mode="reason", learn=False)
            return [r1, r2, r3]
        finally:
            await ff.close()

    r1, r2, r3 = asyncio.run(_run())
    assert r1.flywheel_recorded is False
    assert r2.flywheel_recorded is False
    assert r3.flywheel_recorded is True

    lines = fw_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["prompt"] == "P3"
    assert "99" in rec["chosen"]
    assert "1 / 0" in rec["rejected"]


def test_engine_code_wave2_rescue_records_flywheel_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fw_path = tmp_path / "code_fw.jsonl"
    ff = _make_mock_forge(tmp_path, monkeypatch, record_flywheel=True, flywheel_path=fw_path)

    buggy_w1 = "def fib(n: int) -> int:\n    return n\n"
    fixed_w2 = (
        "def fib(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    rescue_trace = CodeForgeTrace(
        wave1_candidates=[
            CodeCandidate(
                name="radically_simple",
                wave=1,
                code=buggy_w1,
                raw_content=f"```python\n{buggy_w1}```",
                tests_passed=1,
                tests_total=3,
                failed_tests=["test_fib_5", "test_fib_10"],
            )
        ],
        wave2_candidates=[
            CodeCandidate(
                name="w2_surgical_fix",
                wave=2,
                code=fixed_w2,
                raw_content=f"```python\n{fixed_w2}```",
                tests_passed=3,
                tests_total=3,
            )
        ],
        wave2_triggered=True,
        wave2_reason="no perfect wave-1 survivor",
        winner_name="w2_surgical_fix",
        winner_code=fixed_w2,
        recovered_by_redteam_or_wave2=True,
        verified=True,
    )
    no_rescue_trace = CodeForgeTrace(
        wave1_candidates=[
            CodeCandidate(
                name="radically_simple",
                wave=1,
                code=fixed_w2,
                raw_content=fixed_w2,
                tests_passed=3,
                tests_total=3,
            )
        ],
        winner_name="radically_simple",
        winner_code=fixed_w2,
        recovered_by_redteam_or_wave2=False,
        verified=True,
    )
    traces = iter([rescue_trace, no_rescue_trace])

    async def _fake_solve_code(**kwargs: Any) -> CodeForgeTrace:
        return next(traces)

    monkeypatch.setattr("flash_forge.engine.solve_code", _fake_solve_code)

    async def _run() -> tuple[ForgeResult, ForgeResult]:
        try:
            res_rescue = await ff.solve("Implement fib(n)", force_mode="code", learn=False)
            res_clean = await ff.solve("Implement fib(n) again", force_mode="code", learn=False)
            return res_rescue, res_clean
        finally:
            await ff.close()

    res_rescue, res_clean = asyncio.run(_run())
    assert res_rescue.flywheel_recorded is True
    assert res_clean.flywheel_recorded is False

    lines = fw_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["domain"] == "code"
    assert rec["chosen"] == fixed_w2.strip()
    assert rec["rejected"] == buggy_w1.strip()
    assert rec["metadata"]["winner_name"] == "w2_surgical_fix"
    assert rec["metadata"]["rejected_candidate"] == "radically_simple"

    pairs = harvest_from_results_dir(tmp_path)
    dpo_out = tmp_path / "code_dpo.jsonl"
    stats = export_dpo_jsonl(pairs, dpo_out)
    assert stats.exported == 1
    assert stats.domain_counts == {"code": 1}


def test_engine_record_flywheel_false_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fw_path = tmp_path / "should_not_exist.jsonl"
    ff = _make_mock_forge(tmp_path, monkeypatch, record_flywheel=False, flywheel_path=fw_path)

    async def _fake_solve_reasoning(**kwargs: Any) -> ReasonForgeTrace:
        return ReasonForgeTrace(
            wave1_outcomes=[
                BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="1", reasoning="", norm_answer="1")
            ],
            wave2_triggered=True,
            consensus_type="wave2_verified (2 PAL)",
            final_answer="2",
            final_explanation="Verified Answer: **2**",
            recovered_from_disagreement=True,
            verified=True,
        )

    async def _fake_solve_code(**kwargs: Any) -> CodeForgeTrace:
        return CodeForgeTrace(
            wave1_candidates=[
                CodeCandidate(
                    name="w1",
                    wave=1,
                    code="def f(): return 0",
                    raw_content="def f(): return 0",
                    tests_passed=0,
                    tests_total=1,
                    failed_tests=["test_f"],
                )
            ],
            winner_name="w2",
            winner_code="def f(): return 1",
            recovered_by_redteam_or_wave2=True,
            verified=True,
        )

    monkeypatch.setattr("flash_forge.engine.solve_reasoning", _fake_solve_reasoning)
    monkeypatch.setattr("flash_forge.engine.solve_code", _fake_solve_code)

    async def _run() -> tuple[ForgeResult, ForgeResult]:
        try:
            r_reason = await ff.solve("1 + 1?", force_mode="reason", learn=False)
            r_code = await ff.solve("Write f()", force_mode="code", learn=False)
            return r_reason, r_code
        finally:
            await ff.close()

    r_reason, r_code = asyncio.run(_run())
    assert r_reason.flywheel_recorded is False
    assert r_code.flywheel_recorded is False
    assert not fw_path.exists()


def test_engine_flywheel_io_error_never_crashes_solve_and_aliases_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Backwards-compatible aliases
    assert SolveResult is ForgeResult
    assert cortheon.Cortheon is FlashForge
    assert cortheon.CortheonConfig is ForgeConfig
    assert cortheon.CortheonResult is ForgeResult

    fw_path = tmp_path / "fw_io_err.jsonl"
    ff = _make_mock_forge(tmp_path, monkeypatch, record_flywheel=True, flywheel_path=fw_path)

    async def _fake_solve_reasoning(**kwargs: Any) -> ReasonForgeTrace:
        return ReasonForgeTrace(
            wave1_outcomes=[
                BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="wrong", reasoning="", norm_answer="1")
            ],
            wave2_triggered=True,
            consensus_type="wave2_verified (2 PAL)",
            final_answer="2",
            final_explanation="Verified Answer: **2**",
            recovered_from_disagreement=True,
            verified=True,
        )

    def _boom(*args: Any, **kwargs: Any) -> Path:
        raise OSError("Simulated disk full")

    monkeypatch.setattr("flash_forge.engine.solve_reasoning", _fake_solve_reasoning)
    monkeypatch.setattr("flash_forge.engine.record_live_trace", _boom)

    async def _run() -> ForgeResult:
        try:
            return await ff.solve("What is 1 + 1?", force_mode="reason", learn=False)
        finally:
            await ff.close()

    res = asyncio.run(_run())
    assert res.answer == "2"
    assert res.verified is True
    assert res.flywheel_recorded is False
    assert not fw_path.exists()


def test_engine_flywheel_default_path_uses_home_cortheon_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    ff = _make_mock_forge(tmp_path, monkeypatch, record_flywheel=True, flywheel_path=None)

    async def _fake_solve_reasoning(**kwargs: Any) -> ReasonForgeTrace:
        return ReasonForgeTrace(
            wave1_outcomes=[
                BranchOutcome(
                    name="pal_exact",
                    wave=1,
                    kind="pal",
                    raw_content="ANSWER: 10",
                    reasoning="",
                    norm_answer="10",
                )
            ],
            wave2_triggered=True,
            consensus_type="wave2_verified (2 PAL)",
            final_answer="12",
            final_explanation="",  # fallback to final_answer
            verified=True,
        )

    monkeypatch.setattr("flash_forge.engine.solve_reasoning", _fake_solve_reasoning)

    async def _run() -> ForgeResult:
        try:
            return await ff.solve("Find 3 * 4", force_mode="reason", learn=False)
        finally:
            await ff.close()

    res = asyncio.run(_run())
    assert res.flywheel_recorded is True
    default_log = fake_home / ".cortheon" / "flywheel.jsonl"
    assert default_log.is_file()
    rec = json.loads(default_log.read_text(encoding="utf-8").strip())
    assert rec["chosen"] == "12"
    assert rec["rejected"] == "ANSWER: 10"


def test_install_mcp_codex_quoted_header_and_inline_conflict(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_toml = home / ".codex" / "config.toml"
    codex_toml.parent.mkdir(parents=True)

    # Quoted table key + inline comment must be replaced cleanly in place
    codex_toml.write_text(
        '[mcp_servers."cortheon"] # existing quoted header\n'
        'command = "old"\n'
        'args = ["mcp"]\n',
        encoding="utf-8",
    )
    install_mcp(["codex"], command="/opt/My Tools/cortheon", url="http://127.0.0.1:8080/v1", home_dir=home)
    updated = codex_toml.read_text(encoding="utf-8")
    parsed = tomllib.loads(updated)
    assert parsed["mcp_servers"]["cortheon"]["command"] == "/opt/My Tools/cortheon"
    assert updated.count("cortheon]") == 1

    # Conflicting inline key inside [mcp_servers] raises ValueError without clobbering file
    conflict_proj = tmp_path / "conflict_proj"
    conflict_toml = conflict_proj / ".codex" / "config.toml"
    conflict_toml.parent.mkdir(parents=True)
    raw_conflict = '[mcp_servers]\ncortheon = { command = "inline" }\n'
    conflict_toml.write_text(raw_conflict, encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        install_mcp(["codex"], command="/bin/cortheon", project_dir=conflict_proj, home_dir=home)
    assert str(conflict_toml.resolve()) in str(exc_info.value)
    assert conflict_toml.read_text(encoding="utf-8") == raw_conflict

