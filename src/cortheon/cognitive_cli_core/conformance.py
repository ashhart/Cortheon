"""Executable host-integration conformance probes."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from cortheon import cognitive_cli as surface
from cortheon.cognitive_cli_core.diagnostics import runtime_identity


def host_conformance(
    runtime_url: str,
    *,
    token: str,
    hosts: list[str] | tuple[str, ...],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Exercise every selected host boundary without invoking a model."""

    from cortheon.cognitive_install import host_installation_status

    if not 1 <= timeout_seconds <= 120:
        raise ValueError("timeout_seconds must be between 1 and 120")
    selected = list(dict.fromkeys(host.casefold() for host in hosts))
    if not selected or "all" in selected:
        selected = list(surface.SUPPORTED_HOSTS)
    unknown = sorted(set(selected) - set(surface.SUPPORTED_HOSTS))
    if unknown:
        raise ValueError("unsupported host(s): " + ", ".join(unknown))

    capabilities = surface.protocol_capabilities()
    assurance = capabilities["evidence_assurance"]
    assert isinstance(assurance, dict)
    installation = host_installation_status()
    assets = surface._asset_paths()
    results: dict[str, dict[str, Any]] = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                **kwargs,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            **(
                {"error": (completed.stderr.strip() or completed.stdout.strip())[-500:]}
                if completed.returncode != 0
                else {}
            ),
            "_stdout": completed.stdout,
        }

    if "opencode" in selected:
        node = shutil.which("node")
        probe = (
            execute(
                [
                    node,
                    "--input-type=module",
                    "-e",
                    "import(process.argv[1])",
                    Path(assets["opencode_plugin"]).resolve().as_uri() + "?conformance=1",
                ]
            )
            if node
            else {"ok": False, "error": "node is not installed"}
        )
        probe.pop("_stdout", None)
        results["opencode"] = {
            "assurance": assurance["opencode"],
            "configured": installation["opencode"]["configured"],
            "adapter_load": probe,
        }

    if "pi" in selected:
        pi = shutil.which("pi")
        probe = (
            execute(
                [
                    pi,
                    "--no-extensions",
                    "--extension",
                    assets["pi_extension"],
                    "--no-session",
                    "--print",
                    "/cortheon status",
                ]
            )
            if pi
            else {"ok": False, "error": "pi is not installed"}
        )
        probe.pop("_stdout", None)
        results["pi"] = {
            "assurance": assurance["pi"],
            "configured": installation["pi"]["configured"],
            "adapter_load": probe,
        }

    if "codex" in selected:
        codex = shutil.which("codex")
        probe = (
            execute([codex, "plugin", "list"])
            if codex
            else {"ok": False, "error": "codex is not installed", "_stdout": ""}
        )
        listed = "cortheon@cortheon-local" in str(probe.pop("_stdout", ""))
        plugin = Path(assets["codex_plugin"])
        try:
            manifest = json.loads(
                (plugin / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
            )
            hooks = json.loads((plugin / "hooks" / "hooks.json").read_text(encoding="utf-8"))
            ast.parse((plugin / "hooks" / "cortheon_hook.py").read_text(encoding="utf-8"))
            plugin_valid = (
                manifest.get("name") == "cortheon"
                and isinstance(hooks.get("hooks"), dict)
                and bool(hooks["hooks"])
            )
        except (OSError, SyntaxError, json.JSONDecodeError):
            plugin_valid = False
        results["codex"] = {
            "assurance": assurance["codex"],
            "configured": installation["codex"]["configured"],
            "adapter_load": {
                **probe,
                "plugin_listed": listed,
                "plugin_assets_valid": plugin_valid,
                "ok": bool(probe.get("ok") and listed and plugin_valid),
            },
        }

    if "generic" in selected:
        mcp = shutil.which("cortheon-mcp")
        initialize = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            separators=(",", ":"),
        )
        probe = (
            execute([mcp], input=initialize + "\n")
            if mcp
            else {"ok": False, "error": "cortheon-mcp is not installed", "_stdout": ""}
        )
        try:
            response = json.loads(str(probe.pop("_stdout", "")).splitlines()[0])
            initialized = response.get("result", {}).get("serverInfo", {}).get("name") == "cortheon"
        except (IndexError, json.JSONDecodeError):
            initialized = False
        results["generic"] = {
            "assurance": assurance["stdio_mcp"],
            "configured": installation["generic"]["configured"],
            "adapter_load": {
                **probe,
                "mcp_initialized": initialized,
                "ok": bool(probe.get("ok") and initialized),
            },
        }

    if "omp" in selected:
        mcp = shutil.which("cortheon-mcp")
        initialize = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            separators=(",", ":"),
        )
        probe = (
            execute([mcp], input=initialize + "\n")
            if mcp
            else {"ok": False, "error": "cortheon-mcp is not installed", "_stdout": ""}
        )
        try:
            response = json.loads(str(probe.pop("_stdout", "")).splitlines()[0])
            initialized = response.get("result", {}).get("serverInfo", {}).get("name") == "cortheon"
        except (IndexError, json.JSONDecodeError):
            initialized = False
        results["omp"] = {
            "assurance": assurance["stdio_mcp"],
            "configured": installation["omp"]["configured"],
            "skill_present": installation["omp"]["skill_present"],
            "adapter_load": {
                **probe,
                "mcp_initialized": initialized,
                "ok": bool(probe.get("ok") and initialized),
            },
        }

    runtime_required = any(host not in {"generic", "omp"} for host in selected)
    runtime = (
        surface._runtime_health(runtime_url, token=token)
        if runtime_required
        else {"ok": True, "required": False, "reason": "stdio MCP owns its runtime process"}
    )
    _, identity_matches = runtime_identity(runtime) if runtime_required else (None, True)
    invariant_contract = {
        "storage": capabilities["storage"],
        "owns_project_tools": capabilities["owns_project_tools"],
        "owns_project_files": capabilities["owns_project_files"],
        "persists_task_state": capabilities["persists_task_state"],
    }
    for result in results.values():
        result["contract"] = invariant_contract
        result["ok"] = bool(
            result["configured"]
            and result["adapter_load"]["ok"]
            and invariant_contract
            == {
                "storage": "memory_only",
                "owns_project_tools": False,
                "owns_project_files": False,
                "persists_task_state": False,
            }
        )
    return {
        "schema_version": 1,
        "ok": identity_matches and all(item["ok"] for item in results.values()),
        "runtime": runtime,
        "runtime_required": runtime_required,
        "runtime_identity_matches": identity_matches,
        "selected_hosts": selected,
        "hosts": results,
        "cross_host_contract_consistent": all(
            item["contract"] == invariant_contract for item in results.values()
        ),
    }
