"""Live Pi fixtures for host-owned web evidence tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pi_recovery_helpers import Servers, parse_events, run_pi

ROOT = Path(__file__).parents[1]
EXTENSION = ROOT / "src" / "cortheon" / "pi_extension.ts"
PROMPT = "Research the current release evidence and report what the sources say."
CERTIFIED = "CORTHEON CERTIFIED: current sources were checked."


def web_request(capability: str = "search") -> dict[str, Any]:
    return {
        "request_id": "web-req-1",
        "capability": capability,
        "query": "current release evidence",
        "parameters": {"purpose": "check current release evidence"},
    }


def write_web_extension(
    tmp_path: Path,
    *,
    tool: str,
    content: str,
    details: dict[str, Any] | None,
    throws: bool = False,
) -> Path:
    target = tmp_path / f"cortheon-{tool}.ts"
    execute = (
        'throw new Error("simulated web transport failure");'
        if throws
        else "return "
        + json.dumps(
            {
                "content": [{"type": "text", "text": content}],
                "details": details,
            }
        )
        + ";"
    )
    target.write_text(
        f"""
import {{ Type }} from "@earendil-works/pi-ai";
import cortheon from {json.dumps(EXTENSION.as_uri())};

export default function (pi) {{
  cortheon(pi);
  pi.registerTool({{
    name: {json.dumps(tool)},
    label: {json.dumps(tool)},
    description: "Test-only host web tool",
    parameters: Type.Object({{
      query: Type.Optional(Type.String()),
      url: Type.Optional(Type.String()),
    }}),
    async execute() {{ {execute} }},
  }});
}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return target


def runtime_script(
    request: dict[str, Any],
    *,
    observe_result: str = "finish",
):
    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return 200, {
                "session_id": "web-session-1",
                "status": "observing",
                "session": {"deliverable": "answer"},
                "next_action": {
                    "type": "harness_tool",
                    "instruction": "Use the host web tool once.",
                    "request": request,
                },
            }
        if path == "/v1/observe" and observe_result == "reset":
            return "connection-reset"
        if path == "/v1/observe" and observe_result == "repeat":
            return 200, {
                "session_id": "web-session-1",
                "status": "observing",
                "next_action": {
                    "type": "harness_tool",
                    "instruction": "Try the same unavailable web tool again.",
                    "request": request,
                },
            }
        if path == "/v1/observe":
            observations = body.get("observations", [])
            clean = [item for item in observations if item.get("status") == "observed"]
            evidence = [
                {
                    "evidence_id": f"web-ev-{index}",
                    "source": item.get("url", "host"),
                    "content": item.get("content", ""),
                    **(
                        {"quarantine_flags": ["instruction_shaped_content"]}
                        if "Ignore previous instructions" in item.get("content", "")
                        else {}
                    ),
                }
                for index, item in enumerate(clean, start=1)
            ]
            return 200, {
                "session_id": "web-session-1",
                "status": "observing",
                "accepted_evidence_ids": [item["evidence_id"] for item in evidence],
                "context": {"evidence": evidence},
                "next_action": {"type": "finish"},
            }
        if path == "/v1/complete":
            return 200, {
                "session_id": "web-session-1",
                "status": "complete",
                "answer": CERTIFIED,
            }
        return 200, {"status": "ok"}

    return script


def run_web_case(
    tmp_path: Path,
    *,
    extension: Path,
    request: dict[str, Any],
    tool_call: tuple[str, dict[str, Any]],
    observe_result: str = "finish",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"tool_calls": [tool_call]}, {"text": "Host research complete."}],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": runtime_script(request, observe_result=observe_result),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            extension,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace,
            tmp_path=tmp_path,
            timeout=45,
        )
    return completed, model_state, runtime_state


def observe_bodies(runtime_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [body for path, body in runtime_state["records"] if path == "/v1/observe"]


def tool_end_events(completed: Any, tool: str) -> list[dict[str, Any]]:
    return [
        event
        for event in parse_events(completed.stdout)
        if event.get("type") == "tool_execution_end" and event.get("toolName") == tool
    ]
