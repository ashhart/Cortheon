"""Build helpers and the MCP surface probe shared by the packaging suite.

The MCP runtime is the product's primary shipped surface, so a built artifact
must not merely import: the compact and advanced tool catalogues, JSON-RPC
framing, error mapping, host-receipt certification, the argument repair path,
and the whole start/observe/complete lifecycle must behave exactly as they do
in source mode. The probe prints only derived values, never a session or
request identifier, so its output is deterministic across processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

MCP_SURFACE_PROBE = r"""
import inspect, io, json
import cortheon.cognitive_mcp as mcp

def call(server, cid, name, arguments):
    return server.handle({"jsonrpc": "2.0", "id": cid, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})

report = {}
report["surface"] = sorted(n for n in vars(mcp) if not n.startswith("__"))
report["has_all"] = hasattr(mcp, "__all__")
report["module_doc"] = mcp.__doc__
report["constants"] = {
    n: sorted(v) if isinstance(v, set) else v
    for n, v in sorted(vars(mcp).items())
    if n.isupper()
}
report["tools_compact"] = mcp.tool_definitions()
report["tools_advanced"] = mcp.tool_definitions(advanced=True)
report["facade_type"] = type(mcp).__name__

# Signatures and docstrings of everything the facade re-exports from the
# product, plus the server's own methods: an installed artifact must answer
# help(), inspect.signature(), and __doc__ exactly as source mode does.
def described(value):
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None

report["signatures"] = {}
report["docs"] = {"<module>": mcp.__doc__}
for name, value in sorted(vars(mcp).items()):
    if name.startswith("_") or not getattr(value, "__module__", "").startswith("cortheon"):
        continue
    report["signatures"][name] = described(value)
    report["docs"][name] = value.__doc__
    if isinstance(value, type):
        for attribute, member in sorted(vars(value).items()):
            if attribute.startswith("_") and attribute != "__init__":
                continue
            target = getattr(member, "__func__", member)
            if not callable(target):
                continue
            report["signatures"][f"{name}.{attribute}"] = described(target)
            report["docs"][f"{name}.{attribute}"] = target.__doc__

server = mcp.CortheonMcpServer(mcp.CognitiveRuntime())
init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18"}})
report["initialize"] = init
report["ping"] = server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}})
report["notification_is_silent"] = server.handle(
    {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
report["unknown_method"] = server.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})
report["tools_list"] = server.handle(
    {"jsonrpc": "2.0", "id": 30, "method": "tools/list", "params": {}})
report["tools_list_advanced"] = mcp.CortheonMcpServer(
    mcp.CognitiveRuntime(), advanced=True).handle(
    {"jsonrpc": "2.0", "id": 31, "method": "tools/list", "params": {}})
report["bad_params"] = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": []})
report["unknown_tool"] = call(server, 5, "cortheon_nope", {})
report["missing_session"] = call(server, 6, "cortheon_finish", {"session_id": "missing"})

started = call(server, 7, "cortheon_start", {"goal": "Does src/example.py import pathlib?"})
content = started["result"]["structuredContent"]
sid = content["session"]["session_id"]
rid = content["next_action"]["request"]["request_id"]
report["start_storage"] = content["session"]["storage"]
report["start_next_action_type"] = content["next_action"]["type"]

observed = call(server, 8, "cortheon_observe", {
    "session_id": sid, "request_id": rid,
    "observations": [{"kind": "code", "content": "No matches found.",
                      "host_receipt": {"tool": "grep", "outcome": "no_match",
                                       "args": {"pattern": "pathlib", "path": "src/example.py"}}}],
})["result"]["structuredContent"]
report["accepted_evidence_ids"] = observed["accepted_evidence_ids"]
report["observe_submit_via"] = observed["next_action"]["submit_via"]

# JSON-string coercion repair path
report["coerce"] = mcp._coerce_json_array('["ev1"]', "x")
completed = call(server, 9, "cortheon_complete", {
    "session_id": sid, "answer": "No.",
    "claims": json.dumps([{"claim": "No pathlib import.", "evidence_ids": ["ev1"]}]),
    "hypotheses": [{"statement": "No pathlib import.", "falsification_test": "Search it.",
                    "status": "supported", "evidence_ids": ["ev1"]}],
    "completion_evidence_ids": ["ev1"],
})["result"]["structuredContent"]
report["complete_status"] = completed["status"]
report["sessions_discarded"] = server.runtime.active_sessions

# malformed observe -> healing example + waiver
s2 = call(server, 10, "cortheon_start", {"goal": "Another bounded question?"})["result"]["structuredContent"]
bad = call(server, 11, "cortheon_observe",
           {"session_id": s2["session"]["session_id"],
            "request_id": s2["next_action"]["request"]["request_id"], "observations": []})
report["malformed_code"] = bad["error"]["code"]
report["malformed_has_example"] = "Correct example call" in bad["error"]["message"]

# stdio framing, including the oversize guard
out = io.StringIO()
mcp.serve(mcp.CortheonMcpServer(mcp.CognitiveRuntime()),
          io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 12, "method": "initialize", "params": {}})
                      + "\n" + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
                      + "\n" + "{not json}\n"), out)
report["stdio"] = [json.loads(line) for line in out.getvalue().splitlines()]

# Monkeypatching the facade must still reach the code that resolves the name,
# and assigning the original back must restore it - the seam the pre-split
# module had. Run last so nothing above sees a patched binding.
def list_tools(cid):
    return mcp.CortheonMcpServer(mcp.CognitiveRuntime()).handle(
        {"jsonrpc": "2.0", "id": cid, "method": "tools/list", "params": {}})

original = mcp.tool_definitions
sentinel = [{"name": "sentinel"}]
mcp.tool_definitions = lambda **_: sentinel
report["patch_reaches_handle"] = list_tools(40)["result"]["tools"] == sentinel
mcp.tool_definitions = original
report["patch_undo_restores_binding"] = mcp.tool_definitions is original
report["patch_undo_restores_behavior"] = (
    list_tools(41)["result"]["tools"] == report["tools_compact"])

original_error = mcp._error
mcp._error = lambda request_id, code, message: {"patched": [request_id, code]}
report["error_patch_reaches_handle"] = server.handle(
    {"jsonrpc": "2.0", "id": 42, "method": "nope"})
mcp._error = original_error
report["error_undo_restores"] = (
    mcp._error is original_error
    and server.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"}) == report["unknown_method"])

print(json.dumps(report, sort_keys=True, indent=1, default=str))
"""


def build_wheel(target: Path, dist_dir: Path) -> Path:
    """Build a wheel from a repository checkout or an sdist archive."""

    dist_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist_dir),
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return next(dist_dir.glob("cortheon-*.whl"))


def build_sdist(dist_dir: Path) -> Path:
    """Invoke the in-tree PEP 517 backend's build_sdist hook."""

    dist_dir.mkdir(parents=True, exist_ok=True)
    hook = subprocess.run(
        [
            sys.executable,
            "-c",
            "import build_backend, sys;print(build_backend.build_sdist(sys.argv[1]))",
            str(dist_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return dist_dir / hook.stdout.strip().splitlines()[-1]


def install(wheel: Path, target: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return target


def mcp_surface(python_path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", MCP_SURFACE_PROBE],
        env={**os.environ, "PYTHONPATH": str(python_path)},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return completed.stdout
