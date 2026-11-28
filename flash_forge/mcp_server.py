"""Native Zero-Dependency Stdio MCP Server (`cortheon mcp`).

Implements JSON-RPC 2.0 Model Context Protocol over stdio with automatic
dual-framing support:
  - Newline-delimited JSON-RPC (NDJSON — MCP 2024-11-05+ stdio transport)
  - ``Content-Length`` header framing (LSP-style / legacy MCP clients)

Works out-of-the-box with Claude Code, Cursor, OMP, Pi, Codex, and MCP Inspector.

Exposed tools:
  - ``cortheon_verify``       — Multi-stream consensus & sandboxed verification
  - ``cortheon_guard``        — Zero-FP proof-carrying adversarial diff breaker
  - ``cortheon_doctor``       — Local runtime diagnostics (sandbox, solvers, backend)
  - ``cortheon_spec_verify``  — Two-tier speculative drafter -> verifier pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import io
import json
import os
import select
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, BinaryIO, Callable, Optional

from .client import CallSpec, TensorFoldClient
from .doctor import DoctorReport, format_doctor_report, run_doctor
from .engine import FlashForge, ForgeConfig
from .guard import GuardReport, format_guard_report, run_guard
from .memory import EpisodicMemory
from .sandbox import kill_active_sandbox_children, sandbox_scope
from .skill_loader import SkillPack

DEFAULT_UPSTREAM_URL = "http://127.0.0.1:8080/v1"
SERVER_NAME = "cortheon"
SERVER_VERSION = "2.5.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

_VALID_VERIFY_MODES = frozenset({"auto", "code", "reason", "think", "fast"})

# JSON-RPC 2.0 standard error codes
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "cortheon_verify",
        "description": (
            "Solve and mechanically verify a hard coding, math, logic, debugging, or architecture "
            "prompt using Cortheon's parallel multi-stream consensus and OS-sandboxed verification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The problem statement, code task, or question to verify.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "code", "reason", "think", "fast"],
                    "description": "Optional routing override (defaults to automatic classification).",
                },
                "streams": {
                    "type": "integer",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 16,
                    "description": "Number of parallel verification streams (2=fast, 4=balanced, 8=max diversity).",
                },
                "url": {
                    "type": "string",
                    "description": "Optional OpenAI-compatible upstream model URL override.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cortheon_guard",
        "description": (
            "Zero-false-positive proof-carrying diff breaker. Hunts bugs in a git diff inside an "
            "isolated ephemeral worktree + OS sandbox, reporting ONLY bugs proven by failing executable "
            "Python reproducers (and optionally generating a verified surgical fix)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the git repository to inspect.",
                },
                "base_ref": {
                    "type": "string",
                    "default": "HEAD",
                    "description": "Git reference to diff against (default: HEAD).",
                },
                "num_adversaries": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Number of parallel adversarial bug hunters to run (default: 3).",
                },
                "auto_fix": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, attempt to synthesize and verify a unified patch fixing all proven bugs.",
                },
                "url": {
                    "type": "string",
                    "description": "Optional OpenAI-compatible upstream model URL override.",
                },
            },
            "required": ["repo_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cortheon_doctor",
        "description": (
            "Run Cortheon local runtime diagnostics: OS sandbox containment, local solver stack "
            "(SymPy/Z3/NumPy), coding harness integrations, backend capabilities, and optional live probe."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Optional OpenAI-compatible upstream model URL to probe.",
                },
                "skip_live": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip the live completion & throughput probe (default: false).",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "cortheon_spec_verify",
        "description": (
            "Speculative multi-model verification: fan out parallel candidate drafts on a fast "
            "drafter model and critically verify/select the final answer on a stronger verifier model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to draft and verify.",
                },
                "drafter_url": {
                    "type": "string",
                    "description": "OpenAI-compatible URL of the fast drafter model.",
                },
                "verifier_url": {
                    "type": "string",
                    "description": "OpenAI-compatible URL of the stronger verifier model.",
                },
                "drafter_streams": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 16,
                    "description": "Number of parallel drafts to generate on the drafter (default: 3).",
                },
            },
            "required": ["prompt", "drafter_url", "verifier_url"],
            "additionalProperties": False,
        },
    },
]

_TOOL_NAMES = frozenset(t["name"] for t in TOOL_DEFINITIONS)


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _jsonrpc_progress_notification(
    progress_token: str | int,
    progress: int | float,
    message: Optional[str] = None,
    total: Optional[int | float] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "progressToken": progress_token,
        "progress": progress,
    }
    if total is not None:
        params["total"] = total
    if message:
        params["message"] = message
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": params,
    }


def _extract_progress_token(params: dict[str, Any]) -> Optional[str | int]:
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    token = meta.get("progressToken")
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token
    if isinstance(token, str) and token:
        return token
    return None


def _req_key(req_id: Any) -> tuple[type, Any]:
    return (type(req_id), req_id)


def _accepts_kwarg(fn: Callable[..., Any], kwarg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == kwarg and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _format_progress_message(event: dict[str, Any]) -> str:
    ev = str(event.get("event") or "progress")
    engine = str(event.get("engine") or "cortheon")
    if ev == "start":
        streams = event.get("streams")
        mode = event.get("mode") or engine
        if streams is not None:
            return f"Starting {mode} ({streams} parallel streams)"
        return f"Starting {mode}"
    if ev == "slot_done":
        slot = event.get("slot") or event.get("name") or "stream"
        status = event.get("status") or ("ok" if event.get("ok") else "done")
        lat = event.get("latency_s")
        lat_str = f" in {float(lat):.1f}s" if isinstance(lat, (int, float)) else ""
        return f"Slot {slot} finished ({status}{lat_str})"
    if ev == "early_abort":
        cnt = event.get("consensus_count") or event.get("passing")
        cancelled = event.get("cancelled")
        parts = [f"Early-abort consensus reached"]
        if cnt is not None:
            parts.append(f"({cnt} agreeing)")
        if cancelled:
            parts.append(f"— cancelled {cancelled} lagging slot(s)")
        return " ".join(parts)
    if ev == "wave2_start":
        reason = str(event.get("reason") or "disagreement").strip()
        return f"Escalating to Wave 2 verification ({reason[:120]})"
    if ev == "drafter_accepted":
        cnt = event.get("consensus_count")
        suffix = f" ({cnt} scripts, 0 verifier calls)" if cnt is not None else " (0 verifier calls)"
        return f"Drafter consensus accepted{suffix}"
    if ev == "escalated":
        reason = str(event.get("reason") or "no drafter consensus").strip()
        return f"Drafter escalated to big verifier ({reason[:120]})"
    if ev == "bug_proven":
        title = str(event.get("title") or "regression").strip()
        file_ref = event.get("file")
        where = f" in {file_ref}" if file_ref else ""
        return f"Proven bug{where}: {title[:120]}"
    if ev == "done":
        wall = event.get("wall_s")
        wall_str = f" in {float(wall):.1f}s" if isinstance(wall, (int, float)) else ""
        return f"{engine} complete{wall_str}"
    return f"{engine}: {ev}"


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _serialize_any(obj: Any) -> Any:
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


async def _default_spec_verify(
    prompt: str,
    drafter_url: str,
    verifier_url: str,
    drafter_streams: int = 3,
    api_key: Optional[str] = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Delegate to ``flash_forge.spec_verify`` if available, else run two-tier drafter->verifier."""
    try:
        mod = importlib.import_module("flash_forge.spec_verify")
    except ModuleNotFoundError:
        mod = None

    if mod is not None and callable(getattr(mod, "solve_spec_verify", None)):
        solve_fn = getattr(mod, "solve_spec_verify")
        drafter = TensorFoldClient(
            base_url=drafter_url,
            api_key=api_key,
            max_parallel=max(4, drafter_streams),
        )
        verifier = TensorFoldClient(base_url=verifier_url, api_key=api_key)
        skills = SkillPack()
        memory = EpisodicMemory()
        try:
            solve_kwargs: dict[str, Any] = {
                "drafter_client": drafter,
                "verifier_client": verifier,
                "skills": skills,
                "memory": memory,
                "problem": prompt,
                "drafter_streams": drafter_streams,
            }
            if on_event is not None and _accepts_kwarg(solve_fn, "on_event"):
                solve_kwargs["on_event"] = on_event
            trace = await solve_fn(**solve_kwargs)
            verified = bool(
                trace.accepted_by_drafter
                or (trace.verifier_trace is not None and getattr(trace.verifier_trace, "verified", False))
            )
            return {
                "answer": trace.answer,
                "verified": verified,
                "accepted_by_drafter": trace.accepted_by_drafter,
                "escalated": trace.escalated,
                "escalation_reason": trace.escalation_reason,
                "drafter_model": trace.drafter_model,
                "verifier_model": trace.verifier_model,
                "drafter_consensus_count": trace.drafter_consensus_count,
                "drafter_verified_scripts": trace.drafter_verified_scripts,
                "drafter_wall_s": round(trace.drafter_wall_s, 3),
                "verifier_wall_s": round(trace.verifier_wall_s, 3),
                "total_wall_s": round(trace.total_wall_s, 3),
                "drafter_calls": trace.drafter_calls,
                "drafter_tokens": trace.drafter_tokens,
                "verifier_calls": trace.verifier_calls,
                "verifier_tokens": trace.verifier_tokens,
                "drafter_graveyard": list(trace.drafter_graveyard),
                "drafter_url": drafter_url.rstrip("/"),
                "verifier_url": verifier_url.rstrip("/"),
                "drafter_streams": drafter_streams,
            }
        finally:
            await drafter.close()
            await verifier.close()
            memory.close()

    if mod is not None:
        fn = getattr(mod, "run_spec_verify", None) or getattr(mod, "spec_verify", None)
        if callable(fn):
            sig = inspect.signature(fn)
            params = sig.parameters
            kwargs: dict[str, Any] = {}
            if "prompt" in params:
                kwargs["prompt"] = prompt
            if "drafter_url" in params:
                kwargs["drafter_url"] = drafter_url
            if "verifier_url" in params:
                kwargs["verifier_url"] = verifier_url
            if "drafter_streams" in params:
                kwargs["drafter_streams"] = drafter_streams
            elif "streams" in params:
                kwargs["streams"] = drafter_streams

            if "drafter_client" in params and "verifier_client" in params:
                drafter = TensorFoldClient(base_url=drafter_url, api_key=api_key, max_parallel=max(4, drafter_streams))
                verifier = TensorFoldClient(base_url=verifier_url, api_key=api_key)
                try:
                    kwargs["drafter_client"] = drafter
                    kwargs["verifier_client"] = verifier
                    out = fn(**kwargs)
                    if inspect.isawaitable(out):
                        out = await out
                    return _serialize_any(out) if isinstance(_serialize_any(out), dict) else {"result": _serialize_any(out)}
                finally:
                    await drafter.close()
                    await verifier.close()
            else:
                out = fn(**kwargs)
                if inspect.isawaitable(out):
                    out = await out
                ser = _serialize_any(out)
                return ser if isinstance(ser, dict) else {"result": ser}

    # Built-in two-tier speculative verification fallback
    t0 = time.perf_counter()
    drafter = TensorFoldClient(base_url=drafter_url, api_key=api_key, max_parallel=max(4, drafter_streams))
    verifier = TensorFoldClient(base_url=verifier_url, api_key=api_key)
    try:
        draft_specs = [
            CallSpec(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fast speculative drafter. Produce a complete, rigorous, "
                            "self-contained solution with clear final answer or code."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tag=f"spec_draft_{i + 1}",
                max_tokens=3072,
                temperature=round(0.25 + 0.15 * i, 2),
                seed=100 + i * 37,
                enable_thinking=True,
                thinking_budget=1200,
                loop_guard=True,
            )
            for i in range(drafter_streams)
        ]
        draft_results = await drafter.batch(draft_specs)
        valid_drafts = [r.content.strip() for r in draft_results if r.ok and r.content.strip()]

        draft_section = "\n\n".join(
            f"--- DRAFT {idx + 1} ---\n{txt[:4000]}"
            for idx, txt in enumerate(valid_drafts)
        )
        verify_prompt = (
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"CANDIDATE DRAFTS FROM FAST DRAFTER ({len(valid_drafts)}/{drafter_streams}):\n"
            f"{draft_section or '(no drafts succeeded; solve from scratch)'}\n\n"
            "Critically verify every candidate draft against all requirements and edge cases. "
            "Reject flawed reasoning, correct any errors, and output ONLY the final verified answer."
        )
        ver_res = await verifier.complete(
            CallSpec(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict principal verifier. Audit candidate drafts carefully and "
                            "produce the single verified, accurate final response."
                        ),
                    },
                    {"role": "user", "content": verify_prompt},
                ],
                tag="spec_verifier",
                max_tokens=4096,
                temperature=0.1,
                enable_thinking=True,
                thinking_budget=2048,
                loop_guard=True,
            )
        )
        if not ver_res.ok:
            raise RuntimeError(f"Verifier backend error: {ver_res.raw_error}")

        return {
            "answer": ver_res.content,
            "verified": True,
            "drafter_url": drafter_url.rstrip("/"),
            "verifier_url": verifier_url.rstrip("/"),
            "drafter_streams": drafter_streams,
            "drafts_generated": len(valid_drafts),
            "wall_s": round(time.perf_counter() - t0, 3),
            "drafter_stats": drafter.stats_dict(),
            "verifier_stats": verifier.stats_dict(),
        }
    finally:
        await drafter.close()
        await verifier.close()


class MCPServer:
    """Zero-dependency Cortheon JSON-RPC 2.0 MCP Server."""

    def __init__(
        self,
        default_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        verify_fn: Optional[Callable[..., Awaitable[dict[str, Any]]]] = None,
        guard_fn: Optional[Callable[..., Awaitable[GuardReport]]] = None,
        doctor_fn: Optional[Callable[..., Awaitable[DoctorReport]]] = None,
        spec_verify_fn: Optional[Callable[..., Awaitable[dict[str, Any]]]] = None,
        notify_fn: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        self._explicit_url = default_url.rstrip("/") if default_url else None
        self.api_key = api_key
        self.model = model
        self._verify_fn = verify_fn
        self._guard_fn = guard_fn
        self._doctor_fn = doctor_fn
        self._spec_verify_fn = spec_verify_fn
        self._notify_fn = notify_fn
        self.initialized = False
        self._in_flight: dict[tuple[type, Any], tuple[asyncio.Task[Any], str]] = {}
        self._cancelled_keys: set[tuple[type, Any]] = set()

    def register_in_flight(self, req_id: Any, task: asyncio.Task[Any], scope_id: str) -> None:
        self._in_flight[_req_key(req_id)] = (task, scope_id)

    def unregister_in_flight(self, req_id: Any) -> None:
        key = _req_key(req_id)
        self._in_flight.pop(key, None)
        self._cancelled_keys.discard(key)

    def is_cancelled(self, req_id: Any) -> bool:
        return _req_key(req_id) in self._cancelled_keys

    def cancel_request(self, request_id: Any, reason: Optional[str] = None) -> bool:
        """Cancel an in-flight tool call by JSON-RPC requestId and kill its sandbox children."""
        del reason  # reserved for optional diagnostics
        if request_id is None:
            return False
        key = _req_key(request_id)
        entry = self._in_flight.get(key)
        if entry is None:
            return False
        self._cancelled_keys.add(key)
        task, scope_id = entry
        kill_active_sandbox_children(scope=scope_id)
        if not task.done():
            task.cancel()
        return True

    def cancel_all_in_flight(self) -> int:
        """Cancel all in-flight tool tasks and kill their scoped sandboxed children."""
        entries = list(self._in_flight.items())
        cancelled = 0
        for key, (task, scope_id) in entries:
            self._cancelled_keys.add(key)
            kill_active_sandbox_children(scope=scope_id)
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def _make_progress_callback(
        self,
        progress_token: Optional[str | int],
        notify: Optional[Callable[[dict[str, Any]], Any]],
    ) -> tuple[
        Optional[Callable[[dict[str, Any]], None]],
        Callable[[str], None],
        Callable[[], Awaitable[None]],
        Callable[[], None],
    ]:
        """Build a thread-safe ``on_event`` callback that emits MCP ``notifications/progress``.

        Returns ``(on_event, emit_message, flush_pending, mute)``.
        """
        async def _noop_flush() -> None:
            return None

        if progress_token is None or notify is None:
            return None, (lambda _msg: None), _noop_flush, (lambda: None)

        loop = asyncio.get_running_loop()
        step_counter = [0]
        muted = [False]
        bg_tasks: set[asyncio.Task[Any]] = set()

        def _dispatch_payload(payload: dict[str, Any]) -> None:
            if muted[0]:
                return
            try:
                res = notify(payload)
                if inspect.isawaitable(res):
                    t = loop.create_task(res)
                    bg_tasks.add(t)
                    t.add_done_callback(bg_tasks.discard)
            except Exception:
                pass

        def _emit_message(message: str) -> None:
            if muted[0]:
                return
            step = step_counter[0]
            step_counter[0] += 1
            payload = _jsonrpc_progress_notification(progress_token, step, message=message)
            try:
                if asyncio.get_running_loop() is loop:
                    _dispatch_payload(payload)
                    return
            except RuntimeError:
                pass
            try:
                loop.call_soon_threadsafe(_dispatch_payload, payload)
            except RuntimeError:
                pass

        def _on_event(event: dict[str, Any]) -> None:
            if not isinstance(event, dict):
                return
            _emit_message(_format_progress_message(event))

        async def _flush() -> None:
            while bg_tasks:
                batch = list(bg_tasks)
                await asyncio.gather(*batch, return_exceptions=True)

        def _mute() -> None:
            muted[0] = True

        return _on_event, _emit_message, _flush, _mute

    @property
    def default_url(self) -> str:
        return self._resolve_url(None)

    def _resolve_url(self, override: Optional[str] = None) -> str:
        raw = override or self._explicit_url or os.environ.get("CORTHEON_URL") or DEFAULT_UPSTREAM_URL
        return str(raw).strip().rstrip("/")

    async def handle_request(
        self,
        msg: dict[str, Any],
        *,
        notify: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Process a single decoded JSON-RPC 2.0 message and return a response dict (or None for notifications)."""
        if not isinstance(msg, dict):
            return _jsonrpc_error(None, ERR_INVALID_REQUEST, "Invalid Request: expected JSON object")

        has_id = "id" in msg
        req_id = msg.get("id")
        method = msg.get("method")

        if not isinstance(method, str) or not method:
            if not has_id:
                return None
            return _jsonrpc_error(req_id, ERR_INVALID_REQUEST, "Invalid Request: missing method string")

        params = msg.get("params")
        if params is None:
            params = {}

        # Notifications (no id or notifications/*) produce no response per JSON-RPC 2.0 / MCP spec
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method == "notifications/cancelled":
            if isinstance(params, dict):
                self.cancel_request(params.get("requestId"), reason=params.get("reason"))
            return None
        if method.startswith("notifications/") and not has_id:
            return None

        if method == "initialize":
            if not isinstance(params, dict):
                return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: expected object")
            client_proto = params.get("protocolVersion")
            negotiated = (
                client_proto
                if isinstance(client_proto, str) and client_proto in SUPPORTED_PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION
            )
            self.initialized = True
            return _jsonrpc_result(
                req_id,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            )

        if method == "ping":
            return _jsonrpc_result(req_id, {}) if has_id else None

        if method == "tools/list":
            if not isinstance(params, dict):
                return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: expected object")
            return _jsonrpc_result(req_id, {"tools": TOOL_DEFINITIONS}) if has_id else None

        if method == "tools/call":
            if not isinstance(params, dict):
                return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: expected object")
            return await self._handle_tools_call(req_id, params, has_id=has_id, notify=notify)

        if not has_id:
            return None
        return _jsonrpc_error(req_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _handle_tools_call(
        self,
        req_id: Any,
        params: dict[str, Any],
        *,
        has_id: bool,
        notify: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> Optional[dict[str, Any]]:
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: 'name' must be a non-empty string")
        if tool_name not in _TOOL_NAMES:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, f"Unknown tool: {tool_name}")

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: 'arguments' must be an object")

        # Validate arguments strictly before executing
        try:
            validated = self._validate_tool_args(tool_name, arguments)
        except ValueError as exc:
            return _jsonrpc_error(req_id, ERR_INVALID_PARAMS, str(exc))

        eff_notify = notify if notify is not None else self._notify_fn
        progress_token = _extract_progress_token(params)
        on_event, emit_msg, flush_progress, mute_progress = self._make_progress_callback(
            progress_token, eff_notify
        )
        if progress_token is not None and eff_notify is not None:
            emit_msg(f"Starting {tool_name}")

        try:
            payload = await self._dispatch_tool(tool_name, validated, on_event=on_event)
            await flush_progress()
            if not has_id or (has_id and self.is_cancelled(req_id)):
                return None
            return _jsonrpc_result(
                req_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, indent=2),
                        }
                    ],
                    "isError": False,
                },
            )
        except asyncio.CancelledError:
            mute_progress()
            raise
        except Exception as exc:
            await flush_progress()
            if not has_id or (has_id and self.is_cancelled(req_id)):
                return None
            return _jsonrpc_result(
                req_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool '{tool_name}' failed: {type(exc).__name__}: {exc}",
                        }
                    ],
                    "isError": True,
                },
            )

    def _validate_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "cortheon_verify":
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("cortheon_verify requires a non-empty string 'prompt'")
            mode = args.get("mode")
            if mode is not None and (not isinstance(mode, str) or mode not in _VALID_VERIFY_MODES):
                raise ValueError(
                    f"cortheon_verify 'mode' must be one of {sorted(_VALID_VERIFY_MODES)}"
                )
            streams = args.get("streams", 4)
            if not _is_strict_int(streams) or not (1 <= streams <= 16):
                raise ValueError("cortheon_verify 'streams' must be an integer between 1 and 16")
            url = args.get("url")
            if url is not None and (not isinstance(url, str) or not url.strip()):
                raise ValueError("cortheon_verify 'url' must be a non-empty string when provided")
            return {
                "prompt": prompt,
                "mode": mode,
                "streams": streams,
                "url": self._resolve_url(url),
            }

        if tool_name == "cortheon_guard":
            repo_path = args.get("repo_path")
            if not isinstance(repo_path, str) or not repo_path.strip():
                raise ValueError("cortheon_guard requires a non-empty string 'repo_path'")
            base_ref = args.get("base_ref", "HEAD")
            if not isinstance(base_ref, str) or not base_ref.strip():
                raise ValueError("cortheon_guard 'base_ref' must be a non-empty string")
            num_adversaries = args.get("num_adversaries", 3)
            if not _is_strict_int(num_adversaries) or not (1 <= num_adversaries <= 8):
                raise ValueError("cortheon_guard 'num_adversaries' must be an integer between 1 and 8")
            auto_fix = args.get("auto_fix", False)
            if not isinstance(auto_fix, bool):
                raise ValueError("cortheon_guard 'auto_fix' must be a boolean")
            url = args.get("url")
            if url is not None and (not isinstance(url, str) or not url.strip()):
                raise ValueError("cortheon_guard 'url' must be a non-empty string when provided")
            return {
                "repo_path": repo_path,
                "base_ref": base_ref,
                "num_adversaries": num_adversaries,
                "auto_fix": auto_fix,
                "url": self._resolve_url(url),
            }

        if tool_name == "cortheon_doctor":
            url = args.get("url")
            if url is not None and (not isinstance(url, str) or not url.strip()):
                raise ValueError("cortheon_doctor 'url' must be a non-empty string when provided")
            skip_live = args.get("skip_live", False)
            if not isinstance(skip_live, bool):
                raise ValueError("cortheon_doctor 'skip_live' must be a boolean")
            return {
                "url": self._resolve_url(url),
                "skip_live": skip_live,
            }

        if tool_name == "cortheon_spec_verify":
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("cortheon_spec_verify requires a non-empty string 'prompt'")
            drafter_url = args.get("drafter_url")
            if not isinstance(drafter_url, str) or not drafter_url.strip():
                raise ValueError("cortheon_spec_verify requires a non-empty string 'drafter_url'")
            verifier_url = args.get("verifier_url")
            if not isinstance(verifier_url, str) or not verifier_url.strip():
                raise ValueError("cortheon_spec_verify requires a non-empty string 'verifier_url'")
            drafter_streams = args.get("drafter_streams", 3)
            if not _is_strict_int(drafter_streams) or not (1 <= drafter_streams <= 16):
                raise ValueError("cortheon_spec_verify 'drafter_streams' must be an integer between 1 and 16")
            return {
                "prompt": prompt,
                "drafter_url": drafter_url.strip().rstrip("/"),
                "verifier_url": verifier_url.strip().rstrip("/"),
                "drafter_streams": drafter_streams,
            }

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        if tool_name == "cortheon_verify":
            if self._verify_fn is not None:
                call_args = dict(args)
                if on_event is not None and _accepts_kwarg(self._verify_fn, "on_event"):
                    call_args["on_event"] = on_event
                return await self._verify_fn(**call_args)
            cfg = ForgeConfig(
                base_url=args["url"],
                model=self.model,
                api_key=self.api_key,
                num_streams=args["streams"],
            )
            force_mode = None if args["mode"] in (None, "auto") else args["mode"]
            async with FlashForge(cfg) as ff:
                res = await ff.solve(args["prompt"], force_mode=force_mode, on_event=on_event)
            return {
                "mode": res.mode,
                "answer": res.answer,
                "code": res.code,
                "verified": res.verified,
                "wall_s": round(res.wall_s, 3),
                "token_stats": res.token_stats,
                "lesson_learned": res.lesson_learned,
            }

        if tool_name == "cortheon_guard":
            if self._guard_fn is not None:
                call_args = dict(args)
                if on_event is not None and _accepts_kwarg(self._guard_fn, "on_event"):
                    call_args["on_event"] = on_event
                report = await self._guard_fn(**call_args)
            else:
                client = TensorFoldClient(
                    base_url=args["url"],
                    model=self.model,
                    api_key=self.api_key,
                    max_parallel=max(4, args["num_adversaries"]),
                )
                try:
                    report = await run_guard(
                        client,
                        args["repo_path"],
                        base_ref=args["base_ref"],
                        num_adversaries=args["num_adversaries"],
                        auto_fix=args["auto_fix"],
                        on_event=on_event,
                    )
                finally:
                    await client.close()
            out = report.to_dict()
            out["formatted"] = format_guard_report(report, use_color=False)
            return out

        if tool_name == "cortheon_doctor":
            if self._doctor_fn is not None:
                call_args = dict(args)
                if on_event is not None and _accepts_kwarg(self._doctor_fn, "on_event"):
                    call_args["on_event"] = on_event
                report = await self._doctor_fn(**call_args)
            else:
                report = await run_doctor(
                    url=args["url"],
                    api_key=self.api_key,
                    model=self.model,
                    skip_live=args["skip_live"],
                )
            out = report.to_dict()
            out["formatted"] = format_doctor_report(report, use_color=False)
            return out

        if tool_name == "cortheon_spec_verify":
            if self._spec_verify_fn is not None:
                call_args = dict(args)
                if on_event is not None and _accepts_kwarg(self._spec_verify_fn, "on_event"):
                    call_args["on_event"] = on_event
                return await self._spec_verify_fn(**call_args)
            return await _default_spec_verify(
                prompt=args["prompt"],
                drafter_url=args["drafter_url"],
                verifier_url=args["verifier_url"],
                drafter_streams=args["drafter_streams"],
                api_key=self.api_key,
                on_event=on_event,
            )

        raise ValueError(f"Unknown tool: {tool_name}")


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def read_stdio_message(stream: BinaryIO) -> tuple[Optional[bytes], str]:
    """Read one framed JSON-RPC message from ``stream``.

    Automatically detects both:
      - Newline-delimited JSON-RPC (``"ndjson"``)
      - ``Content-Length: N\\r\\n\\r\\n<bytes>`` framing (``"content-length"``)

    Returns ``(None, "eof")`` on end-of-stream.
    """
    while True:
        line = stream.readline()
        if not line:
            return None, "eof"
        stripped = line.strip()
        if not stripped:
            continue

        # Newline-delimited JSON object or array
        if stripped.startswith((b"{", b"[")):
            return stripped, "ndjson"

        # Content-Length / HTTP-style headers
        if b":" in stripped:
            headers: dict[str, str] = {}
            cur_line = stripped
            while cur_line:
                if b":" in cur_line:
                    k_raw, v_raw = cur_line.split(b":", 1)
                    headers[k_raw.decode("ascii", "replace").strip().lower()] = (
                        v_raw.decode("ascii", "replace").strip()
                    )
                next_line = stream.readline()
                if not next_line:
                    break
                cur_line = next_line.strip()

            cl_raw = headers.get("content-length")
            if cl_raw is not None:
                try:
                    length = int(cl_raw)
                except ValueError:
                    return b"", "content-length"
                if length <= 0:
                    return b"", "content-length"
                body = _read_exact(stream, length)
                return body, "content-length"

        # Fallback: pass malformed non-header text through so caller emits JSON-RPC Parse Error
        return stripped, "ndjson"


def write_stdio_message(stream: BinaryIO, payload: dict[str, Any], framing: str = "ndjson") -> None:
    """Write a JSON-RPC message using either NDJSON or ``Content-Length`` framing."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if framing == "content-length":
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        stream.write(header)
        stream.write(body)
    else:
        stream.write(body)
        stream.write(b"\n")
    stream.flush()


def _is_output_pipe_closed(stream: BinaryIO) -> bool:
    """Detect whether the peer closed the read end of a pipe stdout without writing bytes."""
    if getattr(stream, "closed", False):
        return True
    try:
        fd = stream.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError, ValueError):
        return False
    if not hasattr(select, "poll"):
        return False
    try:
        poller = select.poll()
        poller.register(fd, select.POLLERR | select.POLLHUP)
        events = poller.poll(0)
        return any(ev & (select.POLLERR | select.POLLHUP | select.POLLNVAL) for _, ev in events)
    except OSError:
        return False


async def serve_stdio(
    server: MCPServer,
    *,
    stdin: Optional[BinaryIO] = None,
    stdout: Optional[BinaryIO] = None,
) -> None:
    """Async loop serving JSON-RPC 2.0 MCP messages over binary stdio streams.

    Runs ``tools/call`` requests as concurrent tracked tasks so the stdin loop stays
    responsive to ``notifications/cancelled`` and emits live ``notifications/progress``
    updates without interleaving frames on stdout.
    """
    in_stream = stdin if stdin is not None else sys.stdin.buffer
    out_stream = stdout if stdout is not None else sys.stdout.buffer

    write_lock = asyncio.Lock()
    active_tasks: set[asyncio.Task[None]] = set()
    pipe_broken = [False]

    async def _write_payload(payload: dict[str, Any], framing: str) -> None:
        if pipe_broken[0]:
            return
        async with write_lock:
            if pipe_broken[0]:
                return
            try:
                write_stdio_message(out_stream, payload, framing=framing)
            except (BrokenPipeError, OSError, ValueError):
                pipe_broken[0] = True
                server.cancel_all_in_flight()

    async def _write_batch(responses: list[dict[str, Any]], framing: str) -> None:
        if pipe_broken[0] or not responses:
            return
        body = json.dumps(responses, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        async with write_lock:
            if pipe_broken[0]:
                return
            try:
                if framing == "content-length":
                    out_stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
                else:
                    out_stream.write(body + b"\n")
                out_stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                pipe_broken[0] = True
                server.cancel_all_in_flight()

    def _spawn_tools_call(msg: dict[str, Any], framing: str) -> None:
        has_id = "id" in msg
        req_id = msg.get("id")
        params = msg.get("params")
        scope_id = f"mcp-{uuid.uuid4().hex}"

        async def _notify_for_framing(payload: dict[str, Any]) -> None:
            await _write_payload(payload, framing)

        async def _runner() -> None:
            with sandbox_scope(scope_id):
                try:
                    if params is None:
                        eff_params: Any = {}
                    else:
                        eff_params = params
                    if not isinstance(eff_params, dict):
                        resp = _jsonrpc_error(req_id, ERR_INVALID_PARAMS, "Invalid params: expected object")
                    else:
                        resp = await server._handle_tools_call(
                            req_id,
                            eff_params,
                            has_id=has_id,
                            notify=_notify_for_framing,
                        )
                    if resp is not None and not (has_id and server.is_cancelled(req_id)):
                        await _write_payload(resp, framing)
                except asyncio.CancelledError:
                    kill_active_sandbox_children(scope=scope_id)
                finally:
                    if has_id:
                        server.unregister_in_flight(req_id)

        task = asyncio.create_task(_runner())
        if has_id:
            server.register_in_flight(req_id, task, scope_id)
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)

    try:
        while not pipe_broken[0]:
            raw_bytes, framing = await asyncio.to_thread(read_stdio_message, in_stream)
            if raw_bytes is None:
                break

            try:
                msg = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                err_resp = _jsonrpc_error(None, ERR_PARSE, f"Parse error: {exc}")
                await _write_payload(err_resp, framing)
                continue

            if isinstance(msg, list):
                # Support JSON-RPC 2.0 batch requests cleanly
                async def _notify_batch(payload: dict[str, Any]) -> None:
                    await _write_payload(payload, framing)

                responses: list[dict[str, Any]] = []
                for item in msg:
                    resp = await server.handle_request(item, notify=_notify_batch)
                    if resp is not None:
                        responses.append(resp)
                await _write_batch(responses, framing)
                continue

            if isinstance(msg, dict) and msg.get("method") == "tools/call":
                _spawn_tools_call(msg, framing)
                continue

            resp = await server.handle_request(msg)
            if resp is not None:
                await _write_payload(resp, framing)

        # Drain in-flight tasks on stdin EOF unless the client closed stdout (dead pipe).
        while active_tasks:
            if pipe_broken[0] or _is_output_pipe_closed(out_stream):
                server.cancel_all_in_flight()
                break
            await asyncio.wait(list(active_tasks), timeout=0.1)
    finally:
        if active_tasks:
            server.cancel_all_in_flight()
            await asyncio.gather(*list(active_tasks), return_exceptions=True)


def run_stdio_server(
    default_url: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    server: Optional[MCPServer] = None,
    stdin: Optional[BinaryIO] = None,
    stdout: Optional[BinaryIO] = None,
) -> None:
    """Start the Cortheon stdio MCP server until EOF / KeyboardInterrupt."""
    srv = server or MCPServer(default_url=default_url, api_key=api_key, model=model)
    try:
        asyncio.run(serve_stdio(srv, stdin=stdin, stdout=stdout))
    except KeyboardInterrupt:
        pass


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cortheon-mcp",
        description="Cortheon Native Zero-Dependency Stdio MCP Server",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Default OpenAI-compatible upstream URL (overrides CORTHEON_URL; fallback http://127.0.0.1:8080/v1)",
    )
    parser.add_argument("--api-key", default=None, help="Optional Bearer API key for upstream backend")
    parser.add_argument("--model", default=None, help="Optional model ID override")
    args = parser.parse_args(argv)
    run_stdio_server(default_url=args.url, api_key=args.api_key, model=args.model)


if __name__ == "__main__":
    main()
