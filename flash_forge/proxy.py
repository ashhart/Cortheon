"""Universal Drop-In OpenAI + Anthropic Proxy Server wrapping any local LLM with FlashForge.

Works with TensorFold, Ollama, llama.cpp, vLLM, and LM Studio.
Point ANY client straight at http://127.0.0.1:18652:
  - OpenAI SDK / Cursor / OpenCode / tensorfold chat -> http://127.0.0.1:18652/v1
  - Claude Code / Anthropic SDK                      -> ANTHROPIC_BASE_URL=http://127.0.0.1:18652

Virtual model IDs:
  - flash-forge         (auto-routes fast / code / reason / think in ~100ms)
  - flash-forge:code    (forces CodeForge parallel race + brute fuzzer + red-team)
  - flash-forge:reason  (forces ReasonForge neuro-symbolic Python/SymPy/Z3)
  - flash-forge:think   (forces ThinkForge B=4 adversarial council + synthesis)
  - flash-forge:deep    (auto-routes with larger thinking budgets)
  - flash-forge:fast    (direct B=1 passthrough)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .cockpit import CockpitBus, create_cockpit_router
from .code_forge import CodeForgeTrace
from .engine import FlashForge, ForgeConfig
from .progress import format_sse_comment
from .reason_forge import ReasonForgeTrace
from .rescue import detect_death_spiral, inject_rescue_note, run_clean_context_rescue
from .sandbox import sandbox_backend
from .think_forge import ThinkForgeTrace


def _flatten_messages(messages: list[dict[str, Any]], system_text: str = "") -> str:
    parts: list[str] = []
    if system_text:
        parts.append(f"[SYSTEM]\n{system_text}")
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            chunks: list[str] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                ptype = p.get("type", "")
                if p.get("text"):
                    chunks.append(str(p["text"]))
                elif ptype == "tool_use":
                    chunks.append(f"[TOOL_CALL id={p.get('id','')} name={p.get('name','')}]\n{json.dumps(p.get('input', {}))}")
                elif ptype == "tool_result":
                    chunks.append(f"[TOOL_RESULT id={p.get('tool_use_id','')}]\n{p.get('content', '')}")
            content = "\n".join(chunks)
        if role == "assistant" and m.get("tool_calls"):
            tc_str = json.dumps(m["tool_calls"])
            content = f"{content}\n[TOOL_CALLS]\n{tc_str}".strip()
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}")
        elif role == "tool":
            parts.append(f"[TOOL_RESULT id={m.get('tool_call_id','')}]\n{content}")
        else:
            parts.append(str(content))
    return "\n\n".join(parts).strip()


def _select_mode(model_req: str) -> tuple[Optional[str], bool]:
    if model_req.endswith(":fast"):
        return "fast", False
    if model_req.endswith(":code"):
        return "code", False
    if model_req.endswith(":reason"):
        return "reason", False
    if model_req.endswith(":think"):
        return "think", False
    if model_req.endswith(":deep"):
        return None, True
    return None, False


def _format_reply_and_meta(result: Any) -> tuple[str, dict[str, Any]]:
    if result.mode == "reason" and isinstance(result.raw_trace, ReasonForgeTrace):
        reply_text = result.raw_trace.final_explanation or result.answer
        meta = {
            "mode": "reason",
            "consensus": result.raw_trace.consensus_type,
            "wave2_triggered": result.raw_trace.wave2_triggered,
            "final_answer": result.answer,
            "wall_s": round(result.wall_s, 2),
            "lesson_learned": result.lesson_learned,
        }
    elif result.mode == "code" and isinstance(result.raw_trace, CodeForgeTrace):
        reply_text = result.answer
        meta = {
            "mode": "code",
            "winner": result.raw_trace.winner_name,
            "redteam_kills": len(result.raw_trace.redteam_valid_kills),
            "wave2_triggered": result.raw_trace.wave2_triggered,
            "wall_s": round(result.wall_s, 2),
            "lesson_learned": result.lesson_learned,
        }
    elif result.mode == "think" and isinstance(result.raw_trace, ThinkForgeTrace):
        reply_text = result.answer
        meta = {
            "mode": "think",
            "council_members": list(result.raw_trace.council_outputs.keys()),
            "wall_s": round(result.wall_s, 2),
        }
    else:
        reply_text = result.answer
        meta = {"mode": result.mode, "wall_s": round(result.wall_s, 2)}
    return reply_text, meta


def _streams_profile_label(num_streams: int) -> str:
    if num_streams <= 2:
        return f"--streams {num_streams} (fast sanity check, ~2-3 calls)"
    if num_streams == 4:
        return "--streams 4 (balanced default, ~4-7 calls)"
    if num_streams >= 8:
        return f"--streams {num_streams} (max diversity, ~{num_streams}-{num_streams + 3} calls)"
    return f"--streams {num_streams} (custom, ~{num_streams}-{num_streams + 3} calls)"


def format_proxy_startup_banner(
    *,
    upstream_url: str,
    upstream_model: Optional[str] = None,
    referee_url: Optional[str] = None,
    num_streams: int = 4,
    auto_rescue: bool = True,
    record_flywheel: bool = False,
    cockpit_url: Optional[str] = None,
    sandbox: Optional[str] = None,
) -> str:
    """Format the startup runtime summary banner for `cortheon proxy`."""
    active_sandbox = sandbox or sandbox_backend()
    rescue_label = "ON (B=3 death-spiral rescue on tool loops)" if auto_rescue else "OFF (--no-auto-rescue)"
    flywheel_label = "ON (auto-logging Wave-2 rescues to flywheel.jsonl)" if record_flywheel else "OFF (pass --record-flywheel to harvest DPO pairs)"
    bar = "━" * 74
    upstream_line = f"{upstream_url} (model={upstream_model})" if upstream_model else upstream_url
    lines = [
        bar,
        "  Cortheon Universal Proxy v2.4.0",
        f"  Upstream    : {upstream_line}",
    ]
    if referee_url:
        lines.append(f"  Referee     : {referee_url}")
    if cockpit_url:
        lines.append(f"  Live Cockpit: {cockpit_url}")
    lines.extend(
        [
            f"  Streams     : {_streams_profile_label(num_streams)}",
            f"  OS Sandbox  : {active_sandbox}",
            f"  Auto-Rescue : {rescue_label}",
            f"  Flywheel    : {flywheel_label}",
            bar,
        ]
    )
    return "\n".join(lines)


def create_app(
    upstream_url: str = "http://127.0.0.1:8080/v1",
    referee_url: Optional[str] = None,
    upstream_model: Optional[str] = None,
    num_streams: int = 4,
    auto_rescue: bool = True,
    record_flywheel: bool = False,
    flywheel_path: Optional[Any] = None,
    cockpit_url: Optional[str] = None,
) -> FastAPI:
    forge_default: Optional[FlashForge] = None
    forge_deep: Optional[FlashForge] = None
    cockpit_bus = CockpitBus(version="2.4.0")

    def _record_solve_to_cockpit(run_id: str, result: Any) -> None:
        try:
            aborted = 0
            if isinstance(result.raw_trace, CodeForgeTrace):
                aborted = int(getattr(result.raw_trace, "early_aborted_slots", 0))
            tok = int(result.token_stats.get("prompt_tokens", 0)) + int(result.token_stats.get("completion_tokens", 0))
            cockpit_bus.record_run(
                {
                    "id": run_id,
                    "mode": result.mode,
                    "wall_s": round(float(result.wall_s), 2),
                    "verified": bool(result.verified),
                    "calls": int(result.token_stats.get("calls", 0)),
                    "tokens": tok,
                    "aborted_slots": aborted,
                    "summary": str(result.answer or "")[:120],
                }
            )
        except Exception:
            pass

    def _ensure_engines() -> tuple[FlashForge, FlashForge]:
        nonlocal forge_default, forge_deep
        if forge_default is None:
            forge_default = FlashForge(
                ForgeConfig(
                    base_url=upstream_url,
                    referee_url=referee_url,
                    model=upstream_model,
                    num_streams=num_streams,
                    deep=False,
                    record_flywheel=record_flywheel,
                    flywheel_path=flywheel_path,
                )
            )
        if forge_deep is None:
            forge_deep = FlashForge(
                ForgeConfig(
                    base_url=upstream_url,
                    referee_url=referee_url,
                    model=upstream_model,
                    num_streams=num_streams,
                    deep=True,
                    record_flywheel=record_flywheel,
                    flywheel_path=flywheel_path,
                )
            )
        return forge_default, forge_deep

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal forge_default, forge_deep
        default_engine, deep_engine = _ensure_engines()
        caps = await default_engine.client.probe()
        await deep_engine.client.probe()
        if default_engine.referee_client:
            await default_engine.referee_client.probe()
        if deep_engine.referee_client:
            await deep_engine.referee_client.probe()
        print(
            format_proxy_startup_banner(
                upstream_url=upstream_url,
                upstream_model=caps.model_id,
                referee_url=referee_url,
                num_streams=num_streams,
                auto_rescue=auto_rescue,
                record_flywheel=record_flywheel,
                cockpit_url=cockpit_url or "/ui",
            ),
            flush=True,
        )
        try:
            yield
        finally:
            await default_engine.close()
            await deep_engine.close()
            forge_default = None
            forge_deep = None

    app = FastAPI(title="Cortheon Universal Proxy", version="2.4.0", lifespan=lifespan)
    app.state.cockpit_bus = cockpit_bus
    app.include_router(create_cockpit_router(cockpit_bus, flywheel_path=flywheel_path))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        default_engine, _ = _ensure_engines()
        caps = await default_engine.client.probe()
        return {
            "status": "ok",
            "version": "2.4.0",
            "upstream": upstream_url,
            "upstream_model": caps.model_id,
            "referee_url": referee_url,
            "memory_lessons": default_engine.memory.count(),
            "sandbox_backend": sandbox_backend(),
            "streams": num_streams,
            "auto_rescue": auto_rescue,
            "record_flywheel": record_flywheel,
            "cockpit_url": cockpit_url or "/ui",
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        ids = [
            "cortheon",
            "cortheon:code",
            "cortheon:reason",
            "cortheon:think",
            "cortheon:deep",
            "cortheon:fast",
            "flash-forge",
        ]
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "created": now, "owned_by": "cortheon"} for m in ids],
        }

    _VIRTUAL_PREFIXES = ("cortheon", "flash-forge")
    _rescued_fingerprints: set[str] = set()

    async def _passthrough_openai(body: dict[str, Any]) -> Any:
        """Transparently forward OpenAI client-side tool-calling loops (OMP, Pi, OpenCode, Cursor) to upstream.
        When `auto_rescue` is enabled and >= 2 consecutive failing tool outputs appear in the active turn,
        runs a fast clean-context B=3 Graveyard Backtracking diagnosis and injects it into the system prompt.
        """
        default_engine, _ = _ensure_engines()
        caps = await default_engine.client.probe()
        fwd = dict(body)
        if auto_rescue and isinstance(fwd.get("messages"), list):
            spiral = detect_death_spiral(fwd["messages"])
            if spiral is not None and spiral.fingerprint not in _rescued_fingerprints:
                _rescued_fingerprints.add(spiral.fingerprint)
                if len(_rescued_fingerprints) > 256:
                    _rescued_fingerprints.clear()
                try:
                    note = await run_clean_context_rescue(default_engine.client, spiral)
                    if note:
                        fwd["messages"] = inject_rescue_note(fwd["messages"], note)
                        cockpit_bus.publish(
                            {
                                "engine": "rescue",
                                "event": "rescue_triggered",
                                "summary": f"B=3 death-spiral rescue injected ({spiral.consecutive_failures} consecutive failures)",
                            }
                        )
                except Exception:
                    pass
        req_model = str(fwd.get("model", ""))
        if not req_model or req_model.startswith(_VIRTUAL_PREFIXES):
            fwd["model"] = caps.model_id
        upstream_endpoint = f"{upstream_url.rstrip('/')}/chat/completions"
        if not bool(fwd.get("stream", False)):
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                r = await client.post(upstream_endpoint, json=fwd)
                return Response(
                    content=r.content,
                    status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"),
                )

        async def _stream_upstream() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                async with client.stream("POST", upstream_endpoint, json=fwd) as r:
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            yield chunk

        return StreamingResponse(_stream_upstream(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # OpenAI Chat Completions API (/v1/chat/completions)
    # ------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        if body.get("tools") or body.get("functions"):
            return await _passthrough_openai(body)
        model_req = str(body.get("model", "cortheon"))
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))

        prompt = _flatten_messages(messages)
        force_mode, is_deep = _select_mode(model_req)
        default_engine, deep_engine = _ensure_engines()
        engine = deep_engine if is_deep else default_engine
        cid = f"chatcmpl-ff-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if not stream:
            result = await engine.solve(prompt, force_mode=force_mode, on_event=cockpit_bus.publish)  # type: ignore[arg-type]
            _record_solve_to_cockpit(cid, result)
            reply_text, meta = _format_reply_and_meta(result)
            usage = {
                "prompt_tokens": result.token_stats.get("prompt_tokens", 0),
                "completion_tokens": result.token_stats.get("completion_tokens", 0),
                "total_tokens": result.token_stats.get("prompt_tokens", 0)
                + result.token_stats.get("completion_tokens", 0),
            }
            return JSONResponse(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_req,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                    "x_flash_forge": meta,
                }
            )

        async def _sse() -> AsyncIterator[str]:
            live_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)

            def _on_ev(ev: dict[str, Any]) -> None:
                cockpit_bus.publish(ev)
                try:
                    live_q.put_nowait(ev)
                except asyncio.QueueFull:
                    pass

            solve_task = asyncio.create_task(engine.solve(prompt, force_mode=force_mode, on_event=_on_ev))  # type: ignore[arg-type]
            try:
                while not solve_task.done():
                    if await request.is_disconnected():
                        solve_task.cancel()
                        return
                    emitted = False
                    while not live_q.empty():
                        yield format_sse_comment(live_q.get_nowait())
                        emitted = True
                    if not emitted:
                        yield ": keepalive\n\n"
                    done, _ = await asyncio.wait({solve_task}, timeout=2.0)
                    if done:
                        break
                while not live_q.empty():
                    yield format_sse_comment(live_q.get_nowait())
                result = await solve_task
            except asyncio.CancelledError:
                solve_task.cancel()
                return
            _record_solve_to_cockpit(cid, result)
            reply_text, meta = _format_reply_and_meta(result)
            usage = {
                "prompt_tokens": result.token_stats.get("prompt_tokens", 0),
                "completion_tokens": result.token_stats.get("completion_tokens", 0),
                "total_tokens": result.token_stats.get("prompt_tokens", 0)
                + result.token_stats.get("completion_tokens", 0),
            }
            chunk1 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_req,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk1)}\n\n"
            chunk2 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_req,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
                "x_flash_forge": meta,
            }
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Anthropic Messages API (/v1/messages) — Drop-in for Claude Code & SDKs!
    # ------------------------------------------------------------------
    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        body = await request.json()
        if body.get("tools"):
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "FlashForge virtual synthesis models do not yet support Anthropic client-side `tools`; use `flash-forge replica` for autonomous tool execution or call without `tools`.",
                    },
                },
            )
        model_req = str(body.get("model", "flash-forge"))
        messages = body.get("messages", [])
        sys_raw = body.get("system", "")
        if isinstance(sys_raw, list):
            sys_text = "\n".join(str(b.get("text", "")) for b in sys_raw if isinstance(b, dict))
        else:
            sys_text = str(sys_raw or "")
        stream = bool(body.get("stream", False))

        prompt = _flatten_messages(messages, system_text=sys_text)
        force_mode, is_deep = _select_mode(model_req)
        default_engine, deep_engine = _ensure_engines()
        engine = deep_engine if is_deep else default_engine
        mid = f"msg_ff_{uuid.uuid4().hex[:16]}"

        if not stream:
            result = await engine.solve(prompt, force_mode=force_mode, on_event=cockpit_bus.publish)  # type: ignore[arg-type]
            _record_solve_to_cockpit(mid, result)
            reply_text, meta = _format_reply_and_meta(result)
            in_tok = result.token_stats.get("prompt_tokens", 0)
            out_tok = result.token_stats.get("completion_tokens", 0)
            return JSONResponse(
                {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": model_req,
                    "content": [{"type": "text", "text": reply_text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                    "x_flash_forge": meta,
                }
            )

        async def _anthropic_sse() -> AsyncIterator[str]:
            yield "event: message_start\ndata: " + json.dumps({
                "type": "message_start",
                "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model_req,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }) + "\n\n"
            solve_task = asyncio.create_task(engine.solve(prompt, force_mode=force_mode, on_event=cockpit_bus.publish))  # type: ignore[arg-type]
            try:
                while not solve_task.done():
                    if await request.is_disconnected():
                        solve_task.cancel()
                        return
                    yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"
                    done, _ = await asyncio.wait({solve_task}, timeout=4.0)
                    if done:
                        break
                result = await solve_task
            except asyncio.CancelledError:
                solve_task.cancel()
                return
            _record_solve_to_cockpit(mid, result)
            reply_text, meta = _format_reply_and_meta(result)
            out_tok = result.token_stats.get("completion_tokens", 0)
            yield "event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""},
            }) + "\n\n"
            yield "event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reply_text},
            }) + "\n\n"
            yield "event: content_block_stop\ndata: " + json.dumps({"type": "content_block_stop", "index": 0}) + "\n\n"
            yield "event: message_delta\ndata: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": out_tok},
            }) + "\n\n"
            yield "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n"

        return StreamingResponse(_anthropic_sse(), media_type="text/event-stream")

    return app
