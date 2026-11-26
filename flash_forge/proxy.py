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
import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .code_forge import CodeForgeTrace
from .engine import FlashForge, ForgeConfig
from .reason_forge import ReasonForgeTrace
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


def create_app(
    upstream_url: str = "http://127.0.0.1:8080/v1",
    referee_url: Optional[str] = None,
    upstream_model: Optional[str] = None,
    num_streams: int = 4,
) -> FastAPI:
    app = FastAPI(title="Cortheon Universal Proxy", version="2.0.0")
    forge_default = FlashForge(
        ForgeConfig(
            base_url=upstream_url,
            referee_url=referee_url,
            model=upstream_model,
            num_streams=num_streams,
            deep=False,
        )
    )
    forge_deep = FlashForge(
        ForgeConfig(
            base_url=upstream_url,
            referee_url=referee_url,
            model=upstream_model,
            num_streams=num_streams,
            deep=True,
        )
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await forge_default.client.probe()
        await forge_deep.client.probe()
        if forge_default.referee_client:
            await forge_default.referee_client.probe()
        if forge_deep.referee_client:
            await forge_deep.referee_client.probe()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await forge_default.close()
        await forge_deep.close()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        caps = await forge_default.client.probe()
        return {
            "status": "ok",
            "upstream": upstream_url,
            "upstream_model": caps.model_id,
            "referee_url": referee_url,
            "memory_lessons": forge_default.memory.count(),
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

    async def _passthrough_openai(body: dict[str, Any]) -> Any:
        """Transparently forward OpenAI client-side tool-calling loops (OMP, Pi, OpenCode, Cursor) to upstream."""
        caps = await forge_default.client.probe()
        fwd = dict(body)
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
        engine = forge_deep if is_deep else forge_default
        cid = f"chatcmpl-ff-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if not stream:
            result = await engine.solve(prompt, force_mode=force_mode)  # type: ignore[arg-type]
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
            solve_task = asyncio.create_task(engine.solve(prompt, force_mode=force_mode))  # type: ignore[arg-type]
            try:
                while not solve_task.done():
                    if await request.is_disconnected():
                        solve_task.cancel()
                        return
                    # Emit immediate SSE keepalive comment so clients never time out waiting for multi-wave synthesis
                    yield ": keepalive\n\n"
                    done, _ = await asyncio.wait({solve_task}, timeout=4.0)
                    if done:
                        break
                result = await solve_task
            except asyncio.CancelledError:
                solve_task.cancel()
                return
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
        engine = forge_deep if is_deep else forge_default
        mid = f"msg_ff_{uuid.uuid4().hex[:16]}"

        if not stream:
            result = await engine.solve(prompt, force_mode=force_mode)  # type: ignore[arg-type]
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
            solve_task = asyncio.create_task(engine.solve(prompt, force_mode=force_mode))  # type: ignore[arg-type]
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
