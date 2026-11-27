"""Async TensorFold-native client exploiting deduplicated B=2..16 MoE batching,
prompt caching, libtfgrammar structured outputs, thinking budgets, and loop_guard.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class ModelCapabilities:
    model_id: str
    context_length: int = 32768
    structures: bool = False
    thinking_budget: bool = False
    reasoning: bool = False
    thinking_switch: bool = False
    loop_guard: bool = False
    answer_limit: bool = False
    prompt_cache: bool = False
    logprobs: bool = False
    decides: bool = False
    supported_efforts: list[str] = field(default_factory=list)


@dataclass
class CompletionResult:
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    latency_s: float
    raw_error: Optional[str] = None
    tag: str = ""

    @property
    def ok(self) -> bool:
        return self.raw_error is None

    def json_payload(self) -> Optional[dict[str, Any]]:
        """Parse JSON from content with defensive extraction."""
        text = (self.content or "").strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            # Fallback: grab outermost JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
            return None


@dataclass
class CallSpec:
    messages: list[dict[str, str]]
    tag: str = ""
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: Optional[float] = None
    seed: Optional[int] = None
    enable_thinking: bool = True
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[str] = None
    json_schema: Optional[dict[str, Any]] = None
    schema_name: str = "response"
    guided_choice: Optional[list[str]] = None
    grammar_budget: bool = True
    loop_guard: bool = True
    stop: Optional[list[str]] = None


class TensorFoldClient:
    """Async HTTP client tuned for TensorFold native Metal/CUDA servers."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 360.0,
        max_parallel: int = 8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.requested_model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_parallel = max_parallel
        self._sem = asyncio.Semaphore(max_parallel)
        self._caps: Optional[ModelCapabilities] = None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            headers=headers,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        # Running telemetry
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.total_reasoning_tokens = 0

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "TensorFoldClient":
        await self.probe()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.total_reasoning_tokens = 0

    def stats_dict(self) -> dict[str, int]:
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
            "cached_tokens": self.total_cached_tokens,
        }

    async def probe(self) -> ModelCapabilities:
        if self._caps is not None:
            return self._caps
        r = await self._http.get(f"{self.base_url}/models")
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise RuntimeError(f"No models reported by {self.base_url}/models")
        entry = data[0]
        if self.requested_model:
            for item in data:
                if item.get("id") == self.requested_model:
                    entry = item
                    break
        caps_raw = entry.get("capabilities", {})
        reasoning_info = entry.get("reasoning", {})
        # Detect whether backend is TensorFold native vs generic OpenAI-compatible (Ollama/vLLM/llama.cpp/LM Studio)
        is_tensorfold = bool(caps_raw) or (entry.get("owned_by") == "tensorfold")
        self._caps = ModelCapabilities(
            model_id=entry["id"],
            context_length=int(entry.get("context_length", entry.get("max_model_len", 32768))),
            structures=bool(caps_raw.get("structures", False)) if is_tensorfold else True,
            thinking_budget=bool(caps_raw.get("thinking_budget", False)),
            reasoning=bool(caps_raw.get("reasoning", False)),
            thinking_switch=bool(caps_raw.get("thinking_switch", False)),
            loop_guard=bool(caps_raw.get("loop_guard", False)),
            answer_limit=bool(caps_raw.get("answer_limit", False)),
            prompt_cache=bool(caps_raw.get("prompt_cache", False)),
            logprobs=bool(caps_raw.get("logprobs", False)),
            decides=bool(caps_raw.get("decides", False)),
            supported_efforts=list(reasoning_info.get("supported_efforts", [])),
        )
        self._is_tensorfold = is_tensorfold
        return self._caps

    @property
    def caps(self) -> ModelCapabilities:
        if self._caps is None:
            raise RuntimeError("Call await client.probe() before accessing caps")
        return self._caps

    def _build_payload(self, spec: CallSpec) -> dict[str, Any]:
        caps = self.caps
        payload: dict[str, Any] = {
            "model": caps.model_id,
            "messages": spec.messages,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
        }
        if spec.temperature > 0:
            payload["top_p"] = spec.top_p
            payload["top_k"] = spec.top_k
            if spec.min_p is not None:
                payload["min_p"] = spec.min_p
        if spec.seed is not None:
            payload["seed"] = spec.seed
        if spec.stop:
            payload["stop"] = spec.stop

        # Thinking control (works across TensorFold thinking_switch, reasoning_effort, and OpenAI-compatible servers)
        if caps.thinking_switch:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(spec.enable_thinking)}
        elif not spec.enable_thinking:
            if "none" in caps.supported_efforts:
                payload["reasoning_effort"] = "none"
            if getattr(self, "_is_tensorfold", False):
                payload["chat_template_kwargs"] = {"enable_thinking": False}

        if spec.enable_thinking:
            if spec.thinking_budget is not None and caps.thinking_budget:
                payload["thinking_budget"] = spec.thinking_budget
            if spec.reasoning_effort and spec.reasoning_effort in caps.supported_efforts:
                payload["reasoning_effort"] = spec.reasoning_effort
            elif spec.thinking_budget is not None and caps.supported_efforts:
                # Map token budgets to reasoning_effort on any server advertising supported_efforts (DeepSeek-V4.1, GLM-5.3, etc.)
                if spec.thinking_budget <= 550 and "minimal" in caps.supported_efforts:
                    payload["reasoning_effort"] = "minimal"
                elif spec.thinking_budget <= 1200 and "low" in caps.supported_efforts:
                    payload["reasoning_effort"] = "low"
                elif "medium" in caps.supported_efforts:
                    payload["reasoning_effort"] = "medium"
                elif "high" in caps.supported_efforts:
                    payload["reasoning_effort"] = "high"
            # Only boost max_tokens when server lacks hard thinking_budget (e.g. soft-effort DeepSeek-V4.1);
            # servers with thinking_budget=True (Qwen/GLM) already reserve reasoning separately.
            if not caps.thinking_budget and caps.supported_efforts:
                payload["max_tokens"] = max(int(payload["max_tokens"]), 6144)

        # Loop guard
        if spec.loop_guard and caps.loop_guard:
            payload["loop_guard"] = True

        # Structured outputs / guided choice (TensorFold libtfgrammar OR standard OpenAI json_schema on Ollama/vLLM/llama.cpp!)
        is_tf = getattr(self, "_is_tensorfold", True)
        if caps.structures:
            if spec.guided_choice and is_tf:
                payload["guided_choice"] = spec.guided_choice
            elif spec.json_schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": spec.schema_name,
                        "schema": spec.json_schema,
                    },
                }
                if spec.grammar_budget and is_tf:
                    payload["grammar_budget"] = True

        # Prevent server-side double-nesting when talking to a TensorFold server started with --cortheon.
        # Only sent to TensorFold backends so Ollama/vLLM/llama.cpp never see an unknown field.
        if getattr(self, "_is_tensorfold", False):
            payload["cortheon"] = False

        return payload

    async def complete(self, spec: CallSpec) -> CompletionResult:
        await self.probe()
        payload = self._build_payload(spec)
        async with self._sem:
            t0 = time.perf_counter()
            try:
                resp = await self._http.post(f"{self.base_url}/chat/completions", json=payload)
            except Exception as exc:
                return CompletionResult(
                    content="",
                    reasoning="",
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_tokens=0,
                    cached_tokens=0,
                    latency_s=time.perf_counter() - t0,
                    raw_error=str(exc),
                    tag=spec.tag,
                )
            dt = time.perf_counter() - t0
            if resp.status_code != 200:
                # Automatic graceful fallback if a grammar schema rejected
                if resp.status_code == 400 and ("response_format" in payload or "guided_choice" in payload):
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    fallback_payload.pop("guided_choice", None)
                    fallback_payload.pop("grammar_budget", None)
                    resp2 = await self._http.post(f"{self.base_url}/chat/completions", json=fallback_payload)
                    dt = time.perf_counter() - t0
                    if resp2.status_code == 200:
                        resp = resp2
                    else:
                        return CompletionResult(
                            content="",
                            reasoning="",
                            finish_reason="error",
                            prompt_tokens=0,
                            completion_tokens=0,
                            reasoning_tokens=0,
                            cached_tokens=0,
                            latency_s=dt,
                            raw_error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                            tag=spec.tag,
                        )
                else:
                    return CompletionResult(
                        content="",
                        reasoning="",
                        finish_reason="error",
                        prompt_tokens=0,
                        completion_tokens=0,
                        reasoning_tokens=0,
                        cached_tokens=0,
                        latency_s=dt,
                        raw_error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                        tag=spec.tag,
                    )

            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            usage = data.get("usage", {})
            prompt_tok = int(usage.get("prompt_tokens", 0))
            comp_tok = int(usage.get("completion_tokens", 0))
            cached_tok = int(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            reason_tok = int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0))

            self.total_calls += 1
            self.total_prompt_tokens += prompt_tok
            self.total_completion_tokens += comp_tok
            self.total_cached_tokens += cached_tok
            self.total_reasoning_tokens += reason_tok

            return CompletionResult(
                content=(msg.get("content") or ""),
                reasoning=(msg.get("reasoning_content") or ""),
                finish_reason=str(choice.get("finish_reason") or "stop"),
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                reasoning_tokens=reason_tok,
                cached_tokens=cached_tok,
                latency_s=dt,
                tag=spec.tag,
            )

    async def warm_prefix(self, messages: list[dict[str, str]], enable_thinking: bool = True) -> None:
        """Populate TensorFold's prompt cache for a shared prefix before B=4..8 parallel fan-out."""
        await self.probe()
        if not self.caps.prompt_cache:
            return
        await self.complete(
            CallSpec(
                messages=messages,
                tag="warm_cache",
                max_tokens=1,
                temperature=0.0,
                enable_thinking=enable_thinking,
                loop_guard=False,
            )
        )

    async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
        """Fire B=2..16 requests concurrently so TensorFold deduplicates routed MoE experts."""
        await self.probe()
        if not specs:
            return []
        return list(await asyncio.gather(*(self.complete(s) for s in specs)))

    async def decide_choice(
        self,
        prompt: str,
        choices: list[str],
        system: str = "Respond with exactly one choice label.",
        tag: str = "choice",
    ) -> tuple[str, CompletionResult]:
        """Fast (~100ms) grammar-constrained decision (with clean headroom on non-grammar backends)."""
        await self.probe()
        use_grammar = bool(self.caps.structures and getattr(self, "_is_tensorfold", True))
        spec = CallSpec(
            messages=[
                {
                    "role": "system",
                    "content": system if use_grammar else f"{system} Output ONLY the single label word, nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            tag=tag,
            max_tokens=16 if use_grammar else 64,
            temperature=0.0,
            enable_thinking=False,
            guided_choice=choices,
            grammar_budget=True,
            loop_guard=False,
        )
        res = await self.complete(spec)
        if not res.ok:
            return "", res
        raw_text = res.content.strip()
        picked = raw_text.strip("`\"' \t\r\n.,:;!?*")
        if picked in choices:
            return picked, res
        if res.finish_reason == "length":
            return "", res
        # Check last non-empty line first (handles brief non-grammar preambles ending in the label)
        lines = [ln.strip().strip("`\"' \t.,:;!?*") for ln in raw_text.splitlines() if ln.strip()]
        if lines:
            last_line = lines[-1]
            if last_line in choices:
                return last_line, res
            line_matches = [
                c
                for c in sorted(choices, key=len, reverse=True)
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", last_line)
            ]
            if len(line_matches) == 1:
                return line_matches[0], res
        # Exact word-boundary fallback (longest choice first); require a UNIQUE match
        matches = [
            c
            for c in sorted(choices, key=len, reverse=True)
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", raw_text)
        ]
        if len(matches) == 1:
            return matches[0], res
        return "", res

    async def pairwise_vote(
        self,
        rubric_prompt_ab: str,
        rubric_prompt_ba: str,
        tag_prefix: str = "judge",
    ) -> str:
        """Order-swapped pairwise tournament vote in parallel (cancels position bias).
        Returns 'A', 'B', or 'TIE'.
        """
        await self.probe()
        use_grammar = bool(self.caps.structures and getattr(self, "_is_tensorfold", True))
        sys_msg = (
            "You are a strict, impartial verifier. Compare Candidate A and Candidate B "
            "strictly against the problem specification and edge cases. Reply with only A or B."
        )
        tok_cap = 16 if use_grammar else 64
        spec_ab = CallSpec(
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": rubric_prompt_ab}],
            tag=f"{tag_prefix}_ab",
            max_tokens=tok_cap,
            temperature=0.0,
            enable_thinking=False,
            guided_choice=["A", "B"],
        )
        spec_ba = CallSpec(
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": rubric_prompt_ba}],
            tag=f"{tag_prefix}_ba",
            max_tokens=tok_cap,
            temperature=0.0,
            enable_thinking=False,
            guided_choice=["A", "B"],
        )
        r_ab, r_ba = await self.batch([spec_ab, spec_ba])
        if not (r_ab.ok and r_ba.ok):
            return "TIE"
        vote_ab = r_ab.content.strip()
        # In BA order, 'A' means original B, and 'B' means original A
        vote_ba_raw = r_ba.content.strip()
        vote_ba_mapped = "B" if vote_ba_raw == "A" else ("A" if vote_ba_raw == "B" else "TIE")
        if vote_ab == vote_ba_mapped and vote_ab in ("A", "B"):
            return vote_ab
        return "TIE"
