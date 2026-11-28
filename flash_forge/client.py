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


_REASONING_MODEL_ID_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:deepseek|ds\d|r1|qwq|qwen3|glm[-_]?5|nemotron|inkling|o[134](?:[-_]|$)|thinking|reasoner)",
    re.IGNORECASE,
)


_TENSORFOLD_CAP_KEYS = frozenset(
    {
        "structures",
        "thinking_budget",
        "reasoning_optional",
        "thinking_switch",
        "loop_guard",
        "answer_limit",
        "prompt_cache",
        "decides",
    }
)


def _looks_like_reasoning_model(model_id: str) -> bool:
    """Heuristic prior for reasoning models on generic OpenAI-compatible backends
    that omit TensorFold's `/v1/models` capability metadata.
    """
    return bool(model_id and _REASONING_MODEL_ID_RE.search(model_id))


@dataclass
class ModelCapabilities:
    model_id: str
    context_length: int = 32768
    structures: bool = False
    thinking_budget: bool = False
    reasoning: bool = False
    reasoning_optional: bool = True
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
        # Adaptive runtime flags for generic OpenAI-compatible backends (oMLX, vLLM, SGLang, llama.cpp, Ollama, LM Studio)
        self._observed_reasoning: bool = False
        self._reasoning_token_floor: int = 0
        self._chat_template_kwargs_rejected: bool = False
        self._reasoning_effort_rejected: bool = False
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
        caps_raw = entry.get("capabilities")
        caps_dict = caps_raw if isinstance(caps_raw, dict) else {}
        caps_list = {str(x).lower() for x in caps_raw if isinstance(x, str)} if isinstance(caps_raw, list) else set()
        reasoning_raw = entry.get("reasoning")
        reasoning_info = reasoning_raw if isinstance(reasoning_raw, dict) else {}
        # Detect whether backend is TensorFold native vs generic OpenAI-compatible (Ollama/vLLM/llama.cpp/LM Studio)
        is_tensorfold = (entry.get("owned_by") == "tensorfold") or bool(caps_dict.keys() & _TENSORFOLD_CAP_KEYS)
        meta_raw = entry.get("meta")
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        ctx_len = int(
            entry.get("context_length")
            or entry.get("max_model_len")
            or meta.get("n_ctx")
            or 32768
        )
        efforts_raw = reasoning_info.get("supported_efforts")
        supported_efforts = [str(e) for e in efforts_raw] if isinstance(efforts_raw, (list, tuple)) else []
        reasoning_flag = (
            bool(caps_dict.get("reasoning", False))
            if is_tensorfold
            else (
                bool(caps_dict.get("reasoning", False))
                or (reasoning_raw is True)
                or bool(caps_list & {"reasoning", "thinking"})
            )
        )
        self._caps = ModelCapabilities(
            model_id=entry["id"],
            context_length=ctx_len,
            structures=bool(caps_dict.get("structures", False)) if is_tensorfold else True,
            thinking_budget=bool(caps_dict.get("thinking_budget", False)) if is_tensorfold else False,
            reasoning=reasoning_flag,
            reasoning_optional=bool(caps_dict.get("reasoning_optional", True)) if is_tensorfold else True,
            thinking_switch=bool(caps_dict.get("thinking_switch", False)) if is_tensorfold else False,
            loop_guard=bool(caps_dict.get("loop_guard", False)) if is_tensorfold else False,
            answer_limit=bool(caps_dict.get("answer_limit", False)) if is_tensorfold else False,
            prompt_cache=bool(caps_dict.get("prompt_cache", False)) if is_tensorfold else False,
            logprobs=bool(caps_dict.get("logprobs", False)) or ("logprobs" in caps_list),
            decides=bool(caps_dict.get("decides", False)) if is_tensorfold else False,
            supported_efforts=supported_efforts,
        )
        self._is_tensorfold = is_tensorfold
        return self._caps

    @property
    def caps(self) -> ModelCapabilities:
        if self._caps is None:
            raise RuntimeError("Call await client.probe() before accessing caps")
        return self._caps

    def _needs_reasoning_headroom(self) -> bool:
        caps = self.caps
        is_tf = getattr(self, "_is_tensorfold", True)
        if caps.thinking_budget:
            return False
        if caps.supported_efforts:
            return True
        return (not is_tf) and (
            getattr(self, "_observed_reasoning", False) or _looks_like_reasoning_model(caps.model_id)
        )

    def _build_payload(self, spec: CallSpec) -> dict[str, Any]:
        caps = self.caps
        is_tf = getattr(self, "_is_tensorfold", True)
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
            if is_tf or not getattr(self, "_chat_template_kwargs_rejected", False):
                payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Always-on reasoning models (e.g. Inkling-2 where reasoning_optional=False) ignore enable_thinking=False;
        # cap their thinking budget / effort to minimal so short fast-route calls do not starve.
        if not spec.enable_thinking and not caps.reasoning_optional:
            if caps.thinking_budget:
                payload["thinking_budget"] = 0
            if "none" in caps.supported_efforts:
                payload["reasoning_effort"] = "none"
            elif "minimal" in caps.supported_efforts:
                payload["reasoning_effort"] = "minimal"

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
            elif (
                not is_tf
                and not caps.thinking_budget
                and not getattr(self, "_reasoning_effort_rejected", False)
                and (getattr(self, "_observed_reasoning", False) or _looks_like_reasoning_model(caps.model_id))
            ):
                if spec.reasoning_effort in ("low", "medium", "high"):
                    payload["reasoning_effort"] = spec.reasoning_effort
                elif spec.thinking_budget is not None:
                    payload["reasoning_effort"] = "low" if spec.thinking_budget <= 1200 else "medium"
            # Boost max_tokens when server lacks hard thinking_budget (soft-effort DeepSeek-V4.1 on TensorFold,
            # or reasoning models behind generic OpenAI-compatible servers like oMLX/vLLM/SGLang/llama.cpp/Ollama);
            # servers with thinking_budget=True (Qwen/GLM on TensorFold) already reserve reasoning separately.
            if spec.max_tokens > 1 and self._needs_reasoning_headroom():
                base_floor = max(6144, getattr(self, "_reasoning_token_floor", 0))
                max_safe = max(int(payload["max_tokens"]), min(max(1024, caps.context_length // 2), 16384))
                payload["max_tokens"] = max(int(payload["max_tokens"]), min(base_floor, max_safe))

        # Loop guard
        if spec.loop_guard and caps.loop_guard:
            payload["loop_guard"] = True

        # Structured outputs / guided choice (TensorFold libtfgrammar OR standard OpenAI json_schema on Ollama/vLLM/llama.cpp!)
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

    async def _post_with_fallback(self, payload: dict[str, Any]) -> tuple[Optional[httpx.Response], Optional[str]]:
        """POST to /chat/completions with graceful 400/422 fallback for optional grammar or thinking-control fields."""
        try:
            resp = await self._http.post(f"{self.base_url}/chat/completions", json=payload)
        except Exception as exc:
            return None, str(exc)

        if resp.status_code == 200:
            return resp, None

        if resp.status_code in (400, 422):
            cur_payload = dict(payload)
            grammar_keys = ("response_format", "guided_choice", "grammar_budget")
            if any(k in cur_payload for k in grammar_keys):
                for k in grammar_keys:
                    cur_payload.pop(k, None)
                try:
                    resp2 = await self._http.post(f"{self.base_url}/chat/completions", json=cur_payload)
                except Exception as exc:
                    return None, str(exc)
                if resp2.status_code == 200:
                    return resp2, None
                resp = resp2

            if resp.status_code in (400, 422):
                had_ctk = "chat_template_kwargs" in cur_payload
                had_effort = "reasoning_effort" in cur_payload
                if had_ctk or had_effort:
                    cur_payload.pop("chat_template_kwargs", None)
                    cur_payload.pop("reasoning_effort", None)
                    try:
                        resp3 = await self._http.post(f"{self.base_url}/chat/completions", json=cur_payload)
                    except Exception as exc:
                        return None, str(exc)
                    if resp3.status_code == 200:
                        if had_ctk:
                            self._chat_template_kwargs_rejected = True
                        if had_effort:
                            self._reasoning_effort_rejected = True
                        return resp3, None
                    resp = resp3

        return None, f"HTTP {resp.status_code}: {resp.text[:300]}"

    @staticmethod
    def _parse_completion_json(data: dict[str, Any]) -> tuple[str, str, str, int, int, int, int]:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        prompt_tok = int(usage.get("prompt_tokens") or 0)
        comp_tok = int(usage.get("completion_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        comp_details = usage.get("completion_tokens_details") or {}
        cached_tok = int(prompt_details.get("cached_tokens") or 0)
        reason_tok = int(comp_details.get("reasoning_tokens") or 0)
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish_reason = str(choice.get("finish_reason") or "stop")
        return content, reasoning, finish_reason, prompt_tok, comp_tok, cached_tok, reason_tok

    async def complete(self, spec: CallSpec) -> CompletionResult:
        await self.probe()
        caps = self.caps
        payload = self._build_payload(spec)
        async with self._sem:
            t0 = time.perf_counter()
            resp, err = await self._post_with_fallback(payload)
            dt = time.perf_counter() - t0
            if resp is None:
                return CompletionResult(
                    content="",
                    reasoning="",
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_tokens=0,
                    cached_tokens=0,
                    latency_s=dt,
                    raw_error=err,
                    tag=spec.tag,
                )

            content, reasoning, finish_reason, prompt_tok, comp_tok, cached_tok, reason_tok = (
                self._parse_completion_json(resp.json())
            )
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tok
            self.total_completion_tokens += comp_tok
            self.total_cached_tokens += cached_tok
            self.total_reasoning_tokens += reason_tok

            if reason_tok > 0 or bool(reasoning):
                self._observed_reasoning = True

            # Adaptive recovery when a reasoning model behind a backend without hard thinking_budget
            # (or an always-on reasoning model with reasoning_optional=False on a fast-route call)
            # consumes the entire max_tokens budget inside reasoning_content (leaving empty content).
            if (
                finish_reason == "length"
                and not content.strip()
                and (reason_tok > 0 or bool(reasoning))
                and (not caps.thinking_budget or (not spec.enable_thinking and not caps.reasoning_optional))
                and spec.max_tokens > 1
            ):
                sent_max = int(payload.get("max_tokens", spec.max_tokens))
                retry_cap = min(16384, max(sent_max, max(1024, caps.context_length // 2)))
                retry_payload: Optional[dict[str, Any]] = None
                if not spec.enable_thinking:
                    next_max = min(max(sent_max * 4, 1024), retry_cap)
                    if next_max > sent_max:
                        retry_payload = dict(payload)
                        retry_payload["max_tokens"] = next_max
                elif sent_max < 6144 and retry_cap > sent_max:
                    next_max = min(max(sent_max * 2, 6144), retry_cap)
                    self._reasoning_token_floor = max(self._reasoning_token_floor, next_max)
                    retry_payload = dict(payload)
                    retry_payload["max_tokens"] = next_max
                elif not getattr(self, "_chat_template_kwargs_rejected", False):
                    # Already spent >=6144 tokens thinking without emitting visible content:
                    # run a fast enable_thinking=False commit pass seeded with the scratchpad tail.
                    notes_tail = reasoning[-2400:].strip() if reasoning else ""
                    commit_prompt = (
                        "Your reasoning scratchpad reached the token limit before emitting the final answer. "
                        "Do not continue brainstorming. Output ONLY the final requested answer/code directly now."
                    )
                    if notes_tail:
                        commit_prompt += f"\n\nScratchpad excerpt:\n{notes_tail}"
                    commit_msgs = [dict(m) for m in spec.messages]
                    if commit_msgs and commit_msgs[-1].get("role") == "user":
                        commit_msgs[-1]["content"] = f"{commit_msgs[-1].get('content', '')}\n\n{commit_prompt}"
                    else:
                        commit_msgs.append({"role": "user", "content": commit_prompt})
                    commit_spec = CallSpec(
                        messages=commit_msgs,
                        tag=spec.tag,
                        max_tokens=max(2048, min(spec.max_tokens, 4096)),
                        temperature=min(spec.temperature, 0.3),
                        top_p=spec.top_p,
                        top_k=spec.top_k,
                        min_p=spec.min_p,
                        seed=spec.seed,
                        enable_thinking=False,
                        json_schema=spec.json_schema,
                        schema_name=spec.schema_name,
                        guided_choice=spec.guided_choice,
                        grammar_budget=spec.grammar_budget,
                        loop_guard=spec.loop_guard,
                        stop=spec.stop,
                    )
                    retry_payload = self._build_payload(commit_spec)
                elif retry_cap > sent_max:
                    next_max = min(sent_max * 2, retry_cap)
                    self._reasoning_token_floor = max(self._reasoning_token_floor, next_max)
                    retry_payload = dict(payload)
                    retry_payload["max_tokens"] = next_max

                if retry_payload is not None:
                    resp2, _ = await self._post_with_fallback(retry_payload)
                    dt = time.perf_counter() - t0
                    if resp2 is not None:
                        c2, r2, fr2, pt2, ct2, cat2, rt2 = self._parse_completion_json(resp2.json())
                        self.total_calls += 1
                        self.total_prompt_tokens += pt2
                        self.total_completion_tokens += ct2
                        self.total_cached_tokens += cat2
                        self.total_reasoning_tokens += rt2
                        prompt_tok += pt2
                        comp_tok += ct2
                        cached_tok += cat2
                        reason_tok += rt2
                        if c2.strip():
                            content = c2
                            finish_reason = fr2
                        if r2 and not reasoning:
                            reasoning = r2

            return CompletionResult(
                content=content,
                reasoning=reasoning,
                finish_reason=finish_reason,
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
