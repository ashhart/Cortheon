"""Bounded OpenAI-compatible model client for the generic MCP host."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelTurn:
    provider_id: str
    model_id: str
    content: str
    tool_calls: tuple[ModelToolCall, ...]
    finish_reason: str
    tokens: int
    identity_provenance: str = "evaluator_requested_endpoint_response_model"
    cost_usd: float | None = None


class OpenAiModelClient:
    evaluator_executes_exact_tools = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        provider_id: str,
        model_id: str,
        timeout_seconds: float,
        output_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_id = provider_id
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.output_tokens = output_tokens
        self.endpoint_sha256 = hashlib.sha256(self.base_url.encode()).hexdigest()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        if tool_choice != "auto" and not any(
            tool.get("function", {}).get("name") == tool_choice for tool in tools
        ):
            raise ValueError("forced tool choice is absent from the offered catalogue")
        body = {
            "model": self.model_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": (
                "auto"
                if tool_choice == "auto"
                else {"type": "function", "function": {"name": tool_choice}}
            ),
            "temperature": 0,
            "max_tokens": self.output_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, separators=(",", ":")).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2_000_001)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise RuntimeError(f"model endpoint failed: {exc}") from exc
        if len(raw) > 2_000_000:
            raise RuntimeError("model response exceeded 2 MB")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("model response was invalid JSON") from exc
        return self._turn(payload)

    def _turn(self, payload: Any) -> ModelTurn:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        allowed_payload = {
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "system_fingerprint",
            "cost",
        }
        if (
            not isinstance(payload, dict)
            or not {"model", "choices", "usage"} <= set(payload)
            or not set(payload) <= allowed_payload
        ):
            raise RuntimeError("model response envelope fields were invalid")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        if (
            not isinstance(choice, dict)
            or not {"message", "finish_reason"} <= set(choice)
            or not set(choice) <= {"index", "message", "finish_reason", "logprobs"}
        ):
            raise RuntimeError("model choice fields were invalid")
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("model response lacked one assistant message")
        assert isinstance(choice, dict)
        if (
            message.get("role") != "assistant"
            or not set(message) <= {"role", "content", "reasoning_content", "tool_calls"}
            or not ({"content", "tool_calls"} & set(message))
        ):
            raise RuntimeError("assistant message fields were invalid")
        observed_model = payload.get("model")
        if observed_model != self.model_id:
            raise RuntimeError("model response identity did not match the requested model")
        content = message.get("content") or ""
        if not isinstance(content, str) or len(content) > 200_000:
            raise RuntimeError("assistant content was not bounded text")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list) or len(raw_calls) > 16:
            raise RuntimeError("assistant tool calls were malformed or oversized")
        calls: list[ModelToolCall] = []
        for raw in raw_calls:
            function = raw.get("function") if isinstance(raw, dict) else None
            call_id = raw.get("id") if isinstance(raw, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if (
                not isinstance(raw, dict)
                or set(raw) != {"id", "type", "function"}
                or raw.get("type") != "function"
                or not isinstance(function, dict)
                or set(function) != {"name", "arguments"}
                or not all(isinstance(item, str) for item in (call_id, name, arguments))
            ):
                raise RuntimeError("assistant tool call lacked exact string fields")
            assert isinstance(arguments, str)
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError("assistant tool arguments were invalid JSON") from exc
            if not isinstance(decoded, dict):
                raise RuntimeError("assistant tool arguments must decode to an object")
            calls.append(ModelToolCall(str(call_id), str(name), decoded))
        usage = payload.get("usage")
        allowed_usage = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
            "cost",
            "input_tokens",
            "output_tokens",
            "total_time",
            "model_load_duration",
        }
        if not isinstance(usage, dict) or not set(usage) <= allowed_usage:
            raise RuntimeError("model response usage fields were invalid")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens < 0:
            raise RuntimeError("model response lacked valid token usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (prompt_tokens, completion_tokens)
        ):
            raise RuntimeError("model response token components were invalid")
        if (
            usage.get("input_tokens", prompt_tokens) != prompt_tokens
            or usage.get("output_tokens", completion_tokens) != completion_tokens
        ):
            raise RuntimeError("model response duplicate token fields conflicted")
        if (
            prompt_tokens is not None
            and completion_tokens is not None
            and tokens != prompt_tokens + completion_tokens
        ):
            raise RuntimeError("model response total token count conflicted")
        for timing_key in ("total_time", "model_load_duration"):
            timing = usage.get(timing_key)
            if timing is not None and (
                isinstance(timing, bool)
                or not isinstance(timing, (int, float))
                or not math.isfinite(timing)
                or timing < 0
            ):
                raise RuntimeError(f"model response {timing_key} was invalid")
        for details_key in ("prompt_tokens_details", "completion_tokens_details"):
            details = usage.get(details_key)
            if details is not None and (
                not isinstance(details, dict)
                or len(json.dumps(details, separators=(",", ":"))) > 10_000
            ):
                raise RuntimeError("model response token details were invalid")
        reason = choice.get("finish_reason")
        if not isinstance(reason, str) or len(reason) > 128:
            raise RuntimeError("model response lacked a bounded finish reason")
        raw_cost = usage.get("cost", payload.get("cost"))
        if raw_cost is None:
            cost = None
        elif isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise RuntimeError("model response cost was invalid")
        else:
            cost = float(raw_cost)
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise RuntimeError("model response cost was invalid")
        return ModelTurn(
            self.provider_id,
            self.model_id,
            content,
            tuple(calls),
            reason,
            tokens,
            "evaluator_requested_endpoint_response_model",
            cost,
        )
