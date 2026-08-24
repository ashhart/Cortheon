"""Runtime and model-endpoint health probes with postflight checks."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


def _runtime_health(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=2) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cortheon runtime is unavailable at {url}: {exc}") from exc
    if payload.get("storage") != "memory_only":
        raise ValueError("Cortheon runtime did not report memory_only storage")
    return payload


def _model_endpoint_health(
    base_url: str,
    *,
    api_key: str,
    model_id: str,
    inference_timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read(1_000_001)
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError(f"model endpoint is unavailable at {base_url}: {exc}") from exc
    if len(raw) > 1_000_000:
        raise ValueError("model endpoint response exceeded 1 MB")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("model endpoint returned invalid JSON") from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    model_ids = (
        {
            item.get("id")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if isinstance(models, list)
        else set()
    )
    if model_id not in model_ids:
        raise ValueError(f"model endpoint does not advertise requested model {model_id!r}")
    probe = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
            }
        ).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(probe, timeout=inference_timeout) as response:
            raw_probe = response.read(1_000_001)
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError(
            f"model {model_id!r} is listed but failed a live inference probe: {exc}"
        ) from exc
    probe_latency = time.perf_counter() - started
    if len(raw_probe) > 1_000_000:
        raise ValueError("model inference probe response exceeded 1 MB")
    try:
        probe_payload = json.loads(raw_probe)
    except json.JSONDecodeError as exc:
        raise ValueError("model inference probe returned invalid JSON") from exc
    choices = probe_payload.get("choices") if isinstance(probe_payload, dict) else None
    message = (
        choices[0].get("message")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else None
    )
    probe_text = (
        str(message.get("content") or message.get("reasoning_content") or "")
        if isinstance(message, dict)
        else ""
    )
    if not probe_text.strip():
        raise ValueError("model inference probe returned no assistant content")
    return {
        "ok": True,
        "model_id": model_id,
        "models_reported": len(model_ids),
        "inference_probe_ok": True,
        "inference_probe_latency_seconds": round(probe_latency, 4),
    }


def _postflight_probe(probe: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Return a reportable postflight result instead of discarding completed runs."""

    try:
        return probe()
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
