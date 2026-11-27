"""Automatic Death-Spiral Rescue for live coding-agent tool loops (OMP, Pi, OpenCode, Claude Code, Cursor).

Detects when a coding harness hits >= 2 consecutive failing test/compiler/traceback tool results
within the active user turn and runs a fast clean-context B=3 Graveyard Backtracking diagnosis
before forwarding the next tool turn upstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .client import CallSpec, TensorFoldClient

_FAILURE_MARKERS: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "AssertionError",
    "SyntaxError",
    "TypeError",
    "NameError",
    "ValueError",
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "FAILED ",
    " failed",
    "exit 1",
    "exit code 1",
    "Command exited with code 1",
    "error: ",
    "panic:",
)


def _msg_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
        return "\n".join(parts)
    return str(content or "")


def is_failing_tool_output(text: str) -> bool:
    if not text or len(text) < 8:
        return False
    return any(marker in text for marker in _FAILURE_MARKERS)


@dataclass
class SpiralDetection:
    consecutive_failures: int
    latest_user_goal: str
    failure_snippets: list[str]
    fingerprint: str


def detect_death_spiral(messages: list[dict[str, Any]], threshold: int = 2) -> SpiralDetection | None:
    """Return a SpiralDetection when >= `threshold` consecutive trailing tool outputs failed."""
    if not messages:
        return None

    latest_user_idx = -1
    latest_user_goal = ""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            txt = _msg_text(messages[i]).strip()
            if txt:
                latest_user_idx = i
                latest_user_goal = txt[:2000]
                break

    consecutive = 0
    snippets: list[str] = []
    for i in range(len(messages) - 1, latest_user_idx, -1):
        m = messages[i]
        role = m.get("role", "")
        if role == "assistant":
            continue
        if role == "tool":
            txt = _msg_text(m).strip()
            if is_failing_tool_output(txt):
                consecutive += 1
                if len(snippets) < 2:
                    snippets.append(txt[-900:])
            else:
                break

    if consecutive < threshold:
        return None

    h = hashlib.sha256(
        json.dumps({"goal": latest_user_goal[:400], "n": consecutive, "last": snippets[0][:300]}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return SpiralDetection(
        consecutive_failures=consecutive,
        latest_user_goal=latest_user_goal,
        failure_snippets=list(reversed(snippets)),
        fingerprint=h,
    )


async def run_clean_context_rescue(client: TensorFoldClient, spiral: SpiralDetection) -> str:
    """Run B=3 low-effort clean-context Graveyard Backtracking diagnosis over the last failures."""
    tombstones = "\n\n".join(
        f"--- FAILURE #{i + 1} ---\n{snip}" for i, snip in enumerate(spiral.failure_snippets)
    )
    sys_prompt = (
        "You are a principal debugging specialist performing clean-context graveyard backtracking. "
        "Identify the EXACT root cause and minimal surgical fix in <= 8 concise bullet points. "
        "Do NOT repeat failed patches."
    )
    roles = [
        ("root_cause", 0.15, 801, "Diagnose the true root cause from first principles and specify the exact surgical fix."),
        ("invariant_skeptic", 0.45, 802, "Assume prior edits patched a symptom instead of the broken invariant or edge case. What assumption is false and what exact code change fixes it?"),
        ("minimal_fix", 0.30, 803, "Give the smallest zero-regression fix and exact verification command to break this failure loop."),
    ]
    specs = [
        CallSpec(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"GOAL:\n{spiral.latest_user_goal}\n\nRECENT CONSECUTIVE FAILURES:\n{tombstones}\n\n{instruction}",
                },
            ],
            tag=f"rescue_{name}",
            max_tokens=900,
            temperature=temp,
            seed=seed,
            enable_thinking=True,
            thinking_budget=400,
            reasoning_effort="low",
            loop_guard=True,
        )
        for name, temp, seed, instruction in roles
    ]
    results = await client.batch(specs)
    sections: list[str] = []
    for (name, _, _, _), res in zip(roles, results):
        if res.ok and res.content.strip():
            sections.append(f"### {name}\n{res.content.strip()[:1000]}")
    if not sections:
        return ""
    return (
        "[Cortheon Auto-Rescue — Clean-Context B=3 Graveyard Backtracking]\n"
        f"Detected {spiral.consecutive_failures} consecutive failing tool/test runs. "
        "Step back from prior patches and apply one decisive root-cause fix grounded in these independent diagnoses:\n\n"
        + "\n\n".join(sections)
    )


def inject_rescue_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """Return a copy of `messages` with the rescue guidance appended cleanly to the system prompt (safe for strict templates)."""
    if not note:
        return messages
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        prev = _msg_text(out[0])
        out[0]["content"] = f"{prev}\n\n{note}".strip()
        return out
    return [{"role": "system", "content": note}] + out
