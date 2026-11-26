"""Fast (~100ms) adaptive task router using TensorFold guided_choice."""

from __future__ import annotations

import re
from typing import Literal

from .client import TensorFoldClient

TaskMode = Literal["fast", "code", "reason", "think"]

_GREETING_ONLY_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|thanks|thank you|ping|say hello|good morning|good evening)[!.,?\s]*$",
    re.IGNORECASE,
)


def _head_tail_excerpt(text: str, max_chars: int = 3200) -> str:
    """Preserve BOTH opening intent AND trailing output-format/constraint instructions."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head - 32
    return f"{text[:head]}\n\n...[truncated {len(text) - head - tail} chars]...\n\n{text[-tail:]}"


async def classify_task(client: TensorFoldClient, prompt: str) -> TaskMode:
    """Classify a user prompt into 'fast', 'code', 'reason', or 'think' in ~100ms."""
    text = prompt.strip()
    if not text or _GREETING_ONLY_RE.fullmatch(text):
        return "fast"

    excerpt = _head_tail_excerpt(text, max_chars=3200)
    router_prompt = (
        "Classify the user's REQUESTED ACTION (not merely whether code appears in the prompt) into exactly ONE route:\n"
        "- code: asks you to WRITE, IMPLEMENT, FIX, DEBUG, REFACTOR, or OPTIMIZE executable code/functions/classes and return code.\n"
        "- reason: asks for a SINGLE exact final answer, number, string, output, or formal proof — including math, probability, combinatorics, "
        "logic puzzles, CSP/optimization, scheduling, OR tracing/computing the exact output/sizeof/result of a provided Python/C/SQL/regex snippet (verifiable by executing Python/SymPy/Z3). NEVER choose `reason` for system design or architecture audits.\n"
        "- think: asks for open-ended system design, architecture, capacity planning, security/protocol audit, code/PR review, engineering trade-offs, research analysis, explanation, or technical writing — INCLUDING when sizing/latency/throughput math is requested alongside a design or audit.\n"
        "- fast: simple greeting, quick factual lookup, or trivial 1-line chat.\n\n"
        f"REQUEST:\n{excerpt}\n\nRoute:"
    )
    choice, res = await client.decide_choice(
        prompt=router_prompt,
        choices=["code", "reason", "think", "fast"],
        system="Output exactly one label: code, reason, think, or fast.",
        tag="router",
    )
    if res.ok and choice in ("code", "reason", "think", "fast"):
        return choice  # type: ignore[return-value]
    return "think"
