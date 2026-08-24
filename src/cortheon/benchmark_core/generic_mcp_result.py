"""Result contract for one evaluator-owned generic MCP task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenericHostResult:
    events: tuple[dict[str, Any], ...]
    final_text: str
    delivered: bool
    process_error: str | None
    tokens: int
    cost_usd: float | None
    model_steps: int
    tool_calls: int
