"""Sticky bounded terminal state for the evaluator-owned generic MCP host."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cortheon.benchmark_core.generic_mcp_protocol import terminal_payload


@dataclass(slots=True)
class StickyTerminal:
    max_continuations: int = 1
    continuations: int = 0
    disposition: dict[str, object] | None = None

    def _set(
        self,
        disposition: Literal["release", "withhold", "fail_open"],
        text: str,
        *,
        provenance: str,
        finish_reason: str,
        runtime_closed: bool,
    ) -> dict[str, object]:
        if self.disposition is not None:
            raise RuntimeError("terminal disposition is sticky")
        self.disposition = terminal_payload(
            disposition,
            text,
            provenance=provenance,
            finish_reason=finish_reason,
            runtime_closed=runtime_closed,
        )
        return self.disposition

    def certified(self, answer: str, *, runtime_closed: bool) -> dict[str, object]:
        return self._set(
            "release",
            answer,
            provenance="cortheon_complete",
            finish_reason="certified",
            runtime_closed=runtime_closed,
        )

    def released(self, answer: str, *, runtime_closed: bool) -> dict[str, object]:
        return self._set(
            "release",
            answer,
            provenance="generic_mcp_model",
            finish_reason="stop",
            runtime_closed=runtime_closed,
        )

    def premature_final(self) -> Literal["continue", "withhold"]:
        if self.disposition is not None:
            raise RuntimeError("terminal disposition is sticky")
        if self.continuations < self.max_continuations:
            self.continuations += 1
            return "continue"
        return "withhold"

    def withheld(self, reason: str, *, runtime_closed: bool) -> dict[str, object]:
        return self._set(
            "withhold",
            f"[Cortheon withheld: {reason[:500]}]",
            provenance="generic_mcp_wrapper",
            finish_reason="bounded_incomplete",
            runtime_closed=runtime_closed,
        )

    def fail_open(self, candidate: str, reason: str) -> dict[str, object]:
        return self._set(
            "fail_open",
            candidate,
            provenance="generic_mcp_wrapper",
            finish_reason=f"degraded_{reason[:96]}",
            runtime_closed=False,
        )
