"""Offline unit tests for CodeForge v8:
- Inline Wave-1 Executable TIR Self-Repair (fixes failing sample in-place without discarding sound algorithm)
- Soundness gate rejecting a buggy brute-force oracle (0% false-positive rate — original winner untouched)
- Sound brute-force fuzzer promoting a correct sample-passing sibling over a buggy sample-passer
- Sound brute-force fuzzer catching a boundary bug in ALL sample-passers and fixing via minimal counterexample TIR repair
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from flash_forge.client import CallSpec, CompletionResult
from flash_forge.code_forge import solve_code
from flash_forge.memory import EpisodicMemory
from flash_forge.skill_loader import SkillPack


def _mk_res(content: str, *, tag: str = "", tokens: int = 120, latency_s: float = 0.02) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=tokens,
        reasoning_tokens=10,
        cached_tokens=0,
        latency_s=latency_s,
        tag=tag,
    )


class _ScriptedCodeClient:
    def __init__(self, handler: Callable[[CallSpec], Any]) -> None:
        self._handler = handler
        self.seen_tags: list[str] = []
        self.seen_prompts: dict[str, str] = {}

    async def complete(self, spec: CallSpec) -> CompletionResult:
        self.seen_tags.append(spec.tag)
        self.seen_prompts[spec.tag] = spec.messages[-1]["content"]
        out = self._handler(spec)
        if asyncio.iscoroutine(out):
            return await out
        return out


# Problem where samples only test positive x > 0, hiding the x == 0 boundary bug!
_ABS_PLUS_ONE_PROBLEM = (
    "Read a single non-negative integer N from standard input and print max(1, N * 2) to standard output.\n\n"
    "### Sample Input 1\n```\n3\n```\n"
    "### Sample Output 1\n```\n6\n```\n\n"
    "### Sample Input 2\n```\n5\n```\n"
    "### Sample Output 2\n```\n10\n```\n"
)

_SOUND_BRUTE_FUZZER = """```python
import random

def solve_brute(inp: str) -> str:
    n = int(inp.strip())
    return str(1 if n == 0 else n * 2)

def gen_tests() -> list[str]:
    rng = random.Random(42)
    cases = ["0", "1", "2"]
    for _ in range(12):
        cases.append(str(rng.randint(0, 20)))
    return cases
```"""

_BUGGY_BRUTE_FUZZER = """```python
import random

def solve_brute(inp: str) -> str:
    # WRONG on Sample 1 (3 -> 9 instead of 6): MUST be discarded by soundness gate!
    n = int(inp.strip())
    return str(n * 3)

def gen_tests() -> list[str]:
    return [str(i) for i in range(15)]
```"""


def test_v8_inline_wave1_tir_repair_rescues_failing_sample_in_place(tmp_path: Path):
    """When Wave-1 slots fail an official sample on try 1, inline TIR repair fixes them
    in-place with exact sample diff feedback and skips Wave 2 completely."""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag in ("w1_radically_simple", "w1_first_principles", "w1_edge_skeptic", "w1_alt_algorithm"):
            # Initial off-by-one bug: prints 2*n + 1 -> fails Sample 1 (got '7' exp '6')
            return _mk_res("```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(n * 2 + 1)\n```", tag=spec.tag)
        if spec.tag.endswith("_fix1"):
            prompt = spec.messages[-1]["content"]
            assert "sample 1" in prompt.lower() or "got='7'" in prompt
            return _mk_res(
                "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(max(1, n * 2))\n```",
                tag=spec.tag,
            )
        if spec.tag == "v8_brute_fuzzer":
            return _mk_res(_SOUND_BRUTE_FUZZER, tag=spec.tag)
        raise AssertionError(f"Unexpected call: {spec.tag}")

    client = _ScriptedCodeClient(_handle)
    trace = asyncio.run(
        solve_code(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            _ABS_PLUS_ONE_PROBLEM,
            num_wave1_streams=2,
            max_code_repairs=1,
            enable_brute_fuzz=True,
        )
    )
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert trace.v8_brute_checked is True
    assert "max(1, n * 2)" in trace.winner_code
    assert any(t.endswith("_fix1") for t in client.seen_tags)


def test_v8_soundness_gate_discards_buggy_brute_fuzzer_with_zero_false_positives(tmp_path: Path):
    """If solve_brute disagrees with ANY official public sample, CodeForge v8 MUST discard it
    completely and leave the winning candidate untouched (0% false-positive rate)."""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag.startswith("w1_"):
            return _mk_res(
                "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(max(1, n * 2))\n```",
                tag=spec.tag,
            )
        if spec.tag == "v8_brute_fuzzer":
            return _mk_res(_BUGGY_BRUTE_FUZZER, tag=spec.tag)
        raise AssertionError(f"Buggy fuzzer must NOT trigger counterexample repair! Got: {spec.tag}")

    client = _ScriptedCodeClient(_handle)
    trace = asyncio.run(
        solve_code(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            _ABS_PLUS_ONE_PROBLEM,
            num_wave1_streams=2,
            max_code_repairs=0,
            enable_brute_fuzz=True,
        )
    )
    assert trace.verified is True
    assert trace.v8_brute_checked is False
    assert trace.v8_brute_ce_fixed is False
    assert trace.wave2_triggered is False
    assert "v8_ce_surgical" not in client.seen_tags


def test_v8_sound_fuzzer_promotes_correct_sibling_over_shorter_buggy_sample_passer(tmp_path: Path):
    """Two candidates both pass official samples (n=3, n=5), but the shorter one fails n=0.
    Sound brute-force fuzzer MUST promote the correct sibling without even needing extra repair calls!"""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_radically_simple":
            # Shorter code, passes n=3 and n=5, FAILS n=0 (prints 0 instead of 1)!
            return _mk_res("```python\nprint(int(input())*2)\n```", tag=spec.tag)
        if spec.tag == "w1_first_principles":
            # Slightly longer, handles n=0 properly!
            return _mk_res(
                "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(1 if n == 0 else n * 2)\n```",
                tag=spec.tag,
            )
        if spec.tag == "v8_brute_fuzzer":
            return _mk_res(_SOUND_BRUTE_FUZZER, tag=spec.tag)
        raise AssertionError(f"Sibling already passed 100% of brute tests; unexpected call: {spec.tag}")

    client = _ScriptedCodeClient(_handle)
    trace = asyncio.run(
        solve_code(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            _ABS_PLUS_ONE_PROBLEM,
            num_wave1_streams=2,
            max_code_repairs=0,
            enable_brute_fuzz=True,
        )
    )
    assert trace.verified is True
    assert trace.v8_brute_checked is True
    assert trace.winner_name == "first_principles"
    assert "1 if n == 0 else" in trace.winner_code


def test_v8_sound_fuzzer_catches_universal_boundary_bug_and_repairs_via_counterexample(tmp_path: Path):
    """When ALL sample-passing candidates share the n=0 boundary bug, sound brute-force fuzzer
    discovers the exact minimal counterexample (inp='0', exp='1', got='0') and repairs it cleanly."""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag.startswith("w1_"):
            # Both Wave-1 candidates pass samples 3 and 5, but BOTH fail n=0!
            return _mk_res("```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```", tag=spec.tag)
        if spec.tag == "v8_brute_fuzzer":
            return _mk_res(_SOUND_BRUTE_FUZZER, tag=spec.tag)
        if spec.tag == "v8_ce_surgical":
            prompt = spec.messages[-1]["content"]
            assert "MINIMAL PROVEN COUNTEREXAMPLE" in prompt
            assert "EXPECTED OUTPUT:\n```\n1\n```" in prompt
            return _mk_res(
                "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(max(1, n * 2))\n```",
                tag=spec.tag,
            )
        if spec.tag == "v8_ce_rewrite":
            # Worse rewrite that still fails n=0 -> MUST NOT beat v8_ce_surgical
            return _mk_res("```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```", tag=spec.tag)
        raise AssertionError(f"Unexpected call: {spec.tag}")

    client = _ScriptedCodeClient(_handle)
    trace = asyncio.run(
        solve_code(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            _ABS_PLUS_ONE_PROBLEM,
            num_wave1_streams=2,
            max_code_repairs=0,
            enable_brute_fuzz=True,
        )
    )
    assert trace.verified is True
    assert trace.v8_brute_checked is True
    assert trace.v8_brute_ce_fixed is True
    assert trace.winner_name == "v8_ce_surgical"
    assert "max(1, n * 2)" in trace.winner_code
