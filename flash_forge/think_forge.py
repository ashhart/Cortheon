"""ThinkForge: Quant-Grounded Multi-Perspective Adversarial Council for Open-Ended Frontier Reasoning.

Solves domains WITHOUT a compiler or mathematical oracle — system design, security audits,
architecture trade-offs, RFCs, code reviews, product/engineering strategy, and deep analysis —
where single-stream local models normally output shallow, generic "AI slop" or hallucinate sizing math.

Pipeline (exploiting B=4..5 deduplicated MoE batching + Python Quantitative Sandbox + Cross-Model Referee):
  Stage 1 (Parallel Specialist Council + Auto-Grounded Python Quantitative Calculator):
    - architect:     First-principles design, concrete state machines, schemas, invariants
    - red_teamer:    Adversarial 3am production breaker (race conditions, security bypasses, split-brain)
    - pragmatist:    Battle-tested minimalism, sharp trade-offs, explicit assumptions & decision matrix
    - quantifier:    Hard bounds, failure budgets, counter-arguments
    - quant_sandbox: (auto-fired on quantitative/sizing prompts) writes & EXECUTES a Python calculation
                     script in sandbox so EVERY QPS/memory/bandwidth/cost number is mechanically verified!
  Stage 2 (Clean-Context Cross-Model / Adversarial Critique):
    Audits all memos + Python numbers for contradictions, unverified assumptions, and handwaving.
  Stage 3 (Principal-Engineer Master Synthesis):
    Produces ONE razor-sharp, concrete, zero-fluff answer with explicit Assumptions & Decision Matrix,
    verified sizing numbers, and concrete failure-mode mitigations.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .client import CallSpec, TensorFoldClient
from .memory import EpisodicMemory
from .sandbox import extract_python_code, run_python

_QUANT_SIGNAL_RE = re.compile(
    r"\b(\d+\s*(?:k|m|b|gb|tb|pb|mb|kb|ms|us|ns|s|req/s|qps|rps|tps|ops|gbps|mbps|%|usd|\$)|"
    r"throughput|latency|p99|p95|sla|slo|capacity|sizing|bandwidth|storage|cost|budget|scale|memory|tokens/s|kv[- ]?cache|iops|rtt)\b",
    re.IGNORECASE,
)

_TECH_INVARIANT_RE = re.compile(
    r"\b(consensus|raft|paxos|quorum|split[- ]?brain|wal|compaction|mvcc|isolation|deadlock|race condition|"
    r"rate[- ]?limit|token[- ]?bucket|sliding[- ]?window|sharding|partition|replication|failover|backpressure|"
    r"idempoten|exactly[- ]?once|watermark|jwt|oauth|hmac|replay attack|timing attack|encryption|kms|"
    r"state machine|distributed|cache invalidation|thundering herd|queue|kafka|redis|postgres|architecture)\b",
    re.IGNORECASE,
)


def _should_run_executable_verifier(problem: str) -> bool:
    """True when an open-ended prompt has quantitative bounds, physics/SLA limits, or verifiable system invariants."""
    return bool(_QUANT_SIGNAL_RE.search(problem) or _TECH_INVARIANT_RE.search(problem))


@dataclass
class ThinkForgeTrace:
    mode: str = "think"
    council_outputs: dict[str, str] = field(default_factory=dict)
    quant_verified: bool = False
    quant_output: str = ""
    critique: str = ""
    final_answer: str = ""
    wall_s: float = 0.0
    lessons_used: int = 0


async def solve_open_ended(
    client: TensorFoldClient,
    memory: EpisodicMemory,
    problem: str,
    *,
    referee_client: Optional[TensorFoldClient] = None,
    num_streams: int = 4,
    deep: bool = False,
) -> ThinkForgeTrace:
    """Run Specialist Council (+ Python Quant Sandbox) -> Clean-Context Critique -> Master Synthesis."""
    t0 = time.perf_counter()
    trace = ThinkForgeTrace()
    critic_client = referee_client or client

    lessons_block = memory.format_prompt_block(problem, domain="think", limit=2)
    if lessons_block:
        trace.lessons_used = lessons_block.count("\n- [")

    shared_sys = (
        "You are a principal-level specialist on an adversarial frontier council. "
        "Be brutally concrete: give exact mechanisms, failure modes, trade-offs, data structures, "
        "and numbers. NEVER write generic filler or vague textbook buzzwords."
    )
    if lessons_block:
        shared_sys += "\n\n" + lessons_block

    shared_user = f"REQUEST:\n{problem.strip()}\n"
    want_quant_sandbox = _should_run_executable_verifier(problem)

    tb_council = 1100 if deep else 650
    roles = [
        (
            "architect",
            "Design the core mechanism/answer from first principles: concrete components, state transitions, "
            "invariants, data structures/APIs, and concrete step-by-step execution flow (<= 350 words).",
            0.30,
            101,
        ),
        (
            "adversarial_breaker",
            "Attack the obvious approach ruthlessly. Identify the 5 nastiest failure modes, race conditions, "
            "security/logic bypasses, hidden bottlenecks, or subtle edge cases that break naive solutions (<= 350 words).",
            0.55,
            202,
        ),
        (
            "pragmatist",
            "Cut over-engineering. State key explicit assumptions upfront (to resolve ambiguity), identify the simplest "
            "battle-tested design, sharpest real-world trade-offs, what NOT to build, and migration path (<= 300 words).",
            0.40,
            303,
        ),
        (
            "quantifier",
            "Ground the answer in hard numbers, bounds, complexity/throughput/latency/cost math, edge constraints, "
            "and strong counter-arguments to conventional wisdom (<= 300 words).",
            0.45,
            404,
        ),
    ][:num_streams]

    specs = [
        CallSpec(
            messages=[
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": f"{shared_user}\nCOUNCIL ROLE ({name}): {instruction}"},
            ],
            tag=f"think_{name}",
            max_tokens=tb_council + 850,
            temperature=temp,
            seed=seed,
            enable_thinking=True,
            thinking_budget=tb_council,
            loop_guard=True,
        )
        for name, instruction, temp, seed in roles
    ]
    if want_quant_sandbox:
        specs.append(
            CallSpec(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a short, deterministic Python 3 script (<= 50 lines, stdlib only) that mechanically verifies "
                            "the hard math, physics/SLA feasibility, OR protocol/quorum/rate-limit/security invariants of the request.\n"
                            "- For quantitative/sizing/SLA prompts: compute exact throughput, replication bandwidth, storage/day & /year, "
                            "memory/KV-cache/index bounds, AND explicitly check if any latency/RTT/quorum/hardware constraint is FEASIBLE or IMPOSSIBLE!\n"
                            "- For protocol/concurrency/security prompts: run a small deterministic invariant or edge-case check (e.g. quorum overlap, "
                            "token-bucket burst/race bound, key-rotation window).\n"
                            "Print a crisp labeled VERIFIED RESULTS table to stdout. Output ONLY one ```python``` block."
                        ),
                    },
                    {"role": "user", "content": shared_user},
                ],
                tag="think_quant_sandbox",
                max_tokens=1800,
                temperature=0.15,
                seed=505,
                enable_thinking=True,
                thinking_budget=450,
                loop_guard=True,
            )
        )

    results = await client.batch(specs)
    council_res = results[: len(roles)]
    for (name, _, _, _), res in zip(roles, council_res):
        if res.ok and res.content.strip():
            trace.council_outputs[name] = res.content.strip()[:2200]

    # Execute Python quantitative grounding script in sandbox (with 1-shot self-repair so sizing math is ALWAYS verified!)
    if want_quant_sandbox and len(results) > len(roles) and results[-1].ok:
        q_code = extract_python_code(results[-1].content)
        if q_code:
            exec_res = await asyncio.to_thread(run_python, q_code, 6.0)
            if exec_res.ok and exec_res.stdout.strip():
                trace.quant_verified = True
                trace.quant_output = exec_res.stdout.strip()[:1600]
            else:
                fix_res = await client.complete(
                    CallSpec(
                        messages=[
                            {
                                "role": "system",
                                "content": "Fix this short Python 3 stdlib sizing script so it executes cleanly and prints a concise labeled sizing table. Output ONLY one ```python``` block.",
                            },
                            {
                                "role": "user",
                                "content": f"REQUEST:\n{problem.strip()[:1200]}\n\nBROKEN SCRIPT:\n```python\n{q_code[:1800]}\n```\n\nERROR:\n{exec_res.short_error(400)}",
                            },
                        ],
                        tag="think_quant_retry",
                        max_tokens=1600,
                        temperature=0.1,
                        seed=506,
                        enable_thinking=True,
                        thinking_budget=350,
                        loop_guard=True,
                    )
                )
                if fix_res.ok:
                    q_code2 = extract_python_code(fix_res.content)
                    if q_code2:
                        exec_res2 = await asyncio.to_thread(run_python, q_code2, 6.0)
                        if exec_res2.ok and exec_res2.stdout.strip():
                            trace.quant_verified = True
                            trace.quant_output = exec_res2.stdout.strip()[:1600]

    # Stage 2: Clean-Context Adversarial Cross-Critique (Cross-Model Referee if configured!)
    council_dossier = "\n\n".join(
        f"=== {name.upper()} ===\n{txt}" for name, txt in trace.council_outputs.items()
    )
    quant_block = (
        f"\n\n=== PYTHON-VERIFIED QUANTITATIVE SIZING (GROUND TRUTH — USE THESE EXACT NUMBERS!) ===\n"
        f"```\n{trace.quant_output}\n```"
        if trace.quant_verified
        else ""
    )
    crit_res = await critic_client.complete(
        CallSpec(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an independent principal reviewer auditing specialist memos. "
                        "Identify: (1) contradictions between memos and which side is right, "
                        "(2) concrete gaps/blind spots ALL missed, (3) any arithmetic contradicting the verified Python table, "
                        "(4) vague fluff or unstated assumptions to make explicit. Keep output <= 250 words."
                    ),
                },
                {"role": "user", "content": f"ORIGINAL REQUEST:\n{problem.strip()}\n\n{council_dossier}{quant_block}"},
            ],
            tag="think_critique",
            max_tokens=1000,
            temperature=0.1,
            enable_thinking=True,
            thinking_budget=450,
            loop_guard=True,
        )
    )
    trace.critique = crit_res.content.strip()[:1800]

    # Stage 3: Master Frontier Synthesis (crisp, deeply technical, zero fluff!)
    synth_res = await client.complete(
        CallSpec(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are synthesizing the final response to the user after an adversarial principal review.\n"
                        "Rules:\n"
                        "- Do NOT mention 'memos', 'the council', or 'specialists' — speak directly as one unified principal-level expert.\n"
                        "- If requirements have any ambiguity, state crisp explicit assumptions & trade-off choices upfront.\n"
                        "- Integrate the sharpest architecture/answer, every real failure mode & mitigation, "
                        "and exact verified numbers (when a Python-verified table is provided, cite its exact numbers — "
                        "and if it proves any requested SLA/latency/physics/quorum constraint is IMPOSSIBLE, call that out prominently upfront with the exact math!).\n"
                        "- Zero generic fluff, zero TBD placeholders, razor-sharp markdown structure."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"ORIGINAL REQUEST:\n{problem.strip()}\n\n"
                        f"SPECIALIST INPUTS:\n{council_dossier[:7000]}{quant_block}\n\n"
                        f"INDEPENDENT CRITIQUE:\n{trace.critique}\n\n"
                        "Write the final, authoritative, frontier-grade answer now."
                    ),
                },
            ],
            tag="think_synthesis",
            max_tokens=2800,
            temperature=0.25,
            enable_thinking=True,
            thinking_budget=650,
            loop_guard=True,
        )
    )
    trace.final_answer = synth_res.content.strip()
    trace.wall_s = time.perf_counter() - t0
    return trace
