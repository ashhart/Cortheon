"""Speculative Execution-Verification (`SpecVerify`).

A test-time compute architecture that drafts parallel executable Python/SymPy/Z3
scripts on a tiny fast drafter model (e.g. 2B-9B) and verifies them mechanically
inside Cortheon's OS sandbox (`macos-seatbelt` / `bwrap` / POSIX rlimit):

  Stage 1 — Tiny Drafter Speculation (ZERO big-model calls):
    - Fires B=3 diverse executable-solver calls (`pal_exact`, `pal_alt_z3`,
      `pal_sympy`) concurrently on `drafter_client` with low latency
      (`thinking_budget=400`, `max_tokens=2048`).
    - Executes each extracted Python block inside the OS sandbox.
    - Supports Wave-1 streaming early-abort when `drafter_streams >= 3`,
      cancelling lagging drafter slots once `min_consensus` independent scripts
      cleanly pass without conflict.

  Stage 2 — Zero-LLM Mechanical Acceptance Gate:
    - Filters to scripts exiting 0 (`res.ok`) with a non-empty normalized answer
      passing `_answer_satisfies_prompt_format(problem, ans)`.
    - Clusters equivalent answers via `_answers_equivalent` (exact arbitrary-
      precision rationals — never degrading 25+ digit Fractions to float64).
    - Accepts IMMEDIATELY (zero calls to `verifier_client`) only on strict
      unique majority consensus (`>= min_consensus` scripts, `> second_best`,
      and `>= 2` distinct solver roles when `require_distinct_solvers=True`).
    - Otherwise builds a compact non-anchoring `drafter_graveyard` and
      escalates to `verifier_client` via full `solve_reasoning`.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .client import CallSpec, TensorFoldClient
from .memory import EpisodicMemory
from .reason_forge import (
    BranchOutcome,
    ReasonForgeTrace,
    _answer_satisfies_prompt_format,
    _answers_equivalent,
    _cluster_answers,
    _has_executable_assert,
    extract_answer_from_stdout,
    normalize_answer,
    solve_reasoning,
)
from .sandbox import extract_python_code, run_python
from .skill_loader import SkillPack

# Canonical aliases matching internal helper names
_normalize_answer = normalize_answer
_extract_answer_from_stdout = extract_answer_from_stdout
solve_reason = solve_reasoning

_DRAFTER_SHARED_SYS = (
    "You are a fast, exact computational solver running inside an isolated Python 3 sandbox "
    "with sympy, z3, fractions, itertools, math, and numpy available.\n"
    "Translate the problem directly into a short, exact Python script and print "
    "`ANSWER: <exact_value>` on the last line of stdout. Do NOT compute large arithmetic by hand."
)

_PAL_EXACT_FALLBACK = (
    "Solve the problem by writing a clean, exact Python script using `int`, `fractions.Fraction`, "
    "`sympy`, `math`, `itertools`, or `z3`.\n"
    "- Keep thinking brief — translate the problem directly into Python.\n"
    "- Only assert structural problem constraints (never a hand-calculated guess).\n"
    "- Print `ANSWER: <exact_value>` on the LAST line of stdout.\n"
    "- Output ONLY one concise fenced ```python``` block."
)

_PAL_ALT_Z3_FALLBACK = (
    "Solve the problem with an INDEPENDENT computational method (`z3`, exact DP, exact rational "
    "linear system, or direct enumeration) different from a single closed-form formula.\n"
    "- Keep the code compact (<85 lines) and assert problem constraints directly.\n"
    "- Print `ANSWER: <exact_value>` on the LAST line of stdout.\n"
    "- Output ONLY one concise fenced ```python``` block."
)

_PAL_SYMPY_CARD = (
    "Solve the problem symbolically and exactly using Python + `sympy` (`sp.Rational`, `sp.solve`, "
    "`sp.simplify`, `sp.Matrix`, `sp.Poly`, `sympy.ntheory`, `fractions.Fraction`).\n"
    "Rules:\n"
    "- Keep thinking brief — translate equations, recurrences, or algebraic constraints directly "
    "into compact SymPy code (<80 lines).\n"
    "- Never rely on floating-point approximations when exact symbolic or rational arithmetic is possible.\n"
    "- Assert key problem constraints on the computed solution; never assert a hand-calculated guess.\n"
    "- Format the final answer exactly as requested and print `ANSWER: <exact_value>` on the LAST line of stdout.\n"
    "- Output ONLY one concise fenced ```python``` block."
)

_PAL_SKEPTIC_FALLBACK = (
    "Re-read EVERY word of the problem for traps (inclusive/exclusive bounds, symmetry, base cases).\n"
    "- Translate EVERY constraint directly into compact exact Python (`Fraction`/`sympy`/`z3`/`int`).\n"
    "- Assert EVERY constraint holds on the final solution.\n"
    "- Print `ANSWER: <exact_value>` on the LAST line of stdout.\n"
    "- Output ONLY one concise fenced ```python``` block."
)

_SANDBOX_TMP_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/private)?/var/folders/[^'\"\s/]+/[^'\"\s/]+/T/(?:ff_exec_[^'\"\s/]+|tmp[^'\"\s/]+)|"
    r"(?<![A-Za-z0-9_.-])(?:/private)?/tmp/(?:ff_exec_[^'\"\s/]+|tmp[A-Za-z0-9_]+)"
)
_HOST_HOME_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:/Users|/home)/[^'\"\s),;:]+")


def _scrub_paths(text: str) -> str:
    if not text:
        return ""
    cleaned = _SANDBOX_TMP_RE.sub("<TMPDIR>", text)
    cleaned = _HOST_HOME_RE.sub("<HOME>", cleaned)
    return cleaned


@dataclass
class SpecVerifyTrace:
    answer: str = ""
    accepted_by_drafter: bool = False
    escalated: bool = False
    escalation_reason: str = ""
    drafter_model: str = ""
    verifier_model: str = ""
    drafter_consensus_count: int = 0
    drafter_verified_scripts: int = 0
    drafter_wall_s: float = 0.0
    verifier_wall_s: float = 0.0
    total_wall_s: float = 0.0
    drafter_calls: int = 0
    drafter_tokens: int = 0
    verifier_calls: int = 0
    verifier_tokens: int = 0
    drafter_candidates: list[dict[str, Any]] = field(default_factory=list)
    drafter_graveyard: list[str] = field(default_factory=list)
    verifier_trace: Optional[ReasonForgeTrace] = None


def _stats_snapshot(client: Any) -> dict[str, int]:
    fn = getattr(client, "stats_dict", None)
    if callable(fn):
        raw = fn()
        if isinstance(raw, dict):
            return {
                "calls": int(raw.get("calls", 0)),
                "prompt_tokens": int(raw.get("prompt_tokens", 0)),
                "completion_tokens": int(raw.get("completion_tokens", 0)),
                "reasoning_tokens": int(raw.get("reasoning_tokens", 0)),
                "cached_tokens": int(raw.get("cached_tokens", 0)),
            }
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }


def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: max(0, int(after.get(k, 0)) - int(before.get(k, 0))) for k in before}


def _client_model_name(client: Any) -> str:
    caps = getattr(client, "_caps", None)
    if caps is not None and getattr(caps, "model_id", None):
        return str(caps.model_id)
    req = getattr(client, "requested_model", None)
    if req:
        return str(req)
    model_attr = getattr(client, "model_id", None) or getattr(client, "model", None)
    if model_attr:
        return str(model_attr)
    return "unknown"


def _load_card(skills: SkillPack, name: str, fallback: str) -> str:
    try:
        return skills.card(name)
    except FileNotFoundError:
        return fallback


def _build_drafter_roles(skills: SkillPack, drafter_streams: int) -> list[tuple[str, str, float, int]]:
    base_roles: list[tuple[str, str, float, int]] = [
        ("pal_exact", _load_card(skills, "pal_exact", _PAL_EXACT_FALLBACK), 0.20, 101),
        ("pal_alt_z3", _load_card(skills, "pal_alt_z3", _PAL_ALT_Z3_FALLBACK), 0.50, 202),
        ("pal_sympy", _load_card(skills, "pal_sympy", _PAL_SYMPY_CARD), 0.35, 303),
        ("pal_skeptic", _load_card(skills, "pal_skeptic", _PAL_SKEPTIC_FALLBACK), 0.45, 404),
    ]
    if drafter_streams <= len(base_roles):
        return base_roles[:drafter_streams]
    roles = list(base_roles)
    idx = len(roles)
    while len(roles) < drafter_streams:
        b_name, b_card, b_temp, b_seed = base_roles[idx % len(base_roles)]
        rep = idx // len(base_roles) + 1
        roles.append((f"{b_name}_{rep}", b_card, min(0.75, b_temp + 0.05 * rep), b_seed + 100 * rep))
        idx += 1
    return roles


def _execute_drafter_outcome(outcome: BranchOutcome, problem: str, timeout_s: float) -> None:
    """Extract Python code, execute inside OS sandbox, normalize answer, and enforce format gate."""
    code = extract_python_code(outcome.raw_content)
    outcome.code = code
    if not code:
        outcome.error = "No valid Python code block extracted"
        return
    res = run_python(code, timeout_s=timeout_s)
    outcome.exec_result = res
    if not res.ok:
        outcome.error = _scrub_paths(res.short_error(600))
        return
    raw_ans = extract_answer_from_stdout(res.stdout)
    if not raw_ans:
        preview = _scrub_paths(res.stdout[:200])
        outcome.error = f"Script exited 0 but printed no ANSWER line (stdout={preview!r})"
        return
    norm_ans = normalize_answer(raw_ans)
    outcome.raw_answer = raw_ans
    outcome.norm_answer = norm_ans
    outcome.verified_constraints = _has_executable_assert(code)
    if not norm_ans:
        outcome.error = f"Empty normalized answer from raw={raw_ans!r}"
        return
    if norm_ans.lower() in {"nan", "inf", "-inf", "+inf"}:
        outcome.error = f"Format gate rejected non-finite answer {norm_ans!r}"
        return
    if not _answer_satisfies_prompt_format(problem, norm_ans):
        outcome.error = f"Format gate rejected normalized answer {norm_ans!r}"


def evaluate_drafter_consensus(
    outcomes: list[BranchOutcome],
    problem: str,
    *,
    min_consensus: int = 2,
    require_distinct_solvers: bool = True,
) -> tuple[bool, str, int, int, str, list[tuple[str, list[BranchOutcome]]]]:
    """Zero-LLM mechanical acceptance gate over sandbox-executed drafter scripts.

    Returns:
        (accepted, answer, consensus_count, verified_count, reason, clusters)
    """
    valid = [
        o
        for o in outcomes
        if o.exec_result is not None
        and o.exec_result.ok
        and not o.error
        and bool(o.norm_answer)
        and _answer_satisfies_prompt_format(problem, o.norm_answer)
    ]
    verified_count = len(valid)
    if verified_count == 0:
        return (
            False,
            "",
            0,
            0,
            "No drafter script passed sandbox execution and format validation",
            [],
        )

    clusters = _cluster_answers(valid)
    if not clusters:
        return (
            False,
            "",
            0,
            verified_count,
            "No clusterable normalized answers from drafter scripts",
            [],
        )

    top_ans, top_members = clusters[0]
    top_count = len(top_members)
    second_best_count = len(clusters[1][1]) if len(clusters) >= 2 else 0
    distinct_solvers = len({m.name for m in top_members})
    required_count = max(1, min_consensus)

    if top_count < required_count:
        if len(clusters) >= 2:
            summary = ", ".join(f"{ans} ({len(ms)})" for ans, ms in clusters)
            return (
                False,
                "",
                top_count,
                verified_count,
                f"Drafter scripts disagreed without consensus ({summary})",
                clusters,
            )
        return (
            False,
            "",
            top_count,
            verified_count,
            f"Insufficient drafter consensus ({top_count}/{required_count} verified scripts)",
            clusters,
        )

    if top_count <= second_best_count:
        summary = ", ".join(f"{ans} ({len(ms)})" for ans, ms in clusters)
        return (
            False,
            "",
            top_count,
            verified_count,
            f"Drafter top cluster lacked strict unique majority ({summary})",
            clusters,
        )

    if require_distinct_solvers and distinct_solvers < 2:
        return (
            False,
            "",
            top_count,
            verified_count,
            f"Drafter consensus lacked distinct solver diversity ({distinct_solvers} distinct role)",
            clusters,
        )

    return (True, top_ans, top_count, verified_count, "", clusters)


def _candidate_record(outcome: BranchOutcome, problem: str) -> dict[str, Any]:
    exec_ok = bool(outcome.exec_result and outcome.exec_result.ok)
    fmt_ok = bool(outcome.norm_answer and _answer_satisfies_prompt_format(problem, outcome.norm_answer))
    return {
        "name": outcome.name,
        "ok": bool(exec_ok and fmt_ok and not outcome.error),
        "exec_ok": exec_ok,
        "returncode": outcome.exec_result.returncode if outcome.exec_result is not None else None,
        "timed_out": bool(outcome.exec_result.timed_out) if outcome.exec_result is not None else False,
        "raw_answer": outcome.raw_answer,
        "norm_answer": outcome.norm_answer,
        "format_valid": fmt_ok,
        "verified_constraints": bool(outcome.verified_constraints),
        "error": _scrub_paths(outcome.error),
        "latency_s": round(float(outcome.latency_s), 3),
        "exec_elapsed_s": round(float(outcome.exec_result.elapsed_s), 3) if outcome.exec_result is not None else 0.0,
    }


def _extract_error_signature(err: str) -> str:
    """Extract a compact, non-anchoring execution error summary (exception class / timeout)."""
    cleaned = _scrub_paths(err).strip()
    if not cleaned:
        return ""
    if cleaned.startswith("TIMEOUT"):
        return cleaned.splitlines()[0][:140]
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    if tail.startswith("AssertionError"):
        return "AssertionError (constraint check failed)"
    # Prefer pure exception class + non-numeric summary so no wrong drafter numbers leak into prompt
    m_exc = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b(?::\s*(.*))?$", tail)
    if m_exc:
        exc_name = m_exc.group(1)
        exc_msg = (m_exc.group(2) or "").strip()
        if exc_msg and not re.search(r"\b\d{2,}\b", exc_msg):
            return f"{exc_name}: {exc_msg[:120]}"
        return exc_name
    return tail[:120] if not re.search(r"\b\d{2,}\b", tail) else "Runtime execution failure"


def _build_graveyard_and_escalation_prompt(
    problem: str,
    outcomes: list[BranchOutcome],
    clusters: list[tuple[str, list[BranchOutcome]]],
    reason: str,
) -> tuple[list[str], str]:
    """Build detailed `drafter_graveyard` tombstones and a strictly non-anchoring escalated prompt."""
    graveyard: list[str] = []
    pitfall_hints: list[str] = []

    saw_timeout = False
    saw_format_reject = False
    exec_sigs: list[str] = []

    for o in outcomes:
        if o.error == "cancelled_early_consensus":
            continue
        if o.error:
            err_clean = _scrub_paths(o.error)
            if o.exec_result is not None and o.exec_result.timed_out:
                graveyard.append(f"[{o.name}] TIMEOUT: {err_clean[:240]}")
                saw_timeout = True
            elif err_clean.startswith("Format gate rejected"):
                graveyard.append(f"[{o.name}] FORMAT_REJECTED: {err_clean[:240]}")
                saw_format_reject = True
            else:
                graveyard.append(f"[{o.name}] FAILED: {err_clean[:240]}")
                sig = _extract_error_signature(err_clean)
                if sig and not sig.startswith("No valid Python") and sig not in exec_sigs:
                    exec_sigs.append(sig)

    if len(clusters) >= 2:
        cluster_desc = "; ".join(
            f"{ans} ({', '.join(m.name for m in ms)})" for ans, ms in clusters
        )
        graveyard.append(f"[consensus] DISAGREEMENT: {cluster_desc}")
    elif reason:
        graveyard.append(f"[consensus] REJECTED: {reason}")

    # Build a brief, strictly NON-ANCHORING note for the big verifier:
    # NEVER include unverified candidate numbers from the drafter so the verifier cannot anchor on wrong values.
    if saw_timeout:
        pitfall_hints.append(
            "A fast draft script timed out; avoid slow brute-force loops and prefer exact algebra, DP, Z3, or SymPy."
        )
    if saw_format_reject:
        pitfall_hints.append(
            "A fast draft violated the requested output format; verify your final ANSWER matches the exact format required."
        )
    if exec_sigs:
        pitfall_hints.append(
            "Draft execution failures to avoid: " + " | ".join(exec_sigs[:2])
        )
    if len(clusters) >= 2:
        pitfall_hints.append(
            "Independent fast draft methods produced conflicting results; re-read every constraint carefully and verify from first principles."
        )

    base_problem = problem.strip()
    if not pitfall_hints:
        return graveyard, base_problem

    note_lines = "\n".join(f"- {h}" for h in pitfall_hints)
    escalated_prompt = (
        f"{base_problem}\n\n"
        f"[Pre-check sandbox pitfalls — solve cleanly from first principles]\n"
        f"{note_lines}"
    )
    return graveyard, escalated_prompt


def _emit(
    on_event: Optional[Callable[[dict[str, Any]], None]],
    payload: dict[str, Any],
) -> None:
    """Invoke an optional live event callback, swallowing any UI/pipe failure."""
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception:
        pass


async def solve_spec_verify(
    drafter_client: TensorFoldClient,
    verifier_client: TensorFoldClient,
    skills: SkillPack,
    memory: EpisodicMemory,
    problem: str,
    *,
    drafter_streams: int = 3,
    verifier_streams: int = 4,
    min_consensus: int = 2,
    require_distinct_solvers: bool = True,
    sandbox_timeout_s: float = 15.0,
    deep: bool = False,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> SpecVerifyTrace:
    """Solve `problem` using Speculative Execution-Verification (`SpecVerify`).

    Drafts parallel executable Python/SymPy/Z3 scripts on `drafter_client` and
    executes them in Cortheon's OS sandbox. Returns immediately without invoking
    `verifier_client` when mechanical consensus passes; otherwise escalates to
    `verifier_client` via full `solve_reasoning`.
    """
    if drafter_streams < 0:
        raise ValueError(f"drafter_streams must be >= 0, got {drafter_streams}")
    if min_consensus < 1:
        raise ValueError(f"min_consensus must be >= 1, got {min_consensus}")

    t_total_0 = time.perf_counter()
    drafter_model = _client_model_name(drafter_client)
    verifier_model = _client_model_name(verifier_client)

    # ------------------------------------------------------------------
    # Stage 1: Tiny Drafter Speculation (ZERO big-model calls yet!)
    # ------------------------------------------------------------------
    _emit(
        on_event,
        {
            "engine": "spec",
            "event": "start",
            "mode": "spec",
            "streams": drafter_streams,
        },
    )

    d_before = _stats_snapshot(drafter_client)
    t_d0 = time.perf_counter()

    drafter_outcomes: list[BranchOutcome] = []
    if drafter_streams > 0:
        lessons_block = memory.format_prompt_block(problem, domain="reason", limit=2)
        shared_sys = _DRAFTER_SHARED_SYS + (f"\n\n{lessons_block}" if lessons_block else "")

        exemplar_block = skills.match_exemplars("pal_verify", problem, limit=1)
        shared_user_prefix = f"PROBLEM:\n{problem.strip()}"
        if exemplar_block:
            shared_user_prefix += f"\n\n{exemplar_block}"

        drafter_roles = _build_drafter_roles(skills, drafter_streams)
        specs: list[CallSpec] = [
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": f"{shared_user_prefix}\n\nPROTOCOL ({name}):\n{card_text}"},
                ],
                tag=f"spec_draft_{name}",
                max_tokens=2048,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=400,
                loop_guard=True,
            )
            for name, card_text, temp, seed in drafter_roles
        ]

        async def _run_drafter_slot(
            idx: int, name: str, spec: CallSpec
        ) -> tuple[int, BranchOutcome, int, float]:
            slot_t0 = time.perf_counter()
            try:
                res = await drafter_client.complete(spec)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                slot_elapsed = time.perf_counter() - slot_t0
                return (
                    idx,
                    BranchOutcome(
                        name=name,
                        wave=1,
                        kind="pal",
                        raw_content="",
                        reasoning="",
                        error=_scrub_paths(f"{type(exc).__name__}: {exc}"),
                    ),
                    0,
                    slot_elapsed,
                )
            tok = int(getattr(res, "completion_tokens", 0) or 0)
            out = BranchOutcome(
                name=name,
                wave=1,
                kind="pal",
                raw_content=res.content,
                reasoning=res.reasoning,
                latency_s=res.latency_s,
                error=_scrub_paths(res.raw_error or ""),
            )
            if res.ok:
                await asyncio.to_thread(_execute_drafter_outcome, out, problem, sandbox_timeout_s)
            slot_elapsed = max(time.perf_counter() - slot_t0, float(out.latency_s or 0.0))
            return idx, out, tok, slot_elapsed

        slot_tasks: list[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = [
            asyncio.create_task(_run_drafter_slot(i, name, spec))
            for i, ((name, _, _, _), spec) in enumerate(zip(drafter_roles, specs))
        ]
        ordered_outcomes: list[Optional[BranchOutcome]] = [None] * len(drafter_roles)
        early_aborted = False

        pending: set[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = set(slot_tasks)
        try:
            while pending:
                done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                # Always unpack every completed task in this wakeup before checking early-abort
                # so already-completed disagreements are never missed.
                completed_batch: list[tuple[int, BranchOutcome, int, float]] = [
                    done_task.result() for done_task in done_set
                ]
                completed_batch.sort(key=lambda item: item[0])
                for idx, out, tok, slot_elapsed in completed_batch:
                    ordered_outcomes[idx] = out
                    exec_ok = bool(out.exec_result and out.exec_result.ok)
                    fmt_ok = bool(out.norm_answer and _answer_satisfies_prompt_format(problem, out.norm_answer))
                    ok_flag = bool(exec_ok and fmt_ok and not out.error)
                    if ok_flag:
                        summary = "PAL verified"
                    elif out.error:
                        summary = _scrub_paths(out.error).splitlines()[0][:80]
                    elif out.norm_answer and not fmt_ok:
                        summary = "format gate rejected"
                    else:
                        summary = "draft failed"
                    _emit(
                        on_event,
                        {
                            "engine": "spec",
                            "event": "slot_done",
                            "slot": idx + 1,
                            "role": out.name,
                            "ok": ok_flag,
                            "elapsed_s": round(max(0.0, float(slot_elapsed)), 2),
                            "tokens": tok,
                            "answer": out.norm_answer or "",
                            "summary": summary,
                        },
                    )

                if drafter_streams >= 3 and pending:
                    completed_so_far = [o for o in ordered_outcomes if o is not None]
                    accepted_now, ans_now, top_cnt, _, _, clusters_now = evaluate_drafter_consensus(
                        completed_so_far,
                        problem,
                        min_consensus=min_consensus,
                        require_distinct_solvers=require_distinct_solvers,
                    )
                    if accepted_now and len(clusters_now) == 1 and top_cnt >= min_consensus:
                        early_aborted = True
                        n_cancelled = len(pending)
                        for lag in pending:
                            lag.cancel()
                        _emit(
                            on_event,
                            {
                                "engine": "spec",
                                "event": "early_abort",
                                "cancelled": n_cancelled,
                                "elapsed_s": round(time.perf_counter() - t_d0, 2),
                                "consensus_answer": ans_now,
                                "reason": f"{top_cnt}/{drafter_streams} drafter PAL consensus",
                            },
                        )
                        break

            if early_aborted and pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except BaseException:
            for lag in pending:
                lag.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise

        for i, (name, _, _, _) in enumerate(drafter_roles):
            out_i = ordered_outcomes[i]
            if out_i is None:
                out_i = BranchOutcome(
                    name=name,
                    wave=1,
                    kind="pal",
                    raw_content="",
                    reasoning="",
                    error="cancelled_early_consensus",
                )
            drafter_outcomes.append(out_i)

    drafter_wall_s = time.perf_counter() - t_d0
    d_after = _stats_snapshot(drafter_client)
    d_delta = _stats_delta(d_before, d_after)
    drafter_model = _client_model_name(drafter_client)
    drafter_calls = d_delta["calls"]
    drafter_tokens = d_delta["prompt_tokens"] + d_delta["completion_tokens"]
    drafter_candidates = [_candidate_record(o, problem) for o in drafter_outcomes]

    # ------------------------------------------------------------------
    # Stage 2: Zero-LLM Mechanical Acceptance Gate
    # ------------------------------------------------------------------
    accepted, accepted_ans, consensus_count, verified_count, reject_reason, clusters = (
        evaluate_drafter_consensus(
            drafter_outcomes,
            problem,
            min_consensus=min_consensus,
            require_distinct_solvers=require_distinct_solvers,
        )
    )

    if accepted:
        total_wall_s = time.perf_counter() - t_total_0
        trace = SpecVerifyTrace(
            answer=accepted_ans,
            accepted_by_drafter=True,
            escalated=False,
            escalation_reason="",
            drafter_model=drafter_model,
            verifier_model=verifier_model,
            drafter_consensus_count=consensus_count,
            drafter_verified_scripts=verified_count,
            drafter_wall_s=drafter_wall_s,
            verifier_wall_s=0.0,
            total_wall_s=total_wall_s,
            drafter_calls=drafter_calls,
            drafter_tokens=drafter_tokens,
            verifier_calls=0,
            verifier_tokens=0,
            drafter_candidates=drafter_candidates,
            drafter_graveyard=[],
            verifier_trace=None,
        )
        _emit(
            on_event,
            {
                "engine": "spec",
                "event": "drafter_accepted",
                "drafter_model": trace.drafter_model,
                "answer": trace.answer,
                "elapsed_s": round(trace.drafter_wall_s, 2),
                "verifier_calls_saved": verifier_streams,
            },
        )
        _emit(
            on_event,
            {
                "engine": "spec",
                "event": "done",
                "elapsed_s": round(trace.total_wall_s, 2),
                "verified": True,
                "summary": f"drafter_accepted ({trace.drafter_consensus_count}/{drafter_streams} PAL) -> {trace.answer}",
            },
        )
        return trace

    # ------------------------------------------------------------------
    # Escalation: Sandbox falsified drafter -> wake big verifier model
    # ------------------------------------------------------------------
    escalation_reason = reject_reason or "Drafter failed mechanical acceptance gate"
    if drafter_streams == 0:
        escalation_reason = "drafter_streams=0; escalated directly to verifier"

    drafter_graveyard, escalated_prompt = _build_graveyard_and_escalation_prompt(
        problem,
        drafter_outcomes,
        clusters,
        escalation_reason,
    )

    _emit(
        on_event,
        {
            "engine": "spec",
            "event": "escalated",
            "reason": escalation_reason,
            "verifier_model": verifier_model,
        },
    )

    v_before = _stats_snapshot(verifier_client)
    t_v0 = time.perf_counter()
    v_trace = await solve_reasoning(
        verifier_client,
        skills,
        memory,
        escalated_prompt,
        num_wave1_streams=verifier_streams,
        sandbox_timeout_s=sandbox_timeout_s,
        deep=deep,
        on_event=on_event,
    )
    verifier_wall_s = time.perf_counter() - t_v0
    v_after = _stats_snapshot(verifier_client)
    v_delta = _stats_delta(v_before, v_after)

    verifier_model = _client_model_name(verifier_client)
    verifier_calls = v_delta["calls"]
    verifier_tokens = v_delta["prompt_tokens"] + v_delta["completion_tokens"]
    final_answer = normalize_answer(v_trace.final_answer) or v_trace.final_answer
    total_wall_s = time.perf_counter() - t_total_0

    trace = SpecVerifyTrace(
        answer=final_answer,
        accepted_by_drafter=False,
        escalated=True,
        escalation_reason=escalation_reason,
        drafter_model=drafter_model,
        verifier_model=verifier_model,
        drafter_consensus_count=consensus_count,
        drafter_verified_scripts=verified_count,
        drafter_wall_s=drafter_wall_s,
        verifier_wall_s=verifier_wall_s,
        total_wall_s=total_wall_s,
        drafter_calls=drafter_calls,
        drafter_tokens=drafter_tokens,
        verifier_calls=verifier_calls,
        verifier_tokens=verifier_tokens,
        drafter_candidates=drafter_candidates,
        drafter_graveyard=drafter_graveyard,
        verifier_trace=v_trace,
    )
    _emit(
        on_event,
        {
            "engine": "spec",
            "event": "done",
            "elapsed_s": round(trace.total_wall_s, 2),
            "verified": bool(v_trace.verified),
            "summary": f"escalated ({v_trace.consensus_type}) -> {trace.answer}",
        },
    )
    return trace
