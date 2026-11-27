"""Unified FlashForge orchestrator: routing, parallel verification, repo worktree mode,
and zero-weight episodic reflexion learning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .client import CallSpec, TensorFoldClient
from .code_forge import CodeCandidate, CodeForgeTrace, solve_code
from .flywheel import record_live_trace
from .memory import EpisodicMemory
from .reason_forge import ReasonForgeTrace, _answers_equivalent, solve_reasoning
from .router import TaskMode, classify_task
from .sandbox import extract_python_code, run_in_ephemeral_worktree
from .schemas import LESSON_DISTILL_SCHEMA
from .skill_loader import SkillPack
from .think_forge import ThinkForgeTrace, solve_open_ended


@dataclass
class ForgeConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    referee_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    num_streams: int = 4
    deep: bool = False
    enable_memory: bool = True
    memory_db: Optional[Path] = None
    sandbox_timeout_s: float = 10.0
    record_flywheel: bool = False
    flywheel_path: Optional[Path] = None


@dataclass
class ForgeResult:
    mode: TaskMode
    answer: str
    code: str = ""
    raw_trace: Any = None
    wall_s: float = 0.0
    token_stats: dict[str, int] = field(default_factory=dict)
    lesson_learned: Optional[str] = None
    verified: bool = True
    flywheel_recorded: bool = False


SolveResult = ForgeResult


def _is_failing_code_candidate(cand: CodeCandidate) -> bool:
    return bool(
        cand.syntax_error
        or cand.fatal_error
        or cand.redteam_fatal
        or cand.failed_tests
        or cand.redteam_failures
        or (cand.total_checks > 0 and not cand.perfect)
    )


def _format_code_failure_reason(cand: CodeCandidate) -> str:
    if cand.syntax_error:
        return f"syntax_error: {cand.syntax_error[:200]}"
    if cand.fatal_error:
        return f"fatal_error: {cand.fatal_error[:200]}"
    if cand.redteam_fatal:
        return f"redteam_fatal: {cand.redteam_fatal[:200]}"
    if cand.redteam_failures:
        return f"redteam_failures ({cand.name}): {', '.join(cand.redteam_failures[:4])}"
    if cand.failed_tests:
        return f"failed_tests ({cand.name}): {', '.join(cand.failed_tests[:4])}"
    if cand.total_checks > 0 and not cand.perfect:
        return f"partial_pass ({cand.name}): {cand.total_passed}/{cand.total_checks}"
    return f"wave1_failed ({cand.name})"


def _extract_reason_flywheel_pair(
    trace: ReasonForgeTrace,
) -> Optional[tuple[str, str, str, dict[str, Any]]]:
    """Return `(chosen, rejected, rejected_reason, metadata)` for a Wave-2 ReasonForge rescue."""
    if not trace.wave2_triggered:
        return None
    final_ans = (trace.final_answer or "").strip()
    chosen = (trace.final_explanation or "").strip() or final_ans
    if not chosen:
        return None

    consensus_low = (trace.consensus_type or "").lower()
    w2_consensus = bool(
        final_ans
        and consensus_low
        and consensus_low != "fallback_extract"
        and (
            "wave2" in consensus_low
            or "consensus" in consensus_low
            or "pairwise_tiebreak" in consensus_low
        )
    )
    if not (trace.verified or w2_consensus):
        return None

    rejected = ""
    rejected_reason = ""
    rejected_branch = ""

    # Pass 1: Prefer a Wave-1 branch with a distinct wrong answer
    for o in trace.wave1_outcomes:
        o_ans = (o.norm_answer or o.raw_answer or "").strip()
        if o_ans and final_ans and not _answers_equivalent(o_ans, final_ans):
            cand = (o.raw_content or "").strip() or (o.code or "").strip() or f"ANSWER: {o_ans}"
            if cand and cand != chosen:
                rejected = cand
                rejected_branch = o.name
                rejected_reason = f"wave1_wrong_answer ({o.name}: got={o_ans!r}, winner={final_ans!r})"
                break

    # Pass 2: Wave-1 branch that errored / failed sandbox execution / produced no answer
    if not rejected:
        for o in trace.wave1_outcomes:
            exec_failed = o.exec_result is not None and not o.exec_result.ok
            no_ans = not (o.norm_answer or o.raw_answer or "").strip()
            if o.error or exec_failed or no_ans:
                err_txt = (
                    (o.error or "").strip()
                    or (o.exec_result.short_error(250) if exec_failed and o.exec_result else "")
                )
                cand = (o.raw_content or "").strip() or (o.code or "").strip() or err_txt
                if cand and cand != chosen:
                    rejected = cand
                    rejected_branch = o.name
                    rejected_reason = err_txt or f"wave1_failed ({o.name})"
                    break

    # Pass 3: Graveyard fallback when wave1_outcomes had no extractable rejected text
    if not rejected and trace.graveyard:
        gv = "\n".join(g.strip() for g in trace.graveyard[:4] if g and g.strip()).strip()
        if gv and gv != chosen:
            rejected = gv
            rejected_reason = "wave1_graveyard"

    if not rejected or rejected == chosen:
        return None

    metadata = {
        "mode": "reason",
        "wave2_triggered": bool(trace.wave2_triggered),
        "wave2_reason": trace.wave2_reason,
        "consensus_type": trace.consensus_type,
        "final_answer": final_ans,
        "trace_verified": bool(trace.verified),
        "recovered_from_disagreement": bool(trace.recovered_from_disagreement),
        "rejected_branch": rejected_branch,
    }
    return chosen, rejected, rejected_reason, metadata


def _extract_code_flywheel_pair(
    trace: CodeForgeTrace,
) -> Optional[tuple[str, str, str, dict[str, Any]]]:
    """Return `(chosen, rejected, rejected_reason, metadata)` for a Red-Team/Wave-2 CodeForge rescue."""
    if not trace.recovered_by_redteam_or_wave2:
        return None
    chosen = (trace.winner_code or "").strip()
    if not chosen:
        return None

    failing_w1 = [c for c in trace.wave1_candidates if _is_failing_code_candidate(c)]
    rejected = ""
    rejected_reason = ""
    rejected_name = ""

    # Pass 1: Prefer a failing Wave-1 candidate with distinct executable code
    for cand in failing_w1:
        cand_code = (cand.code or "").strip()
        if cand_code and cand_code != chosen:
            rejected = cand_code
            rejected_name = cand.name
            rejected_reason = _format_code_failure_reason(cand)
            break

    # Pass 2: Fallback to raw model output if failing candidate had no extracted code block
    if not rejected:
        for cand in failing_w1:
            cand_raw = (cand.raw_content or "").strip()
            if cand_raw and cand_raw != chosen:
                rejected = cand_raw
                rejected_name = cand.name
                rejected_reason = _format_code_failure_reason(cand)
                break

    if not rejected or rejected == chosen:
        return None

    metadata = {
        "mode": "code",
        "winner_name": trace.winner_name,
        "wave2_triggered": bool(trace.wave2_triggered),
        "wave2_reason": trace.wave2_reason,
        "redteam_valid_kills": list(trace.redteam_valid_kills),
        "trace_verified": bool(trace.verified),
        "recovered_by_redteam_or_wave2": bool(trace.recovered_by_redteam_or_wave2),
        "rejected_candidate": rejected_name,
    }
    return chosen, rejected, rejected_reason, metadata


class FlashForge:
    """High-level frozen-weights reasoning environment for Qwen3.8-Flash-Next."""

    def __init__(self, config: Optional[ForgeConfig] = None) -> None:
        self.config = config or ForgeConfig()
        self.client = TensorFoldClient(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key=self.config.api_key,
            max_parallel=max(8, self.config.num_streams),
        )
        self.referee_client = (
            TensorFoldClient(base_url=self.config.referee_url, api_key=self.config.api_key)
            if self.config.referee_url
            else None
        )
        self.skills = SkillPack()
        self.memory = EpisodicMemory(
            db_path=self.config.memory_db,
            enabled=self.config.enable_memory,
        )

    async def __aenter__(self) -> "FlashForge":
        await self.client.probe()
        if self.referee_client:
            await self.referee_client.probe()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self.client.close()
        if self.referee_client:
            await self.referee_client.close()
        self.memory.close()

    async def _distill_lesson(self, domain: str, problem: str, graveyard: list[str], fix_note: str) -> Optional[str]:
        if not self.config.enable_memory or not graveyard:
            return None
        prompt = (
            f"DOMAIN: {domain}\n"
            f"TASK SUMMARY: {problem[:600]}\n"
            f"WHAT FAILED IN WAVE 1:\n" + "\n".join(graveyard[:4]) + "\n"
            f"HOW WAVE 2 / VERIFICATION FIXED IT:\n{fix_note[:600]}\n\n"
            "Distill ONE short, generalizable engineering/reasoning lesson (NOT problem-specific constants!) "
            "that prevents this mistake class in future tasks."
        )
        res = await self.client.complete(
            CallSpec(
                messages=[{"role": "user", "content": prompt}],
                tag="distill_lesson",
                max_tokens=220,
                temperature=0.0,
                enable_thinking=False,
                json_schema=LESSON_DISTILL_SCHEMA,
                schema_name="lesson",
            )
        )
        obj = res.json_payload()
        if not obj:
            return None
        rule = str(obj.get("verified_rule", "")).strip()
        if len(rule) < 12:
            return None
        self.memory.add(
            domain=domain,
            symptom=str(obj.get("symptom", "")),
            dead_end=str(obj.get("dead_end", "")),
            verified_rule=rule,
        )
        return rule

    def _current_model_id(self) -> str:
        caps = getattr(self.client, "_caps", None)
        if caps is not None and getattr(caps, "model_id", ""):
            return str(caps.model_id)
        return str(self.config.model or getattr(self.client, "requested_model", None) or "")

    def _maybe_record_flywheel(self, prompt: str, trace: Any) -> bool:
        """Record a verified Wave-2/Red-Team rescue pair into the local flywheel JSONL store.

        Never raises: any I/O or extraction failure is swallowed so solves never crash.
        """
        if not self.config.record_flywheel:
            return False
        try:
            if isinstance(trace, ReasonForgeTrace):
                extracted = _extract_reason_flywheel_pair(trace)
                domain = "math"
                source = "live_reason_forge"
            elif isinstance(trace, CodeForgeTrace):
                extracted = _extract_code_flywheel_pair(trace)
                domain = "code"
                source = "live_code_forge"
            else:
                return False

            if extracted is None:
                return False

            chosen, rejected, rejected_reason, metadata = extracted
            log_path = (
                Path(self.config.flywheel_path)
                if self.config.flywheel_path is not None
                else Path.home() / ".cortheon" / "flywheel.jsonl"
            )
            record_live_trace(
                log_path,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                domain=domain,
                model=self._current_model_id(),
                chosen_verified=True,
                rejected_reason=rejected_reason,
                source=source,
                metadata=metadata,
            )
            return True
        except Exception:
            return False

    async def solve(
        self,
        prompt: str,
        *,
        force_mode: Optional[TaskMode] = None,
        external_test_code: Optional[str] = None,
        learn: bool = True,
        on_event: Optional[Any] = None,
    ) -> ForgeResult:
        t0 = time.perf_counter()
        self.client.reset_stats()
        await self.client.probe()

        def _emit(ev: dict[str, Any]) -> None:
            if on_event is None:
                return
            try:
                on_event(ev)
            except Exception:
                pass

        mode: TaskMode = force_mode or await classify_task(self.client, prompt)
        _emit({"engine": mode, "event": "start", "mode": mode, "streams": self.config.num_streams})

        if mode == "fast":
            res = await self.client.complete(
                CallSpec(
                    messages=[{"role": "user", "content": prompt}],
                    tag="fast_pass",
                    max_tokens=2048,
                    temperature=0.3,
                    enable_thinking=False,
                    loop_guard=True,
                )
            )
            wall_s = time.perf_counter() - t0
            _emit({"engine": "fast", "event": "done", "elapsed_s": wall_s, "summary": "fast pass complete", "verified": False})
            return ForgeResult(
                mode="fast",
                answer=res.content,
                wall_s=wall_s,
                token_stats=self.client.stats_dict(),
            )

        if mode == "code":
            trace: CodeForgeTrace = await solve_code(
                client=self.client,
                skills=self.skills,
                memory=self.memory,
                problem=prompt,
                num_wave1_streams=self.config.num_streams,
                sandbox_timeout_s=self.config.sandbox_timeout_s,
                deep=self.config.deep,
                external_test_code=external_test_code,
                on_event=on_event,
            )
            lesson = None
            if learn and trace.recovered_by_redteam_or_wave2 and trace.graveyard:
                lesson = await self._distill_lesson(
                    "code",
                    prompt,
                    trace.graveyard,
                    f"Winner `{trace.winner_name}` passed checks after Red-Team/Wave-2 repair.",
                )
            flywheel_recorded = self._maybe_record_flywheel(prompt, trace)
            formatted = f"```python\n{trace.winner_code}\n```"
            return ForgeResult(
                mode="code",
                answer=formatted,
                code=trace.winner_code,
                raw_trace=trace,
                wall_s=time.perf_counter() - t0,
                token_stats=self.client.stats_dict(),
                lesson_learned=lesson,
                verified=trace.verified,
                flywheel_recorded=flywheel_recorded,
            )

        if mode == "think":
            ttrace: ThinkForgeTrace = await solve_open_ended(
                client=self.client,
                memory=self.memory,
                problem=prompt,
                referee_client=self.referee_client,
                num_streams=self.config.num_streams,
                deep=self.config.deep,
            )
            return ForgeResult(
                mode="think",
                answer=ttrace.final_answer,
                raw_trace=ttrace,
                wall_s=time.perf_counter() - t0,
                token_stats=self.client.stats_dict(),
            )

        # mode == "reason"
        rtrace: ReasonForgeTrace = await solve_reasoning(
            client=self.client,
            skills=self.skills,
            memory=self.memory,
            problem=prompt,
            num_wave1_streams=self.config.num_streams,
            sandbox_timeout_s=self.config.sandbox_timeout_s,
            deep=self.config.deep,
        )
        lesson = None
        if learn and rtrace.recovered_from_disagreement and rtrace.graveyard:
            lesson = await self._distill_lesson(
                "reason",
                prompt,
                rtrace.graveyard,
                f"Resolved to `{rtrace.final_answer}` via {rtrace.consensus_type}.",
            )
        flywheel_recorded = self._maybe_record_flywheel(prompt, rtrace)
        return ForgeResult(
            mode="reason",
            answer=rtrace.final_answer,
            raw_trace=rtrace,
            wall_s=time.perf_counter() - t0,
            token_stats=self.client.stats_dict(),
            lesson_learned=lesson,
            verified=rtrace.verified,
            flywheel_recorded=flywheel_recorded,
        )

    async def solve_repo_file(
        self,
        repo_dir: str | Path,
        target_rel_path: str,
        task_prompt: str,
        verify_cmd: str,
        *,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Parallel role-seeded patching in isolated ephemeral git worktrees with hard reset."""
        t0 = time.perf_counter()
        self.client.reset_stats()
        await self.client.probe()

        repo = Path(repo_dir).resolve()
        orig_content = (repo / target_rel_path).read_text(encoding="utf-8")

        shared_sys = (
            "You are a principal engineer patching a real repository file. "
            "Output ONLY the complete updated file inside a fenced code block. "
            "Never truncate functions or leave placeholders."
        )
        base_user = (
            f"TASK:\n{task_prompt}\n\n"
            f"TARGET FILE: `{target_rel_path}`\n"
            f"VERIFY COMMAND: `{verify_cmd}`\n\n"
            f"CURRENT FILE CONTENT:\n```python\n{orig_content}\n```"
        )

        roles = [
            ("minimal_patch", self.skills.card("race_minimal"), 0.20, 101),
            ("root_cause", self.skills.card("race_first_principles"), 0.55, 202),
            ("skeptic_patch", self.skills.card("race_skeptic"), 0.45, 303),
            ("alt_patch", self.skills.card("race_alt"), 0.65, 404),
        ][: self.config.num_streams]

        specs = [
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": f"{base_user}\n\nROLE ({name}):\n{card}"},
                ],
                tag=f"repo_w1_{name}",
                max_tokens=4096,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=1500,
                loop_guard=True,
            )
            for name, card, temp, seed in roles
        ]
        results = await self.client.batch(specs)

        graveyard: list[str] = []
        passing: list[tuple[str, str, float]] = []
        for (name, _, _, _), res in zip(roles, results):
            code = extract_python_code(res.content)
            if not code:
                graveyard.append(f"[{name}] No code block produced")
                continue
            exec_res = run_in_ephemeral_worktree(repo, {target_rel_path: code}, verify_cmd, timeout_s=timeout_s)
            if exec_res.ok:
                passing.append((name, code, exec_res.elapsed_s))
            else:
                graveyard.append(f"[{name}] verify failed: {exec_res.short_error(350)}")

        # Wave 2 hard reset if none passed
        if not passing:
            w2_user = (
                f"{base_user}\n\n"
                f"GRAVEYARD OF WAVE-1 FAILURES:\n" + "\n".join(graveyard)
                + "\n\nFix the exact root cause shown above. Output the complete updated file."
            )
            w2_specs = [
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {"role": "user", "content": f"{w2_user}\n\nROLE ({name}):\n{card}"},
                    ],
                    tag=f"repo_w2_{name}",
                    max_tokens=4500,
                    temperature=temp,
                    seed=seed + 1000,
                    enable_thinking=True,
                    thinking_budget=1800,
                    loop_guard=True,
                )
                for name, card, temp, seed in roles
            ]
            w2_res = await self.client.batch(w2_specs)
            for (name, _, _, _), res in zip(roles, w2_res):
                code = extract_python_code(res.content)
                if not code:
                    continue
                exec_res = run_in_ephemeral_worktree(repo, {target_rel_path: code}, verify_cmd, timeout_s=timeout_s)
                if exec_res.ok:
                    passing.append((f"w2_{name}", code, exec_res.elapsed_s))
                else:
                    graveyard.append(f"[w2_{name}] verify failed: {exec_res.short_error(350)}")

        return {
            "ok": bool(passing),
            "winner": passing[0][0] if passing else None,
            "code": passing[0][1] if passing else None,
            "passing_count": len(passing),
            "graveyard": graveyard,
            "wall_s": time.perf_counter() - t0,
            "token_stats": self.client.stats_dict(),
        }
