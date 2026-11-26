"""Unified FlashForge orchestrator: routing, parallel verification, repo worktree mode,
and zero-weight episodic reflexion learning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .client import CallSpec, TensorFoldClient
from .code_forge import CodeForgeTrace, solve_code
from .memory import EpisodicMemory
from .reason_forge import ReasonForgeTrace, solve_reasoning
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

    async def solve(
        self,
        prompt: str,
        *,
        force_mode: Optional[TaskMode] = None,
        external_test_code: Optional[str] = None,
        learn: bool = True,
    ) -> ForgeResult:
        t0 = time.perf_counter()
        self.client.reset_stats()
        await self.client.probe()

        mode: TaskMode = force_mode or await classify_task(self.client, prompt)

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
            return ForgeResult(
                mode="fast",
                answer=res.content,
                wall_s=time.perf_counter() - t0,
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
            )
            lesson = None
            if learn and trace.recovered_by_redteam_or_wave2 and trace.graveyard:
                lesson = await self._distill_lesson(
                    "code",
                    prompt,
                    trace.graveyard,
                    f"Winner `{trace.winner_name}` passed checks after Red-Team/Wave-2 repair.",
                )
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
        return ForgeResult(
            mode="reason",
            answer=rtrace.final_answer,
            raw_trace=rtrace,
            wall_s=time.perf_counter() - t0,
            token_stats=self.client.stats_dict(),
            lesson_learned=lesson,
            verified=rtrace.verified,
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
