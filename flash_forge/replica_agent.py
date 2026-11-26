"""ReplicaAgent: Autonomous Frontier-Replica Cognitive Loop for Qwen3.8-Flash-Next.

Replicates how a frontier coding/research agent thinks, investigates, probes, builds,
debugs, and referees real projects — engineered specifically to defeat every local-MoE failure mode:
  1. Compacted Working Memory (`BrainState`): Turn 25 stays as sharp as Turn 1 (<5k tok fixed context).
  2. Strict Causal Tool Schema (`libtfgrammar` + `grammar_budget: true`): zero malformed tool JSON.
  3. Parallel Strategic Thought-Forking (`B=3` at Turn 1, errors, stuck loops, or Referee rejections):
     Investigator vs Direct Builder vs Edge-Case Skeptic -> 99ms `guided_choice` selector!
  4. Recursive Forge Escalation (`forge_code` / `forge_reason` tools): delegates hard algorithms
     or math/Z3 sub-problems directly to our 100%-verified parallel Forge engines.
  5. Independent Clean-Context Adversarial Referee Gate: intercepts `finish()`, verifies citations
     or runs newly synthesized adversarial pytest edge cases, and rejects until truly CONFIRMED!
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .brain_state import BrainState
from .client import CallSpec, TensorFoldClient
from .code_forge import solve_code
from .memory import EpisodicMemory
from .reason_forge import solve_reasoning
from .referee import RefereeVerdict, run_referee_gate
from .skill_loader import SkillPack
from .tools import AgentToolbelt, ToolResult


AGENT_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "observed_facts": {"type": "array", "items": {"type": "string"}},
        "next_hypothesis": {"type": "string"},
        "new_verified_fact": {"type": "string"},
        "new_dead_end": {"type": "string"},
        "plan_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"]},
                },
                "required": ["title", "status"],
                "additionalProperties": False,
            },
        },
        "tool": {
            "type": "string",
            "enum": [
                "list_dir",
                "repo_map",
                "search_code",
                "read_file",
                "write_file",
                "edit_file",
                "run_command",
                "python_probe",
                "forge_code",
                "forge_reason",
                "finish",
            ],
        },
        "path": {"type": "string"},
        "pattern": {"type": "string"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
        "cmd": {"type": "string"},
        "edit_target": {"type": "string"},
        "code_or_content": {"type": "string"},
        "final_summary": {"type": "string"},
    },
    "required": [
        "observed_facts",
        "next_hypothesis",
        "new_verified_fact",
        "new_dead_end",
        "plan_updates",
        "tool",
        "path",
        "pattern",
        "start_line",
        "end_line",
        "cmd",
        "edit_target",
        "code_or_content",
        "final_summary",
    ],
    "additionalProperties": False,
}


@dataclass
class ReplicaConfig:
    max_turns: int = 24
    max_referee_rejections: int = 3
    enable_strategic_fork: bool = True
    enable_referee: bool = True
    read_only: bool = False
    think_budget_normal: int = 650
    think_budget_strategic: int = 900
    verbose: bool = True


@dataclass
class ReplicaResult:
    ok: bool
    confirmed_by_referee: bool
    summary: str
    turns_used: int
    strategic_forks: int
    referee_rejections: int
    modified_files: list[str]
    verified_facts: list[str]
    wall_s: float
    token_stats: dict[str, int]
    referee_log: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)


_AGENT_SYSTEM_PROMPT = """You are ReplicaAgent — an elite, disciplined autonomous software engineer and systems investigator.
You operate in a rigorous scientific loop: RECON -> PLAN -> PROBE -> BUILD -> VERIFY -> REFEREE.

TOOLS (set unused string fields to "", unused integer fields to 0):
- list_dir: inspect directory (`path`, `end_line` = depth 1..3)
- repo_map: compact AST symbol outline of a codebase (`path`)
- search_code: fast regex search returning `file:line` (`pattern`, `path`)
- read_file: read exact numbered lines (`path`, `start_line`, `end_line`) — max 180 lines per read!
- write_file: create/overwrite a complete file (`path`, `code_or_content`) — Python syntax is auto-checked
- edit_file: surgical exact SEARCH/REPLACE edit (`path`, `edit_target`, `code_or_content` = replacement)
- run_command: run shell/pytest/build/git command (`cmd`)
- python_probe: run a quick isolated Python/SymPy/Z3 diagnostic script with workspace on PYTHONPATH (`code_or_content`)
- forge_code: delegate a hard standalone algorithm/module to the B=4 parallel verified CodeForge engine (`code_or_content` = full spec, `path` = optional target file to write result)
- forge_reason: delegate a hard math/logic/Z3 sub-problem to the B=4 neuro-symbolic ReasonForge engine (`code_or_content` = problem)
- finish: declare mission complete (`final_summary` with concrete evidence and `file:line` citations). NOTE: An INDEPENDENT ADVERSARIAL REFEREE will audit your work and run its own breaking tests before accepting finish()!

IRON RULES:
1. NEVER hallucinate file paths, line numbers, APIs, or test results. Read and verify first.
2. ALL commands and paths ALREADY execute inside the workspace root `.` — NEVER prepend `cd /var/...` or `cd /private/...`; just run `pytest -q` or read `minidb/wal.py` directly!
3. Files you read or write are AUTOMATICALLY PINNED in `PINNED FILES ALREADY IN YOUR MEMORY`. NEVER re-read a file that is already pinned above! Move immediately to writing, editing, or testing.
4. For multi-file greenfield builds, write each module cleanly (typically 1 turn per file), write a crisp pytest suite covering every spec edge case + crash/error behavior, and run `pytest -q` until 100% green.
5. Keep `plan_updates` crisp (3-6 steps) on Turn 1 or phase transitions; pass `[]` on intermediate turns.
6. Record proven facts in `new_verified_fact` using FULL workspace-relative paths (e.g. `zig/src/families/flashnext/tfq3.zig:45`, NEVER bare basenames like `tfq3.zig:45`) and failed approaches in `new_dead_end`.
7. NEVER repeat an identical failing tool call or spin in a read loop. On read-only investigations, once you have verified all required files/lines, call `finish` promptly with a CONCISE, complete report covering ALL numbered questions (<= 600 words total so nothing truncates) and full-path `path:line` citations.
8. After editing code/kernels, ALWAYS run the build/test/benchmark via `run_command` BEFORE rewriting the file again!
"""


def _summarize_action(action: dict[str, Any]) -> tuple[str, str]:
    tool = str(action.get("tool", "unknown"))
    if tool in ("list_dir", "repo_map"):
        return tool, f"path={action.get('path') or '.'}"
    if tool == "search_code":
        return tool, f"pattern={action.get('pattern', '')!r}, path={action.get('path') or '.'}"
    if tool == "read_file":
        return tool, f"{action.get('path', '')}:{action.get('start_line', 1)}-{action.get('end_line', 0)}"
    if tool == "write_file":
        n = len(str(action.get("code_or_content", "")).splitlines())
        return tool, f"{action.get('path', '')} ({n}L)"
    if tool == "edit_file":
        return tool, f"{action.get('path', '')}"
    if tool == "run_command":
        return tool, f"{action.get('cmd', '')[:100]}"
    if tool == "python_probe":
        first = str(action.get("code_or_content", "")).strip().splitlines()[:1]
        return tool, f"{(first[0] if first else '')[:80]}"
    if tool in ("forge_code", "forge_reason"):
        return tool, f"path={action.get('path', '')} task={str(action.get('code_or_content', ''))[:70]!r}"
    if tool == "finish":
        return tool, f"summary_len={len(str(action.get('final_summary', '')))}"
    return tool, ""


class ReplicaAgent:
    """Autonomous Frontier-Replica Agent powered by Qwen3.8-Flash-Next on TensorFold."""

    def __init__(
        self,
        client: TensorFoldClient,
        skills: SkillPack,
        memory: EpisodicMemory,
        workspace: str | Path,
        config: Optional[ReplicaConfig] = None,
        on_event: Optional[Callable[[str], None]] = None,
        referee_client: Optional[TensorFoldClient] = None,
    ) -> None:
        self.client = client
        self.referee_client = referee_client or client
        self.skills = skills
        self.memory = memory
        self.cfg = config or ReplicaConfig()
        self.tools = AgentToolbelt(workspace, read_only=self.cfg.read_only)
        self.on_event = on_event

    def _log(self, msg: str) -> None:
        if self.on_event:
            self.on_event(msg)
        elif self.cfg.verbose:
            print(msg, flush=True)

    async def _propose_action_single(self, state: BrainState, sys_prompt: str) -> Optional[dict[str, Any]]:
        user_msg = state.render_prompt(self.cfg.max_turns) + "\n\nChoose the single highest-leverage next action now."
        res = await self.client.complete(
            CallSpec(
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                tag=f"agent_t{state.turn}",
                max_tokens=self.cfg.think_budget_normal + 4800,
                temperature=0.2,
                seed=100 + state.turn,
                enable_thinking=True,
                thinking_budget=self.cfg.think_budget_normal,
                json_schema=AGENT_STEP_SCHEMA,
                schema_name="agent_step",
                grammar_budget=True,
                loop_guard=True,
            )
        )
        return res.json_payload()

    async def _propose_action_strategic_fork(
        self,
        state: BrainState,
        sys_prompt: str,
        reason: str,
    ) -> Optional[dict[str, Any]]:
        """B=3 parallel strategic thought-fork (Investigator vs Builder vs Skeptic) + 99ms selector!"""
        base_user = state.render_prompt(self.cfg.max_turns) + f"\n\nSTRATEGIC CROSSROAD ({reason}):\n"
        await self.client.warm_prefix(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": base_user}],
            enable_thinking=True,
        )
        roles = [
            (
                "investigator",
                "Prioritize PROVING ground truth via targeted search_code, read_file, or python_probe before guessing edits.",
                0.20,
                301 + state.turn,
            ),
            (
                "builder",
                "Take the cleanest, most decisive implementation, fix, or verification step right now.",
                0.35,
                401 + state.turn,
            ),
            (
                "skeptic",
                "Challenge hidden assumptions! Look for subtle edge cases, cross-file side effects, or missing specs.",
                0.55,
                501 + state.turn,
            ),
        ]
        specs = [
            CallSpec(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": base_user + f"STRATEGIC ANGLE ({name}): {hint}"},
                ],
                tag=f"fork_t{state.turn}_{name}",
                max_tokens=self.cfg.think_budget_strategic + 4800,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=self.cfg.think_budget_strategic,
                json_schema=AGENT_STEP_SCHEMA,
                schema_name="agent_step",
                grammar_budget=True,
                loop_guard=True,
            )
            for name, hint, temp, seed in roles
        ]
        results = await self.client.batch(specs)
        candidates: list[tuple[str, dict[str, Any]]] = []
        for (name, _, _, _), res in zip(roles, results):
            obj = res.json_payload()
            if obj and obj.get("tool"):
                candidates.append((name, obj))
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][1]

        # Present compact summaries to 99ms guided_choice Strategic Selector!
        opts_lines: list[str] = []
        labels: list[str] = []
        for idx, (name, obj) in enumerate(candidates, 1):
            lbl = str(idx)
            labels.append(lbl)
            t, arg_s = _summarize_action(obj)
            opts_lines.append(
                f"Option {lbl} ({name}): tool={t}({arg_s}) | hypothesis={str(obj.get('next_hypothesis', ''))[:220]}"
            )
        sel_prompt = (
            f"MISSION: {state.goal[:600]}\n"
            f"SITUATION: {reason}\n"
            f"PINNED FILES ALREADY IN MEMORY: {list(state.pinned_files.keys())}\n"
            f"CONSECUTIVE READ-ONLY TURNS: {state.consecutive_readonly_turns}\n"
            f"RECENT GRAVEYARD: {'; '.join(state.graveyard[-3:]) or 'none'}\n\n"
            "PROPOSED NEXT ACTIONS:\n" + "\n".join(opts_lines) + "\n\n"
            "Pick the SINGLE highest-leverage, non-repetitive next action. "
            "CRITICAL: If the relevant files are ALREADY pinned in memory (or consecutive read-only turns >= 3 in a write task), "
            "STRONGLY prefer concrete edit_file / write_file / run_command(pytest) / python_probe over re-reading files again! "
            "Reply with ONLY the option number."
        )
        picked, _ = await self.client.decide_choice(sel_prompt, labels, tag=f"fork_select_t{state.turn}")
        pick_idx = max(0, min(len(candidates) - 1, int(picked) - 1)) if picked.isdigit() else 0
        self._log(f"  [Fork B=3 ({reason})] picked Option {pick_idx+1} ({candidates[pick_idx][0]}) -> {_summarize_action(candidates[pick_idx][1])}")
        return candidates[pick_idx][1]

    async def _execute_tool(self, action: dict[str, Any]) -> ToolResult:
        tool = str(action.get("tool", ""))
        path = str(action.get("path", "") or ".")
        pattern = str(action.get("pattern", "") or "")
        start_line = int(action.get("start_line") or 1)
        end_line = int(action.get("end_line") or 0)
        cmd = str(action.get("cmd", "") or "")
        edit_target = str(action.get("edit_target", "") or "")
        code_or_content = str(action.get("code_or_content", "") or "")

        if tool == "list_dir":
            return self.tools.list_dir(path, depth=end_line if 1 <= end_line <= 4 else 2)
        if tool == "repo_map":
            return self.tools.repo_map(path)
        if tool == "search_code":
            return self.tools.search_code(pattern, path)
        if tool == "read_file":
            return self.tools.read_file(path, start_line=start_line, end_line=end_line or None)
        if tool == "write_file":
            return self.tools.write_file(path, code_or_content)
        if tool == "edit_file":
            return self.tools.edit_file(path, edit_target, code_or_content)
        if tool == "run_command":
            return self.tools.run_command(cmd)
        if tool == "python_probe":
            return self.tools.python_probe(code_or_content)
        if tool == "forge_code":
            if self.cfg.read_only:
                return ToolResult(False, "forge_code disabled in read-only mode")
            trace = await solve_code(
                client=self.client,
                skills=self.skills,
                memory=self.memory,
                problem=code_or_content,
                num_wave1_streams=4,
            )
            if path and path != "." and trace.winner_code:
                wres = self.tools.write_file(path, trace.winner_code)
                if wres.ok:
                    return ToolResult(
                        True,
                        f"CodeForge succeeded (winner={trace.winner_name}, rt_kills={len(trace.redteam_valid_kills)}) and wrote {path}.",
                        modified_file=wres.modified_file,
                    )
                return wres
            return ToolResult(bool(trace.winner_code), f"CodeForge winner={trace.winner_name}:\n```python\n{trace.winner_code[:2200]}\n```")
        if tool == "forge_reason":
            rtrace = await solve_reasoning(
                client=self.client,
                skills=self.skills,
                memory=self.memory,
                problem=code_or_content,
                num_wave1_streams=4,
            )
            return ToolResult(bool(rtrace.final_answer), rtrace.final_explanation or f"Verified Answer: {rtrace.final_answer}")
        return ToolResult(False, f"Unknown tool: {tool}")

    async def run(self, goal: str) -> ReplicaResult:
        t0 = time.perf_counter()
        self.client.reset_stats()
        await self.client.probe()

        state = BrainState(
            goal=goal,
            workspace=str(self.tools.root),
            read_only=self.cfg.read_only,
        )

        # Inject relevant episodic lessons cleanly into system prompt
        lessons = self.memory.format_prompt_block(goal, limit=3)
        sys_prompt = _AGENT_SYSTEM_PROMPT + (f"\n\n{lessons}" if lessons else "")
        if self.cfg.read_only:
            sys_prompt += "\n\nMODE: READ-ONLY INVESTIGATION. Do not call write_file/edit_file/forge_code. Cite exact `path:line` for every factual claim."

        # Turn 0 deterministic workspace bootstrap: list files AND auto-pin small repo source files (<7k chars total)!
        boot = self.tools.list_dir(".", depth=2)
        state.record_observation("list_dir", "path=.", boot.ok, boot.output)
        if not self.cfg.read_only:
            from .repo_index import list_repo_files
            repo_files = [f for f in list_repo_files(self.tools.root, max_files=20) if f.endswith((".py", ".c", ".h", ".cpp", ".zig"))]
            if 1 <= len(repo_files) <= 10:
                total_sz = sum((self.tools.root / f).stat().st_size for f in repo_files if (self.tools.root / f).is_file())
                if total_sz <= 12000:
                    for rel in repo_files:
                        state.refresh_pinned_file(self.tools.root, rel)

        strategic_forks = 0
        referee_rejections = 0
        referee_log: list[str] = []
        trajectory: list[dict[str, Any]] = []
        last_step_failed = False
        just_rejected_by_referee = False
        final_summary = ""
        confirmed = False

        for turn in range(1, self.cfg.max_turns + 1):
            state.turn = turn
            stuck, stuck_reason = state.is_stuck()

            # Decide whether to trigger B=3 Strategic Thought-Forking
            use_fork = False
            fork_reason = ""
            if self.cfg.enable_strategic_fork:
                if turn == 1:
                    use_fork, fork_reason = True, "Initial planning & recon"
                elif just_rejected_by_referee:
                    use_fork, fork_reason = True, "Referee rejected finish() — diagnose and fix root cause"
                elif stuck:
                    use_fork, fork_reason = True, f"Stuck detector triggered ({stuck_reason})"
                elif last_step_failed:
                    use_fork, fork_reason = True, "Previous command/edit failed — diagnose before patching"

            if use_fork:
                strategic_forks += 1
                action = await self._propose_action_strategic_fork(state, sys_prompt, fork_reason)
            else:
                action = await self._propose_action_single(state, sys_prompt)

            just_rejected_by_referee = False

            if not action or not action.get("tool"):
                state.add_dead_end(f"Turn {turn}: model emitted empty/unparseable action")
                last_step_failed = True
                continue

            # Update BrainState from model's cognitive fields
            if action.get("plan_updates"):
                state.set_steps(action["plan_updates"])
            if action.get("new_verified_fact"):
                state.add_fact(str(action["new_verified_fact"]))
            if action.get("new_dead_end"):
                state.add_dead_end(str(action["new_dead_end"]))

            tool_name, arg_sum = _summarize_action(action)
            self._log(f"[Replica T{turn:02d}/{self.cfg.max_turns}] {tool_name}({arg_sum}) | {str(action.get('next_hypothesis',''))[:110]}")

            # Intercept `finish` with the Independent Adversarial Referee Gate!
            if tool_name == "finish":
                proposed_summary = str(action.get("final_summary", "") or action.get("next_hypothesis", ""))
                proposed_evidence = "\n".join(state.verified_facts) + "\n" + state.last_test_output
                if not self.cfg.enable_referee:
                    final_summary = proposed_summary
                    confirmed = True
                    trajectory.append({"turn": turn, "tool": "finish", "confirmed": True})
                    break
                if referee_rejections >= self.cfg.max_referee_rejections:
                    # Fail-closed: hitting max referee rejections terminates as UNCONFIRMED (ok=False)!
                    final_summary = proposed_summary
                    confirmed = False
                    trajectory.append({"turn": turn, "tool": "finish_rejected_cap", "confirmed": False})
                    break

                self._log(f"  [REFEREE GATE] Auditing workspace in clean zero-history context...")
                verdict: RefereeVerdict = await run_referee_gate(
                    client=self.referee_client,
                    tools=self.tools,
                    state=state,
                    proposed_summary=proposed_summary,
                    proposed_evidence=proposed_evidence,
                )
                referee_log.append(f"T{turn}: {'CONFIRMED' if verdict.confirmed else 'REJECTED'} — {verdict.reason[:300]}")
                trajectory.append({
                    "turn": turn,
                    "tool": "finish_audit",
                    "confirmed": verdict.confirmed,
                    "reason": verdict.reason[:400],
                })
                if verdict.confirmed:
                    self._log(f"  [REFEREE] CONFIRMED: {verdict.reason[:180]}")
                    final_summary = proposed_summary
                    confirmed = True
                    break
                else:
                    referee_rejections += 1
                    just_rejected_by_referee = True
                    self._log(f"  [REFEREE] REJECTED ({referee_rejections}/{self.cfg.max_referee_rejections}): {verdict.reason[:220]}")
                    state.referee_feedback.append(verdict.reason)
                    state.add_dead_end(f"Referee rejected finish() at T{turn}: {verdict.reason[:180]}")
                    state.record_observation("referee_gate", "finish()", False, verdict.reason)
                    last_step_failed = True
                    continue

            # Execute real tool!
            tres = await self._execute_tool(action)
            if tres.modified_file:
                state.mark_file_modified(tres.modified_file)
                state.refresh_pinned_file(self.tools.root, tres.modified_file)
            elif tool_name == "read_file" and tres.ok:
                rel_read = str(action.get("path", "") or "").strip().lstrip("./")
                if rel_read:
                    state.pin_file(rel_read, tres.output)
            if tres.is_test_run:
                state.record_test_run(tres.test_cmd, tres.ok, tres.output)
            state.record_observation(tool_name, arg_sum, tres.ok, tres.output)
            if not tres.ok:
                state.add_dead_end(f"T{turn} {tool_name}({arg_sum[:60]}) failed: {tres.output[:140]}")
                last_step_failed = True
            else:
                last_step_failed = False

            trajectory.append({
                "turn": turn,
                "tool": tool_name,
                "args": arg_sum,
                "ok": tres.ok,
                "forked": use_fork,
            })

        # Fallback synthesis if max_turns hit before confirmed finish (especially valuable in read-only investigations!)
        if not final_summary:
            if self.cfg.read_only:
                synth = await self.client.complete(
                    CallSpec(
                        messages=[
                            {"role": "system", "content": "Synthesize a crisp, complete technical report answering EVERY part of the mission strictly from the PINNED FILES and VERIFIED FACTS below. Cite ONLY exact full workspace-relative `path:line` locations actually present in the pinned files."},
                            {"role": "user", "content": state.render_prompt(self.cfg.max_turns)},
                        ],
                        tag="agent_final_synth",
                        max_tokens=2200,
                        temperature=0.0,
                        enable_thinking=True,
                        thinking_budget=600,
                        loop_guard=True,
                    )
                )
                final_summary = synth.content.strip()
                if self.cfg.enable_referee and final_summary:
                    v_final = await run_referee_gate(self.referee_client, self.tools, state, final_summary, "\n".join(state.verified_facts))
                    referee_log.append(f"FINAL_SYNTH: {'CONFIRMED' if v_final.confirmed else 'REJECTED'} — {v_final.reason[:260]}")
                    confirmed = v_final.confirmed
            else:
                final_summary = (
                    f"Reached turn limit ({self.cfg.max_turns}). "
                    f"Modified files: {state.modified_files}. "
                    f"Verified facts:\n" + "\n".join(f"- {f}" for f in state.verified_facts)
                )

        ok_final = (
            confirmed
            if self.cfg.enable_referee
            else (bool(final_summary) and (state.read_only or (state.last_test_ok and state.edits_since_last_test == 0)))
        )
        return ReplicaResult(
            ok=ok_final,
            confirmed_by_referee=confirmed,
            summary=final_summary,
            turns_used=state.turn,
            strategic_forks=strategic_forks,
            referee_rejections=referee_rejections,
            modified_files=state.modified_files,
            verified_facts=state.verified_facts,
            wall_s=time.perf_counter() - t0,
            token_stats=self.client.stats_dict(),
            referee_log=referee_log,
            trajectory=trajectory,
        )
