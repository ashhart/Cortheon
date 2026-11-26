"""Compacted Cognitive Working Memory (`BrainState` v2) for long-horizon local agents.

Combines fixed-budget state discipline with a **Pinned File Cache + Live Workspace API Map**:
- Turn 25 stays crisp (bounded context, zero raw chat-transcript bloat).
- Every file read or written stays pinned (with auto-refresh on edit!) up to a clean 14k-char
  budget so the local model NEVER suffers amnesia or re-reads the same file in a loop!
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .repo_index import build_repo_map


StepStatus = Literal["pending", "in_progress", "done", "blocked"]


@dataclass
class PlanStep:
    title: str
    status: StepStatus = "pending"


@dataclass
class Observation:
    turn: int
    tool: str
    args_summary: str
    ok: bool
    output: str


@dataclass
class BrainState:
    goal: str
    workspace: str
    read_only: bool = False
    steps: list[PlanStep] = field(default_factory=list)
    verified_facts: list[str] = field(default_factory=list)
    graveyard: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    pinned_files: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    edits_since_last_test: int = 0
    consecutive_readonly_turns: int = 0
    last_test_cmd: str = ""
    last_test_ok: bool = False
    last_test_output: str = ""
    recent_observations: list[Observation] = field(default_factory=list)
    referee_feedback: list[str] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)
    turn: int = 0

    def pin_file(self, rel_path: str, numbered_content: str, max_total_chars: int = 24000) -> None:
        """Keep full/sliced contents of active files pinned in working memory (no premature middle truncation!)."""
        key = rel_path.strip().lstrip("./")
        if not key:
            return
        body = numbered_content.strip()
        if len(body) > 10500:
            body = body[:6000] + "\n...(middle truncated)...\n" + body[-4000:]
        # Never overwrite a full pinned file with a shorter partial slice of the same file
        if key in self.pinned_files and len(self.pinned_files[key]) > len(body) + 400 and "lines 1-" not in body[:90]:
            self.pinned_files.move_to_end(key)
            return
        self.pinned_files[key] = body
        self.pinned_files.move_to_end(key)
        while sum(len(v) for v in self.pinned_files.values()) > max_total_chars and len(self.pinned_files) > 1:
            self.pinned_files.popitem(last=False)

    def refresh_pinned_file(self, root: Path, rel_path: str) -> None:
        """Automatically refresh a pinned file right after write_file / edit_file so memory is never stale."""
        key = rel_path.strip().lstrip("./")
        p = (root / key).resolve()
        if not p.is_file():
            return
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        numbered = "\n".join(f"{i:4d} | {ln}" for i, ln in enumerate(lines[:300], 1))
        if len(lines) > 300:
            numbered += f"\n... ({len(lines) - 300} more lines)"
        self.pin_file(key, f"=== PINNED FILE: {key} ({len(lines)}L) ===\n{numbered}")

    def add_fact(self, fact: str) -> None:
        cleaned = fact.strip()[:260]
        if not cleaned:
            return
        if cleaned not in self.verified_facts:
            self.verified_facts.append(cleaned)
            if len(self.verified_facts) > 18:
                self.verified_facts = self.verified_facts[-18:]

    def add_dead_end(self, entry: str) -> None:
        cleaned = entry.strip()[:260]
        if not cleaned:
            return
        if cleaned not in self.graveyard:
            self.graveyard.append(cleaned)
            if len(self.graveyard) > 12:
                self.graveyard = self.graveyard[-12:]

    def mark_file_modified(self, rel_path: str) -> None:
        key = rel_path.strip().lstrip("./")
        if key not in self.modified_files:
            self.modified_files.append(key)
        self.edits_since_last_test += 1
        self.consecutive_readonly_turns = 0

    def record_test_run(self, cmd: str, ok: bool, output: str) -> None:
        self.last_test_cmd = cmd
        self.last_test_ok = ok
        self.last_test_output = output[-1600:]
        if ok:
            self.edits_since_last_test = 0

    def record_observation(self, tool: str, args_summary: str, ok: bool, output: str) -> None:
        sig = f"{tool}({args_summary[:100]})"
        self.action_history.append(sig)
        if tool in ("list_dir", "repo_map", "search_code", "read_file"):
            self.consecutive_readonly_turns += 1
        elif tool in ("write_file", "edit_file", "forge_code"):
            self.consecutive_readonly_turns = 0

        compact = output.strip()
        if len(compact) > 2600:
            compact = compact[:1300] + "\n...(middle truncated)...\n" + compact[-1200:]
        self.recent_observations.append(
            Observation(
                turn=self.turn,
                tool=tool,
                args_summary=args_summary[:180],
                ok=ok,
                output=compact,
            )
        )
        if len(self.recent_observations) > 4:
            self.recent_observations = self.recent_observations[-4:]

    def is_stuck(self) -> tuple[bool, str]:
        """Detect repetitive tool loops, read-treadmills, or consecutive failures."""
        if len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            return True, f"Repeated identical action twice: {self.action_history[-1]}"
        if not self.read_only and self.consecutive_readonly_turns >= 4 and self.pinned_files:
            return True, f"{self.consecutive_readonly_turns} consecutive read-only turns while files are already pinned — take concrete build/edit/test action!"
        if len(self.recent_observations) >= 3:
            last3 = self.recent_observations[-3:]
            if all(not o.ok for o in last3):
                return True, "3 consecutive tool failures"
        return False, ""

    def set_steps(self, step_dicts: list[dict[str, str]]) -> None:
        parsed: list[PlanStep] = []
        valid_statuses = {"pending", "in_progress", "done", "blocked"}
        for item in step_dicts[:10]:
            title = str(item.get("title", "")).strip()[:120]
            status = str(item.get("status", "pending")).strip().lower()
            if status not in valid_statuses:
                status = "pending"
            if title:
                parsed.append(PlanStep(title=title, status=status))  # type: ignore[arg-type]
        if parsed:
            self.steps = parsed

    def render_prompt(self, max_turns: int) -> str:
        """Render the complete bounded working memory snapshot for the next cognitive step."""
        root = Path(self.workspace)
        lines: list[str] = [
            f"=== MISSION (Turn {self.turn}/{max_turns}) ===",
            self.goal.strip(),
            f"Workspace: `.` (ALL relative paths & commands already run in workspace root — NEVER `cd` to absolute temp dirs!)",
            f"Mode: {'READ-ONLY INVESTIGATION' if self.read_only else 'FULL READ/WRITE/TEST'}",
        ]
        remaining = max_turns - self.turn
        if remaining <= 2:
            lines.append(
                f"\n⚠️ TURN BUDGET WARNING: ONLY {remaining} TURN(S) LEFT! "
                "Stop extra exploration and either run your final verification or call `finish` NOW with complete, "
                "workspace-relative `path:line` citations (e.g. `zig/src/.../file.zig:42`, NEVER bare basenames!)."
            )

        if self.steps:
            lines.append("\n=== LIVE PLAN ===")
            icons = {"done": "[x]", "in_progress": "[>]", "pending": "[ ]", "blocked": "[!]"}
            for i, st in enumerate(self.steps, 1):
                lines.append(f"{i}. {icons.get(st.status, '[ ]')} {st.title}")

        # Always show compact live AST map for small/medium workspaces (<=40 files)
        if not self.read_only and root.is_dir():
            rmap = build_repo_map(root, max_chars=2400)
            if len(rmap) > 15:
                lines.append(f"\n=== LIVE WORKSPACE API MAP ===\n{rmap}")

        if self.verified_facts:
            lines.append("\n=== VERIFIED FACTS (PROVEN BY TOOLS) ===")
            for f in self.verified_facts[-14:]:
                lines.append(f"- {f}")

        if self.modified_files:
            lines.append(
                f"\n=== WORKSPACE STATE ===\n"
                f"Modified/Created files: {', '.join(self.modified_files)}\n"
                f"Edits since last passing test: {self.edits_since_last_test}"
            )
            if self.edits_since_last_test >= 1 and len(self.modified_files) == 1:
                lines.append(
                    "⚠️ You have ALREADY edited this file — DO NOT blindly rewrite it again! "
                    "RUN the build/test/benchmark command via `run_command` FIRST to measure real exactness & speed!"
                )
            if self.last_test_cmd:
                lines.append(
                    f"Last test `{self.last_test_cmd}`: {'PASS' if self.last_test_ok else 'FAIL'}"
                )

        if self.pinned_files:
            lines.append("\n=== PINNED FILES ALREADY IN YOUR MEMORY (DO NOT RE-READ THESE!) ===")
            for body in self.pinned_files.values():
                lines.append(body)

        if self.graveyard:
            lines.append("\n=== GRAVEYARD (DEAD ENDS — DO NOT REPEAT) ===")
            for g in self.graveyard[-8:]:
                lines.append(f"- {g}")

        if self.referee_feedback:
            lines.append("\n=== INDEPENDENT REFEREE REJECTION (MUST FIX BEFORE FINISH) ===")
            for fb in self.referee_feedback[-2:]:
                lines.append(fb[:1600])

        if self.recent_observations:
            lines.append("\n=== RECENT TOOL OBSERVATIONS ===")
            for idx, obs in enumerate(self.recent_observations):
                budget = 2200 if idx == len(self.recent_observations) - 1 else 600
                out = obs.output
                if len(out) > budget:
                    out = out[: budget // 2] + "\n...(compacted)...\n" + out[-(budget // 2) :]
                status = "OK" if obs.ok else "ERROR"
                lines.append(f"[Turn {obs.turn}] {obs.tool}({obs.args_summary}) -> {status}\n{out}")

        return "\n".join(lines)
