"""One-command `/cortheon` slash-command & skill installer for all major coding harnesses.

Supported harnesses:
  - omp       (oh-my-pi: native TS extension -> `/cortheon`, `/cortheon build`, `cortheon_verify` tool + live TUI widget)
  - pi        (pi-coding-agent: same native TS extension -> `/cortheon` + `cortheon_verify` tool + live widget)
  - opencode  (OpenCode: native Markdown slash command `command/cortheon.md`)
  - claude    (Claude Code: native Markdown slash command `commands/cortheon.md`)
  - cursor    (Cursor: native Markdown slash command `commands/cortheon.md`)
  - codex     (Codex / generic agents: `skills/cortheon/SKILL.md` + `cortheon` skill cards)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

ALL_HARNESSES: tuple[str, ...] = ("omp", "pi", "opencode", "claude", "cursor", "codex")

_PACKAGE_DIR = Path(__file__).resolve().parent
_INTEGRATIONS_DIR = _PACKAGE_DIR / "_integrations"
_SKILL_DIR = _PACKAGE_DIR / "_skill"


def _render_markdown_command(harness: str, default_url: str) -> str:
    url_expr = f'${{CORTHEON_URL:-{default_url}}}'
    if harness == "opencode":
        frontmatter = (
            "---\n"
            "description: Run Cortheon v2 B=4 execution-verified consensus (ReasonForge / CodeForge v5 / ThinkForge) on your local model\n"
            "---\n\n"
        )
    elif harness == "claude":
        frontmatter = (
            "---\n"
            "description: Run Cortheon v2 B=4 execution-verified parallel consensus on your local model\n"
            "argument-hint: <question or task to verify>\n"
            "allowed-tools: Bash(cortheon:*)\n"
            "---\n\n"
        )
    else:
        frontmatter = (
            "---\n"
            "description: Run Cortheon v2 B=4 execution-verified parallel consensus on your local model\n"
            "---\n\n"
        )

    body = (
        "# Cortheon v2 — Execution-Verified Consensus (`/cortheon`)\n\n"
        "Run Cortheon v2 `B=4` execution-verified parallel consensus on the user's request below BEFORE answering or editing code.\n\n"
        "## Steps\n"
        "1. Formulate a concise, self-contained specification of the math, algorithm, protocol, or systems-sizing question from `$ARGUMENTS`.\n"
        "2. Execute Cortheon synchronously via bash:\n"
        "```sh\n"
        f'cortheon solve --url "{url_expr}" "$ARGUMENTS"\n'
        "```\n"
        "   *(If serving TensorFold native with `--cortheon --cortheon-exec`, you can also POST tool-less JSON directly to `"
        + url_expr
        + "/chat/completions`.)*\n"
        "3. Report the verified answer, consensus status, and key invariants cleanly. If `$ARGUMENTS` asks to implement or fix code, ground your implementation in Cortheon's verified result and run real tests before finishing.\n\n"
        "## Request\n"
        "$ARGUMENTS\n"
    )
    return frontmatter + body


def install_commands(
    harnesses: Iterable[str] = ALL_HARNESSES,
    *,
    home_dir: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    omp_profile: Optional[str] = None,
    default_url: str = "http://127.0.0.1:18651/v1",
) -> list[Path]:
    """Install `/cortheon` slash commands & extensions for the requested harnesses."""
    home = (home_dir or Path.home()).expanduser().resolve()
    proj = project_dir.expanduser().resolve() if project_dir is not None else None
    ts_src = _INTEGRATIONS_DIR / "cortheon.ts"
    if not ts_src.is_file():
        raise FileNotFoundError(f"Bundled OMP/Pi extension missing: {ts_src}")

    written: list[Path] = []
    chosen = [h.strip().lower() for h in harnesses if h.strip()]
    for h in chosen:
        if h not in ALL_HARNESSES:
            raise ValueError(f"Unknown harness '{h}'. Supported: {', '.join(ALL_HARNESSES)}")

        if h == "omp":
            if proj is not None:
                dest = proj / ".omp" / "extensions" / "cortheon.ts"
            elif omp_profile:
                dest = home / ".omp" / "profiles" / omp_profile / "agent" / "extensions" / "cortheon.ts"
            else:
                dest = home / ".omp" / "agent" / "extensions" / "cortheon.ts"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ts_src, dest)
            written.append(dest)

        elif h == "pi":
            dest = (proj / ".pi" / "extensions" / "cortheon.ts") if proj is not None else (home / ".pi" / "agent" / "extensions" / "cortheon.ts")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ts_src, dest)
            written.append(dest)

        elif h == "opencode":
            dest = (proj / ".opencode" / "command" / "cortheon.md") if proj is not None else (home / ".config" / "opencode" / "command" / "cortheon.md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_render_markdown_command("opencode", default_url), encoding="utf-8")
            written.append(dest)

        elif h == "claude":
            dest = (proj / ".claude" / "commands" / "cortheon.md") if proj is not None else (home / ".claude" / "commands" / "cortheon.md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_render_markdown_command("claude", default_url), encoding="utf-8")
            written.append(dest)

        elif h == "cursor":
            dest = (proj / ".cursor" / "commands" / "cortheon.md") if proj is not None else (home / ".cursor" / "commands" / "cortheon.md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_render_markdown_command("cursor", default_url), encoding="utf-8")
            written.append(dest)

        elif h == "codex":
            skill_dest_dir = (proj / ".agents" / "skills" / "cortheon") if proj is not None else (home / ".agents" / "skills" / "cortheon")
            skill_dest_dir.mkdir(parents=True, exist_ok=True)
            skill_md = _SKILL_DIR / "SKILL.md"
            if skill_md.is_file():
                dst = skill_dest_dir / "SKILL.md"
                shutil.copy2(skill_md, dst)
                written.append(dst)
            cards_src = _SKILL_DIR / "cards"
            if cards_src.is_dir():
                cards_dst = skill_dest_dir / "cards"
                cards_dst.mkdir(parents=True, exist_ok=True)
                for cf in sorted(cards_src.glob("*.md")):
                    shutil.copy2(cf, cards_dst / cf.name)

    return written
