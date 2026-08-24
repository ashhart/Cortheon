from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    runtime_command: str
    description: str
    usage: str
    argument_hint: str
    notes: tuple[str, ...]


SPECS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(
        name="cortheon-answer",
        runtime_command="answer",
        description="Pool current evidence for a task and return allow / needs_evidence / block.",
        usage="/cortheon-answer <task> or /cortheon-answer <task> :: <proposed action>",
        argument_hint="<task> [:: proposed action]",
        notes=(
            "Use this before coding when the right library, API, architecture, or current best option matters.",
            "If the verdict is needs_evidence, do not continue into production code.",
        ),
    ),
    SlashCommandSpec(
        name="cortheon-decide",
        runtime_command="decide",
        description="Gate a proposed action with bounded evidence agents.",
        usage="/cortheon-decide <task> :: <proposed action>",
        argument_hint="<task> :: <proposed action>",
        notes=(
            "Use this when the model is about to act and needs a permission gate.",
            "The separator :: is required so the substrate can distinguish task from action.",
        ),
    ),
    SlashCommandSpec(
        name="cortheon-research",
        runtime_command="research",
        description="Run a bounded live research mission and summarize evidence gaps.",
        usage="/cortheon-research <topic>",
        argument_hint="<topic>",
        notes=(
            "Use this for frontier or broad technical questions before committing to an architecture.",
            "The result is evidence, not permission to act; use /cortheon-decide before a risky action.",
        ),
    ),
    SlashCommandSpec(
        name="cortheon-api",
        runtime_command="api",
        description="Check source-derived package API evidence.",
        usage="/cortheon-api <package> :: <symbol>",
        argument_hint="<package> :: <symbol>",
        notes=(
            "Use this before writing production code against a package-specific class, method, or function.",
            "No matches means do not use that symbol unless another source-derived proof is found.",
        ),
    ),
    SlashCommandSpec(
        name="cortheon-recommend",
        runtime_command="recommend",
        description="Recommend current package options for a task profile.",
        usage="/cortheon-recommend <task>",
        argument_hint="<task>",
        notes=(
            "Use this for narrow package/library selection.",
            "For open-ended research or scientific claims, prefer /cortheon-answer or /cortheon-research.",
        ),
    ),
)


def command_specs() -> tuple[SlashCommandSpec, ...]:
    return SPECS


def render_command(spec: SlashCommandSpec, root: Path | None = None) -> str:
    python_path = f'PYTHONPATH="{root / "src"}" ' if root is not None else ""
    command = f'{python_path}python3 -m cortheon.slash {spec.runtime_command} "$ARGUMENTS"'
    note_lines = "\n".join(f"- {note}" for note in spec.notes)
    return f"""---
name: {spec.name}
description: {spec.description}
argument-hint: "{spec.argument_hint}"
agent: build
subtask: false
---

Run Cortheon for the user arguments.

Usage: `{spec.usage}`

Rules:
{note_lines}
- Run the command below with the exact user arguments substituted for `$ARGUMENTS`.
- Reply with the Cortheon output and obey its final Agent Instruction.
- Do not invent sources, APIs, or evidence that are not in the output.

```bash
{command}
```
"""


def install_targets(selected: list[str] | None = None) -> list[tuple[str, Path]]:
    targets = {
        "pi": Path(os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser() / "prompts",
        "opencode": Path("~/.config/opencode/commands").expanduser(),
    }
    names = selected or ["pi", "opencode"]
    output: list[tuple[str, Path]] = []
    for name in names:
        if name not in targets:
            raise ValueError(f"Unknown slash command target: {name}")
        output.append((name, targets[name]))
    return output


def write_command_pack(
    targets: list[str] | None = None,
    *,
    root: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for _, directory in install_targets(targets):
        if not dry_run:
            directory.mkdir(parents=True, exist_ok=True)
        for spec in SPECS:
            path = directory / f"{spec.name}.md"
            written.append(path)
            if not dry_run:
                path.write_text(render_command(spec, root=root), encoding="utf-8")
    return written


def export_command_pack(
    directory: Path, *, root: Path | None = None, clean: bool = False
) -> list[Path]:
    if clean and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in SPECS:
        path = directory / f"{spec.name}.md"
        path.write_text(render_command(spec, root=root), encoding="utf-8")
        written.append(path)
    return written
