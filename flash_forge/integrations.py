"""One-command `/cortheon` slash-command, skill, and MCP server installer for major coding harnesses.

Supported slash-command harnesses (`install_commands`):
  - omp       (oh-my-pi: native TS extension -> `/cortheon`, `/cortheon build`, `cortheon_verify` tool + live TUI widget)
  - pi        (pi-coding-agent: same native TS extension -> `/cortheon` + `cortheon_verify` tool + live widget)
  - opencode  (OpenCode: native Markdown slash command `command/cortheon.md`)
  - claude    (Claude Code: native Markdown slash command `commands/cortheon.md`)
  - cursor    (Cursor: native Markdown slash command `commands/cortheon.md`)
  - codex     (Codex / generic agents: `skills/cortheon/SKILL.md` + `cortheon` skill cards)

Supported MCP server harnesses (`install_mcp`):
  - claude    (`~/.claude.json` or `<project>/.mcp.json`)
  - cursor    (`~/.cursor/mcp.json` or `<project>/.cursor/mcp.json`)
  - opencode  (`~/.config/opencode/opencode.json` or `<project>/opencode.json`)
  - codex     (`~/.codex/config.toml` or `<project>/.codex/config.toml`)
  - windsurf  (`~/.codeium/windsurf/mcp_config.json` or `<project>/.windsurf/mcp_config.json`)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Iterable, Optional

ALL_HARNESSES: tuple[str, ...] = ("omp", "pi", "opencode", "claude", "cursor", "codex")
MCP_HARNESSES: tuple[str, ...] = ("claude", "cursor", "opencode", "codex", "windsurf")

_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

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


def _atomic_write_text(dest: Path, content: str, *, sensitive: bool = False) -> None:
    """Atomically write UTF-8 text via a sibling `.tmp` file + `replace()`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = (dest.stat().st_mode & 0o777) if dest.exists() else (0o600 if sensitive else 0o644)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp_path, mode)
        except OSError:
            pass
        tmp_path.replace(dest)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _resolve_mcp_command(
    *,
    url: str,
    api_key: Optional[str] = None,
    command: Optional[str] = None,
) -> tuple[str, list[str], dict[str, str]]:
    """Resolve executable, CLI args, and env vars for the Cortheon stdio MCP server."""
    clean_url = str(url).strip()
    clean_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None

    if command is not None and str(command).strip():
        cmd = str(command).strip()
        args = ["mcp", "--url", clean_url]
    else:
        found = shutil.which("cortheon")
        if found:
            cmd = found
            args = ["mcp", "--url", clean_url]
        else:
            cmd = sys.executable
            args = ["-m", "flash_forge.cli", "mcp", "--url", clean_url]

    env: dict[str, str] = {"CORTHEON_URL": clean_url}
    if clean_key is not None:
        args.extend(["--api-key", clean_key])
        env["CORTHEON_API_KEY"] = clean_key

    return cmd, args, env


def _merge_json_mcp_config(
    dest: Path,
    section_key: str,
    entry: dict[str, Any],
    *,
    sensitive: bool = False,
) -> Path:
    """Non-destructively merge `entry` under `data[section_key]['cortheon']`."""
    data: dict[str, Any] = {}
    if dest.exists():
        raw = dest.read_text(encoding="utf-8")
        if raw.strip():
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {dest}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Malformed JSON in {dest}: expected top-level JSON object, got {type(loaded).__name__}"
                )
            data = loaded

    existing_section = data.get(section_key)
    if existing_section is None and section_key not in data:
        section: dict[str, Any] = {}
    elif isinstance(existing_section, dict):
        section = dict(existing_section)
    else:
        raise ValueError(
            f"Malformed JSON in {dest}: expected '{section_key}' to be a JSON object, got {type(existing_section).__name__}"
        )

    section["cortheon"] = entry
    data[section_key] = section
    _atomic_write_text(dest, json.dumps(data, indent=2, ensure_ascii=False) + "\n", sensitive=sensitive)
    return dest


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_key(key: str) -> str:
    return key if _TOML_BARE_KEY_RE.match(key) else _toml_str(key)


def _render_codex_cortheon_block(cmd: str, args: list[str], env: dict[str, str]) -> list[str]:
    args_literal = "[" + ", ".join(_toml_str(a) for a in args) + "]"
    env_items = ", ".join(f"{_toml_key(k)} = {_toml_str(v)}" for k, v in env.items())
    return [
        "[mcp_servers.cortheon]",
        f"command = {_toml_str(cmd)}",
        f"args = {args_literal}",
        f"env = {{ {env_items} }}",
    ]


def _extract_toml_header_path(line: str) -> Optional[tuple[str, tuple[str, ...]]]:
    """Parse a standalone TOML table or array-of-tables header into `(kind, key_path)`."""
    stripped = line.lstrip()
    if not stripped.startswith("["):
        return None
    try:
        parsed = tomllib.loads(f"{stripped}\n")
    except tomllib.TOMLDecodeError:
        return None
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return None

    kind = "array" if stripped.startswith("[[") else "table"
    path: list[str] = []
    cur: Any = parsed
    while isinstance(cur, dict) and len(cur) == 1:
        k, v = next(iter(cur.items()))
        path.append(k)
        if kind == "table" and isinstance(v, dict) and not v:
            return kind, tuple(path)
        if kind == "array" and isinstance(v, list) and len(v) == 1 and v[0] == {}:
            return kind, tuple(path)
        cur = v
    return None


def _advance_toml_state(
    line: str,
    in_multiline: Optional[str],
    bracket_depth: int,
) -> tuple[Optional[str], int]:
    """Track multiline TOML strings and array bracket nesting across lines."""
    i = 0
    n = len(line)
    while i < n:
        if in_multiline is not None:
            q = in_multiline[0]
            if in_multiline == '"""' and line[i] == "\\":
                i += 2
                continue
            if line[i : i + 3] == in_multiline:
                # Handle 4 or 5 consecutive quotes where 1-2 quotes belong inside the string
                while i + 3 < n and line[i + 3] == q:
                    i += 1
                in_multiline = None
                i += 3
                continue
            i += 1
            continue

        ch = line[i]
        if ch == "#":
            break
        if line[i : i + 3] in ('"""', "'''"):
            in_multiline = line[i : i + 3]
            i += 3
            continue
        if ch == '"':
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "'":
            i += 1
            while i < n and line[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        i += 1
    return in_multiline, bracket_depth


def _scan_toml_headers(lines: list[str]) -> list[tuple[int, str, tuple[str, ...]]]:
    """Return `(line_idx, kind, key_path)` for every TOML table/array header outside strings."""
    headers: list[tuple[int, str, tuple[str, ...]]] = []
    in_multiline: Optional[str] = None
    bracket_depth = 0
    for idx, line in enumerate(lines):
        if in_multiline is None and bracket_depth == 0:
            hdr = _extract_toml_header_path(line)
            if hdr is not None:
                headers.append((idx, hdr[0], hdr[1]))
                continue
        in_multiline, bracket_depth = _advance_toml_state(line, in_multiline, bracket_depth)
    return headers


def _upsert_codex_toml(
    dest: Path,
    cmd: str,
    args: list[str],
    env: dict[str, str],
    *,
    sensitive: bool = False,
) -> Path:
    """Idempotently upsert `[mcp_servers.cortheon]` in Codex's TOML config without disturbing other entries."""
    existing_text = ""
    if dest.exists():
        existing_text = dest.read_text(encoding="utf-8")
        if existing_text.strip():
            try:
                tomllib.loads(existing_text)
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f"Malformed TOML in {dest}: {exc}") from exc

    block_lines = _render_codex_cortheon_block(cmd, args, env)
    if not existing_text.strip():
        updated_text = "\n".join(block_lines) + "\n"
        _atomic_write_text(dest, updated_text, sensitive=sensitive)
        return dest

    lines = existing_text.splitlines()
    headers = _scan_toml_headers(lines)

    target_header_indices = [
        h_i
        for h_i, (_, _, key_path) in enumerate(headers)
        if key_path == ("mcp_servers", "cortheon")
        or (len(key_path) > 2 and key_path[:2] == ("mcp_servers", "cortheon"))
    ]

    if target_header_indices:
        primary_h_i = next(
            (h_i for h_i in target_header_indices if headers[h_i][2] == ("mcp_servers", "cortheon")),
            target_header_indices[0],
        )
        # Compute replacement/removal spans from bottom to top so indices remain valid
        spans: list[tuple[int, int, bool]] = []
        for h_i in target_header_indices:
            start = headers[h_i][0]
            raw_end = headers[h_i + 1][0] if h_i + 1 < len(headers) else len(lines)
            # Preserve blank lines & comments immediately preceding the next section or EOF
            end = raw_end
            while end > start + 1 and (
                lines[end - 1].strip() == "" or lines[end - 1].lstrip().startswith("#")
            ):
                end -= 1
            spans.append((start, end, h_i == primary_h_i))

        for start, end, is_primary in sorted(spans, key=lambda item: item[0], reverse=True):
            if is_primary:
                lines[start:end] = block_lines
            else:
                del lines[start:end]
                # Collapse duplicate blank line left by deleting a legacy sub-table (e.g. [mcp_servers.cortheon.env])
                while start < len(lines) and lines[start].strip() == "" and (
                    start == 0 or lines[start - 1].strip() == ""
                ):
                    del lines[start]
    else:
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines:
            lines.append("")
        lines.extend(block_lines)

    updated_text = "\n".join(lines).rstrip("\n") + "\n"
    try:
        tomllib.loads(updated_text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"Could not safely update TOML in {dest} due to conflicting mcp_servers.cortheon definition: {exc}"
        ) from exc

    _atomic_write_text(dest, updated_text, sensitive=sensitive)
    return dest


def install_mcp(
    harnesses: Iterable[str] | str = MCP_HARNESSES,
    *,
    url: str = "http://127.0.0.1:8080/v1",
    api_key: Optional[str] = None,
    command: Optional[str] = None,
    project_dir: Optional[str | Path] = None,
    home_dir: Optional[str | Path] = None,
) -> list[Path]:
    """Non-destructively install the Cortheon stdio MCP server into native harness config files.

    Existing user MCP servers, comments, and unrelated settings are preserved intact.
    All writes are atomic via `.tmp` + `replace()`.
    """
    home = (Path(home_dir) if home_dir is not None else Path.home()).expanduser().resolve()
    proj = Path(project_dir).expanduser().resolve() if project_dir is not None else None

    if isinstance(harnesses, str):
        raw_items = [h.strip().lower() for h in harnesses.split(",") if h.strip()]
    else:
        raw_items = [str(h).strip().lower() for h in harnesses if str(h).strip()]

    chosen: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        expanded = MCP_HARNESSES if item == "all" else (item,)
        for h in expanded:
            if h not in MCP_HARNESSES:
                raise ValueError(f"Unknown MCP harness '{h}'. Supported: {', '.join(MCP_HARNESSES)}")
            if h not in seen:
                seen.add(h)
                chosen.append(h)

    cmd, args, env = _resolve_mcp_command(url=url, api_key=api_key, command=command)
    sensitive = "CORTHEON_API_KEY" in env

    standard_entry: dict[str, Any] = {
        "command": cmd,
        "args": list(args),
        "env": dict(env),
    }
    opencode_entry: dict[str, Any] = {
        "type": "local",
        "command": [cmd, *args],
        "enabled": True,
        "environment": dict(env),
    }

    written: list[Path] = []
    for h in chosen:
        if h == "claude":
            dest = (proj / ".mcp.json") if proj is not None else (home / ".claude.json")
            written.append(_merge_json_mcp_config(dest, "mcpServers", dict(standard_entry), sensitive=sensitive))
        elif h == "cursor":
            dest = (proj / ".cursor" / "mcp.json") if proj is not None else (home / ".cursor" / "mcp.json")
            written.append(_merge_json_mcp_config(dest, "mcpServers", dict(standard_entry), sensitive=sensitive))
        elif h == "opencode":
            dest = (proj / "opencode.json") if proj is not None else (home / ".config" / "opencode" / "opencode.json")
            written.append(_merge_json_mcp_config(dest, "mcp", dict(opencode_entry), sensitive=sensitive))
        elif h == "codex":
            dest = (proj / ".codex" / "config.toml") if proj is not None else (home / ".codex" / "config.toml")
            written.append(_upsert_codex_toml(dest, cmd, args, env, sensitive=sensitive))
        elif h == "windsurf":
            dest = (
                (proj / ".windsurf" / "mcp_config.json")
                if proj is not None
                else (home / ".codeium" / "windsurf" / "mcp_config.json")
            )
            written.append(_merge_json_mcp_config(dest, "mcpServers", dict(standard_entry), sensitive=sensitive))

    return written
