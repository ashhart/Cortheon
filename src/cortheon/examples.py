from __future__ import annotations

import ast
import re
import textwrap

MAX_EXAMPLE_CHARS = 2_000
FENCED_BLOCK = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
PYTHON_LANGS = {"", "python", "py", "python3", "pycon"}
# Reject examples that need user-supplied secrets/placeholders or interactive
# input: they cannot pass unattended and their failures would be noise, not
# evidence.
PLACEHOLDER_MARKERS = (
    "<your",
    "your_api_key",
    "your-api-key",
    "your_token",
    "changeme",
    "todo",
    "xxxx",
)
BLOCKED_SNIPPETS = ("input(", "getpass", "sys.argv", "argparse")
ANGLE_PLACEHOLDER = re.compile(r"<[a-zA-Z_][a-zA-Z_ .-]*>")


def extract_runnable_examples(
    description: str | None,
    import_names: list[str],
    limit: int = 3,
) -> list[str]:
    """Extract runnable official example scripts from a package README/description.

    Only fenced python blocks that parse, import the target package, and carry
    no placeholder/interactive markers qualify. Doctest-style blocks are
    converted to plain scripts.
    """
    if not description or limit <= 0:
        return []
    examples: list[str] = []
    seen: set[str] = set()
    for lang, body in FENCED_BLOCK.findall(description):
        if lang.lower() not in PYTHON_LANGS:
            continue
        code = textwrap.dedent(body).strip()
        if ">>>" in code:
            code = doctest_to_script(code) or ""
        if not code or len(code) > MAX_EXAMPLE_CHARS:
            continue
        if not is_runnable_candidate(code, import_names):
            continue
        key = re.sub(r"\s+", " ", code)
        if key in seen:
            continue
        seen.add(key)
        examples.append(code)
        if len(examples) >= limit:
            break
    return examples


def doctest_to_script(block: str) -> str | None:
    lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(">>> ") or stripped.startswith("... "):
            lines.append(stripped[4:])
        elif stripped in {">>>", "..."}:
            continue
        # Non-prompt lines are expected output, not code.
    script = "\n".join(lines).strip()
    return script or None


def is_runnable_candidate(code: str, import_names: list[str]) -> bool:
    lower = code.lower()
    if any(marker in lower for marker in PLACEHOLDER_MARKERS):
        return False
    if any(snippet in code for snippet in BLOCKED_SNIPPETS):
        return False
    if ANGLE_PLACEHOLDER.search(code):
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return imports_target(tree, import_names)


def imports_target(tree: ast.AST, import_names: list[str]) -> bool:
    targets = {name.lower() for name in import_names if name}
    if not targets:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0].lower() in targets:
                    return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0].lower() in targets
        ):
            return True
    return False
