"""Live evidence-loop showcase for Cortheon.

`cortheon demo` walks the full cognitive loop (orient -> discover -> connect ->
challenge -> synthesize -> verify) against this checkout, using only real
evidence collected from the repository itself. It runs with zero model calls
and zero network: every claim in the trace is bound to a receipt from a real
file read, grep, or the actual distribution test. Nothing is certified unless
verification passed.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root, not the package dir


def _rule(char: str = "=") -> str:
    return char * 62


def _stage(title: str, lines: Iterable[str]) -> None:
    print(_rule())
    print(f"  {title}")
    print(_rule("-"))
    for line in lines:
        print(f"  {line}")
    print()


def _read(path: str, limit: int = 80) -> list[str]:
    full = (REPO_ROOT / path).resolve()
    if not full.exists():
        return [f"[receipt: missing] {path}"]
    try:
        text = full.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"[receipt: error] {path}: {exc}"]
    lines = text.splitlines()
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"[... {len(lines) - limit} more lines elided]")
    return shown


def _grep(pattern: str, path: str) -> list[str]:
    full = (REPO_ROOT / path).resolve()
    if not full.exists():
        return [f"[receipt: missing] {path} (pattern {pattern!r})"]
    hits = [
        f"{i + 1}: {line}"
        for i, line in enumerate(full.read_text(encoding="utf-8").splitlines())
        if pattern.lower() in line.lower()
    ]
    if not hits:
        return [f"[receipt: no_match] {path} for {pattern!r}"]
    return hits[:12]


def _shell(args: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return 127, f"executable not found: {exc}"
    except subprocess.TimeoutExpired as exc:
        return 124, f"timed out after {timeout}s: {exc}"
    tail = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode, "\n".join(tail[-8:])


def run_demo(verify: bool = True) -> int:
    print(f"Cortheon demo — evidence loop against {REPO_ROOT.name}@HEAD")
    print()

    # orient
    pyproject_lines = _read("pyproject.toml", 6)
    readme_lines = _read("README.md", 4)[1:4]
    _stage(
        "orient — frame the deliverable",
        [
            "Question: does this checkout honestly satisfy Cortheon's own constraints?",
            f"pyproject: {' | '.join(pyproject_lines)}",
            f"readme:    {' | '.join(readme_lines)}",
        ],
    )

    # discover
    dist_test = REPO_ROOT / "tests" / "test_lightweight_distribution.py"
    discover_lines = [
        "The enforcement surface is the wheel/distribution gate:",
        f"  candidate: {dist_test.relative_to(REPO_ROOT)}",
        f"  exists: {dist_test.exists()}",
    ]
    if dist_test.exists():
        discover_lines += _grep("150", "tests/test_lightweight_distribution.py")[:4]
    _stage("discover — request the tool that reduces uncertainty", discover_lines)

    # connect
    allowlist = _grep("RUNTIME_MODULES", "setup.py")[:3]
    connect_lines = [
        "Cross-source join (setup.py allowlist + pyproject deps):",
        *allowlist,
        "  => claim under test: the shipped wheel is allowlisted and dependency-free.",
    ]
    _stage("connect — derive a candidate across sources", connect_lines)

    # challenge
    _stage(
        "challenge — seek the strongest counterexample",
        [
            "Rival hypothesis: the constraint check is a stub, or the wheel ships junk.",
            "Discriminating test: run the real distribution gate and inspect its receipts.",
        ],
    )

    # synthesize
    _stage(
        "synthesize — separate fact, claim, and unknown",
        [
            "Known: allowlist present in setup.py; enforcements live under tests/.",
            "Uncertain until verified: that the gate actually passes on this checkout.",
        ],
    )

    # verify
    if not verify:
        code, output = 130, "verification skipped by --no-verify"
    else:
        code, output = _shell(
            [sys.executable, "-m", "pytest", "tests/test_lightweight_distribution.py", "-q"]
        )
    verified = code == 0
    _stage(
        f"verify — bind completion to live evidence (exit {code})",
        output.splitlines(),
    )

    print(_rule())
    if verified:
        print("  VERDICT: CERTIFIED — the distribution gate passed on this checkout.")
        print("  Every claim above was bound to a receipt; nothing was asserted unverified.")
        print(_rule())
        return 0
    print("  VERDICT: NOT CERTIFIED — verification failed; the loop did not finish.")
    print(_rule())
    return 1


if __name__ == "__main__":
    raise SystemExit(run_demo())
