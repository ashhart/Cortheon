"""Zero-dependency ANSI Cognitive HUD for crisp live terminal output & X demos."""

from __future__ import annotations

import sys


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"


def _use_color() -> bool:
    return sys.stdout.isatty()


def style(text: str, *codes: str) -> str:
    if not _use_color():
        return text
    return "".join(codes) + text + _C.RESET


def banner(title: str, subtitle: str = "") -> None:
    bar = "━" * 74
    print(style(bar, _C.CYAN, _C.BOLD))
    print(style(f"  ⚡ {title}", _C.CYAN, _C.BOLD))
    if subtitle:
        print(style(f"     {subtitle}", _C.DIM))
    print(style(bar, _C.CYAN, _C.BOLD), flush=True)


def event(tag: str, msg: str, color: str = _C.BLUE) -> None:
    print(f"  {style(f'[{tag}]', color, _C.BOLD)} {msg}", flush=True)


def badge_pass(text: str = "PASS") -> str:
    return style(f"✔ {text}", _C.GREEN, _C.BOLD)


def badge_fail(text: str = "FAIL") -> str:
    return style(f"✘ {text}", _C.RED, _C.BOLD)


def badge_warn(text: str) -> str:
    return style(f"▲ {text}", _C.YELLOW, _C.BOLD)


def badge_fork(text: str) -> str:
    return style(f"⑂ {text}", _C.MAGENTA, _C.BOLD)
