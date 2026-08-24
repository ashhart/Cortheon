"""The split MCP runtime must survive both packaging paths intact.

A wheel built from the repository and a wheel rebuilt from the sdist must each
ship every ``cognitive_mcp_core`` module, exclude the repository-only modules,
and expose an MCP surface byte-identical to source mode.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from cognitive_mcp_packaging_support import (
    ROOT,
    build_sdist,
    build_wheel,
    install,
    mcp_surface,
)

CORE_DIR = ROOT / "src/cortheon/cognitive_mcp_core"
REPOSITORY_ONLY = (
    "cortheon/cognitive_benchmark.py",
    "cortheon/benchmark.py",
    "cortheon/parity.py",
)


def _shipped_mcp_modules(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    for excluded in REPOSITORY_ONLY:
        assert excluded not in names, excluded
    assert "cortheon/cognitive_mcp.py" in names
    return {name for name in names if name.startswith("cortheon/cognitive_mcp_core/")}


def test_wheel_and_sdist_rebuilt_wheel_expose_the_same_mcp_surface(tmp_path: Path) -> None:
    expected = {
        f"cortheon/cognitive_mcp_core/{path.name}"
        for path in sorted(CORE_DIR.glob("*.py"))
        if path.name != "__init__.py"
    }
    assert expected, "cognitive_mcp_core must exist"

    wheel = build_wheel(ROOT, tmp_path / "wheel")
    sdist_wheel = build_wheel(build_sdist(tmp_path / "sdist"), tmp_path / "from-sdist")

    assert _shipped_mcp_modules(wheel) == expected
    assert _shipped_mcp_modules(sdist_wheel) == expected

    source_surface = mcp_surface(ROOT / "src")
    assert mcp_surface(install(wheel, tmp_path / "install")) == source_surface
    assert mcp_surface(install(sdist_wheel, tmp_path / "install-sdist")) == source_surface
