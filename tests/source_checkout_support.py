"""Copy this checkout the way a downloaded source tree arrives: with no VCS data.

A release is built from an extracted archive, a CI artifact, or a container
COPY at least as often as from a clone, so the packaging suite has to be able
to stage a tree that was never a git checkout. Deriving the file list from
``git ls-files`` would quietly make ``.git`` a build input; walking the
filesystem and skipping generated and machine-local state does not.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Version-control data, build output, environments, caches, and local runtime
# state: everything a fresh source tree would arrive without.
SKIPPED_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cortheon",
        ".cortheon-test",
        ".learn-layer",
        ".learn-layer-test",
        ".vortex-test",
        ".zcode",
        ".claude",
        ".DS_Store",
    }
)
SKIPPED_SUFFIXES = (".pyc", ".pyo", ".egg-info")


def _ignored(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIPPED_NAMES or name.endswith(SKIPPED_SUFFIXES)}


def copy_source_tree(source: Path, destination: Path) -> Path:
    """Mirror ``source`` into ``destination`` without VCS or build state."""

    shutil.copytree(source, destination, ignore=_ignored, symlinks=True)
    if (destination / ".git").exists():
        raise AssertionError("the staged tree still carries version-control data")
    return destination
