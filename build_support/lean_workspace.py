"""A private, self-removing staging directory for one Cortheon build."""

# setuptools puts every intermediate under one shared ``build/`` tree and
# stages source archives in a ``<name>-<version>/`` directory beside setup.py,
# so two builds in the same checkout write to the same paths: the first to
# finish deletes the tree the second is still reading, and whichever wins
# decides what ships. Each build takes a workspace from here instead and
# points its staging directories inside it, so concurrent builds never share
# a path and a finished build leaves nothing behind. Only comments carry the
# reasoning: build-only modules ship compacted into the source archive, which
# keeps docstrings and drops comments, and this one is never introspected.

from __future__ import annotations

import atexit
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

WORKSPACE_PREFIX = "lean-"
# A workspace this old can only belong to a build that was killed before it
# could clean up: a full build of this project takes seconds.
STALE_AFTER_SECONDS = 6 * 60 * 60


def open_workspace(root: Path) -> Path:
    """Create this build's staging directory under ``root/build``."""

    base = root / "build"
    base.mkdir(parents=True, exist_ok=True)
    discard_abandoned(base)
    workspace = base / f"{WORKSPACE_PREFIX}{os.getpid()}-{uuid.uuid4().hex}"
    # The name is unique per call and never reused: an inherited directory
    # would hand this build whatever the previous one left in it. Not
    # exist_ok, because a collision means the name is not unique after all,
    # which is the one condition this module exists to rule out.
    workspace.mkdir()
    atexit.register(discard_workspace, workspace)
    return workspace


def discard_workspace(path: Path) -> None:
    """Remove a workspace, reporting a failure rather than raising."""

    # This runs at interpreter exit, after the artifact is already written, so
    # raising would turn a disk-space problem into a failed build. Leftovers
    # are still worth saying out loud.
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        print(f"lean_workspace: could not remove {path}", file=sys.stderr)


def discard_abandoned(base: Path) -> None:
    """Drop workspaces left by builds that died before their own cleanup."""

    # Bounds what a checkout can accumulate even when builds are killed
    # outright. Only this module's own directories are considered, and only
    # once they are far older than any build could still be using them.
    cutoff = time.time() - STALE_AFTER_SECONDS
    for candidate in base.glob(f"{WORKSPACE_PREFIX}*"):
        if candidate.is_dir() and candidate.stat().st_mtime < cutoff:
            shutil.rmtree(candidate, ignore_errors=True)
