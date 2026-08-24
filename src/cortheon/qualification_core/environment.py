"""Host, runtime, and repository environment probes used around each cell."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

from cortheon.qualification_core.models import Cell, QualificationError


def _cell_namespace(
    cell: Cell,
    *,
    repository: Path,
    api_key: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        repository=repository,
        host=cell.host,
        opencode=cell.opencode,
        pi=cell.pi,
        provider=cell.provider,
        base_url=cell.base_url,
        api_key=api_key,
        model_id=cell.model_id,
        runtime_url=cell.runtime_url,
        cases=cell.cases,
        repeats=cell.repeats,
        suite=cell.suite,
        seed=cell.seed,
        timeout_seconds=cell.timeout_seconds,
        context_tokens=cell.context_tokens,
        output_tokens=cell.output_tokens,
        max_steps=cell.max_steps,
        reasoning=cell.reasoning,
    )


def _command_version(command: str) -> str:
    try:
        completed = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError(f"host command {Path(command).name!r} is unavailable") from exc
    if completed.returncode != 0:
        raise QualificationError(f"host command {Path(command).name!r} failed its version check")
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0][:200] if lines else "unknown"


def _public_runtime_health(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": value.get("ok") is True,
        "service": value.get("service"),
        "version": value.get("version"),
        "protocol_version": value.get("protocol_version"),
        "source_fingerprint": value.get("source_fingerprint"),
        "storage": value.get("storage"),
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("cortheon")
    except importlib.metadata.PackageNotFoundError:
        return "source-checkout"


def _git_revision(repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None
