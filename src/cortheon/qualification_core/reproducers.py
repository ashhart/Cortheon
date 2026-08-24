"""Minimal, content-free reproducer commands for every observed failure."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from cortheon.qualification_core.models import CellRun, Manifest
from cortheon.qualification_core.taxonomy import _failure_type


def _reproducers(manifest: Manifest, runs: list[CellRun]) -> list[dict[str, Any]]:
    reproducers: list[dict[str, Any]] = []
    try:
        manifest_argument = str(manifest.path.relative_to(Path.cwd().resolve()))
    except ValueError:
        manifest_argument = manifest.path.name
    for run in runs:
        for result in run.results:
            failure_type = _failure_type(result)
            if failure_type is None:
                continue
            identity = {
                "cell": run.cell.cell_id,
                "case_id": result.case_id,
                "repeat": result.repeat,
                "condition": result.condition,
                "failure_type": failure_type,
            }
            fingerprint = hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            command = " ".join(
                [
                    "cortheon-qualify",
                    "--cell",
                    shlex.quote(run.cell.cell_id),
                    "--case-id",
                    shlex.quote(result.case_id),
                    "--repeat",
                    str(result.repeat),
                    "--no-enforce",
                    "--",
                    shlex.quote(manifest_argument),
                ]
            )
            reproducers.append(
                {
                    "failure_id": fingerprint[:16],
                    **identity,
                    "minimal_scope": {
                        "cells": 1,
                        "cases": 1,
                        "repeats": 1,
                        "conditions": len(run.cell.condition_ids),
                    },
                    "command": command,
                }
            )
    return sorted(
        reproducers,
        key=lambda item: (
            item["cell"],
            item["case_id"],
            item["repeat"],
            item["condition"],
        ),
    )
