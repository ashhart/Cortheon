"""Compare current PyPI versions with a prior snapshot and report drift."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cortheon.models import to_jsonable, utc_now

DEFAULT_WATCH = ("fastapi", "pydantic", "httpx")


@dataclass(slots=True)
class DriftSnapshot:
    """One point-in-time reading of the live answer key."""

    taken_at: datetime
    versions: dict[str, str]  # package -> current version


@dataclass(slots=True)
class DriftChange:
    package: str
    from_version: str | None
    to_version: str
    taken_at: datetime


@dataclass(slots=True)
class DriftReport:
    taken_at: datetime
    snapshot: DriftSnapshot
    changes: list[DriftChange] = field(default_factory=list)
    previous_at: datetime | None = None

    @property
    def drifted(self) -> bool:
        return bool(self.changes)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)  # type: ignore[no-any-return]


def fetch_snapshot(engine, watch: tuple[str, ...] = DEFAULT_WATCH) -> DriftSnapshot:
    """Read the current live versions of the watch list right now."""
    versions: dict[str, str] = {}
    for pkg in watch:
        try:
            versions[pkg] = engine.pypi.fetch(pkg)[0].version
        except Exception:  # pragma: no cover - network dependent
            versions[pkg] = "unavailable"
    return DriftSnapshot(taken_at=utc_now(), versions=versions)


def diff_snapshots(current: DriftSnapshot, previous: DriftSnapshot | None) -> list[DriftChange]:
    """Which packages changed version since the last snapshot (or are new)."""
    if previous is None:
        return []
    changes: list[DriftChange] = []
    for pkg, version in current.versions.items():
        old = previous.versions.get(pkg)
        if version == "unavailable":
            continue
        if old != version:
            changes.append(
                DriftChange(
                    package=pkg, from_version=old, to_version=version, taken_at=current.taken_at
                )
            )
    return changes


def load_history(log_path: Path) -> list[DriftSnapshot]:
    """Read the append-only drift log. Empty if none exists yet."""
    if not log_path.exists():
        return []
    try:
        raw = json.loads(log_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    out: list[DriftSnapshot] = []
    for entry in raw if isinstance(raw, list) else []:
        try:
            out.append(
                DriftSnapshot(
                    taken_at=datetime.fromisoformat(entry["taken_at"]),
                    versions=dict(entry.get("versions", {})),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return out


def record_snapshot(log_path: Path, snapshot: DriftSnapshot) -> None:
    """Append a snapshot to the drift log (created if absent)."""
    history = load_history(log_path)
    history.append(snapshot)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"taken_at": s.taken_at.isoformat(), "versions": s.versions} for s in history]
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_drift_check(engine, log_path: Path, watch: tuple[str, ...] = DEFAULT_WATCH) -> DriftReport:
    """One full pass: fetch current, diff against last, record, report."""
    history = load_history(log_path)
    previous = history[-1] if history else None
    current = fetch_snapshot(engine, watch)
    changes = diff_snapshots(current, previous)
    record_snapshot(log_path, current)
    return DriftReport(
        taken_at=current.taken_at,
        snapshot=current,
        changes=changes,
        previous_at=previous.taken_at if previous else None,
    )
