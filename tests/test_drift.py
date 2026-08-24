"""Tests for the drift detector.

The drift detector proves the project's thesis on itself: the live answer key
moves, and only a substrate that re-reads it keeps up. These tests exercise the
snapshot/diff/record logic offline with a fake engine, so they are deterministic.
"""

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from cortheon.drift import (
    DEFAULT_WATCH,
    DriftSnapshot,
    diff_snapshots,
    fetch_snapshot,
    load_history,
    record_snapshot,
    run_drift_check,
)


def _fake_engine(versions: dict[str, str]):
    """An engine whose pypi.fetch returns canned versions."""

    class _Meta:
        def __init__(self, version: str) -> None:
            self.version = version

    class _PyPI:
        def __init__(self, mapping: dict[str, str]) -> None:
            self.mapping = mapping

        def fetch(self, pkg: str):
            return (_Meta(self.mapping[pkg]), None)

    return SimpleNamespace(pypi=_PyPI(versions))


class DriftTests(unittest.TestCase):
    def test_fetch_snapshot_reads_current_versions(self) -> None:
        engine = _fake_engine({"fastapi": "0.139.0", "pydantic": "2.5.0", "httpx": "0.28.1"})
        snap = fetch_snapshot(engine)
        self.assertEqual(snap.versions["fastapi"], "0.139.0")
        self.assertEqual(set(snap.versions), set(DEFAULT_WATCH))
        # An empty watch list yields an empty (but valid) snapshot.
        empty_snap = fetch_snapshot(engine, ())
        self.assertEqual(empty_snap.versions, {})

    def test_no_change_when_versions_identical(self) -> None:
        now = datetime.now(UTC)
        prev = DriftSnapshot(taken_at=now, versions={"fastapi": "0.139.0"})
        curr = DriftSnapshot(taken_at=now + timedelta(hours=1), versions={"fastapi": "0.139.0"})
        self.assertEqual(diff_snapshots(curr, prev), [])

    def test_change_reported_when_version_advances(self) -> None:
        now = datetime.now(UTC)
        prev = DriftSnapshot(taken_at=now, versions={"fastapi": "0.139.0"})
        curr = DriftSnapshot(taken_at=now + timedelta(days=1), versions={"fastapi": "0.140.0"})
        changes = diff_snapshots(curr, prev)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].package, "fastapi")
        self.assertEqual(changes[0].from_version, "0.139.0")
        self.assertEqual(changes[0].to_version, "0.140.0")

    def test_new_package_treated_as_change(self) -> None:
        now = datetime.now(UTC)
        prev = DriftSnapshot(taken_at=now, versions={"httpx": "0.28.1"})
        curr = DriftSnapshot(taken_at=now, versions={"httpx": "0.28.1", "starlette": "0.40.0"})
        changes = diff_snapshots(curr, prev)
        self.assertEqual([c.package for c in changes], ["starlette"])
        self.assertIsNone(changes[0].from_version)

    def test_first_run_records_baseline_and_reports_no_drift(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.json"
            engine = _fake_engine({"fastapi": "0.139.0"})
            report = run_drift_check(engine, log, ("fastapi",))
            self.assertFalse(report.drifted)
            self.assertIsNone(report.previous_at)
            self.assertEqual(len(load_history(log)), 1)

    def test_second_run_detects_drift_and_appends(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.json"
            run_drift_check(_fake_engine({"fastapi": "0.139.0"}), log, ("fastapi",))
            report = run_drift_check(_fake_engine({"fastapi": "0.140.0"}), log, ("fastapi",))
            self.assertTrue(report.drifted)
            self.assertEqual(len(report.changes), 1)
            self.assertEqual(report.changes[0].to_version, "0.140.0")
            self.assertEqual(len(load_history(log)), 2)

    def test_record_and_load_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.json"
            now = datetime.now(UTC)
            record_snapshot(log, DriftSnapshot(taken_at=now, versions={"httpx": "0.28.1"}))
            history = load_history(log)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].versions, {"httpx": "0.28.1"})

    def test_load_history_handles_missing_and_corrupt(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.json"
            self.assertEqual(load_history(log), [])  # missing
            log.write_text("not json", encoding="utf-8")
            self.assertEqual(load_history(log), [])  # corrupt -> empty, not crash


if __name__ == "__main__":
    unittest.main()
