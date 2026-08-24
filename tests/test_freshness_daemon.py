import tempfile
import unittest
from pathlib import Path

from cortheon.engine import CortheonEngine
from cortheon.freshness_daemon import (
    DEFAULT_WATCH_LIST,
    FreshnessDaemon,
    FreshnessEntry,
    run_freshness_check,
)
from cortheon.ledger import EvidenceLedger
from cortheon.models import utc_now


class FreshnessDaemonTests(unittest.TestCase):
    def test_default_watch_list(self) -> None:
        self.assertIn("fastapi", DEFAULT_WATCH_LIST)
        self.assertIn("pydantic", DEFAULT_WATCH_LIST)
        self.assertIn("httpx", DEFAULT_WATCH_LIST)

    def test_scan_active_packages_finds_watch_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            daemon = FreshnessDaemon(engine, watch_list=("fastapi", "pydantic"))
            entries = daemon.scan_active_packages()
            packages = [entry.package for entry in entries]
            self.assertIn("fastapi", packages)
            self.assertIn("pydantic", packages)

    def test_find_stale_entries_with_no_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            daemon = FreshnessDaemon(engine)
            entries = [
                FreshnessEntry(
                    package="testpkg",
                    version="1.0.0",
                    last_fetched=None,
                    expires_at=None,
                    status="unknown",
                    source_types=[],
                ),
            ]
            stale = daemon.find_stale_entries(entries)
            self.assertEqual(len(stale), 1)
            self.assertEqual(stale[0].status, "stale")

    def test_find_stale_entries_with_expired(self) -> None:
        from datetime import timedelta

        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            daemon = FreshnessDaemon(engine)
            entries = [
                FreshnessEntry(
                    package="testpkg",
                    version="1.0.0",
                    last_fetched=utc_now() - timedelta(days=10),
                    expires_at=utc_now() - timedelta(days=3),
                    status="current",
                    source_types=["pypi_package_metadata"],
                ),
            ]
            stale = daemon.find_stale_entries(entries)
            self.assertEqual(len(stale), 1)
            self.assertEqual(stale[0].status, "stale")

    def test_find_stale_entries_with_current(self) -> None:
        from datetime import timedelta

        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            daemon = FreshnessDaemon(engine)
            entries = [
                FreshnessEntry(
                    package="testpkg",
                    version="1.0.0",
                    last_fetched=utc_now(),
                    expires_at=utc_now() + timedelta(days=6),
                    status="current",
                    source_types=["pypi_package_metadata"],
                ),
            ]
            stale = daemon.find_stale_entries(entries)
            self.assertEqual(len(stale), 0)

    def test_find_stale_entries_with_expiring(self) -> None:
        from datetime import timedelta

        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            daemon = FreshnessDaemon(engine, refresh_lead_seconds=7200)
            entries = [
                FreshnessEntry(
                    package="testpkg",
                    version="1.0.0",
                    last_fetched=utc_now(),
                    expires_at=utc_now() + timedelta(seconds=3600),
                    status="current",
                    source_types=["pypi_package_metadata"],
                ),
            ]
            stale = daemon.find_stale_entries(entries)
            self.assertEqual(len(stale), 1)
            self.assertEqual(stale[0].status, "expiring")

    def test_run_once_with_watch_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            daemon = FreshnessDaemon(
                engine,
                watch_list=("fastapi",),
                recent_reports=0,
            )
            report = daemon.run_once()
            self.assertIn("fastapi", report.refreshed)
            self.assertEqual(len(report.errors), 0)

    def test_run_freshness_check_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = CortheonEngine(ledger=EvidenceLedger(Path(tmp) / ".cortheon"))
            report = run_freshness_check(
                engine,
                watch_list=("fastapi",),
                recent_reports=0,
            )
            self.assertIn("fastapi", report.watch_list)
            self.assertGreater(len(report.entries), 0)


if __name__ == "__main__":
    unittest.main()
