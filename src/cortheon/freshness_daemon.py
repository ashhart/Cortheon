"""Refresh evidence for actively recommended packages before it expires."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from cortheon.engine import CortheonEngine
from cortheon.models import utc_now

DEFAULT_SCAN_INTERVAL_SECONDS = 5 * 60
DEFAULT_REFRESH_LEAD_SECONDS = 60 * 60
DEFAULT_WATCH_LIST = ("fastapi", "pydantic", "httpx")
DEFAULT_RECENT_REPORTS = 20


@dataclass(slots=True)
class FreshnessEntry:
    package: str
    version: str | None
    last_fetched: datetime | None
    expires_at: datetime | None
    status: str
    source_types: list[str]


@dataclass(slots=True)
class FreshnessReport:
    scanned_at: datetime
    watch_list: list[str]
    entries: list[FreshnessEntry]
    refreshed: list[str]
    skipped: list[str]
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class FreshnessDaemon:
    """Background daemon that proactively refreshes evidence for active packages."""

    def __init__(
        self,
        engine: CortheonEngine,
        *,
        scan_interval_seconds: int = DEFAULT_SCAN_INTERVAL_SECONDS,
        refresh_lead_seconds: int = DEFAULT_REFRESH_LEAD_SECONDS,
        watch_list: tuple[str, ...] = DEFAULT_WATCH_LIST,
        recent_reports: int = DEFAULT_RECENT_REPORTS,
    ) -> None:
        self.engine = engine
        self.scan_interval_seconds = scan_interval_seconds
        self.refresh_lead_seconds = refresh_lead_seconds
        self.watch_list = list(watch_list)
        self.recent_reports = recent_reports
        self._running = False

    def scan_active_packages(self) -> list[FreshnessEntry]:
        """Scan the ledger for packages that are actively being recommended."""
        entries: list[FreshnessEntry] = []
        seen: set[str] = set()

        reports_dir = self.engine.ledger.reports_dir
        if reports_dir.is_dir():
            report_files = sorted(
                reports_dir.glob("*-recommend-*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[: self.recent_reports]

            for report_path in report_files:
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                for candidate in payload.get("candidates", []):
                    package = candidate.get("package")
                    if not package or package in seen:
                        continue
                    seen.add(package)
                    entries.append(_entry_from_report(candidate))

        for package in self.watch_list:
            if package not in seen:
                seen.add(package)
                entries.append(
                    FreshnessEntry(
                        package=package,
                        version=None,
                        last_fetched=None,
                        expires_at=None,
                        status="watched",
                        source_types=[],
                    )
                )

        return entries

    def find_stale_entries(self, entries: list[FreshnessEntry]) -> list[FreshnessEntry]:
        """Find entries whose evidence is stale or about to expire."""
        now = utc_now()
        stale: list[FreshnessEntry] = []
        for entry in entries:
            if entry.expires_at is None:
                entry.status = "stale"
                stale.append(entry)
                continue
            if entry.expires_at <= now:
                entry.status = "stale"
                stale.append(entry)
                continue
            if entry.expires_at <= now + timedelta(seconds=self.refresh_lead_seconds):
                entry.status = "expiring"
                stale.append(entry)
        return stale

    def refresh_entry(self, entry: FreshnessEntry) -> bool:
        """Refresh evidence for a single package. Returns True on success."""
        try:
            report = self.engine.inspect_package(
                entry.package,
                write_report=True,
            )
            entry.version = report.version
            entry.last_fetched = report.fetched_at
            entry.status = "refreshed"
            entry.source_types = [item.source_type for item in report.evidence]
            for evidence in report.evidence:
                if evidence.expires_at and (
                    entry.expires_at is None or evidence.expires_at > entry.expires_at
                ):
                    entry.expires_at = evidence.expires_at
            return True
        except Exception:
            entry.status = "refresh_failed"
            return False

    def run_once(self) -> FreshnessReport:
        """Run one scan-refresh cycle."""
        scanned_at = utc_now()
        entries = self.scan_active_packages()
        stale = self.find_stale_entries(entries)

        refreshed: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []

        for entry in stale:
            if entry.status in {"watched", "stale", "expiring"}:
                if self.refresh_entry(entry):
                    refreshed.append(entry.package)
                else:
                    errors.append(f"{entry.package}: refresh failed")
            else:
                skipped.append(entry.package)

        skipped.extend(
            entry.package
            for entry in entries
            if entry not in stale and entry.status not in ("stale", "expiring", "watched")
        )

        notes = []
        if refreshed:
            notes.append(f"Refreshed evidence for {len(refreshed)} package(s).")
        if skipped:
            notes.append(f"{len(skipped)} package(s) had current evidence; skipped.")
        if not stale:
            notes.append("No stale or expiring evidence found.")

        return FreshnessReport(
            scanned_at=scanned_at,
            watch_list=self.watch_list,
            entries=entries,
            refreshed=refreshed,
            skipped=skipped,
            errors=errors,
            notes=notes,
        )

    def run_loop(self, *, max_iterations: int | None = None) -> None:
        """Run the daemon loop. Blocks until interrupted or max_iterations reached."""
        self._running = True
        iteration = 0
        try:
            while self._running:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                report = self.run_once()
                self._write_freshness_report(report)
                iteration += 1
                if max_iterations is None:
                    time.sleep(self.scan_interval_seconds)
        except KeyboardInterrupt:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def _write_freshness_report(self, report: FreshnessReport) -> None:
        """Write the freshness report to the ledger."""
        self.engine.ledger.ensure()
        timestamp = report.scanned_at.strftime("%Y%m%dT%H%M%SZ")
        path = self.engine.ledger.reports_dir / f"{timestamp}-freshness.json"
        path.write_text(
            json.dumps(_report_to_dict(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _entry_from_report(candidate: dict[str, Any]) -> FreshnessEntry:
    """Build a FreshnessEntry from a recommendation report candidate dict."""
    evidence_list = candidate.get("evidence", [])
    expires_at = None
    last_fetched = None
    source_types: list[str] = []

    for item in evidence_list:
        source_type = item.get("source_type")
        if source_type:
            source_types.append(source_type)
        retrieved = item.get("retrieved_at")
        if retrieved:
            from cortheon.models import parse_datetime

            parsed = parse_datetime(retrieved)
            if parsed and (last_fetched is None or parsed > last_fetched):
                last_fetched = parsed
        expires = item.get("expires_at")
        if expires:
            from cortheon.models import parse_datetime

            parsed = parse_datetime(expires)
            if parsed and (expires_at is None or parsed > expires_at):
                expires_at = parsed

    return FreshnessEntry(
        package=candidate.get("package", "unknown"),
        version=candidate.get("version"),
        last_fetched=last_fetched,
        expires_at=expires_at,
        status="current" if expires_at else "unknown",
        source_types=source_types,
    )


def _report_to_dict(report: FreshnessReport) -> dict[str, Any]:
    """Convert a FreshnessReport to a JSON-serializable dict."""
    return {
        "scanned_at": report.scanned_at.isoformat(),
        "watch_list": report.watch_list,
        "entries": [
            {
                "package": entry.package,
                "version": entry.version,
                "last_fetched": entry.last_fetched.isoformat() if entry.last_fetched else None,
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "status": entry.status,
                "source_types": entry.source_types,
            }
            for entry in report.entries
        ],
        "refreshed": report.refreshed,
        "skipped": report.skipped,
        "errors": report.errors,
        "notes": report.notes,
    }


def run_freshness_check(
    engine: CortheonEngine,
    *,
    watch_list: tuple[str, ...] = DEFAULT_WATCH_LIST,
    recent_reports: int = DEFAULT_RECENT_REPORTS,
) -> FreshnessReport:
    """Run a single freshness check (for CLI/cron use)."""
    daemon = FreshnessDaemon(
        engine,
        watch_list=watch_list,
        recent_reports=recent_reports,
    )
    return daemon.run_once()
