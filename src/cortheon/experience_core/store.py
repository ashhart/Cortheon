"""Experience recording and aggregate lesson queries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cortheon.experience_core._compat import facade
from cortheon.experience_core.models import (
    FailureSignature,
    RecoveryStrategy,
    VerificationContract,
)
from cortheon.experience_core.persistence import ExperiencePersistence


class ExperienceStore(ExperiencePersistence):
    """Tenant-namespaced, append-only SQLite experience store."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        max_events: int = 100_000,
    ) -> None:
        if not 1 <= max_events <= 1_000_000:
            raise ValueError("max_events must be between 1 and 1000000")
        self.path = Path(path).expanduser().resolve()
        self.namespace = facade()._namespace(namespace)
        self.max_events = max_events
        self._ensure()

    def record_failure(
        self,
        signature: FailureSignature,
        *,
        verification: VerificationContract,
        attempted_strategy: RecoveryStrategy | None = None,
        latency_ms: float | int | None = None,
    ) -> dict[str, Any]:
        """Record one failed attempt without retaining its request or response."""

        return self._append(
            result="failure",
            signature=signature,
            strategy=attempted_strategy,
            verification=verification,
            latency_bucket=facade()._latency_bucket(latency_ms),
        )

    def record_recovery(
        self,
        signature: FailureSignature,
        *,
        strategy: RecoveryStrategy,
        verification: VerificationContract,
        latency_ms: float | int | None = None,
    ) -> dict[str, Any]:
        """Record a strategy only after a strong verification contract passes."""

        if not verification.satisfied:
            raise ValueError(
                "a recovery is learnable only after its verification contract is satisfied"
            )
        return self._append(
            result="recovered",
            signature=signature,
            strategy=strategy,
            verification=verification,
            latency_bucket=facade()._latency_bucket(latency_ms),
        )

    def record_attempt(
        self,
        signature: FailureSignature,
        *,
        outcome: dict[str, Any],
        strategy: RecoveryStrategy | None = None,
        latency_ms: float | int | None = None,
    ) -> dict[str, Any]:
        """Map an outcome contract to a failure or verified recovery event."""

        verification = facade().VerificationContract.from_outcome(outcome)
        if outcome.get("verified_completion") is True:
            if strategy is None:
                raise ValueError("a verified recovery requires a named recovery strategy")
            return self.record_recovery(
                signature,
                strategy=strategy,
                verification=verification,
                latency_ms=latency_ms,
            )
        return self.record_failure(
            signature,
            verification=verification,
            attempted_strategy=strategy,
            latency_ms=latency_ms,
        )

    def relevant_lessons(
        self,
        *,
        capability: str,
        task_family: str | None = None,
        context_tags: Iterable[str] = (),
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Return ranked, aggregate lessons without returning individual events."""

        api = facade()
        capability_id = api._identifier(capability, "capability")
        task_family_id = (
            api._identifier(task_family, "task_family") if task_family is not None else None
        )
        query_tags = frozenset(api._identifiers(context_tags, "context_tags", maximum=12))
        bounded_limit = api._limit(limit)
        where = ["namespace = ?", "capability = ?"]
        parameters: list[Any] = [self.namespace, capability_id]
        if task_family_id is not None:
            where.append("task_family = ?")
            parameters.append(task_family_id)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    signature_key, capability, task_family, stage,
                    failure_kind, failure_code, context_tags_json,
                    COUNT(*) AS events,
                    SUM(CASE WHEN result = 'failure' THEN 1 ELSE 0 END)
                        AS failures,
                    SUM(CASE WHEN result = 'recovered' THEN 1 ELSE 0 END)
                        AS recoveries,
                    MAX(recorded_at) AS last_seen
                FROM experience_events
                WHERE {" AND ".join(where)}
                GROUP BY
                    signature_key, capability, task_family, stage,
                    failure_kind, failure_code, context_tags_json
                """,
                parameters,
            ).fetchall()
            candidates: list[dict[str, Any]] = []
            for row in rows:
                row_tags = frozenset(api.json.loads(row["context_tags_json"]))
                overlap = len(query_tags.intersection(row_tags))
                failures = int(row["failures"])
                recoveries = int(row["recoveries"])
                candidates.append(
                    {
                        "signature": row["signature_key"],
                        "capability": row["capability"],
                        "task_family": row["task_family"],
                        "stage": row["stage"],
                        "failure_kind": row["failure_kind"],
                        "failure_code": row["failure_code"],
                        "context_tags": sorted(row_tags),
                        "recurrences": failures,
                        "verified_recoveries": recoveries,
                        "recovery_rate": api._recovery_rate(recoveries, failures),
                        "unresolved_recurrences": max(failures - recoveries, 0),
                        "last_seen": row["last_seen"],
                        "_overlap": overlap,
                    }
                )
            candidates.sort(
                key=lambda item: (
                    item["_overlap"],
                    item["verified_recoveries"],
                    item["recurrences"],
                    item["last_seen"],
                ),
                reverse=True,
            )
            selected = candidates[:bounded_limit]
            for lesson in selected:
                lesson["strategies"] = self._strategies(connection, lesson["signature"])
                lesson.pop("_overlap", None)
            return selected

    def lessons_for(
        self,
        signature: FailureSignature,
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        api = facade()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    signature_key, capability, task_family, stage,
                    failure_kind, failure_code, context_tags_json,
                    COUNT(*) AS events,
                    SUM(CASE WHEN result = 'failure' THEN 1 ELSE 0 END)
                        AS failures,
                    SUM(CASE WHEN result = 'recovered' THEN 1 ELSE 0 END)
                        AS recoveries,
                    MAX(recorded_at) AS last_seen
                FROM experience_events
                WHERE namespace = ? AND signature_key = ?
                GROUP BY
                    signature_key, capability, task_family, stage,
                    failure_kind, failure_code, context_tags_json
                """,
                (self.namespace, signature.key),
            ).fetchone()
            if row is None:
                return []
            failures = int(row["failures"])
            recoveries = int(row["recoveries"])
            return [
                {
                    "signature": row["signature_key"],
                    "capability": row["capability"],
                    "task_family": row["task_family"],
                    "stage": row["stage"],
                    "failure_kind": row["failure_kind"],
                    "failure_code": row["failure_code"],
                    "context_tags": list(api.json.loads(row["context_tags_json"])),
                    "recurrences": failures,
                    "verified_recoveries": recoveries,
                    "recovery_rate": api._recovery_rate(recoveries, failures),
                    "unresolved_recurrences": max(failures - recoveries, 0),
                    "last_seen": row["last_seen"],
                    "strategies": self._strategies(
                        connection,
                        signature.key,
                        limit=api._limit(limit),
                    ),
                }
            ]

    def capability_outcomes(self) -> dict[str, Any]:
        """Summarize failures and verified recoveries for this tenant only."""

        api = facade()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    capability,
                    COUNT(*) AS attempts,
                    SUM(CASE WHEN result = 'failure' THEN 1 ELSE 0 END)
                        AS failures,
                    SUM(CASE WHEN result = 'recovered' THEN 1 ELSE 0 END)
                        AS recoveries,
                    COUNT(DISTINCT signature_key) AS failure_classes,
                    MAX(recorded_at) AS last_seen
                FROM experience_events
                WHERE namespace = ?
                GROUP BY capability
                ORDER BY capability
                """,
                (self.namespace,),
            ).fetchall()
            capabilities: dict[str, Any] = {}
            total_failures = 0
            total_recoveries = 0
            for row in rows:
                failures = int(row["failures"])
                recoveries = int(row["recoveries"])
                total_failures += failures
                total_recoveries += recoveries
                capabilities[str(row["capability"])] = {
                    "attempts": int(row["attempts"]),
                    "failure_recurrences": failures,
                    "verified_recoveries": recoveries,
                    "recovery_rate": api._recovery_rate(recoveries, failures),
                    "unresolved_recurrences": max(failures - recoveries, 0),
                    "failure_classes": int(row["failure_classes"]),
                    "last_seen": row["last_seen"],
                }
            return {
                "schema_version": api.EXPERIENCE_SCHEMA_VERSION,
                "namespace": self.namespace,
                "max_events": self.max_events,
                "stored_events": total_failures + total_recoveries,
                "failure_recurrences": total_failures,
                "verified_recoveries": total_recoveries,
                "recovery_rate": api._recovery_rate(total_recoveries, total_failures),
                "capabilities": capabilities,
            }
