"""Append-only SQLite mechanics for the experience store."""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING, Any

from cortheon.experience_core._compat import facade

if TYPE_CHECKING:
    from cortheon.experience_core.models import (
        FailureSignature,
        RecoveryStrategy,
        VerificationContract,
    )


class ExperiencePersistence:
    """Private persistence half of ``ExperienceStore``."""

    path: Any
    namespace: str
    max_events: int

    def _append(
        self,
        *,
        result: str,
        signature: FailureSignature,
        strategy: RecoveryStrategy | None,
        verification: VerificationContract,
        latency_bucket: str,
    ) -> dict[str, Any]:
        api = facade()
        if result not in api._RESULTS:
            raise ValueError(f"unsupported experience result: {result}")
        if result == "recovered" and strategy is None:
            raise ValueError("recovered events require a recovery strategy")
        event_id = "experience_" + api.uuid.uuid4().hex
        recorded_at = api.datetime.now(api.UTC).replace(microsecond=0).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stored_events = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM experience_events
                    WHERE namespace = ?
                    """,
                    (self.namespace,),
                ).fetchone()[0]
            )
            if stored_events >= self.max_events:
                raise RuntimeError(
                    "experience store capacity reached; rotate the tenant "
                    "experience database before learning more events"
                )
            connection.execute(
                """
                INSERT INTO experience_events(
                    event_id, namespace, recorded_at, signature_key,
                    capability, task_family, stage, failure_kind, failure_code,
                    context_tags_json, result, strategy_id, action_ids_json,
                    assurance, required_checks_json, passed_checks_json,
                    evidence_kinds_json, evidence_count, contract_satisfied,
                    verified_completion, latency_bucket
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?)
                """,
                (
                    event_id,
                    self.namespace,
                    recorded_at,
                    signature.key,
                    signature.capability,
                    signature.task_family,
                    signature.stage,
                    signature.failure_kind,
                    signature.failure_code,
                    api.json.dumps(signature.context_tags),
                    result,
                    strategy.strategy_id if strategy is not None else None,
                    api.json.dumps(strategy.action_ids if strategy is not None else ()),
                    verification.assurance,
                    api.json.dumps(verification.required_checks),
                    api.json.dumps(verification.passed_checks),
                    api.json.dumps(verification.evidence_kinds),
                    verification.evidence_count,
                    int(verification.satisfied),
                    int(result == "recovered" and verification.satisfied),
                    latency_bucket,
                ),
            )
        return {
            "schema_version": api.EXPERIENCE_SCHEMA_VERSION,
            "event_id": event_id,
            "recorded_at": recorded_at,
            "signature": signature.key,
            "result": result,
            "verified_completion": (result == "recovered" and verification.satisfied),
        }

    def _strategies(
        self,
        connection: sqlite3.Connection,
        signature_key: str,
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        api = facade()
        rows = connection.execute(
            """
            SELECT
                strategy_id, action_ids_json,
                MAX(CASE WHEN result = 'recovered' THEN
                    CASE assurance
                        WHEN 'independent_grader' THEN 6
                        WHEN 'repository_tests' THEN 5
                        WHEN 'behavioral' THEN 4
                        WHEN 'agent_tools' THEN 3
                        WHEN 'runtime_bind' THEN 3
                        WHEN 'patch_applied' THEN 2
                        WHEN 'structural' THEN 2
                        WHEN 'policy' THEN 1
                        ELSE 0
                    END
                ELSE 0 END) AS assurance_rank,
                MAX(CASE WHEN result = 'recovered'
                    THEN required_checks_json END) AS required_checks_json,
                MAX(CASE WHEN result = 'recovered'
                    THEN evidence_kinds_json END) AS evidence_kinds_json,
                COUNT(*) AS attempts,
                SUM(CASE
                    WHEN result = 'recovered' AND verified_completion = 1
                    THEN 1 ELSE 0
                END) AS successes,
                SUM(CASE WHEN result = 'failure' THEN 1 ELSE 0 END)
                    AS failures,
                MAX(CASE WHEN result = 'recovered' THEN recorded_at END)
                    AS last_verified
            FROM experience_events
            WHERE namespace = ?
              AND signature_key = ?
              AND strategy_id IS NOT NULL
            GROUP BY
                strategy_id, action_ids_json
            HAVING successes > 0
            ORDER BY
                successes DESC,
                assurance_rank DESC,
                last_verified DESC
            LIMIT ?
            """,
            (self.namespace, signature_key, api._limit(limit)),
        ).fetchall()
        return [
            {
                "strategy_id": row["strategy_id"],
                "action_ids": list(api.json.loads(row["action_ids_json"])),
                "attempts": int(row["attempts"]),
                "failed_attempts": int(row["failures"]),
                "verified_successes": int(row["successes"]),
                "verified_success_rate": api._rate(int(row["successes"]), int(row["attempts"])),
                "assurance": api._assurance_for_rank(int(row["assurance_rank"])),
                "required_checks": list(api.json.loads(row["required_checks_json"])),
                "evidence_kinds": list(api.json.loads(row["evidence_kinds_json"])),
                "last_verified": row["last_verified"],
            }
            for row in rows
        ]

    def _ensure(self) -> None:
        api = facade()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            self.path.parent.chmod(0o700)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experience_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    namespace TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    signature_key TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    task_family TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    failure_kind TEXT NOT NULL,
                    failure_code TEXT NOT NULL,
                    context_tags_json TEXT NOT NULL,
                    result TEXT NOT NULL CHECK(result IN ('failure', 'recovered')),
                    strategy_id TEXT,
                    action_ids_json TEXT NOT NULL,
                    assurance TEXT NOT NULL,
                    required_checks_json TEXT NOT NULL,
                    passed_checks_json TEXT NOT NULL,
                    evidence_kinds_json TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL CHECK(evidence_count >= 0),
                    contract_satisfied INTEGER NOT NULL CHECK(
                        contract_satisfied IN (0, 1)
                    ),
                    verified_completion INTEGER NOT NULL CHECK(
                        verified_completion IN (0, 1)
                    ),
                    latency_bucket TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS experience_namespace_capability
                    ON experience_events(namespace, capability, task_family);
                CREATE INDEX IF NOT EXISTS experience_namespace_signature
                    ON experience_events(namespace, signature_key, result);
                CREATE TRIGGER IF NOT EXISTS experience_events_no_update
                BEFORE UPDATE ON experience_events
                BEGIN
                    SELECT RAISE(ABORT, 'experience events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS experience_events_no_delete
                BEFORE DELETE ON experience_events
                BEGIN
                    SELECT RAISE(ABORT, 'experience events are append-only');
                END;
                """
            )
        with contextlib.suppress(OSError):
            api.os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = facade().sqlite3.connect(self.path, timeout=10)
        connection.row_factory = facade().sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
