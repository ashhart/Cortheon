from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from cortheon.telemetry_core._compat import facade


class ProxyMetrics:
    """Thread-safe aggregate and JSONL sink for proxy outcome telemetry."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        max_file_bytes: int = 64 * 1024 * 1024,
        retained_files: int = 4,
    ) -> None:
        if not 1_000_000 <= max_file_bytes <= 10_000_000_000:
            raise ValueError("max_file_bytes must be between 1000000 and 10000000000")
        if not 1 <= retained_files <= 100:
            raise ValueError("retained_files must be between 1 and 100")
        self.path = facade().Path(path) if path is not None else None
        self.max_file_bytes = max_file_bytes
        self.retained_files = retained_files
        self._lock = facade().threading.Lock()
        self._total = 0
        self._verified_completions = 0
        self._agent_runs = 0
        self._agent_completions = 0
        self._agent_tool_calls = 0
        self._agent_successful_tool_calls = 0
        self._labeled = 0
        self._correct_labels = 0
        self._expected: Counter[str] = facade().Counter()
        self._label_errors: Counter[str] = facade().Counter()
        self._outcomes: Counter[str] = facade().Counter()
        self._assurance: Counter[str] = facade().Counter()
        self._contract_verified_completions = 0
        self._unsupported_verified_claims = 0
        self._verification_contracts = 0
        self._verification_evidence = 0
        self._missing_checks: Counter[str] = facade().Counter()
        self._evidence_kinds: Counter[str] = facade().Counter()
        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0
        self._tenant_stats: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        meta: dict[str, Any],
        *,
        request_id: str | None = None,
        expected_verdict: str | None = None,
        case_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        outcome_value = meta.get("outcome")
        timing_value = meta.get("timing_ms")
        outcome: dict[str, Any] = outcome_value if isinstance(outcome_value, dict) else {}
        timing: dict[str, Any] = timing_value if isinstance(timing_value, dict) else {}
        total_latency = facade()._float(timing.get("request_total"))
        observed_verdict = str(outcome.get("verdict") or "")
        label_error = (
            facade().labeled_error_kind(outcome, expected_verdict) if expected_verdict else None
        )
        label_matches = bool(expected_verdict and observed_verdict == expected_verdict)
        event = {
            "schema_version": facade().OUTCOME_SCHEMA_VERSION,
            "recorded_at": facade().datetime.now(facade().UTC).replace(microsecond=0).isoformat(),
            "request_id": request_id,
            "case_id": case_id,
            "tenant_id": tenant_id,
            "expected_verdict": expected_verdict,
            "label_matches": label_matches if expected_verdict else None,
            "label_error": label_error,
            "outcome": outcome,
            "timing_ms": timing,
        }
        agent_value = meta.get("agent")
        agent: dict[str, Any] = agent_value if isinstance(agent_value, dict) else {}
        scorecard_value = agent.get("scorecard")
        agent_scorecard: dict[str, Any] = (
            scorecard_value if isinstance(scorecard_value, dict) else {}
        )
        audit = facade().verification_audit(outcome)
        event["verification_audit"] = audit
        with self._lock:
            self._total += 1
            self._outcomes[str(outcome.get("status") or "unknown")] += 1
            self._assurance[str(outcome.get("assurance") or "none")] += 1
            if outcome.get("verified_completion") is True:
                self._verified_completions += 1
            self._verification_contracts += int(audit["contract_present"])
            self._contract_verified_completions += int(audit["supported_verified"])
            self._unsupported_verified_claims += int(audit["unsupported_verified_claim"])
            self._verification_evidence += int(audit["evidence_count"])
            self._missing_checks.update(audit["missing_checks"])
            self._evidence_kinds.update(audit["evidence_kinds"])
            if agent_scorecard:
                self._agent_runs += 1
                if agent_scorecard.get("completed") is True:
                    self._agent_completions += 1
                self._agent_tool_calls += facade()._int(agent_scorecard.get("tool_calls"))
                self._agent_successful_tool_calls += facade()._int(
                    agent_scorecard.get("successful_tool_calls")
                )
            if expected_verdict:
                self._labeled += 1
                self._expected[expected_verdict] += 1
                if label_matches:
                    self._correct_labels += 1
                else:
                    self._label_errors[label_error or "verdict_mismatch"] += 1
            self._total_latency_ms += total_latency
            self._max_latency_ms = max(self._max_latency_ms, total_latency)
            facade()._update_tenant_stats(
                self._tenant_stats,
                tenant_id or "default",
                outcome=outcome,
                agent_scorecard=agent_scorecard,
                expected_verdict=expected_verdict,
                label_matches=label_matches,
                label_error=label_error,
                total_latency=total_latency,
                verification_audit=audit,
            )
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    encoded = facade().json.dumps(event, sort_keys=True) + "\n"
                    self._rotate_sink(len(encoded.encode("utf-8")))
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(encoded)
                except OSError:
                    # Telemetry must never take the proxy request path down.
                    pass

    def _rotate_sink(self, incoming_bytes: int) -> None:
        if self.path is None or not self.path.exists():
            return
        if self.path.stat().st_size + incoming_bytes <= self.max_file_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.retained_files}")
        oldest.unlink(missing_ok=True)
        for index in range(self.retained_files - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def snapshot(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if tenant_id is not None:
                return facade()._tenant_snapshot(
                    self._tenant_stats.get(tenant_id),
                    tenant_id,
                )
            average = self._total_latency_ms / self._total if self._total else 0.0
            false_allows = self._label_errors["false_allow"]
            false_blocks = self._label_errors["false_block"]
            expected_blocks = self._expected["block"]
            expected_allows = self._expected["allow"]
            return {
                "schema_version": facade().OUTCOME_SCHEMA_VERSION,
                "requests": self._total,
                "verified_completions": self._verified_completions,
                "verified_completion_rate": round(self._verified_completions / self._total, 4)
                if self._total
                else 0.0,
                "agent": {
                    "runs": self._agent_runs,
                    "completed": self._agent_completions,
                    "completion_rate": round(self._agent_completions / self._agent_runs, 4)
                    if self._agent_runs
                    else 0.0,
                    "tool_calls": self._agent_tool_calls,
                    "successful_tool_calls": self._agent_successful_tool_calls,
                    "tool_success_rate": round(
                        self._agent_successful_tool_calls / self._agent_tool_calls,
                        4,
                    )
                    if self._agent_tool_calls
                    else 0.0,
                },
                "evaluation": {
                    "labeled_requests": self._labeled,
                    "correct": self._correct_labels,
                    "accuracy": round(self._correct_labels / self._labeled, 4)
                    if self._labeled
                    else 0.0,
                    "false_allows": false_allows,
                    "false_allow_rate": round(false_allows / expected_blocks, 4)
                    if expected_blocks
                    else 0.0,
                    "false_blocks": false_blocks,
                    "false_block_rate": round(false_blocks / expected_allows, 4)
                    if expected_allows
                    else 0.0,
                    "errors": dict(sorted(self._label_errors.items())),
                },
                "outcomes": dict(sorted(self._outcomes.items())),
                "assurance_levels": dict(sorted(self._assurance.items())),
                "verification": {
                    "contracts": self._verification_contracts,
                    "contract_coverage_rate": round(self._verification_contracts / self._total, 4)
                    if self._total
                    else 0.0,
                    "contract_verified_completions": (self._contract_verified_completions),
                    "unsupported_verified_claims": (self._unsupported_verified_claims),
                    "evidence_units": self._verification_evidence,
                    "evidence_kinds": dict(sorted(self._evidence_kinds.items())),
                    "missing_checks": dict(sorted(self._missing_checks.items())),
                },
                "latency_ms": {
                    "average": round(average, 3),
                    "maximum": round(self._max_latency_ms, 3),
                },
                "telemetry_retention": {
                    "max_file_bytes": self.max_file_bytes,
                    "retained_files": self.retained_files,
                },
            }


def _update_tenant_stats(
    tenant_stats: dict[str, dict[str, Any]],
    tenant_id: str,
    *,
    outcome: dict[str, Any],
    agent_scorecard: dict[str, Any],
    expected_verdict: str | None,
    label_matches: bool,
    label_error: str | None,
    total_latency: float,
    verification_audit: dict[str, Any],
) -> None:
    stats = tenant_stats.setdefault(
        tenant_id,
        {
            "requests": 0,
            "verified": 0,
            "agent_runs": 0,
            "agent_completed": 0,
            "agent_tool_calls": 0,
            "agent_successful_tool_calls": 0,
            "labeled": 0,
            "correct": 0,
            "expected": Counter(),
            "label_errors": Counter(),
            "outcomes": Counter(),
            "assurance": Counter(),
            "verification_contracts": 0,
            "contract_verified": 0,
            "unsupported_verified": 0,
            "verification_evidence": 0,
            "missing_checks": Counter(),
            "evidence_kinds": Counter(),
            "latency_total": 0.0,
            "latency_max": 0.0,
        },
    )
    stats["requests"] += 1
    stats["verified"] += int(outcome.get("verified_completion") is True)
    stats["outcomes"][str(outcome.get("status") or "unknown")] += 1
    stats["assurance"][str(outcome.get("assurance") or "none")] += 1
    stats["verification_contracts"] += int(verification_audit["contract_present"])
    stats["contract_verified"] += int(verification_audit["supported_verified"])
    stats["unsupported_verified"] += int(verification_audit["unsupported_verified_claim"])
    stats["verification_evidence"] += int(verification_audit["evidence_count"])
    stats["missing_checks"].update(verification_audit["missing_checks"])
    stats["evidence_kinds"].update(verification_audit["evidence_kinds"])
    stats["latency_total"] += total_latency
    stats["latency_max"] = max(stats["latency_max"], total_latency)
    if agent_scorecard:
        stats["agent_runs"] += 1
        stats["agent_completed"] += int(agent_scorecard.get("completed") is True)
        stats["agent_tool_calls"] += facade()._int(agent_scorecard.get("tool_calls"))
        stats["agent_successful_tool_calls"] += facade()._int(
            agent_scorecard.get("successful_tool_calls")
        )
    if expected_verdict:
        stats["labeled"] += 1
        stats["expected"][expected_verdict] += 1
        if label_matches:
            stats["correct"] += 1
        else:
            stats["label_errors"][label_error or "verdict_mismatch"] += 1


def _tenant_snapshot(stats: dict[str, Any] | None, tenant_id: str) -> dict[str, Any]:
    values = stats or {
        "requests": 0,
        "verified": 0,
        "agent_runs": 0,
        "agent_completed": 0,
        "agent_tool_calls": 0,
        "agent_successful_tool_calls": 0,
        "labeled": 0,
        "correct": 0,
        "expected": Counter(),
        "label_errors": Counter(),
        "outcomes": Counter(),
        "assurance": Counter(),
        "verification_contracts": 0,
        "contract_verified": 0,
        "unsupported_verified": 0,
        "verification_evidence": 0,
        "missing_checks": Counter(),
        "evidence_kinds": Counter(),
        "latency_total": 0.0,
        "latency_max": 0.0,
    }
    requests = int(values["requests"])
    agent_runs = int(values["agent_runs"])
    tool_calls = int(values["agent_tool_calls"])
    labeled = int(values["labeled"])
    false_allows = values["label_errors"]["false_allow"]
    false_blocks = values["label_errors"]["false_block"]
    expected_blocks = values["expected"]["block"]
    expected_allows = values["expected"]["allow"]
    return {
        "schema_version": facade().OUTCOME_SCHEMA_VERSION,
        "tenant": tenant_id,
        "requests": requests,
        "verified_completions": int(values["verified"]),
        "verified_completion_rate": (round(values["verified"] / requests, 4) if requests else 0.0),
        "agent": {
            "runs": agent_runs,
            "completed": int(values["agent_completed"]),
            "completion_rate": (
                round(values["agent_completed"] / agent_runs, 4) if agent_runs else 0.0
            ),
            "tool_calls": tool_calls,
            "successful_tool_calls": int(values["agent_successful_tool_calls"]),
            "tool_success_rate": (
                round(values["agent_successful_tool_calls"] / tool_calls, 4) if tool_calls else 0.0
            ),
        },
        "evaluation": {
            "labeled_requests": labeled,
            "correct": int(values["correct"]),
            "accuracy": (round(values["correct"] / labeled, 4) if labeled else 0.0),
            "false_allows": false_allows,
            "false_allow_rate": (
                round(false_allows / expected_blocks, 4) if expected_blocks else 0.0
            ),
            "false_blocks": false_blocks,
            "false_block_rate": (
                round(false_blocks / expected_allows, 4) if expected_allows else 0.0
            ),
            "errors": dict(sorted(values["label_errors"].items())),
        },
        "outcomes": dict(sorted(values["outcomes"].items())),
        "assurance_levels": dict(sorted(values["assurance"].items())),
        "verification": {
            "contracts": int(values["verification_contracts"]),
            "contract_coverage_rate": (
                round(values["verification_contracts"] / requests, 4) if requests else 0.0
            ),
            "contract_verified_completions": int(values["contract_verified"]),
            "unsupported_verified_claims": int(values["unsupported_verified"]),
            "evidence_units": int(values["verification_evidence"]),
            "evidence_kinds": dict(sorted(values["evidence_kinds"].items())),
            "missing_checks": dict(sorted(values["missing_checks"].items())),
        },
        "latency_ms": {
            "average": (round(values["latency_total"] / requests, 3) if requests else 0.0),
            "maximum": round(values["latency_max"], 3),
        },
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
