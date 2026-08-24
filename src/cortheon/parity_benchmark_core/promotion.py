from __future__ import annotations

from typing import Any

from cortheon.parity_benchmark_core.pairing import (
    _cell_index,
    _duplicate_count,
    _paired_statistics,
    _stable_integer_seed,
)


def evaluate_promotion(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    candidate_name: str,
    min_improvement: float,
    max_domain_regression: float,
    max_latency_ratio: float,
    max_cost_ratio: float,
    require_external_holdout: bool = False,
) -> dict[str, Any]:
    """Return a machine-checkable, fail-closed capability promotion decision."""

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    if require_external_holdout:
        baseline_bank_value = baseline.get("case_bank")
        current_bank_value = current.get("case_bank")
        baseline_bank: dict[str, Any] = (
            baseline_bank_value if isinstance(baseline_bank_value, dict) else {}
        )
        current_bank: dict[str, Any] = (
            current_bank_value if isinstance(current_bank_value, dict) else {}
        )
        check(
            "external_heldout_case_bank",
            bool(
                baseline_bank.get("source") == "external"
                and current_bank.get("source") == "external"
                and baseline_bank.get("split") == "heldout"
                and current_bank.get("split") == "heldout"
            ),
            baseline_source=baseline_bank.get("source"),
            baseline_split=baseline_bank.get("split"),
            current_source=current_bank.get("source"),
            current_split=current_bank.get("split"),
        )

    baseline_hash = _report_selection_hash(baseline)
    current_hash = _report_selection_hash(current)
    check(
        "same_blinded_case_selection",
        bool(baseline_hash and baseline_hash == current_hash),
        baseline=baseline_hash,
        current=current_hash,
    )
    baseline_methodology = baseline.get("methodology")
    current_methodology = current.get("methodology")
    baseline_method = (
        baseline_methodology.get("grading") if isinstance(baseline_methodology, dict) else None
    )
    current_method = (
        current_methodology.get("grading") if isinstance(current_methodology, dict) else None
    )
    check(
        "contender_blind_deterministic_grading",
        baseline_method == current_method == "deterministic and contender-blind",
        baseline=baseline_method,
        current=current_method,
    )
    baseline_summary = _report_candidate_summary(baseline, candidate_name)
    current_summary = _report_candidate_summary(current, candidate_name)
    if baseline_summary is None or current_summary is None:
        check(
            "candidate_present",
            False,
            candidate=candidate_name,
            baseline_present=baseline_summary is not None,
            current_present=current_summary is not None,
        )
        return {
            "schema_version": 1,
            "candidate": candidate_name,
            "passed": False,
            "checks": checks,
            "failure_reasons": [value["name"] for value in checks if not value["passed"]],
        }
    check("candidate_present", True, candidate=candidate_name)
    baseline_runs = int(baseline_summary.get("runs") or 0)
    current_runs = int(current_summary.get("runs") or 0)
    check(
        "same_nonzero_run_count",
        baseline_runs > 0 and baseline_runs == current_runs,
        baseline=baseline_runs,
        current=current_runs,
    )
    baseline_rate = _metric_float(
        baseline_summary,
        "verified_completion_rate",
    )
    current_rate = _metric_float(current_summary, "verified_completion_rate")
    improvement = (
        current_rate - baseline_rate
        if baseline_rate is not None and current_rate is not None
        else None
    )
    check(
        "verified_completion_improved",
        improvement is not None and improvement > min_improvement,
        baseline=baseline_rate,
        current=current_rate,
        delta=improvement,
        required_strictly_greater_than=min_improvement,
    )
    paired = _paired_promotion_statistics(
        baseline,
        current,
        candidate_name=candidate_name,
    )
    paired_delta = paired.get("verified_completion_rate_delta")
    paired_interval = paired.get("paired_bootstrap_95ci")
    paired_lower = paired_interval.get("lower") if isinstance(paired_interval, dict) else None
    check(
        "paired_blinded_improvement",
        bool(
            paired.get("same_paired_runs")
            and isinstance(paired_delta, (int, float))
            and paired_delta > min_improvement
            and isinstance(paired_lower, (int, float))
            and paired_lower >= 0
        ),
        statistics=paired,
        required_strictly_greater_than=min_improvement,
        required_95ci_lower_bound=0,
    )
    for metric in (
        "false_allows",
        "false_blocks",
        "verdict_mismatches",
        "errors",
    ):
        baseline_value = int(baseline_summary.get(metric) or 0)
        current_value = int(current_summary.get(metric) or 0)
        check(
            f"no_{metric}_regression",
            current_value <= baseline_value,
            baseline=baseline_value,
            current=current_value,
        )
    baseline_domains = baseline_summary.get("by_domain")
    current_domains = current_summary.get("by_domain")
    baseline_domains = baseline_domains if isinstance(baseline_domains, dict) else {}
    current_domains = current_domains if isinstance(current_domains, dict) else {}
    check(
        "same_domain_coverage",
        bool(baseline_domains) and set(baseline_domains) == set(current_domains),
        baseline=sorted(baseline_domains),
        current=sorted(current_domains),
    )
    for domain in sorted(set(baseline_domains) | set(current_domains)):
        baseline_domain_rate = _metric_float(
            baseline_domains.get(domain),
            "verified_completion_rate",
        )
        current_domain_rate = _metric_float(
            current_domains.get(domain),
            "verified_completion_rate",
        )
        regression = (
            baseline_domain_rate - current_domain_rate
            if baseline_domain_rate is not None and current_domain_rate is not None
            else None
        )
        check(
            f"domain_regression:{domain}",
            regression is not None and regression <= max_domain_regression,
            baseline=baseline_domain_rate,
            current=current_domain_rate,
            regression=regression,
            maximum=max_domain_regression,
        )
    _ratio_gate(
        checks,
        name="p95_latency_ratio",
        baseline=_nested_metric(baseline_summary, "latency_ms", "p95"),
        current=_nested_metric(current_summary, "latency_ms", "p95"),
        maximum=max_latency_ratio,
        measurement_required=True,
    )
    _ratio_gate(
        checks,
        name="mean_cost_ratio",
        baseline=_nested_metric(baseline_summary, "cost_usd", "mean"),
        current=_nested_metric(current_summary, "cost_usd", "mean"),
        maximum=max_cost_ratio,
        measurement_required=True,
    )
    return {
        "schema_version": 1,
        "candidate": candidate_name,
        "passed": all(value["passed"] for value in checks),
        "checks": checks,
        "failure_reasons": [value["name"] for value in checks if not value["passed"]],
    }


def _paired_promotion_statistics(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    baseline_alias = _report_candidate_alias(baseline, candidate_name)
    current_alias = _report_candidate_alias(current, candidate_name)
    baseline_rows = baseline.get("rows")
    current_rows = current.get("rows")
    if (
        baseline_alias is None
        or current_alias is None
        or not isinstance(baseline_rows, list)
        or not isinstance(current_rows, list)
    ):
        return {"same_paired_runs": False, "paired_runs": 0}
    typed_baseline_rows = [row for row in baseline_rows if isinstance(row, dict)]
    typed_current_rows = [row for row in current_rows if isinstance(row, dict)]
    baseline_by_key = _cell_index(typed_baseline_rows, baseline_alias)
    current_by_key = _cell_index(typed_current_rows, current_alias)
    duplicate_cells = _duplicate_count(baseline_by_key) + _duplicate_count(current_by_key)
    same_keys = (
        bool(baseline_by_key)
        and set(baseline_by_key) == set(current_by_key)
        and duplicate_cells == 0
    )
    common = sorted(set(baseline_by_key) & set(current_by_key))
    differences: dict[str, list[int]] = {}
    for key in common:
        if len(current_by_key[key]) != 1 or len(baseline_by_key[key]) != 1:
            continue
        differences.setdefault(str(key[0]), []).append(
            int(current_by_key[key][0].get("verified_completion") is True)
            - int(baseline_by_key[key][0].get("verified_completion") is True)
        )
    statistics_payload = _paired_statistics(
        differences,
        left="current",
        right="baseline",
        seed=_stable_integer_seed(
            0,
            _report_selection_hash(current) or "",
            candidate_name,
        ),
        duplicate_cells=duplicate_cells,
        same_paired_runs=same_keys,
    )
    return statistics_payload


def _ratio_gate(
    checks: list[dict[str, Any]],
    *,
    name: str,
    baseline: float | None,
    current: float | None,
    maximum: float,
    measurement_required: bool,
) -> None:
    if baseline is None:
        checks.append(
            {
                "name": name,
                "passed": not measurement_required,
                "status": (
                    "required_measurement_unavailable"
                    if measurement_required
                    else "not_applicable_baseline_unavailable"
                ),
                "baseline": None,
                "current": current,
                "maximum": maximum,
            }
        )
        return
    ratio = (
        current / baseline
        if current is not None and baseline > 0
        else (1.0 if current == baseline == 0 else None)
    )
    passed = ratio is not None and ratio <= maximum
    checks.append(
        {
            "name": name,
            "passed": passed,
            "baseline": baseline,
            "current": current,
            "ratio": ratio,
            "maximum": maximum,
        }
    )


def _report_selection_hash(report: dict[str, Any]) -> str | None:
    case_bank = report.get("case_bank")
    value = case_bank.get("selection_sha256") if isinstance(case_bank, dict) else None
    return value if isinstance(value, str) and len(value) == 64 else None


def _report_candidate_summary(
    report: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any] | None:
    candidates = report.get("candidates")
    summaries = report.get("summary")
    if not isinstance(candidates, dict) or not isinstance(summaries, dict):
        return None
    alias = _report_candidate_alias(report, candidate_name)
    if alias is not None and isinstance(summaries.get(alias), dict):
        return summaries[alias]
    return None


def _report_candidate_alias(
    report: dict[str, Any],
    candidate_name: str,
) -> str | None:
    candidates = report.get("candidates")
    if not isinstance(candidates, dict):
        return None
    for alias, identity in candidates.items():
        if isinstance(identity, dict) and identity.get("name") == candidate_name:
            return str(alias)
    return None


def _metric_float(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _nested_metric(payload: Any, outer: str, inner: str) -> float | None:
    nested = payload.get(outer) if isinstance(payload, dict) else None
    return _metric_float(nested, inner)
