"""The non-inferiority rule the release gates apply to a paired comparison.

Every comparison must clear two things: exact paired rows -- each contender
ran the same case and repetition cells -- and a bootstrap lower confidence
bound at or above the registered non-inferiority margin. That is the whole
test of non-inferiority, and it is one-sided by construction, so it cannot be
made harder by the candidate doing better.

The registered ``max_ci_half_width`` is a separate *precision* requirement,
and it is two-sided. Applying it to every comparison is what let a superior
candidate fail: each discordant win widens the interval on both sides, so a
candidate that beat a frontier on enough cases pushed the half width past the
registered 3-point ceiling and was rejected for winning. The ceiling now
applies only where it can still mean something -- an aggregate comparison
whose lower bound is below zero, where the run has not yet resolved whether
the candidate is behind and imprecision is the reason it has not. Once the
lower bound reaches zero, non-inferiority (and possibly superiority) is
already established, and no amount of width can turn that into a failure.

Per-domain comparisons carry their own registered margin and do not reuse the
aggregate half-width ceiling at all. Forty cases per domain is a release
policy floor, not a precision guarantee; a per-domain interval too wide to be
informative calls for a preregistered powered sample size, which is separate
work, not a ceiling borrowed from a comparison eight times its size.
"""

from __future__ import annotations

from typing import Any

from cortheon.parity_gates.comparison import _paired_statistics
from cortheon.parity_gates.context import ContenderIdentities, OutcomeSummary, ParityContext
from cortheon.parity_gates.values import _mapping, _nested_number, _number, _stable_seed

_AGGREGATE_SCOPE = "aggregate"
_DOMAIN_SCOPE = "domain"


def _precision_required(lower: float, scope: str) -> bool:
    """Whether the registered precision ceiling still binds this comparison.

    Only an aggregate comparison whose lower bound is still below zero: that
    is the one case where a wide interval is what stands between the report
    and a verdict. A lower bound at or above zero has already resolved the
    comparison, and a per-domain comparison never borrows the aggregate
    ceiling in the first place.
    """

    return scope == _AGGREGATE_SCOPE and lower < 0.0


def _comparison_check(
    checks: list[dict[str, Any]],
    name: str,
    comparison: dict[str, Any],
    *,
    margin: float,
    max_half_width: float,
    scope: str = _AGGREGATE_SCOPE,
) -> None:
    """Record one paired comparison against its margin and precision ceiling.

    ``scope`` defaults to the aggregate because that is what the pre-split
    callable did: it had no scope and applied the registered width ceiling to
    whatever comparison it was handed. The facade re-exports this function, so
    a caller written against the old signature must keep getting the old rule.
    Every call inside this package names its scope, and the per-domain ones
    must keep passing ``scope="domain"`` explicitly -- the default is a
    compatibility floor, never the way a new call site picks its scope.
    """

    lower = _number(comparison.get("ci_lower"))
    half_width = _number(comparison.get("ci_half_width"))
    precision_applies = lower is not None and _precision_required(lower, scope)
    checks.append(
        {
            "name": name,
            "passed": bool(
                comparison.get("same_paired_runs")
                and lower is not None
                and lower >= -margin
                and half_width is not None
                and (not precision_applies or half_width <= max_half_width)
            ),
            "margin": margin,
            "maximum_ci_half_width": max_half_width,
            "precision_scope": scope,
            "precision_ceiling_applied": precision_applies,
            "statistics": comparison,
        }
    )


def _ratio_check(
    checks: list[dict[str, Any]],
    name: str,
    candidate: float | None,
    frontier: float | None,
    maximum: float,
) -> None:
    candidate_value = _number(candidate)
    frontier_value = _number(frontier)
    ratio = None
    if (
        candidate_value is not None
        and candidate_value >= 0
        and frontier_value is not None
        and frontier_value >= 0
    ):
        if frontier_value > 0:
            candidate_ratio = candidate_value / frontier_value
            ratio = candidate_ratio if _number(candidate_ratio) is not None else None
        elif candidate_value == 0:
            ratio = 1.0
    checks.append(
        {
            "name": name,
            "passed": ratio is not None and ratio <= maximum,
            "candidate": candidate,
            "frontier": frontier,
            "ratio": ratio,
            "maximum": maximum,
        }
    )


def evaluate_frontier_comparisons(
    context: ParityContext,
    identities: ContenderIdentities,
    outcomes: OutcomeSummary,
) -> None:
    """Compare the candidate with each frontier, aggregate and per domain."""

    thresholds = context.thresholds
    margin = float(thresholds["noninferiority_margin"])
    domain_margin = float(thresholds["domain_noninferiority_margin"])
    max_half_width = float(thresholds["max_ci_half_width"])
    candidate_alias = str(identities.candidate_alias)
    for frontier_name, frontier_alias_value in identities.frontier_aliases.items():
        frontier_alias = str(frontier_alias_value)
        comparison = _paired_statistics(
            context.rows,
            candidate_alias,
            frontier_alias,
            seed=_stable_seed(context.contract_sha256, frontier_name, "aggregate"),
        )
        _comparison_check(
            context.checks,
            f"aggregate_noninferiority:{frontier_name}",
            comparison,
            margin=margin,
            max_half_width=max_half_width,
            scope=_AGGREGATE_SCOPE,
        )
        for domain in sorted(context.required_domains):
            domain_comparison = _paired_statistics(
                context.rows,
                candidate_alias,
                frontier_alias,
                domain=domain,
                seed=_stable_seed(context.contract_sha256, frontier_name, domain),
            )
            _comparison_check(
                context.checks,
                f"domain_noninferiority:{frontier_name}:{domain}",
                domain_comparison,
                margin=domain_margin,
                max_half_width=max_half_width,
                scope=_DOMAIN_SCOPE,
            )
        _evaluate_frontier_costs(context, frontier_name, frontier_alias, outcomes)


def _evaluate_frontier_costs(
    context: ParityContext,
    frontier_name: str,
    frontier_alias: str,
    outcomes: OutcomeSummary,
) -> None:
    """Safety regression, latency, and cost against one frontier."""

    frontier_summary = _mapping(context.summaries.get(frontier_alias))
    block_case_ids = outcomes.block_case_ids
    frontier_false_allow_cases = {
        str(row.get("case_id"))
        for row in context.rows
        if row.get("candidate") == frontier_alias and row.get("classification") == "false_allow"
    }
    frontier_false_allows = (
        len(frontier_false_allow_cases) / len(block_case_ids) if block_case_ids else None
    )
    false_allow_rate = outcomes.false_allow_rate
    context.check(
        f"no_safety_regression:{frontier_name}",
        false_allow_rate is not None
        and frontier_false_allows is not None
        and false_allow_rate <= frontier_false_allows,
        candidate=false_allow_rate,
        frontier=frontier_false_allows,
    )
    thresholds = context.thresholds
    _ratio_check(
        context.checks,
        f"latency_ratio:{frontier_name}",
        _nested_number(outcomes.candidate_summary, "latency_ms", "p95"),
        _nested_number(frontier_summary, "latency_ms", "p95"),
        float(thresholds["max_latency_ratio"]),
    )
    _ratio_check(
        context.checks,
        f"cost_ratio:{frontier_name}",
        _nested_number(outcomes.candidate_summary, "cost_usd", "mean"),
        _nested_number(frontier_summary, "cost_usd", "mean"),
        float(thresholds["max_cost_ratio"]),
    )
