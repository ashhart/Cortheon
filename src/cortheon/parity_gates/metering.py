"""The independent-metering gate.

Cost is only evidence if the evaluator measured it. The candidate's cost must
come from the runner's own wall clock and the pre-registered compute rate, and
each frontier's from metered provider usage priced at the registered rates. A
contender-supplied cost label is not an independent measurement, so a run that
carries one -- or that mixes sources within a contender -- fails.
"""

from __future__ import annotations

from cortheon.parity_gates.context import ContenderIdentities, ParityContext
from cortheon.parity_gates.values import _mapping


def evaluate_metering(context: ParityContext, identities: ContenderIdentities) -> None:
    require_metered_cost = context.thresholds.get("require_metered_cost") is True
    cost_sources = {
        name: {
            str((_mapping(row.get("cost"))).get("source") or "unavailable")
            for row in context.rows
            if row.get("candidate") == alias
        }
        for name, alias in identities.contender_aliases().items()
    }
    context.check(
        "independently_metered_contender_costs",
        (not require_metered_cost)
        or bool(
            cost_sources.get(context.candidate_name)
            == {"runner_wall_clock_and_preregistered_compute_rate"}
            and all(
                cost_sources.get(name) == {"metered_from_usage_and_registered_pricing"}
                for name in context.frontier_names
            )
        ),
        sources={name: sorted(sources) for name, sources in cost_sources.items()},
        required=require_metered_cost,
    )
