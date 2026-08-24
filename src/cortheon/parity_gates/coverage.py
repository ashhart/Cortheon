"""Coverage gates: repetitions, case count, and the domain universe.

Repetitions are stability measurements, not extra independent cases, so the
executed count has to equal the pre-registered one exactly -- too few dilutes
nothing and too many dilutes safety and delivery rates. The case and domain
floors then establish that the run covered the breadth the claim asserts.
"""

from __future__ import annotations

from cortheon.parity_benchmark_core.oracle_taxonomy import TASK_CLASSES
from cortheon.parity_gates.context import ParityContext


def evaluate_coverage(context: ParityContext) -> None:
    case_bank = context.case_bank
    thresholds = context.thresholds
    repetitions = context.repetitions
    min_repetitions = int(thresholds["min_repetitions"])
    context.check(
        "exact_precommitted_repetitions",
        repetitions == min_repetitions
        and int(case_bank.get("execution_repetitions") or 0) == repetitions,
        actual=repetitions,
        precommitted=case_bank.get("execution_repetitions"),
        required=min_repetitions,
    )
    selected_cases = int(case_bank.get("selected_cases") or 0)
    min_cases = int(thresholds["min_cases"])
    context.check(
        "minimum_case_count",
        selected_cases >= min_cases and len(context.cases) == selected_cases,
        selected_cases=selected_cases,
        report_cases=len(context.cases),
        minimum=min_cases,
    )
    observed_domains = {
        str(case.get("domain") or "") for case in context.cases if case.get("domain")
    }
    context.check(
        "required_domain_universe",
        observed_domains == context.required_domains,
        required=sorted(context.required_domains),
        observed=sorted(observed_domains),
    )
    cases_by_domain = {
        domain: sum(str(case.get("domain")) == domain for case in context.cases)
        for domain in sorted(context.required_domains)
    }
    min_cases_per_domain = int(thresholds["min_cases_per_domain"])
    context.check(
        "minimum_cases_per_domain",
        bool(cases_by_domain)
        and all(value >= min_cases_per_domain for value in cases_by_domain.values()),
        counts=cases_by_domain,
        minimum=min_cases_per_domain,
    )
    cases_by_task_class = {
        task_class: sum(
            case.get("task_class") == task_class and case.get("grader_type") is not None
            for case in context.cases
        )
        for task_class in sorted(TASK_CLASSES)
    }
    minimum_per_class = int(thresholds["min_cases_per_task_class"])
    context.check(
        "minimum_proof_cases_per_task_class",
        all(value >= minimum_per_class for value in cases_by_task_class.values()),
        counts=cases_by_task_class,
        minimum=minimum_per_class,
    )
