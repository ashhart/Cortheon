"""Stable facade for the sealed development case bank."""

from cortheon.operator_lift.cases_derivation import _derivation_cases
from cortheon.operator_lift.cases_discrimination import _discrimination_cases
from cortheon.operator_lift.cases_hypothesis import _hypothesis_cases
from cortheon.operator_lift.cases_revision import _revision_cases
from cortheon.operator_lift.cases_stopping import _stopping_cases
from cortheon.operator_lift.models import LiftCase


def development_cases() -> tuple[LiftCase, ...]:
    """Return the immutable 60-case development bank in preregistered order."""

    cases = (
        *_hypothesis_cases(),
        *_discrimination_cases(),
        *_revision_cases(),
        *_derivation_cases(),
        *_stopping_cases(),
    )
    if len(cases) != 60:
        raise AssertionError("operator-lift case bank must contain exactly 60 cases")
    return cases
