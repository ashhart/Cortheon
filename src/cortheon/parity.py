"""Fail-closed frontier-parity release contracts for Cortheon benchmark reports.

Stable import surface: every definition now lives in the repository-only
``cortheon.parity_gates`` package and is re-exported here (public and private
names alike) so existing imports keep working. ``__all__`` is unchanged, so a
star import still yields exactly the ten public names.

``UNIVERSAL_SCALE_REQUIREMENTS`` is a rebinding anchor: the gates resolve it
through this module at call time, so substituting a reduced test-scale policy
here still steers the release-scale gate.
"""

from cortheon.parity_gates.comparison import _instability, _paired_statistics
from cortheon.parity_gates.contract import _validate_contract, load_parity_contract
from cortheon.parity_gates.decision import _decision, evaluate_frontier_parity
from cortheon.parity_gates.errors import (
    _TRUSTED_FRONTIER_HOSTS,
    SUPPORTED_CANDIDATE_HOSTS,
    ParityContractError,
)
from cortheon.parity_gates.noninferiority import _comparison_check, _ratio_check
from cortheon.parity_gates.preregistration import _universal_scale_ok
from cortheon.parity_gates.projection import (
    evaluation_schedule,
    evaluation_schedule_hash,
    public_case_projection,
    public_task_hash,
)
from cortheon.parity_gates.values import (
    _after,
    _is_sha256,
    _mapping,
    _nested_number,
    _number,
    _percentile,
    _stable_seed,
)
from cortheon.parity_scale_policy import (
    UNIVERSAL_SCALE_CEILINGS,
    UNIVERSAL_SCALE_REQUIREMENTS,
)

__all__ = [
    "SUPPORTED_CANDIDATE_HOSTS",
    "UNIVERSAL_SCALE_CEILINGS",
    "UNIVERSAL_SCALE_REQUIREMENTS",
    "ParityContractError",
    "evaluate_frontier_parity",
    "evaluation_schedule",
    "evaluation_schedule_hash",
    "load_parity_contract",
    "public_case_projection",
    "public_task_hash",
]

# Private names the pre-split module exposed as attributes. Naming them here
# keeps the re-exports live -- an unused import would be removed as dead code
# -- without widening ``__all__``, which is the star-import contract.
_COMPATIBILITY_EXPORTS = (
    _TRUSTED_FRONTIER_HOSTS,
    _after,
    _comparison_check,
    _decision,
    _instability,
    _is_sha256,
    _mapping,
    _nested_number,
    _number,
    _paired_statistics,
    _percentile,
    _ratio_check,
    _stable_seed,
    _universal_scale_ok,
    _validate_contract,
)
