"""Manifest-driven, content-free qualification matrix for Cortheon.

Stable import surface: every definition now lives in the repository-only
``cortheon.qualification_core`` package and is re-exported here (public and
private names alike) so existing imports keep working. The facade owns no
implementation, so there is exactly one place to change any behaviour.

``_repository_fingerprint`` is re-exported too. It is not a qualification
definition -- it comes from ``cortheon.cognitive_benchmark`` -- but the
repository has always been able to reach it through this module, so the
compatibility surface keeps it.

Monkeypatching stays facade-level. ``run_qualification`` resolves
``_run_cell``, ``_repository_fingerprint``, and ``_git_revision`` through this
module at call time, and ``_run_cell`` resolves ``_repository_fingerprint`` the
same way for its pre/post workspace checks, so rebinding any of the three here
steers the run exactly as it did before the split.
"""

from cortheon.qualification_core.cell_gates import _cell_gates
from cortheon.qualification_core.cli import _example_manifest, build_parser, main
from cortheon.qualification_core.conditions import (
    ABLATION_OPERATORS,
    AVAILABLE_CONDITIONS,
    CONDITION_REGISTRY_VERSION,
    CONDITIONS,
    CONTRASTS,
    REQUIRED_CONDITIONS,
    closed_registry,
    condition_record,
)
from cortheon.qualification_core.constants import (
    CELL_KEYS,
    ENVIRONMENT_NAME,
    FORBIDDEN_CREDENTIAL_KEYS,
    GATE_KEYS,
    HOSTS,
    IDENTIFIER,
    MAX_CELLS,
    MAX_JOBS,
    MAX_MANIFEST_BYTES,
    REPORT_SCHEMA_VERSION,
    ROOT_KEYS,
    SCHEMA_VERSION,
    SUITES,
    TIER_DEFAULTS,
)
from cortheon.qualification_core.digests import _cell_public_config, _sealed_task_digest
from cortheon.qualification_core.environment import (
    _cell_namespace,
    _command_version,
    _git_revision,
    _package_version,
    _public_runtime_health,
)
from cortheon.qualification_core.execution import _repository_fingerprint, _run_cell
from cortheon.qualification_core.gates import _strict_gate_overrides
from cortheon.qualification_core.manifest import _parse_cell, load_manifest
from cortheon.qualification_core.models import Cell, CellRun, Manifest, QualificationError
from cortheon.qualification_core.pairing import (
    _aggregate_pairing,
    _bootstrap_summary,
    _independent_pairing,
)
from cortheon.qualification_core.report import _cell_report, run_qualification
from cortheon.qualification_core.reproducers import _reproducers
from cortheon.qualification_core.taxonomy import _failure_type, _public_run
from cortheon.qualification_core.validation import (
    _bounded_int,
    _bounded_number,
    _bounded_text,
    _http_url,
    _parse_document,
    _reject_embedded_credentials,
    _reject_unknown,
)

__all__ = [
    "ABLATION_OPERATORS",
    "AVAILABLE_CONDITIONS",
    "CELL_KEYS",
    "CONDITIONS",
    "CONDITION_REGISTRY_VERSION",
    "CONTRASTS",
    "ENVIRONMENT_NAME",
    "FORBIDDEN_CREDENTIAL_KEYS",
    "GATE_KEYS",
    "HOSTS",
    "IDENTIFIER",
    "MAX_CELLS",
    "MAX_JOBS",
    "MAX_MANIFEST_BYTES",
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_CONDITIONS",
    "ROOT_KEYS",
    "SCHEMA_VERSION",
    "SUITES",
    "TIER_DEFAULTS",
    "Cell",
    "CellRun",
    "Manifest",
    "QualificationError",
    "_aggregate_pairing",
    "_bootstrap_summary",
    "_bounded_int",
    "_bounded_number",
    "_bounded_text",
    "_cell_gates",
    "_cell_namespace",
    "_cell_public_config",
    "_cell_report",
    "_command_version",
    "_example_manifest",
    "_failure_type",
    "_git_revision",
    "_http_url",
    "_independent_pairing",
    "_package_version",
    "_parse_cell",
    "_parse_document",
    "_public_run",
    "_public_runtime_health",
    "_reject_embedded_credentials",
    "_reject_unknown",
    "_repository_fingerprint",
    "_reproducers",
    "_run_cell",
    "_sealed_task_digest",
    "_strict_gate_overrides",
    "build_parser",
    "closed_registry",
    "condition_record",
    "load_manifest",
    "main",
    "run_qualification",
]


if __name__ == "__main__":
    raise SystemExit(main())
