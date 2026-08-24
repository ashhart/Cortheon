"""Outcome contracts and aggregate telemetry for repository evaluation paths.

Implementation ownership lives in :mod:`cortheon.telemetry_core`. This facade
keeps the original import path, object identities, and patch points stable.
"""

# Former module globals remain observable patch points for moved code.
# ruff: noqa: F401

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.experience import VerificationContract
from cortheon.telemetry_core.metrics import ProxyMetrics
from cortheon.telemetry_core.metrics import _float as _float
from cortheon.telemetry_core.metrics import _int as _int
from cortheon.telemetry_core.metrics import _tenant_snapshot as _tenant_snapshot
from cortheon.telemetry_core.metrics import _update_tenant_stats as _update_tenant_stats
from cortheon.telemetry_core.outcomes import _outcome as _outcome
from cortheon.telemetry_core.outcomes import (
    agent_completion_outcome,
    agent_inconclusive_outcome,
    decision_outcome,
    enforcement_outcome,
    labeled_error_kind,
    patch_outcome,
    verification_audit,
)

OUTCOME_SCHEMA_VERSION = 2

for _definition in (
    ProxyMetrics,
    agent_completion_outcome,
    agent_inconclusive_outcome,
    decision_outcome,
    enforcement_outcome,
    labeled_error_kind,
    patch_outcome,
    verification_audit,
    _float,
    _int,
    _outcome,
    _tenant_snapshot,
    _update_tenant_stats,
):
    _definition.__module__ = __name__

for _member in vars(ProxyMetrics).values():
    if callable(_member) and hasattr(_member, "__module__"):
        _member.__module__ = __name__
