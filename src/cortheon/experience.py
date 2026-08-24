"""Privacy-preserving, failure-derived experience for Cortheon.

The experience store is intentionally not a conversation log. It accepts only
small taxonomy identifiers, counters, and machine-checkable verification
contracts. Prompts, answers, expected benchmark outputs, secrets, tool output,
and reasoning traces have no field in the schema.

Implementation ownership lives in :mod:`cortheon.experience_core`. This module
keeps the original import path and patch points stable for repository consumers.
"""

# The facade keeps former module globals as observable patch points.
# ruff: noqa: F401

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.experience_core.models import (
    FailureSignature as _FailureSignature,
)
from cortheon.experience_core.models import (
    RecoveryStrategy as _RecoveryStrategy,
)
from cortheon.experience_core.models import (
    VerificationContract as _VerificationContract,
)
from cortheon.experience_core.store import ExperienceStore as _ExperienceStore
from cortheon.experience_core.taxonomy import (
    _CROSS_DOCUMENT_TASK,
    _DOCUMENT_TASK,
    _PACKAGE_TASK,
    _QUANTITATIVE_TASK,
    _REPOSITORY_TASK,
    _RESEARCH_TASK,
    classify_experience_task,
)
from cortheon.experience_core.validation import (
    _ASSURANCE_RANK,
    _IDENTIFIER,
    _NAMESPACE,
    _RESERVED_IDENTIFIERS,
    _RESULTS,
    _VERIFIABLE_ASSURANCE,
    EXPERIENCE_SCHEMA_VERSION,
    _assurance_for_rank,
    _identifier,
    _identifiers,
    _latency_bucket,
    _limit,
    _looks_secret,
    _namespace,
    _rate,
    _recovery_rate,
)

FailureSignature = _FailureSignature
RecoveryStrategy = _RecoveryStrategy
VerificationContract = _VerificationContract
ExperienceStore = _ExperienceStore

__all__ = [
    "EXPERIENCE_SCHEMA_VERSION",
    "ExperienceStore",
    "FailureSignature",
    "RecoveryStrategy",
    "VerificationContract",
    "classify_experience_task",
]

for _public in (
    classify_experience_task,
    FailureSignature,
    RecoveryStrategy,
    VerificationContract,
    ExperienceStore,
):
    _public.__module__ = __name__

for _class in (FailureSignature, RecoveryStrategy, VerificationContract, ExperienceStore):
    for _owner in _class.__mro__[:-1]:
        for _member in vars(_owner).values():
            if isinstance(_member, (classmethod, staticmethod)):
                _member = _member.__func__
            accessors = (
                (_member.fget, _member.fset, _member.fdel)
                if isinstance(_member, property)
                else (_member,)
            )
            for _accessor in accessors:
                if callable(_accessor) and hasattr(_accessor, "__module__"):
                    _accessor.__module__ = __name__
