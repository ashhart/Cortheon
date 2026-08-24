"""Validated failure, recovery, and verification value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cortheon.experience_core._compat import facade


@dataclass(frozen=True, slots=True)
class FailureSignature:
    """A content-free description of a recurring failure class."""

    capability: str
    task_family: str
    stage: str
    failure_kind: str
    failure_code: str = "unspecified"
    context_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        api = facade()
        object.__setattr__(self, "capability", api._identifier(self.capability, "capability"))
        object.__setattr__(self, "task_family", api._identifier(self.task_family, "task_family"))
        object.__setattr__(self, "stage", api._identifier(self.stage, "stage"))
        object.__setattr__(self, "failure_kind", api._identifier(self.failure_kind, "failure_kind"))
        object.__setattr__(self, "failure_code", api._identifier(self.failure_code, "failure_code"))
        object.__setattr__(
            self,
            "context_tags",
            tuple(sorted(api._identifiers(self.context_tags, "context_tags", maximum=12))),
        )

    @property
    def key(self) -> str:
        api = facade()
        canonical = api.json.dumps(
            {
                "capability": self.capability,
                "context_tags": self.context_tags,
                "failure_code": self.failure_code,
                "failure_kind": self.failure_kind,
                "stage": self.stage,
                "task_family": self.task_family,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return "failure_" + api.hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class RecoveryStrategy:
    """A reusable recovery recipe made only from stable action identifiers."""

    strategy_id: str
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        api = facade()
        object.__setattr__(self, "strategy_id", api._identifier(self.strategy_id, "strategy_id"))
        actions = api._identifiers(self.action_ids, "action_ids", maximum=16)
        if not actions:
            raise ValueError("action_ids must contain at least one action")
        object.__setattr__(self, "action_ids", actions)


@dataclass(frozen=True, slots=True)
class VerificationContract:
    """Evidence needed to treat an outcome as machine-verified."""

    assurance: str
    required_checks: tuple[str, ...]
    passed_checks: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        api = facade()
        assurance = api._identifier(self.assurance, "assurance")
        if assurance not in api._ASSURANCE_RANK:
            raise ValueError(f"unsupported assurance level: {assurance}")
        object.__setattr__(self, "assurance", assurance)
        object.__setattr__(
            self,
            "required_checks",
            api._identifiers(self.required_checks, "required_checks", maximum=24),
        )
        object.__setattr__(
            self,
            "passed_checks",
            api._identifiers(self.passed_checks, "passed_checks", maximum=24),
        )
        object.__setattr__(
            self,
            "evidence_kinds",
            api._identifiers(self.evidence_kinds, "evidence_kinds", maximum=24),
        )
        if isinstance(self.evidence_count, bool) or not isinstance(self.evidence_count, int):
            raise TypeError("evidence_count must be an integer")
        if not 0 <= self.evidence_count <= 1_000_000:
            raise ValueError("evidence_count must be between 0 and 1000000")

    @property
    def missing_checks(self) -> tuple[str, ...]:
        return tuple(check for check in self.required_checks if check not in self.passed_checks)

    @property
    def evidence_coverage(self) -> float:
        required = len(self.required_checks)
        if not required:
            return 0.0
        passed = len(set(self.required_checks).intersection(self.passed_checks))
        return round(passed / required, 4)

    @property
    def satisfied(self) -> bool:
        return bool(
            self.assurance in facade()._VERIFIABLE_ASSURANCE
            and self.required_checks
            and not self.missing_checks
            and self.evidence_kinds
            and self.evidence_count > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "assurance": self.assurance,
            "assurance_rank": facade()._ASSURANCE_RANK[self.assurance],
            "required_checks": list(self.required_checks),
            "passed_checks": list(self.passed_checks),
            "missing_checks": list(self.missing_checks),
            "evidence_kinds": list(self.evidence_kinds),
            "evidence_count": self.evidence_count,
            "evidence_coverage": self.evidence_coverage,
            "satisfied": self.satisfied,
        }

    @classmethod
    def from_outcome(cls, outcome: dict[str, Any]) -> VerificationContract:
        raw_value = outcome.get("verification")
        raw: dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
        try:
            return cls(
                assurance=str(raw["assurance"]),
                required_checks=tuple(raw["required_checks"]),
                passed_checks=tuple(raw["passed_checks"]),
                evidence_kinds=tuple(raw["evidence_kinds"]),
                evidence_count=raw["evidence_count"],
            )
        except KeyError as exc:
            raise ValueError("outcome does not contain a verification contract") from exc
