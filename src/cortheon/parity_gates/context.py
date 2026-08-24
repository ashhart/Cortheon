"""The evaluation context and the records the gate stages hand each other.

Every gate reads the same report and contract, and each stage needs a few
values the previous one derived. Collecting them here keeps the stages
independent of one another's internals and keeps ``checks`` append-only: the
order in which a stage records its checks is the order a reviewer reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortheon.parity_gates.values import _mapping


@dataclass
class ParityContext:
    """Everything the gate stages share, derived once from an untrusted report."""

    report: dict[str, Any]
    contract: dict[str, Any]
    contract_sha256: str
    case_bank: dict[str, Any]
    methodology: dict[str, Any]
    candidates: dict[str, Any]
    summaries: dict[str, Any]
    cases: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    candidate_name: str
    frontier_names: list[str]
    frontier_families: dict[str, str]
    contender_models: dict[str, str]
    contender_endpoints: dict[str, str]
    registered_pricing: dict[str, Any]
    required_domains: set[str]
    thresholds: dict[str, Any]
    repetitions: int
    checks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        report: dict[str, Any],
        contract: dict[str, Any],
        contract_sha256: str,
    ) -> ParityContext:
        methodology = _mapping(report.get("methodology"))
        return cls(
            report=report,
            contract=contract,
            contract_sha256=contract_sha256,
            case_bank=_mapping(report.get("case_bank")),
            methodology=methodology,
            candidates=_mapping(report.get("candidates")),
            summaries=_mapping(report.get("summary")),
            cases=[item for item in report.get("cases") or [] if isinstance(item, dict)],
            rows=[item for item in report.get("rows") or [] if isinstance(item, dict)],
            candidate_name=str(contract["candidate"]),
            frontier_names=[str(value) for value in contract["frontiers"]],
            frontier_families={
                str(key): str(value)
                for key, value in _mapping(contract.get("frontier_families")).items()
            },
            contender_models={
                str(key): str(value)
                for key, value in _mapping(contract.get("contender_models")).items()
            },
            contender_endpoints={
                str(key): str(value).rstrip("/")
                for key, value in _mapping(contract.get("contender_endpoints")).items()
            },
            registered_pricing=_mapping(contract.get("pricing_per_million")),
            required_domains={str(value) for value in contract["required_domains"]},
            thresholds=_mapping(contract.get("thresholds")),
            repetitions=int(methodology.get("repetitions") or 0),
        )

    def check(self, name: str, passed: bool, **evidence: Any) -> None:
        """Record one gate outcome with the evidence a reviewer needs to audit it."""

        self.checks.append({"name": name, "passed": bool(passed), **evidence})


@dataclass(frozen=True)
class ContenderIdentities:
    """How the report's opaque aliases map back to the registered contenders."""

    candidate_name: str
    candidate_alias: str | None
    candidate_identity: dict[str, Any]
    frontier_aliases: dict[str, str | None]
    observed_aliases: dict[str, str]

    @property
    def resolved(self) -> bool:
        """True when the candidate and every frontier were found in the report.

        An unresolved report cannot be scored: the outcome, comparison, and
        metering gates would all read empty row sets and report green.
        """

        return self.candidate_alias is not None and all(
            alias is not None for alias in self.frontier_aliases.values()
        )

    def contender_aliases(self) -> dict[str, str | None]:
        """The candidate first, then every frontier, in registration order."""

        return {self.candidate_name: self.candidate_alias, **self.frontier_aliases}


@dataclass(frozen=True)
class OutcomeSummary:
    """Candidate-side results the comparison gates reuse instead of recomputing."""

    candidate_summary: dict[str, Any]
    candidate_rows: list[dict[str, Any]]
    block_case_ids: set[str]
    false_allow_rate: float | None
