from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cortheon.threat_model.catalog import REQUIRED_RISK_IDS, THREAT_MANIFEST
from cortheon.threat_model.validation import validate_threat_model

ROOT = Path(__file__).parents[1]


def _must_not_run(_command: tuple[str, ...], _root: Path) -> subprocess.CompletedProcess[str]:
    raise AssertionError("invalid catalog must fail before pytest collection")


def test_catalog_enumerates_every_fixed_risk_once_with_one_owner() -> None:
    assert tuple(risk.risk_id for risk in THREAT_MANIFEST.risks) == REQUIRED_RISK_IDS
    nodes = [node for risk in THREAT_MANIFEST.risks for node in risk.test_node_ids]
    assert len(THREAT_MANIFEST.risks) == 49
    assert len(nodes) == 67
    assert len(nodes) == len(set(nodes))
    assert all(
        risk.owner and risk.severity in {"critical", "high"} for risk in THREAT_MANIFEST.risks
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "renamed", "duplicate", "unowned", "duplicate_node"]
)
def test_inventory_mutations_fail_closed_before_collection(mutation: str) -> None:
    risks = list(THREAT_MANIFEST.risks)
    if mutation == "missing":
        risks.pop()
    elif mutation == "renamed":
        risks[0] = replace(risks[0], risk_id="renamed_attack")
    elif mutation == "duplicate":
        risks[1] = replace(risks[1], risk_id=risks[0].risk_id)
    elif mutation == "unowned":
        risks[0] = replace(risks[0], owner="")
    else:
        risks[1] = replace(risks[1], test_node_ids=risks[0].test_node_ids)
    manifest = replace(THREAT_MANIFEST, risks=tuple(risks))
    result = validate_threat_model(ROOT, manifest, runner=_must_not_run)
    assert result.valid is False
    assert result.collection_valid is False
    assert result.errors


def test_lower_residuals_are_versioned_owned_dispositions_not_empty_claims() -> None:
    assert {residual.severity for residual in THREAT_MANIFEST.residuals} <= {"medium", "low"}
    assert len({residual.residual_id for residual in THREAT_MANIFEST.residuals}) == len(
        THREAT_MANIFEST.residuals
    )
    assert all(
        residual.statement and residual.disposition for residual in THREAT_MANIFEST.residuals
    )
