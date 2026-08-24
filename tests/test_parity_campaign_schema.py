"""Schema-level validation of campaign registrations and results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cortheon.parity_campaign.errors import CampaignContractError
from cortheon.parity_campaign.results import validate_results
from cortheon.parity_campaign.schema import (
    registration_digest,
    validate_registration,
)

FAMILIES = ("qwen", "llama", "mistral")
HOSTS = ("codex", "generic_mcp", "opencode", "pi")
EVALUATORS = ("lab-a", "lab-b")
_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _registration(**overrides) -> dict:
    cells = [
        _cell(family, host, evaluator)
        for family in FAMILIES
        for host in HOSTS
        for evaluator in EVALUATORS
    ]
    payload = {
        "schema_version": 2,
        "claim": "replicated_broad_frontier_parity",
        "campaign_id": "campaign-test",
        "declared_at": "2026-08-01T00:00:00Z",
        "cells": cells,
    }
    payload.update(overrides)
    return payload


def _cell(family: str, host: str, evaluator: str, **overrides) -> dict:
    cell = {
        "cell_id": f"{family}-{host}-{evaluator}",
        "family": family,
        "model": family,
        "host": host,
        "runtime_sha256": _digest(f"runtime:{family}:{host}"),
        "evaluator": evaluator,
        "evaluator_key_sha256": _digest(f"evaluator-key:{evaluator}"),
        "pack_issuer": f"{evaluator}-grading",
        "pack_id": f"pack-{family}-{host}-{evaluator}",
        "runner_id": f"runner-{family}-{host}-{evaluator}",
        "contract_path": f"contracts/{family}-{host}.json",
        "contract_sha256": _digest(f"contract:{family}:{host}"),
        "pack_path": f"packs/{family}-{host}-{evaluator}.json",
        "pack_sha256": _digest(f"pack:{family}:{host}:{evaluator}"),
        "pack_key_env": f"KEY_{evaluator.replace('-', '_').upper()}_PACK",
        "runner_key_env": (f"KEY_{family}_{host}_{evaluator}_RUNNER".replace("-", "_").upper()),
    }
    cell.update(overrides)
    return cell


def _results(payload: dict, **overrides) -> dict:
    results = [
        {
            "cell_id": cell["cell_id"],
            "submission_path": f"submissions/{cell['cell_id']}.json",
            "submission_sha256": _digest(f"submission:{cell['cell_id']}"),
            "report_path": f"reports/{cell['cell_id']}.json",
            "report_sha256": _digest(f"report:{cell['cell_id']}"),
        }
        for cell in payload["cells"]
    ]
    body = {
        "schema_version": 1,
        "campaign_id": payload["campaign_id"],
        "registration_sha256": registration_digest(payload),
        "results": results,
    }
    body.update(overrides)
    return body


def _validate(payload: dict, base: Path = Path("/tmp/campaign")) -> list:
    return validate_registration(payload, base_dir=base)


def test_valid_registration_and_results_round_trip(tmp_path: Path) -> None:
    payload = _registration()
    cells = _validate(payload, tmp_path)
    assert len(cells) == 24
    assert {cell["host"] for cell in cells} == set(HOSTS)
    digest = registration_digest(payload)
    again = registration_digest(json.loads(json.dumps(payload)))
    assert digest == again
    mapped = validate_results(
        _results(payload),
        cells=cells,
        expected_campaign_id=payload["campaign_id"],
        expected_registration_digest=digest,
        base_dir=tmp_path,
    )
    assert set(mapped) == {cell["cell_id"] for cell in cells}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": 1}, "schema_version must be 2"),
        ({"schema_version": "2"}, "schema_version must be 2"),
        ({"claim": "broad_frontier_parity"}, "claim must be"),
        ({"campaign_id": "  "}, "campaign_id must be a non-empty string"),
        ({"campaign_id": 7}, "campaign_id must be a non-empty string"),
        ({"declared_at": "2026-08-01T00:00:00"}, "must carry a UTC"),
        ({"declared_at": "2026-08-01T00:00:00+02:00"}, "must carry a UTC"),
        ({"declared_at": "not-a-timestamp"}, "ISO-8601"),
        ({"cells": []}, "non-empty cells"),
        ({"extra": True}, "fields must be exactly"),
        ({"declared_at": None}, "ISO-8601"),
    ],
)
def test_registration_header_fails_closed(mutation: dict, message: str) -> None:
    with pytest.raises(CampaignContractError, match=message):
        _validate(_registration(**mutation))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"contract_sha256": "nothex"}, "64 lowercase hex"),
        ({"host": "mainframe"}, "host must be one of"),
        ({"host": ""}, "must be a string"),
        ({"cell_id": "Bad ID"}, "cell_id must match"),
        ({"pack_key_env": "lower_case"}, "environment variable name"),
        ({"runner_key_env": "1BAD"}, "environment variable name"),
        ({"model": 3}, "must be a string"),
        ({"unexpected": "field"}, "fields must be exactly"),
    ],
)
def test_cell_fields_fail_closed(mutation: dict, message: str) -> None:
    payload = _registration()
    payload["cells"][0].update(mutation)
    with pytest.raises(CampaignContractError, match=message):
        _validate(payload)


def test_missing_cell_field_fails_closed() -> None:
    payload = _registration()
    del payload["cells"][5]["pack_id"]
    with pytest.raises(CampaignContractError, match="fields must be exactly"):
        _validate(payload)


def test_registration_cannot_preregister_post_run_digests() -> None:
    payload = _registration()
    payload["cells"][0]["report_sha256"] = _DIGEST
    with pytest.raises(CampaignContractError, match="fields must be exactly"):
        _validate(payload)


def test_duplicate_cell_id_fails_closed() -> None:
    payload = _registration()
    payload["cells"][1]["cell_id"] = payload["cells"][0]["cell_id"]
    with pytest.raises(CampaignContractError, match="reuses cell_id"):
        _validate(payload)


def test_reused_pack_or_runner_fails_closed() -> None:
    payload = _registration()
    payload["cells"][1]["pack_id"] = payload["cells"][0]["pack_id"]
    with pytest.raises(CampaignContractError, match="reuses pack_id"):
        _validate(payload)
    payload = _registration()
    payload["cells"][1]["runner_id"] = payload["cells"][0]["runner_id"]
    with pytest.raises(CampaignContractError, match="reuses runner_id"):
        _validate(payload)
    payload = _registration()
    payload["cells"][1]["pack_sha256"] = payload["cells"][0]["pack_sha256"]
    with pytest.raises(CampaignContractError, match="reuses pack_sha256"):
        _validate(payload)


def test_diagonal_only_matrix_fails_closed() -> None:
    """Four families each on one host is not cross-host replication."""

    cells = [
        _cell(family, host, evaluator)
        for family, host in (
            ("qwen", "pi"),
            ("llama", "codex"),
            ("mistral", "opencode"),
            ("gemma", "generic_mcp"),
        )
        for evaluator in EVALUATORS
    ]
    payload = _registration(cells=cells)
    with pytest.raises(CampaignContractError, match="complete product"):
        _validate(payload)


def test_missing_evaluator_replicate_fails_closed() -> None:
    payload = _registration()
    del payload["cells"][3]
    with pytest.raises(CampaignContractError, match="complete product"):
        _validate(payload)


def test_missing_host_family_or_evaluator_fails_closed() -> None:
    cells = [
        _cell(family, host, evaluator)
        for family in ("qwen", "llama")
        for host in HOSTS
        for evaluator in EVALUATORS
    ]
    with pytest.raises(CampaignContractError, match="at least 3 local model families"):
        _validate(_registration(cells=cells))
    cells = [
        _cell(family, host, evaluator)
        for family in FAMILIES
        for host in ("codex", "opencode", "pi")
        for evaluator in EVALUATORS
    ]
    with pytest.raises(CampaignContractError, match="every supported host"):
        _validate(_registration(cells=cells))
    cells = [_cell(family, host, "lab-a") for family in FAMILIES for host in HOSTS]
    with pytest.raises(CampaignContractError, match="at least 2 evaluators"):
        _validate(_registration(cells=cells))


def test_duplicate_logical_cell_fails_closed() -> None:
    payload = _registration()
    payload["cells"][1] = _cell(
        "qwen",
        "codex",
        "lab-a",
        cell_id="clone",
        pack_id="pack-clone",
        pack_sha256=_digest("pack:clone"),
        runner_id="runner-clone",
        pack_path="packs/clone.json",
        pack_key_env="KEY_LAB_A_PACK",
        runner_key_env="KEY_CLONE_RUNNER",
    )
    with pytest.raises(CampaignContractError, match="more than once"):
        _validate(payload)


def test_inconsistent_evaluator_replicates_fail_closed() -> None:
    payload = _registration()
    payload["cells"][0]["contract_sha256"] = _OTHER_DIGEST
    with pytest.raises(CampaignContractError, match="same inner contract"):
        _validate(payload)
    # A diverging model on an evaluator replicate is caught even earlier, by
    # the one-exact-model-per-family rule that spans hosts and evaluators.
    payload = _registration()
    payload["cells"][0]["model"] = "other-model"
    with pytest.raises(CampaignContractError, match="one exact model"):
        _validate(payload)


def test_contract_shared_across_logical_cells_fails_closed() -> None:
    payload = _registration()
    shared_digest = payload["cells"][4]["contract_sha256"]
    shared_path = payload["cells"][4]["contract_path"]
    for index in (0, 1):
        payload["cells"][index]["contract_sha256"] = shared_digest
        payload["cells"][index]["contract_path"] = shared_path
    with pytest.raises(CampaignContractError, match="exactly one logical"):
        _validate(payload)


def test_one_exact_model_per_family_across_hosts_and_evaluators() -> None:
    """A different exact model on one host is an exact-model confound."""

    payload = _registration()
    payload["cells"][4]["model"] = "qwen-alt"
    with pytest.raises(CampaignContractError, match="one exact model"):
        _validate(payload)
    # Evaluator replicates must stay identical too.
    payload = _registration()
    payload["cells"][1]["model"] = "qwen-alt"
    with pytest.raises(CampaignContractError, match="one exact model"):
        _validate(payload)


def test_evaluator_commitment_identity_fails_closed() -> None:
    # One evaluator must commit to exactly one signing key across its cells.
    payload = _registration()
    payload["cells"][0]["evaluator_key_sha256"] = _OTHER_DIGEST
    with pytest.raises(CampaignContractError, match="exactly one"):
        _validate(payload)
    # Different evaluators must not share a commitment (one secret = one authority).
    payload = _registration()
    for cell in payload["cells"]:
        if cell["evaluator"] == "lab-b":
            cell["evaluator_key_sha256"] = _digest("evaluator-key:lab-a")
    with pytest.raises(CampaignContractError, match="distinct commitments"):
        _validate(payload)
    # A shared pack-key environment variable across evaluators is rejected too.
    payload = _registration()
    for cell in payload["cells"]:
        if cell["evaluator"] == "lab-b":
            cell["pack_key_env"] = "KEY_LAB_A_PACK"
    with pytest.raises(CampaignContractError, match="share pack key"):
        _validate(payload)


def test_shared_pack_key_env_within_one_evaluator_is_allowed() -> None:
    payload = _registration()
    envs = {cell["pack_key_env"] for cell in payload["cells"] if cell["evaluator"] == "lab-a"}
    assert len(envs) == 1
    cells = _validate(payload, Path("/tmp/campaign"))
    assert len(cells) == 24


def test_pack_path_alias_fails_closed() -> None:
    payload = _registration()
    payload["cells"][1]["pack_path"] = "./alias/../" + payload["cells"][0]["pack_path"]
    with pytest.raises(CampaignContractError, match="reuses sealed pack path"):
        _validate(payload, Path("/tmp/campaign").resolve())


def test_results_header_and_binding_fail_closed(tmp_path: Path) -> None:
    payload = _registration()
    cells = _validate(payload, tmp_path)
    digest = registration_digest(payload)
    kwargs = {
        "cells": cells,
        "expected_campaign_id": payload["campaign_id"],
        "expected_registration_digest": digest,
        "base_dir": tmp_path,
    }
    with pytest.raises(CampaignContractError, match="schema_version must be 1"):
        validate_results(_results(payload, schema_version=2), **kwargs)
    with pytest.raises(CampaignContractError, match="campaign_id does not match"):
        validate_results(_results(payload, campaign_id="other"), **kwargs)
    with pytest.raises(CampaignContractError, match="registration_sha256 does not match"):
        validate_results(_results(payload, registration_sha256=_DIGEST), **kwargs)


def test_missing_duplicate_and_extra_results_fail_closed(tmp_path: Path) -> None:
    payload = _registration()
    cells = _validate(payload, tmp_path)
    kwargs = {
        "cells": cells,
        "expected_campaign_id": payload["campaign_id"],
        "expected_registration_digest": registration_digest(payload),
        "base_dir": tmp_path,
    }
    missing = _results(payload)
    del missing["results"][2]
    with pytest.raises(CampaignContractError, match="missing preregistered cells"):
        validate_results(missing, **kwargs)
    duplicate = _results(payload)
    duplicate["results"].append(dict(duplicate["results"][0]))
    with pytest.raises(CampaignContractError, match="more than once"):
        validate_results(duplicate, **kwargs)
    extra = _results(payload)
    extra["results"].append(
        {
            "cell_id": "never-registered",
            "submission_path": "submissions/extra.json",
            "submission_sha256": _DIGEST,
            "report_path": "reports/extra.json",
            "report_sha256": _OTHER_DIGEST,
        }
    )
    with pytest.raises(CampaignContractError, match="never preregistered"):
        validate_results(extra, **kwargs)


def test_result_path_and_digest_reuse_fail_closed(tmp_path: Path) -> None:
    payload = _registration()
    cells = _validate(payload, tmp_path)
    kwargs = {
        "cells": cells,
        "expected_campaign_id": payload["campaign_id"],
        "expected_registration_digest": registration_digest(payload),
        "base_dir": tmp_path,
    }
    aliased = _results(payload)
    aliased["results"][1]["submission_path"] = (
        "./submissions/../" + aliased["results"][0]["submission_path"]
    )
    with pytest.raises(CampaignContractError, match=r"path.*is reused"):
        validate_results(aliased, **kwargs)
    shared_digest = _results(payload)
    shared_digest["results"][1]["report_sha256"] = shared_digest["results"][0]["report_sha256"]
    with pytest.raises(CampaignContractError, match="reuse a report_sha256"):
        validate_results(shared_digest, **kwargs)
    cross_kind = _results(payload)
    cross_kind["results"][0]["report_path"] = "submissions/../" + payload["cells"][0]["pack_path"]
    with pytest.raises(CampaignContractError, match="not be reused across kinds"):
        validate_results(cross_kind, **kwargs)
