"""Loading and strict validation of qualification manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cortheon.qualification_core.conditions import (
    HISTORICAL_CONDITIONS,
    REQUIRED_CONDITIONS,
    condition_record,
    implementation_digest,
)
from cortheon.qualification_core.constants import (
    CELL_KEYS,
    ENVIRONMENT_NAME,
    HOSTS,
    IDENTIFIER,
    MAX_CELLS,
    MAX_JOBS,
    ROOT_KEYS,
    SCHEMA_VERSION,
    SUITES,
    TIER_DEFAULTS,
)
from cortheon.qualification_core.gates import _strict_gate_overrides
from cortheon.qualification_core.models import Cell, Manifest, QualificationError
from cortheon.qualification_core.validation import (
    _bounded_int,
    _bounded_number,
    _bounded_text,
    _http_url,
    _parse_document,
    _reject_embedded_credentials,
    _reject_unknown,
)


def _parse_cell(
    raw: Any,
    *,
    index: int,
    tier: str,
    root_seed: int,
    implementation_sha256: str | None = None,
) -> Cell:
    location = f"manifest.cells[{index}]"
    if not isinstance(raw, dict):
        raise QualificationError(f"{location} must be an object")
    _reject_unknown(raw, CELL_KEYS, location)
    cell_id = _bounded_text(raw.get("id"), field=f"{location}.id", limit=64)
    if not IDENTIFIER.fullmatch(cell_id):
        raise QualificationError(
            f"{location}.id must contain only letters, digits, '.', '_' or '-'"
        )
    suite = _bounded_text(
        raw.get("suite"),
        field=f"{location}.suite",
        default="mixed",
        limit=32,
    )
    if suite not in SUITES:
        raise QualificationError(f"{location}.suite must be one of: " + ", ".join(sorted(SUITES)))
    host = _bounded_text(
        raw.get("host"),
        field=f"{location}.host",
        default="opencode",
        limit=32,
    )
    if host not in HOSTS:
        raise QualificationError(f"{location}.host must be one of: " + ", ".join(sorted(HOSTS)))
    if suite == "research" and host != "opencode":
        raise QualificationError(f"{location}: the research suite requires the OpenCode host")
    api_key_env = raw.get("api_key_env")
    if api_key_env is not None:
        api_key_env = _bounded_text(
            api_key_env,
            field=f"{location}.api_key_env",
            limit=128,
        )
        if not ENVIRONMENT_NAME.fullmatch(api_key_env):
            raise QualificationError(f"{location}.api_key_env is not a valid environment name")
    reasoning = raw.get("reasoning", False)
    if not isinstance(reasoning, bool):
        raise QualificationError(f"{location}.reasoning must be a boolean")
    historical_comparison = raw.get("historical_comparison", False)
    if not isinstance(historical_comparison, bool):
        raise QualificationError(f"{location}.historical_comparison must be a boolean")
    if historical_comparison and host != "opencode":
        raise QualificationError(f"{location}.historical_comparison is available only for OpenCode")
    measured_implementation = implementation_sha256 or implementation_digest()
    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list):
        raise QualificationError(f"{location}.conditions must be a list")
    condition_ids: list[str] = []
    for condition_index, supplied in enumerate(raw_conditions):
        condition_location = f"{location}.conditions[{condition_index}]"
        if not isinstance(supplied, dict) or set(supplied) != {
            "id",
            "config_sha256",
            "implementation_sha256",
        }:
            raise QualificationError(f"{condition_location} has invalid fields")
        condition_id = supplied.get("id")
        if not isinstance(condition_id, str):
            raise QualificationError(f"{condition_location}.id must be a string")
        try:
            expected = condition_record(
                condition_id,
                implementation_sha256=measured_implementation,
                host=host,
            )
        except ValueError as exc:
            raise QualificationError(f"{condition_location}.id is not registered") from exc
        if not expected["available"]:
            raise QualificationError(
                f"{condition_location} is unavailable: {expected['unavailable_reason']}"
            )
        if supplied != {
            "id": condition_id,
            "config_sha256": expected["config_sha256"],
            "implementation_sha256": expected["implementation_sha256"],
        }:
            raise QualificationError(f"{condition_location} digest mismatch")
        condition_ids.append(condition_id)
    expected_conditions = HISTORICAL_CONDITIONS if historical_comparison else REQUIRED_CONDITIONS
    if tuple(condition_ids) != expected_conditions:
        raise QualificationError(
            f"{location}.conditions do not match its preregistered condition matrix"
        )
    return Cell(
        cell_id=cell_id,
        suite=suite,
        host=host,
        provider=_bounded_text(
            raw.get("provider"),
            field=f"{location}.provider",
            default="Local",
            limit=128,
        ),
        base_url=_http_url(
            raw.get("base_url"),
            field=f"{location}.base_url",
            default="http://127.0.0.1:18081/v1",
        ),
        api_key_env=api_key_env,
        model_id=_bounded_text(
            raw.get("model_id"),
            field=f"{location}.model_id",
            default="qwen3-1.7b",
            limit=256,
        ),
        runtime_url=_http_url(
            raw.get("runtime_url"),
            field=f"{location}.runtime_url",
            default="http://127.0.0.1:8743",
        ),
        cases=_bounded_int(
            raw.get("cases", 2),
            field=f"{location}.cases",
            minimum=2,
            maximum=1_000,
        ),
        repeats=_bounded_int(
            raw.get("repeats", TIER_DEFAULTS[tier]["default_repeats"]),
            field=f"{location}.repeats",
            minimum=1,
            maximum=20,
        ),
        seed=_bounded_int(
            raw.get("seed", root_seed),
            field=f"{location}.seed",
            minimum=0,
            maximum=2**31 - 1,
        ),
        timeout_seconds=_bounded_number(
            raw.get("timeout_seconds", 60.0),
            field=f"{location}.timeout_seconds",
            minimum=0.1,
            maximum=3_600.0,
        ),
        context_tokens=_bounded_int(
            raw.get("context_tokens", 32_768),
            field=f"{location}.context_tokens",
            minimum=1_024,
            maximum=10_000_000,
        ),
        output_tokens=_bounded_int(
            raw.get("output_tokens", 2_048),
            field=f"{location}.output_tokens",
            minimum=64,
            maximum=1_000_000,
        ),
        max_steps=_bounded_int(
            raw.get("max_steps", 4),
            field=f"{location}.max_steps",
            minimum=1,
            maximum=32,
        ),
        reasoning=reasoning,
        opencode=_bounded_text(
            raw.get("opencode"),
            field=f"{location}.opencode",
            default="opencode",
            limit=1_024,
        ),
        pi=_bounded_text(
            raw.get("pi"),
            field=f"{location}.pi",
            default="pi",
            limit=1_024,
        ),
        condition_ids=tuple(condition_ids),
        condition_implementation_sha256=measured_implementation,
        historical_comparison=historical_comparison,
    )


def load_manifest(path: Path) -> Manifest:
    """Load and strictly validate a JSON or TOML qualification manifest."""

    path = path.expanduser().resolve()
    raw, digest = _parse_document(path)
    _reject_embedded_credentials(raw)
    _reject_unknown(raw, ROOT_KEYS, "manifest")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise QualificationError(f"manifest.schema_version must be {SCHEMA_VERSION}")
    tier = _bounded_text(
        raw.get("tier"),
        field="manifest.tier",
        default="pr",
        limit=16,
    )
    if tier not in TIER_DEFAULTS:
        raise QualificationError(
            "manifest.tier must be one of: " + ", ".join(sorted(TIER_DEFAULTS))
        )
    root_seed = _bounded_int(
        raw.get("seed", 20_260_726),
        field="manifest.seed",
        minimum=0,
        maximum=2**31 - 1,
    )
    raw_repository = raw.get("repository", ".")
    repository_text = _bounded_text(
        raw_repository,
        field="manifest.repository",
        limit=4_096,
    )
    repository = (path.parent / repository_text).expanduser().resolve()
    if not repository.is_dir():
        raise QualificationError("manifest.repository must resolve to a directory")
    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, list) or not 1 <= len(raw_cells) <= MAX_CELLS:
        raise QualificationError(f"manifest.cells must contain between 1 and {MAX_CELLS} cells")
    measured_implementation = implementation_digest()
    cells = tuple(
        _parse_cell(
            item,
            index=index,
            tier=tier,
            root_seed=root_seed,
            implementation_sha256=measured_implementation,
        )
        for index, item in enumerate(raw_cells)
    )
    identifiers = [cell.cell_id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise QualificationError("manifest cell IDs must be unique")
    total_jobs = sum(cell.cases * cell.repeats * len(cell.condition_ids) for cell in cells)
    if total_jobs > MAX_JOBS:
        raise QualificationError(f"manifest expands to {total_jobs} jobs; the limit is {MAX_JOBS}")
    return Manifest(
        path=path,
        digest=digest,
        tier=tier,
        repository=repository,
        seed=root_seed,
        cells=cells,
        gates=_strict_gate_overrides(tier, raw.get("gates")),
        condition_implementation_sha256=measured_implementation,
    )
