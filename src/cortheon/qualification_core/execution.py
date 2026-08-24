"""Execution of a single matrix cell under preflight/postflight invariants."""

from __future__ import annotations

import json
import os
import random
import sys

from cortheon.cognitive_benchmark import (
    RunResult,
    _model_endpoint_health,
    _runtime_health,
    discover_benchmark_cases,
    run_job,
)

# Borrowed re-export, never a call target here. The facade has always exposed
# the workspace fingerprint, and it sources the name from this module, so the
# binding has to stay -- with the original object identity. The two pre/post
# checks in ``_run_cell`` deliberately go through ``facade()`` instead, so
# rebinding ``cortheon.qualification_factory._repository_fingerprint`` steers
# them exactly as it did when both call sites lived in the pre-split god file.
from cortheon.cognitive_benchmark import _repository_fingerprint as _repository_fingerprint
from cortheon.cognitive_http import _source_fingerprint
from cortheon.cognitive_protocol import CORTHEON_PROTOCOL_VERSION
from cortheon.qualification_core._compat import facade
from cortheon.qualification_core.conditions import (
    CONDITION_REGISTRY_VERSION,
    CONTRASTS,
    OLD_PLANNER,
    execution_profile,
    implementation_digest,
)
from cortheon.qualification_core.digests import _sealed_task_digest
from cortheon.qualification_core.environment import (
    _cell_namespace,
    _command_version,
    _public_runtime_health,
)
from cortheon.qualification_core.frozen_execution import run_frozen_job
from cortheon.qualification_core.frozen_old_planner import frozen_old_planner
from cortheon.qualification_core.models import (
    Cell,
    CellRun,
    Manifest,
    QualificationError,
)
from cortheon.qualification_core.pairing import _independent_pairing


def _run_cell(
    manifest: Manifest,
    cell: Cell,
    *,
    case_filter: str | None,
    repeat_filter: int | None,
    progress: bool,
) -> CellRun:
    if cell.api_key_env:
        api_key = os.environ.get(cell.api_key_env)
        if api_key is None:
            raise QualificationError(
                f"cell {cell.cell_id!r} requires environment variable {cell.api_key_env}"
            )
    else:
        api_key = ""
    args = _cell_namespace(
        cell,
        repository=manifest.repository,
        api_key=api_key,
    )
    implementation_pre_valid = implementation_digest() == cell.condition_implementation_sha256
    runtime_source_pre = _source_fingerprint()
    if not implementation_pre_valid:
        raise QualificationError(
            f"cell {cell.cell_id!r} condition implementation changed after manifest load"
        )
    try:
        runtime_pre = _runtime_health(cell.runtime_url)
        if (
            runtime_pre.get("protocol_version") != CORTHEON_PROTOCOL_VERSION
            or runtime_pre.get("source_fingerprint") != runtime_source_pre
        ):
            raise QualificationError(
                f"cell {cell.cell_id!r} runtime identity does not match evaluator sources"
            )
        inference_pre = _model_endpoint_health(
            cell.base_url,
            api_key=api_key,
            model_id=cell.model_id,
        )
        host_command = cell.opencode if cell.host == "opencode" else cell.pi
        host_pre = _command_version(host_command)
        starting_fingerprint = facade()._repository_fingerprint(manifest.repository)
        cases = discover_benchmark_cases(
            manifest.repository,
            count=cell.cases,
            seed=cell.seed,
            suite=cell.suite,
        )
    except QualificationError:
        raise
    except (OSError, ValueError) as exc:
        raise QualificationError(f"cell {cell.cell_id!r} failed preflight") from exc
    if case_filter is not None:
        cases = [case for case in cases if case.case_id == case_filter]
        if not cases:
            raise QualificationError(f"cell {cell.cell_id!r} does not contain case {case_filter!r}")
    repeats = (repeat_filter,) if repeat_filter is not None else tuple(range(cell.repeats))
    if any(repeat < 0 or repeat >= cell.repeats for repeat in repeats):
        raise QualificationError(
            f"repeat filter is outside cell {cell.cell_id!r}'s configured range"
        )
    jobs = [
        (case, repeat, condition)
        for case in cases
        for repeat in repeats
        for condition in cell.condition_ids
    ]
    random.Random(cell.seed ^ 0x0FA17F1E).shuffle(jobs)
    results: list[RunResult] = []
    for index, (case, repeat, condition) in enumerate(jobs, start=1):
        try:
            if condition == OLD_PLANNER:
                with frozen_old_planner() as historical_runtime:
                    result = run_frozen_job(
                        args,
                        case,
                        repeat=repeat,
                        runtime=historical_runtime,
                    )
            else:
                profile = execution_profile(
                    condition,
                    cell.condition_implementation_sha256,
                )
                result = run_job(
                    args,
                    case,
                    repeat=repeat,
                    treatment=condition != "bare",
                    condition=condition,
                    evaluation_profile=profile,
                )
                result.condition_registry_version = CONDITION_REGISTRY_VERSION
        except (OSError, ValueError) as exc:
            raise QualificationError(
                f"cell {cell.cell_id!r} could not execute a benchmark job"
            ) from exc
        results.append(result)
        if progress:
            print(
                json.dumps(
                    {
                        "cell": cell.cell_id,
                        "progress": index,
                        "total": len(jobs),
                        "case_id": case.case_id,
                        "repeat": repeat,
                        "condition": result.condition,
                        "correct": result.correct,
                        "delivered": result.delivered,
                        "latency_seconds": result.latency_seconds,
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
    try:
        runtime_post = _runtime_health(cell.runtime_url)
        inference_post = _model_endpoint_health(
            cell.base_url,
            api_key=api_key,
            model_id=cell.model_id,
        )
        host_post = _command_version(host_command)
        ending_fingerprint = facade()._repository_fingerprint(manifest.repository)
    except (OSError, ValueError) as exc:
        raise QualificationError(f"cell {cell.cell_id!r} failed postflight") from exc
    contrasts: dict[str, dict[str, object]] = {}
    contrast_case_deltas: dict[str, dict[str, float]] = {}
    contrast_invalid_case_ids: dict[str, set[str]] = {}
    for contrast, comparison in CONTRASTS.items():
        if comparison not in cell.condition_ids:
            continue
        selected = [item for item in results if item.condition in {"full", comparison}]
        paired, deltas, invalid = _independent_pairing(
            selected,
            treatment="full",
            comparison=comparison,
            repeats=repeats,
            seed=cell.seed,
        )
        contrasts[contrast] = paired
        contrast_case_deltas[contrast] = deltas
        contrast_invalid_case_ids[contrast] = invalid
    primary = contrasts["full_vs_bare"]
    primary_deltas = contrast_case_deltas["full_vs_bare"]
    primary_invalid = contrast_invalid_case_ids["full_vs_bare"]
    implementation_post_valid = implementation_digest() == cell.condition_implementation_sha256
    runtime_source_post = _source_fingerprint()
    return CellRun(
        cell=cell,
        case_ids=tuple(case.case_id for case in cases),
        task_digests={case.case_id: _sealed_task_digest(case) for case in cases},
        results=results,
        pairing=primary,
        case_deltas=primary_deltas,
        invalid_case_ids=primary_invalid,
        repository_unchanged=starting_fingerprint == ending_fingerprint,
        environment_stable=(
            host_pre == host_post
            and runtime_pre.get("version") == runtime_post.get("version")
            and runtime_pre.get("protocol_version") == CORTHEON_PROTOCOL_VERSION
            and runtime_post.get("protocol_version") == CORTHEON_PROTOCOL_VERSION
            and runtime_pre.get("source_fingerprint") == runtime_source_pre
            and runtime_post.get("source_fingerprint") == runtime_source_post
            and runtime_source_pre == runtime_source_post
            and runtime_post.get("storage") == "memory_only"
            and inference_pre.get("model_id") == inference_post.get("model_id")
            and inference_post.get("ok") is True
            and implementation_post_valid
        ),
        runtime=_public_runtime_health(runtime_post),
        inference={
            "ok": inference_post.get("ok") is True,
            "model_id": inference_post.get("model_id"),
            "models_reported": inference_post.get("models_reported"),
        },
        host_version=host_post,
        evaluator_runtime_source_fingerprint=runtime_source_post,
        evaluator_runtime_protocol=CORTHEON_PROTOCOL_VERSION,
        contrasts=contrasts,
        contrast_case_deltas=contrast_case_deltas,
        contrast_invalid_case_ids=contrast_invalid_case_ids,
        scheduled_repeats=repeats,
    )
