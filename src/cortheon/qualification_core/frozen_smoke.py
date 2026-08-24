"""Content-free live attestation for the frozen OpenCode comparator."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.health import _model_endpoint_health
from cortheon.benchmark_core.models import PatchCase
from cortheon.qualification_core.frozen_archive import (
    _ROOT,
    ARCHIVE_SHA256,
    FROZEN_COMMIT,
    FROZEN_TREE,
    archive_available,
)
from cortheon.qualification_core.frozen_execution import _run_frozen_evidence
from cortheon.qualification_core.frozen_old_planner import (
    WRAPPER_SHA256,
    frozen_implementation_sha256,
    frozen_old_planner,
    wrapper_sha256,
)
from cortheon.qualification_core.frozen_receipt import (
    SMOKE_BASE_URL,
    SMOKE_MODEL,
    SMOKE_POLICY,
    SMOKE_PROVIDER,
    SMOKE_SCHEMA_VERSION,
    SMOKE_TASK,
    _receipt_sha,
    smoke_endpoint_sha256,
    smoke_task_sha256,
    validate_smoke_receipt,
)

SMOKE_CASE = PatchCase(
    case_id=SMOKE_TASK["case_id"],
    files=tuple((path, content) for path, content in SMOKE_TASK["files"]),
    protected_paths=tuple(SMOKE_TASK["protected_paths"]),
    test_command=SMOKE_TASK["test_command"],
    hidden_assertions=SMOKE_TASK["hidden_assertions"],
    prompt=SMOKE_TASK["prompt"],
)


def _executable_identity(command: str) -> tuple[Path, str, str]:
    located = shutil.which(command)
    if located is None:
        raise ValueError("OpenCode executable was not found")
    executable = Path(located).resolve()
    if not executable.is_file() or executable.stat().st_size > 250_000_000:
        raise ValueError("OpenCode executable is not a bounded regular file")
    digest = _receipt_sha(executable.read_bytes())
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("OpenCode version probe failed") from exc
    version = completed.stdout.strip()
    if completed.returncode != 0 or not 0 < len(version) <= 128 or "\n" in version:
        raise ValueError("OpenCode returned an invalid version")
    return executable, version, digest


def generate_smoke_candidate(
    *,
    opencode: str,
    provider: str,
    model_id: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Run the fixed smoke task while the qualification gate stays closed."""

    if (provider, model_id, base_url) != (SMOKE_PROVIDER, SMOKE_MODEL, SMOKE_BASE_URL):
        raise ValueError("smoke identity does not match the registered fixed identity")
    if not api_key:
        raise ValueError("the smoke model credential is empty")
    if not archive_available() or wrapper_sha256() != WRAPPER_SHA256:
        raise ValueError("frozen comparator artifact verification failed")
    executable, version_pre, executable_sha_pre = _executable_identity(opencode)
    if version_pre != "1.18.18":
        raise ValueError("OpenCode version does not match the registered smoke host")
    inference_pre = _model_endpoint_health(base_url, api_key=api_key, model_id=model_id)
    args = argparse.Namespace(
        host="opencode",
        opencode=str(executable),
        pi="pi",
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        repository=_ROOT,
        runtime_url="http://127.0.0.1:1",
        reasoning=False,
        **SMOKE_POLICY,
    )
    with frozen_old_planner() as runtime:
        result, delta, lifecycle_valid, cleaned = _run_frozen_evidence(
            args,
            SMOKE_CASE,
            repeat=0,
            runtime=runtime,
        )
        active_postflight = runtime.health().get("active_sessions")
        artifact_unchanged = runtime.unchanged()
    inference_post = _model_endpoint_health(base_url, api_key=api_key, model_id=model_id)
    _executable, version_post, executable_sha_post = _executable_identity(str(executable))
    if (
        inference_pre.get("ok") is not True
        or inference_post.get("ok") is not True
        or inference_pre.get("model_id") != model_id
        or inference_post.get("model_id") != model_id
        or version_post != version_pre
        or executable_sha_post != executable_sha_pre
        or not lifecycle_valid
        or cleaned != 1
        or active_postflight != 0
        or not artifact_unchanged
        or result.process_error is not None
        or result.timed_out
        or not result.execution_identity_valid
        or not result.execution_measurements_valid
        or result.inference_provider_id != provider
        or result.inference_model_id != model_id
        or result.evaluator_outcome.terminal_status != "success"
        or not result.delivered
    ):
        raise ValueError("frozen comparator smoke did not meet the attestation contract")
    receipt = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "kind": "cortheon_frozen_old_planner_smoke",
        "content_free": True,
        "commit": FROZEN_COMMIT,
        "git_tree_sha1": FROZEN_TREE,
        "archive_sha256": ARCHIVE_SHA256,
        "wrapper_sha256": WRAPPER_SHA256,
        "implementation_sha256": frozen_implementation_sha256(),
        "sealed_task_sha256": smoke_task_sha256(),
        "host": {
            "id": "opencode",
            "version": version_post,
            "executable_sha256": executable_sha_post,
        },
        "inference": {
            "provider_id": result.inference_provider_id,
            "model_id": result.inference_model_id,
            "endpoint_sha256": smoke_endpoint_sha256(),
            "identity_valid": result.execution_identity_valid,
            "measurements_valid": result.execution_measurements_valid,
        },
        "policy": {
            "max_steps": result.policy_max_steps,
            "max_tool_calls": result.policy_max_tool_calls,
            "timeout_seconds": result.policy_timeout_seconds,
            "context_tokens": result.policy_context_tokens,
            "output_tokens": result.policy_output_tokens,
        },
        "outcome": {
            "terminal_status": result.evaluator_outcome.terminal_status,
            "delivered": result.delivered,
            "candidate_correct": result.candidate_correct,
            "observed_steps": result.observed_steps,
            "observed_tool_calls": result.tool_calls,
        },
        "runtime": {
            "sessions_started": delta["sessions_started"],
            "observations_accepted": delta["observations_accepted"],
            "sessions_completed": delta["sessions_completed"],
            "sessions_evidence_closed": delta["sessions_evidence_closed"],
            "sessions_abandoned": delta["sessions_abandoned"],
            "completion_withheld": delta["completion_withheld"],
            "evaluator_cleanup_sessions": cleaned,
            "active_sessions_postflight": active_postflight,
            "artifact_unchanged": artifact_unchanged,
        },
    }
    if not validate_smoke_receipt(
        receipt,
        frozen_implementation_sha256(),
        WRAPPER_SHA256,
    ):
        raise ValueError("generated frozen comparator receipt is invalid")
    return receipt


def regenerate_smoke(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regenerate-frozen-old-planner-smoke")
    parser.add_argument("--opencode", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env) is None:
        raise SystemExit("--api-key-env is invalid")
    api_key = os.environ.get(args.api_key_env, "")
    try:
        candidate = generate_smoke_candidate(
            opencode=args.opencode,
            provider=args.provider,
            model_id=args.model_id,
            base_url=args.base_url,
            api_key=api_key,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    encoded = (json.dumps(candidate, indent=2) + "\n").encode()
    sys.stdout.buffer.write(encoded)
    print(f"sha256={_receipt_sha(encoded)}", file=sys.stderr)
    return 0
