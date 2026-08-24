from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from cortheon.models import Evidence, PatchReport, PatchTestRun, SupportLevel, utc_now
from cortheon.repo_scanner import SKIP_DIRS, scan_repo
from cortheon.verifier import _scrubbed_env

DEFAULT_TEST_TIMEOUT = 600
DEFAULT_PATCH_TEST_IMAGE = "python:3.12-slim-bookworm"
TEST_ISOLATION_MODES = frozenset({"host", "docker", "disabled"})
_ENV_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)
_SHELL_CONTROL_TOKENS = frozenset({"&", "&&", "(", ")", ";", "<", "<<", ">", ">>", "|", "||"})


def run_patch_verification(
    repo_path: str | Path,
    patch_text: str,
    *,
    test_command: str | None = None,
    run_baseline: bool = True,
    timeout_seconds: int = DEFAULT_TEST_TIMEOUT,
    test_isolation: str = "host",
    sandbox_image: str = DEFAULT_PATCH_TEST_IMAGE,
) -> PatchReport:
    """Verify a proposed patch by applying it to a scratch copy and running tests.

    This closes the loop: the substrate stops judging only the evidence going
    INTO the model and starts judging the code coming OUT of it. The verdict is
    earned from test results, never from how plausible the diff looks.

    ``test_isolation="docker"`` executes the model-influenced scratch tree in a
    no-network, resource-capped container with no inherited host secrets. Host
    execution remains available only as an explicit trust-tier choice for
    developer-controlled repositories. ``disabled`` applies and inspects the
    patch without executing it.
    """
    if test_isolation not in TEST_ISOLATION_MODES:
        return _failed_report(
            str(Path(repo_path).expanduser()),
            ["test_isolation must be one of: " + ", ".join(sorted(TEST_ISOLATION_MODES))],
        )
    root = Path(repo_path).expanduser().resolve()
    errors: list[str] = []
    notes: list[str] = []
    if not root.is_dir():
        return _failed_report(str(root), [f"Repository path does not exist: {root}"])
    if not patch_text.strip():
        return _failed_report(str(root), ["Patch text is empty."])
    if not patch_text.endswith("\n"):
        patch_text += "\n"

    repo = scan_repo(root)
    command = test_command or (repo.test_commands[0] if repo.test_commands else None)
    if command and test_isolation == "disabled":
        notes.append("Test execution is disabled; the patch will not earn a verified verdict.")
        command = None
    if not command:
        notes.append(
            "No test command was detected or provided; the patch can be applied but not verified."
        )

    baseline: PatchTestRun | None = None
    after: PatchTestRun | None = None
    applied = False
    files_changed: list[str] = []
    insertions = 0
    deletions = 0

    with tempfile.TemporaryDirectory(prefix="cortheon-patch-") as tmp:
        scratch = Path(tmp) / "repo"
        shutil.copytree(root, scratch, ignore=shutil.ignore_patterns(*SKIP_DIRS, ".git"))
        ok, output = _git(["init", "-q"], scratch)
        ok = ok and _git(["add", "-A"], scratch)[0]
        ok = ok and _git(["commit", "-q", "-m", "baseline"], scratch)[0]
        if not ok:
            errors.append(f"Could not initialize scratch git repo: {output.strip()[:200]}")
            return _failed_report(str(root), errors)

        if command and run_baseline:
            baseline = _run_tests(
                command,
                scratch,
                timeout_seconds,
                isolation=test_isolation,
                sandbox_image=sandbox_image,
            )
            _clear_runtime_caches(scratch)

        patch_file = Path(tmp) / "proposed.patch"
        patch_file.write_text(patch_text, encoding="utf-8")
        applied, apply_output = _git(["apply", "--whitespace=nowarn", str(patch_file)], scratch)
        if not applied:
            errors.append(f"Patch does not apply: {apply_output.strip()[:400]}")
        else:
            _, names = _git(["diff", "--name-only"], scratch)
            files_changed = [line.strip() for line in names.splitlines() if line.strip()]
            _, numstat = _git(["diff", "--numstat"], scratch)
            for line in numstat.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    insertions += int(parts[0]) if parts[0].isdigit() else 0
                    deletions += int(parts[1]) if parts[1].isdigit() else 0
            if command:
                after = _run_tests(
                    command,
                    scratch,
                    timeout_seconds,
                    isolation=test_isolation,
                    sandbox_image=sandbox_image,
                )

    verdict, verdict_notes, earned = _verdict(applied, command, baseline, after)
    notes.extend(verdict_notes)
    rollback_plan = _rollback_plan(files_changed)
    evidence = [
        _patch_evidence(str(root), applied, command, baseline, after, verdict, files_changed)
    ]
    return PatchReport(
        repo_root=str(root),
        generated_at=utc_now(),
        applied=applied,
        files_changed=files_changed,
        insertions=insertions,
        deletions=deletions,
        baseline=baseline,
        after=after,
        verdict=verdict,
        earned_evidence_tags=earned,
        rollback_plan=rollback_plan,
        notes=notes,
        evidence=evidence,
        errors=errors,
    )


def _verdict(
    applied: bool,
    command: str | None,
    baseline: PatchTestRun | None,
    after: PatchTestRun | None,
) -> tuple[str, list[str], list[str]]:
    if not applied:
        return "block", ["The patch does not apply cleanly; do not proceed with this diff."], []
    if not command or after is None:
        return (
            "needs_evidence",
            [
                "The patch applies, but no test command verified it. Provide --test-command or add tests."
            ],
            [],
        )
    if after.passed and (baseline is None or baseline.passed):
        return "allow", ["Tests pass after the patch."], ["tests_passed"]
    if after.passed and baseline is not None and not baseline.passed:
        return (
            "allow",
            ["Tests were failing before the patch and pass after it — the patch fixes the suite."],
            ["tests_passed"],
        )
    if baseline is not None and baseline.passed and not after.passed:
        return "block", ["REGRESSION: tests passed before the patch and fail after it."], []
    return (
        "needs_evidence",
        [
            "Tests fail both before and after the patch; the failure cannot be attributed to this diff."
        ],
        [],
    )


def _run_tests(
    command: str,
    cwd: Path,
    timeout_seconds: int,
    *,
    isolation: str,
    sandbox_image: str,
) -> PatchTestRun:
    start = time.monotonic()
    environment = _scrubbed_env(cwd)
    if isolation == "docker":
        executed_command = _docker_test_command(
            command,
            cwd,
            sandbox_image,
        )
        environment = os.environ.copy()
    else:
        try:
            executed_command, command_environment = _host_test_invocation(command)
        except ValueError as exc:
            return PatchTestRun(
                command=command,
                ran=False,
                passed=False,
                returncode=None,
                duration_seconds=round(time.monotonic() - start, 3),
                output_tail=f"host test command rejected: {exc}",
            )
        environment.update(command_environment)
    try:
        completed = subprocess.run(
            executed_command,
            shell=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
        passed: bool | None = completed.returncode == 0
        returncode: int | None = completed.returncode
        output = f"{completed.stdout}\n{completed.stderr}"
    except subprocess.TimeoutExpired:
        passed = False
        returncode = None
        output = f"test command timed out after {timeout_seconds}s"
    except OSError as exc:
        passed = False
        returncode = None
        output = (
            "isolated test runner could not start; host fallback is disabled: "
            f"{type(exc).__name__}: {exc}"
        )
    return PatchTestRun(
        command=(f"docker:{sandbox_image}:{command}" if isolation == "docker" else command),
        ran=True,
        passed=passed,
        returncode=returncode,
        duration_seconds=round(time.monotonic() - start, 3),
        output_tail=output.strip()[-1500:],
    )


def _host_test_invocation(command: str) -> tuple[list[str], dict[str, str]]:
    """Parse a host test command without invoking a shell."""

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"invalid quoting: {exc}") from exc
    if not tokens:
        raise ValueError("command is empty")

    environment: dict[str, str] = {}
    while tokens:
        assignment = _ENV_ASSIGNMENT.fullmatch(tokens[0])
        if assignment is None:
            break
        name, value = assignment.groups()
        environment[name] = value
        tokens.pop(0)
    if not tokens:
        raise ValueError("command contains only environment assignments")
    if any(token in _SHELL_CONTROL_TOKENS or "\n" in token for token in tokens):
        raise ValueError(
            "shell operators are not allowed in host mode; use a checked-in test script"
        )
    return tokens, environment


def _docker_test_command(
    command: str,
    cwd: Path,
    image: str,
) -> list[str]:
    """Build a fail-closed test command with no host execution fallback."""

    if shutil.which("docker") is None:
        # A missing executable produces a bounded failed test result rather
        # than silently crossing into the host trust tier.
        return [
            "__cortheon_docker_unavailable__",
        ]
    user = (
        f"{os.getuid()}:{os.getgid()}"
        if hasattr(os, "getuid") and hasattr(os, "getgid")
        else "65534:65534"
    )
    return [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--pids-limit",
        "256",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--user",
        user,
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{cwd}:/workspace:rw",
        "-w",
        "/workspace",
        image,
        "sh",
        "-lc",
        command,
    ]


def _clear_runtime_caches(root: Path) -> None:
    """Prevent a baseline run from masking same-size source edits.

    CPython's timestamp-based bytecode cache can remain valid when a patch
    changes source without changing its size inside the filesystem timestamp
    granularity. The after-test must execute the patched source, never cached
    baseline bytecode.
    """

    for directory in root.rglob("__pycache__"):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
    for bytecode in root.rglob("*.py[co]"):
        with contextlib.suppress(OSError):
            bytecode.unlink()


def _git(args: list[str], cwd: Path) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "user.email=cortheon@local",
                "-c",
                "user.name=cortheon",
                *args,
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except OSError as exc:
        return False, f"git unavailable: {type(exc).__name__}: {exc}"
    return completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"


def _rollback_plan(files_changed: list[str]) -> list[str]:
    plan = [
        "Apply the patch on a branch, never directly on main.",
        "Reverse it with: git apply -R proposed.patch",
    ]
    if files_changed:
        plan.append("Or restore the touched files: git checkout -- " + " ".join(files_changed[:12]))
    return plan


def _patch_evidence(
    root: str,
    applied: bool,
    command: str | None,
    baseline: PatchTestRun | None,
    after: PatchTestRun | None,
    verdict: str,
    files_changed: list[str],
) -> Evidence:
    if verdict == "allow":
        support = SupportLevel.VERIFIED
        claim = (
            f"Proposed patch for {root} applied cleanly and the test command passed "
            f"({command}); verdict allow."
        )
    elif verdict == "block":
        support = SupportLevel.FAILED
        claim = f"Proposed patch for {root} was blocked: " + (
            "it does not apply cleanly." if not applied else "it regresses the test suite."
        )
    else:
        support = SupportLevel.INFERRED
        claim = f"Proposed patch for {root} applied but could not be verified by tests."
    return Evidence(
        claim=claim,
        source_type="patch_verification",
        source_url=None,
        support=support,
        details={
            "repo_root": root,
            "applied": applied,
            "test_command": command,
            "baseline_passed": baseline.passed if baseline else None,
            "after_passed": after.passed if after else None,
            "files_changed": files_changed[:20],
            "verdict": verdict,
        },
    )


def _failed_report(root: str, errors: list[str]) -> PatchReport:
    return PatchReport(
        repo_root=root,
        generated_at=utc_now(),
        applied=False,
        files_changed=[],
        insertions=0,
        deletions=0,
        baseline=None,
        after=None,
        verdict="block",
        earned_evidence_tags=[],
        rollback_plan=[],
        notes=[],
        evidence=[
            Evidence(
                claim=f"Patch verification failed to run for {root}.",
                source_type="patch_verification",
                source_url=None,
                support=SupportLevel.FAILED,
                details={"errors": errors},
            )
        ],
        errors=errors,
    )
