from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from cortheon.models import Evidence, ExampleRunResult, SupportLevel, VerificationResult
from cortheon.verifier import guess_import_name

DEFAULT_IMAGE = "python:3.12-slim"
INSTALL_MEMORY = "1g"
INSTALL_CPUS = "2"
INSTALL_PIDS = "256"
RUN_MEMORY = "512m"
RUN_CPUS = "1"
RUN_PIDS = "128"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_sandboxed_install_import_test(
    package: str,
    version: str,
    import_name: str | None = None,
    timeout_seconds: int = 300,
    examples: list[str] | None = None,
    image: str = DEFAULT_IMAGE,
    example_network: str = "none",
) -> tuple[VerificationResult, list[Evidence]]:
    """Containerized verification: install with network, execute without.

    Phase 1 installs the package into a scratch mount (network required for
    pip). Phase 2 imports and runs examples in fresh containers with
    --network none, a read-only package mount, and memory/pid caps — untrusted
    package code gets no chance to phone home or read host files.
    """
    import_target = import_name or guess_import_name(package)
    start = time.monotonic()
    if not docker_available():
        return _docker_unavailable_result(package, version, import_target)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    install_ok = False
    import_ok: bool | None = None
    example_results: list[ExampleRunResult] = []
    command: list[str] = []

    with tempfile.TemporaryDirectory(prefix="sandbox-", dir=_scratch_root()) as tmp:
        pkgs_dir = Path(tmp) / "pkgs"
        pkgs_dir.mkdir()
        command = docker_install_command(image, pkgs_dir, f"{package}=={version}")
        install = _run_docker(command, timeout_seconds)
        stdout_parts.append(install.stdout)
        stderr_parts.append(install.stderr)
        install_ok = install.returncode == 0

        if install_ok:
            import_command = docker_run_command(
                image,
                pkgs_dir,
                None,
                ["python", "-c", f"import {import_target}; print('import-ok')"],
                network="none",
            )
            imported = _run_docker(import_command, min(120, timeout_seconds))
            stdout_parts.append(imported.stdout)
            stderr_parts.append(imported.stderr)
            import_ok = imported.returncode == 0
        else:
            import_ok = False

        if examples and install_ok and import_ok:
            example_results = _run_sandboxed_examples(
                image,
                pkgs_dir,
                Path(tmp) / "examples",
                examples,
                network=example_network,
                timeout_seconds=min(90, timeout_seconds),
            )

    duration = time.monotonic() - start
    result = VerificationResult(
        package=package,
        version=version,
        install_ran=True,
        install_ok=install_ok,
        import_name=import_target,
        import_ok=import_ok,
        command=command,
        stdout_tail=_tail("\n".join(stdout_parts)),
        stderr_tail=_tail("\n".join(stderr_parts)),
        duration_seconds=round(duration, 3),
        source="docker_sandbox",
        example_results=example_results,
    )
    evidence = _sandbox_evidence(
        result,
        image=image,
        example_network=example_network,
    )
    return result, evidence


@dataclass
class CodeBlockResult:
    """One generated code block executed in the sandbox."""

    index: int
    ok: bool
    returncode: int | None
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str


@dataclass
class ExecutionReport:
    """The execution rung: run generated code in the sandbox after it passes
    structural checks. Catches semantic misuse of *real* methods that parse and
    bind cleanly — the gap structural checks cannot close.

    Opt-in by construction: the proxy only calls this when ``--execute`` is set,
    and the result is always disclosed in the ``cortheon`` meta. A failure
    does not silently rewrite the answer; it ships under a banner.
    """

    ran: bool
    reason: str  # why it did not run (no docker, no blocks, disabled)
    blocks: list[CodeBlockResult]

    @property
    def all_passed(self) -> bool:
        return self.ran and all(b.ok for b in self.blocks)


def execute_answer_code(
    blocks: list[str],
    packages: list[str],
    *,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = 90,
    example_network: str = "none",
) -> ExecutionReport:
    """Execute generated code blocks in an ephemeral sandbox.

    Installs the (package, current-version) specs with network, then runs each
    code block with ``--network none``, a read-only package mount, and memory/pid
    caps — the same isolation contract as example execution. A block that errors
    at runtime is a finding the structural checks provably cannot make (a misused
    *real* method). Docker unavailable is reported honestly, never faked.
    """
    if not blocks:
        return ExecutionReport(ran=False, reason="no code blocks", blocks=[])
    if not docker_available():
        return ExecutionReport(ran=False, reason="docker unavailable", blocks=[])

    with tempfile.TemporaryDirectory(prefix="sandbox-exec-", dir=_scratch_root()) as tmp:
        pkgs_dir = Path(tmp) / "pkgs"
        pkgs_dir.mkdir()
        work_dir = Path(tmp) / "work"
        work_dir.mkdir()
        # Install all referenced packages (best-effort spec; version resolved by
        # the caller or just "latest" if not pinned). One install phase serves
        # every block so import cost is paid once.
        for spec in packages:
            install = _run_docker(docker_install_command(image, pkgs_dir, spec), timeout_seconds)
            if install.returncode != 0:
                return ExecutionReport(
                    ran=False,
                    reason=f"install failed for {spec}: {_tail(install.stderr, 200)}",
                    blocks=[],
                )
        results: list[CodeBlockResult] = []
        for index, code in enumerate(blocks, start=1):
            script = work_dir / f"block-{index}.py"
            script.write_text(code, encoding="utf-8")
            command = docker_run_command(
                image,
                pkgs_dir,
                work_dir,
                ["python", f"/work/block-{index}.py"],
                network=example_network,
            )
            start = time.monotonic()
            completed = _run_docker(command, min(60, timeout_seconds))
            results.append(
                CodeBlockResult(
                    index=index,
                    ok=completed.returncode == 0,
                    returncode=completed.returncode if completed.returncode >= 0 else None,
                    duration_seconds=round(time.monotonic() - start, 3),
                    stdout_tail=_tail(completed.stdout, 500),
                    stderr_tail=_tail(completed.stderr, 500),
                )
            )
        return ExecutionReport(ran=True, reason="executed", blocks=results)


def docker_install_command(image: str, pkgs_dir: Path, spec: str) -> list[str]:
    # Install phase keeps network (pip needs it) but still gets resource caps
    # and only the scratch mount.
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        _container_name(),
        "--memory",
        INSTALL_MEMORY,
        "--cpus",
        INSTALL_CPUS,
        "--pids-limit",
        INSTALL_PIDS,
        "-v",
        f"{pkgs_dir}:/pkgs",
        image,
        "python",
        "-m",
        "pip",
        "install",
        "--target",
        "/pkgs",
        "--no-cache-dir",
        "--quiet",
        spec,
    ]


def docker_run_command(
    image: str,
    pkgs_dir: Path,
    work_dir: Path | None,
    args: list[str],
    network: str = "none",
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        _container_name(),
        "--network",
        network,
        "--memory",
        RUN_MEMORY,
        "--cpus",
        RUN_CPUS,
        "--pids-limit",
        RUN_PIDS,
        "-e",
        "PYTHONPATH=/pkgs",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{pkgs_dir}:/pkgs:ro",
    ]
    if work_dir is not None:
        command.extend(["-v", f"{work_dir}:/work", "-w", "/work"])
    command.append(image)
    command.extend(args)
    return command


def _run_sandboxed_examples(
    image: str,
    pkgs_dir: Path,
    base_dir: Path,
    examples: list[str],
    *,
    network: str,
    timeout_seconds: int,
) -> list[ExampleRunResult]:
    results: list[ExampleRunResult] = []
    for index, code in enumerate(examples, start=1):
        work_dir = base_dir / f"example-{index}"
        work_dir.mkdir(parents=True, exist_ok=True)
        script = work_dir / "example.py"
        script.write_text(code, encoding="utf-8")
        command = docker_run_command(
            image,
            pkgs_dir,
            work_dir,
            ["python", "/work/example.py"],
            network=network,
        )
        start = time.monotonic()
        completed = _run_docker(command, timeout_seconds)
        results.append(
            ExampleRunResult(
                index=index,
                ok=completed.returncode == 0,
                returncode=completed.returncode if completed.returncode >= 0 else None,
                duration_seconds=round(time.monotonic() - start, 3),
                code=code,
                stdout_tail=_tail(completed.stdout, 500),
                stderr_tail=_tail(completed.stderr, 500),
            )
        )
    return results


class _DockerRun:
    __slots__ = ("returncode", "stderr", "stdout")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_docker(command: list[str], timeout_seconds: int) -> _DockerRun:
    name = _name_from_command(command)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return _DockerRun(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        # The host-side timeout fired; make sure the container dies too.
        if name:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        return _DockerRun(-1, stdout, f"timed out after {timeout_seconds}s; container removed")


def _name_from_command(command: list[str]) -> str | None:
    for index, token in enumerate(command):
        if token == "--name" and index + 1 < len(command):
            return command[index + 1]
    return None


def _container_name() -> str:
    return f"cortheon-{secrets.token_hex(6)}"


def _scratch_root() -> str:
    """Scratch mounts must live where the Docker VM can see them.

    Docker Desktop shares /Users, colima shares $HOME — a path under the
    user's home works for both (and for native Linux Docker).
    """
    override = os.environ.get("CORTHEON_SANDBOX_DIR")
    root = Path(override) if override else Path.home() / ".cache" / "cortheon" / "sandbox"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _docker_unavailable_result(
    package: str,
    version: str,
    import_target: str,
) -> tuple[VerificationResult, list[Evidence]]:
    result = VerificationResult(
        package=package,
        version=version,
        install_ran=False,
        install_ok=None,
        import_name=import_target,
        import_ok=None,
        command=[],
        stdout_tail="",
        stderr_tail="Docker CLI was not found on PATH; sandboxed verification did not run.",
        duration_seconds=0.0,
        source="docker_sandbox",
    )
    evidence = [
        Evidence(
            claim=(
                f"Sandboxed verification for {package} {version} did not run because Docker is unavailable. "
                "No fallback to host execution was attempted."
            ),
            source_type="sandbox_execution",
            source_url=None,
            package=package,
            version=version,
            support=SupportLevel.FAILED,
            details={"reason": "docker_unavailable"},
        )
    ]
    return result, evidence


def _sandbox_evidence(
    result: VerificationResult,
    *,
    image: str,
    example_network: str,
) -> list[Evidence]:
    if result.install_ok and result.import_ok:
        claim = (
            f"{result.package} {result.version} installed and imported inside a Docker sandbox "
            f"({image}); import ran with the network disabled."
        )
        support = SupportLevel.VERIFIED
    else:
        claim = f"{result.package} {result.version} sandboxed install/import did not fully pass."
        support = SupportLevel.FAILED
    evidence = [
        Evidence(
            claim=claim,
            source_type="sandbox_execution",
            source_url=None,
            package=result.package,
            version=result.version,
            support=support,
            details={
                "image": image,
                "import_name": result.import_name,
                "install_ok": result.install_ok,
                "import_ok": result.import_ok,
                "duration_seconds": result.duration_seconds,
            },
        )
    ]
    if result.example_results:
        passed = sum(1 for item in result.example_results if item.ok)
        all_ok = passed == len(result.example_results)
        evidence.append(
            Evidence(
                claim=(
                    f"{passed} of {len(result.example_results)} official example(s) for {result.package} "
                    f"{result.version} executed in a Docker sandbox with network={example_network}."
                ),
                source_type="sandbox_example_execution",
                source_url=None,
                package=result.package,
                version=result.version,
                support=SupportLevel.VERIFIED if all_ok else SupportLevel.FAILED,
                details={
                    "passed": passed,
                    "total": len(result.example_results),
                    "network": example_network,
                    "image": image,
                },
            )
        )
    return evidence


def _tail(value: str, limit: int = 3000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[-limit:]
