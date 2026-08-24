from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cortheon.models import Evidence, ExampleRunResult, SupportLevel, VerificationResult


def run_install_import_test(
    package: str,
    version: str,
    import_name: str | None = None,
    timeout_seconds: int = 120,
    examples: list[str] | None = None,
) -> tuple[VerificationResult, list[Evidence]]:
    import_target = import_name or guess_import_name(package)
    start = time.monotonic()
    command: list[str] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    install_ok = False
    import_ok: bool | None = None
    example_results: list[ExampleRunResult] = []

    with tempfile.TemporaryDirectory(prefix="cortheon-venv-") as tmp:
        venv_dir = Path(tmp) / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        python = _venv_python(venv_dir)
        command = [str(python), "-m", "pip", "install", f"{package}=={version}"]
        install = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout_parts.append(install.stdout)
        stderr_parts.append(install.stderr)
        install_ok = install.returncode == 0
        if install_ok:
            import_command = [str(python), "-c", f"import {import_target}; print('import-ok')"]
            imported = subprocess.run(
                import_command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            stdout_parts.append(imported.stdout)
            stderr_parts.append(imported.stderr)
            import_ok = imported.returncode == 0
        else:
            import_ok = False
        if examples and install_ok and import_ok:
            example_results = run_python_examples(
                python,
                examples,
                Path(tmp) / "examples",
                timeout_seconds=min(60, timeout_seconds),
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
        example_results=example_results,
    )
    if install_ok and import_ok:
        claim = f"{package} {version} installed in an ephemeral virtualenv and imported as {import_target}."
        support = SupportLevel.VERIFIED
    else:
        claim = f"{package} {version} install/import smoke test failed or was incomplete."
        support = SupportLevel.FAILED
    evidence = [
        Evidence(
            claim=claim,
            source_type="local_execution",
            source_url=None,
            package=package,
            version=version,
            support=support,
            details={
                "import_name": import_target,
                "install_ok": install_ok,
                "import_ok": import_ok,
                "duration_seconds": round(duration, 3),
            },
        )
    ]
    if example_results:
        passed = sum(1 for item in example_results if item.ok)
        all_ok = passed == len(example_results)
        evidence.append(
            Evidence(
                claim=(
                    f"{passed} of {len(example_results)} official example(s) (README/docs) for {package} {version} "
                    "executed successfully in the ephemeral virtualenv."
                ),
                source_type="local_example_execution",
                source_url=None,
                package=package,
                version=version,
                support=SupportLevel.VERIFIED if all_ok else SupportLevel.FAILED,
                details={
                    "passed": passed,
                    "total": len(example_results),
                    "durations_seconds": [item.duration_seconds for item in example_results],
                },
            )
        )
    return result, evidence


def run_python_examples(
    python: Path | str,
    examples: list[str],
    base_dir: Path,
    timeout_seconds: int = 60,
) -> list[ExampleRunResult]:
    """Run extracted example scripts with a scrubbed environment.

    Each example gets its own working directory and HOME so host secrets and
    user files stay out of reach. This is opt-in execution, same trust tier as
    the install/import smoke test.
    """
    results: list[ExampleRunResult] = []
    for index, code in enumerate(examples, start=1):
        work_dir = base_dir / f"example-{index}"
        work_dir.mkdir(parents=True, exist_ok=True)
        script = work_dir / "example.py"
        script.write_text(code, encoding="utf-8")
        start = time.monotonic()
        try:
            completed = subprocess.run(
                [str(python), str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=work_dir,
                env=_scrubbed_env(work_dir),
            )
            ok = completed.returncode == 0
            returncode: int | None = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            ok = False
            returncode = None
            stdout = _coerce_text(exc.stdout)
            stderr = f"timed out after {timeout_seconds}s"
        results.append(
            ExampleRunResult(
                index=index,
                ok=ok,
                returncode=returncode,
                duration_seconds=round(time.monotonic() - start, 3),
                code=code,
                stdout_tail=_tail(stdout or "", 500),
                stderr_tail=_tail(stderr or "", 500),
            )
        )
    return results


def _scrubbed_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def guess_import_name(package: str) -> str:
    overrides = {
        "django-ninja": "ninja",
        "python-json-logger": "pythonjsonlogger",
        "beautifulsoup4": "bs4",
        "pillow": "PIL",
        "pyyaml": "yaml",
    }
    return overrides.get(package.lower(), package.replace("-", "_"))


def _venv_python(venv_dir: Path) -> Path:
    bin_dir = "Scripts" if sys.platform.startswith("win") else "bin"
    python = venv_dir / bin_dir / "python"
    if not python.exists():
        found = shutil.which("python", path=str(venv_dir / bin_dir))
        if found:
            return Path(found)
    return python


def _tail(value: str, limit: int = 3000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[-limit:]
