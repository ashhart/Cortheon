"""The behavioral rung: classify execution failures and repair them.

Structural checks (``code_check``) and the runtime-bind grader (``runtime_env``)
verify names and signatures — but not semantics. A generated program can pass
both and still raise at runtime (a top-level ``await``, a misused *real* method,
a ``1/0``). The behavioral rung closes that gap: it runs the program, reads the
traceback, classifies it, and — for genuine user-code faults — feeds the
traceback back for one repair pass before re-running.

The uncertainty principle (same as the bind rung): environmental errors are
infrastructure uncertainty, never evidence against the code. The sandbox runs
with ``--network none``, so a ``ConnectionError``/DNS/timeout is *expected*, not
a bug, and must never trigger a repair or a banner.

This module is pure logic plus an executor protocol; it has no I/O of its own
and is fully testable offline. The two executor implementations live in
``sandbox.execute_answer_code`` (Docker) and the host-venv executor wired in the
proxy (opt-in, unsandboxed, disclosed).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from cortheon.sandbox import CodeBlockResult, ExecutionReport

# A block executor: takes (blocks, package_specs) and returns an ExecutionReport.
# Both the Docker sandbox path and the opt-in host-venv path conform to this.
Executor = Callable[[list[str], list[str]], ExecutionReport]


class TracebackClass(StrEnum):
    """What kind of failure a block's traceback represents."""

    USER_CODE_FAULT = "user_code_fault"
    ENVIRONMENTAL = "environmental"
    UNKNOWN = "unknown"


# Environmental signals: the sandbox runs network-off, so these are expected.
# Counting them against the code would punish a correct program for the
# isolation working as designed — the same principle the bind rung follows.
_ENVIRONMENTAL_PATTERNS = (
    "ConnectionError",
    "ConnectError",
    "httpx.ConnectError",
    "requests.exceptions.Connection",
    "URLError",
    "socket.gaierror",
    "socket.timeout",
    "TimeoutError",
    "asyncio.exceptions.TimeoutError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "Name or service not known",
    "Temporary failure in name resolution",
    "nodename nor servname provided",
    "Failed to establish a new connection",
)

# Exception types that, when rooted in the generated file, are unambiguous
# user-code faults (not environmental, not infrastructure).
_USER_FAULT_TYPES = (
    "SyntaxError",
    "IndentationError",
    "NameError",
    "TypeError",
    "AttributeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
    "ImportError",
    "ModuleNotFoundError",
    "UnboundLocalError",
    "NotImplementedError",
    "RecursionError",
)

_FRAME_LINE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)')
_EXC_LINE = re.compile(r"^([\w.]+(?:Error|Exception|Warning))\s*:")


def classify_traceback(stderr: str, source_file: str = "block") -> TracebackClass:
    """Classify a block's traceback into user-fault / environmental / unknown.

    ``source_file`` is the basename (or path) of the generated program the block
    was written to. The last frame rooted in that file points at user code; a
    SyntaxError naming it does too. Environmental signals anywhere in the
    traceback take precedence only when no user-code frame is present — a
    ``NameError`` inside a ``ConnectionError`` stack is still a user bug the
    model can fix.
    """
    if not stderr:
        return TracebackClass.UNKNOWN

    # 1. The final exception line is the strongest signal: a NameError/
    #    TypeError/etc. at the bottom is a user bug even if an environmental
    #    exception appears higher in the stack (e.g. a user NameError raised
    #    while constructing a request that would have connected).
    final_exc = _final_exception_type(stderr)
    if final_exc and final_exc in _USER_FAULT_TYPES:
        return TracebackClass.USER_CODE_FAULT

    # 2. Otherwise, an environmental signal with no user-code frame is the
    #    sandbox's network-off isolation working as designed — not a bug.
    has_user_frame = _last_frame_is_user(stderr, source_file)
    environmental = any(sig in stderr for sig in _ENVIRONMENTAL_PATTERNS)
    if environmental and not has_user_frame:
        return TracebackClass.ENVIRONMENTAL

    # 3. Last frame in the generated file => user code (covers exception types
    #    not in the explicit list, e.g. a custom library error raised from user
    #    code).
    if has_user_frame:
        return TracebackClass.USER_CODE_FAULT

    return TracebackClass.UNKNOWN


def _last_frame_is_user(stderr: str, source_file: str) -> bool:
    """True if the deepest 'File "..."' frame is the generated program."""
    target = source_file.rstrip("/").split("/")[-1]
    last_file = None
    for line in stderr.splitlines():
        m = _FRAME_LINE.match(line)
        if m:
            last_file = m.group("file")
    if last_file is None:
        return False
    return last_file.rstrip("/").split("/")[-1] == target


def _final_exception_type(stderr: str) -> str | None:
    """The exception type on the last non-empty traceback line."""
    last = None
    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _EXC_LINE.match(line)
        if m:
            last = m.group(1)
        elif _FRAME_LINE.match(line.lstrip()):
            # A frame line resets the "last seen" since the actual exception
            # follows the deepest frame.
            last = None
    return last


@dataclass(slots=True)
class BlockClassification:
    index: int
    ok: bool
    classification: TracebackClass
    stderr_tail: str


@dataclass(slots=True)
class ExecutionOutcome:
    """An ExecutionReport plus per-block classification and a repair verdict."""

    report: ExecutionReport
    classifications: list[BlockClassification]

    @classmethod
    def from_report(
        cls, report: ExecutionReport, source_basename: str = "block"
    ) -> ExecutionOutcome:
        classes = [
            BlockClassification(
                index=b.index,
                ok=b.ok,
                classification=TracebackClass.USER_CODE_FAULT
                if b.ok
                else classify_traceback(b.stderr_tail, source_basename),
                stderr_tail=b.stderr_tail,
            )
            for b in report.blocks
        ]
        return cls(report=report, classifications=classes)

    @property
    def failing(self) -> list[BlockClassification]:
        return [c for c in self.classifications if not c.ok]

    @property
    def repairable(self) -> bool:
        """Ran, with at least one user-code fault and no environmental-only block.

        Environmental/unknown failures are not repairable by re-asking the model
        — they are infrastructure uncertainty (the sandbox is network-off), and
        re-prompting would just burn a retry without a signal the model can use.
        """
        if not self.report.ran or not self.failing:
            return False
        return any(c.classification is TracebackClass.USER_CODE_FAULT for c in self.failing)


def build_repair_feedback(outcome: ExecutionOutcome) -> str:
    """Feedback for the model's one repair pass: the tracebacks it must fix.

    Mirrors the tone of the existing feedback-retry rung — concrete, scoped to
    the failure, and explicit about keeping the rest of the answer identical.
    """
    faults = [c for c in outcome.failing if c.classification is TracebackClass.USER_CODE_FAULT]
    parts = ["Your code ran but raised these errors when executed:"]
    for c in faults[:3]:
        tb = c.stderr_tail.strip()
        parts.append(f"  block {c.index}:\n{tb}")
    parts.append(
        "Fix the bug(s) above. Give your COMPLETE answer again, keeping everything else "
        "exactly as it was (prose, requirements, other code blocks), and changing only what "
        "is needed so the code runs without raising. Output the corrected ```python block(s)."
    )
    return "\n".join(parts)


def sandbox_executor(blocks: list[str], packages: list[str]) -> ExecutionReport:
    """The default executor: run blocks in an ephemeral Docker sandbox.

    Thin wrapper over ``sandbox.execute_answer_code`` so the proxy can treat
    sandboxed and host execution as one pluggable protocol. Keeps the security
    win (network-off, read-only package mount, capped).
    """
    from cortheon.sandbox import execute_answer_code

    return execute_answer_code(blocks, packages)


def host_venv_executor_factory(pool) -> Executor:
    """Build a host-venv executor bound to a ``RuntimeEnvPool``.

    Opt-in and UNSANDBOXED: installs the program's packages into a cached host
    venv and runs each block directly under a scrubbed environment (no host
    secrets, but no network isolation either). Faster than Docker, but it runs
    generated code on the host — the proxy always discloses which executor ran
    in ``meta['execution']['executor']`` so the choice is never silent.

    The pool is injected (not imported) so this module stays importable offline
    and the executor is testable with a fake pool.
    """

    def _execute(blocks: list[str], packages: list[str]) -> ExecutionReport:
        import subprocess
        import tempfile
        import time
        from pathlib import Path

        from cortheon.verifier import _scrubbed_env, _tail

        if not blocks:
            return ExecutionReport(ran=False, reason="no code blocks", blocks=[])
        specs = list(packages)
        python = pool.python_for_specs(specs, wait=False)
        if python is None:
            # Cold env building in the background; don't block the request path
            # (same contract as the bind rung's wait=False).
            return ExecutionReport(ran=False, reason="env building, try again", blocks=[])
        results: list[CodeBlockResult] = []
        with tempfile.TemporaryDirectory(prefix="behavioral-host-") as base:
            base_dir = Path(base)
            for index, code in enumerate(blocks, start=1):
                work_dir = base_dir / f"block-{index}"
                work_dir.mkdir(parents=True, exist_ok=True)
                script = work_dir / "block.py"
                script.write_text(code, encoding="utf-8")
                start = time.monotonic()
                try:
                    completed = subprocess.run(
                        [str(python), str(script)],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=work_dir,
                        env=_scrubbed_env(work_dir),
                    )
                    ok = completed.returncode == 0
                    returncode: int | None = completed.returncode
                    stdout, stderr = completed.stdout, completed.stderr
                except subprocess.TimeoutExpired:
                    ok = False
                    returncode = None
                    stdout = ""
                    stderr = "timed out after 60s"
                except OSError as exc:
                    ok = False
                    returncode = None
                    stdout = ""
                    stderr = str(exc)
                results.append(
                    CodeBlockResult(
                        index=index,
                        ok=ok,
                        returncode=returncode
                        if returncode is not None and returncode >= 0
                        else None,
                        duration_seconds=round(time.monotonic() - start, 3),
                        stdout_tail=_tail(stdout or "", 500),
                        stderr_tail=_tail(stderr or "", 500),
                    )
                )
        return ExecutionReport(ran=True, reason="executed (host venv)", blocks=results)

    return _execute
