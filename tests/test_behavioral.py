"""Tests for the behavioral rung: traceback classification + repair feedback.

The classifier is pure string logic, so these are deterministic and offline.
One end-to-end test exercises the host executor + classifier with a real
stdlib venv and a deliberate ``1/0`` (no Docker required).
"""

import sys
import unittest
from pathlib import Path

from cortheon.behavioral import (
    ExecutionOutcome,
    TracebackClass,
    build_repair_feedback,
    classify_traceback,
    host_venv_executor_factory,
)
from cortheon.sandbox import CodeBlockResult, ExecutionReport


def _tb(
    exc: str, file: str = "/work/block-1.py", line: int = 2, body: str = "print(undefined)"
) -> str:
    return (
        "Traceback (most recent call last):\n"
        f'  File "{file}", line {line}, in <module>\n'
        f"    {body}\n"
        f"{exc}\n"
    )


class ClassifyTracebackTests(unittest.TestCase):
    def test_syntax_error_in_user_file_is_user_fault(self) -> None:
        stderr = (
            '  File "/work/block-1.py", line 3\n'
            "    await foo()\n"
            "    ^^^^\n"
            "SyntaxError: 'await' outside async function\n"
        )
        self.assertIs(classify_traceback(stderr, "block-1.py"), TracebackClass.USER_CODE_FAULT)

    def test_name_error_in_user_file_is_user_fault(self) -> None:
        self.assertIs(
            classify_traceback(_tb("NameError: name 'x' is not defined"), "block-1.py"),
            TracebackClass.USER_CODE_FAULT,
        )

    def test_type_error_in_user_file_is_user_fault(self) -> None:
        self.assertIs(
            classify_traceback(_tb("TypeError: unsupported operand"), "block-1.py"),
            TracebackClass.USER_CODE_FAULT,
        )

    def test_zero_division_is_user_fault(self) -> None:
        stderr = _tb("ZeroDivisionError: division by zero", body="1/0")
        self.assertIs(classify_traceback(stderr, "block-1.py"), TracebackClass.USER_CODE_FAULT)

    def test_connection_error_no_user_frame_is_environmental(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/pkgs/httpx/_client.py", line 12, in send\n'
            "    ...\n"
            "httpx.ConnectError: [Errno -2] Name or service not known\n"
        )
        self.assertIs(classify_traceback(stderr, "block-1.py"), TracebackClass.ENVIRONMENTAL)

    def test_socket_gaierror_is_environmental(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/pkgs/urllib/request.py", line 100\n'
            "socket.gaierror: [Errno 8] nodename nor servname provided\n"
        )
        self.assertIs(classify_traceback(stderr, "block-1.py"), TracebackClass.ENVIRONMENTAL)

    def test_empty_stderr_is_unknown(self) -> None:
        self.assertIs(classify_traceback("", "block-1.py"), TracebackClass.UNKNOWN)

    def test_last_frame_in_stdlib_not_user_file_is_unknown(self) -> None:
        # No user-fault exception type, no user frame, no environmental signal.
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.12/os.py", line 600\n'
            "OSError: [Errno 1] something\n"
        )
        self.assertIs(classify_traceback(stderr, "block-1.py"), TracebackClass.UNKNOWN)

    def test_user_fault_wins_over_environmental_keyword(self) -> None:
        # A NameError at the bottom of the traceback is the real bug, even if an
        # environmental exception appears higher in the same stderr.
        stderr = _tb("NameError: name 'x' is not defined") + "ConnectionError: ...\n"
        # The final exception line is the NameError -> user fault.
        self.assertIs(classify_traceback(stderr, "block-1.py"), TracebackClass.USER_CODE_FAULT)


class ExecutionOutcomeTests(unittest.TestCase):
    def test_repairable_requires_ran_and_user_fault(self) -> None:
        report = ExecutionReport(
            ran=True,
            reason="executed",
            blocks=[CodeBlockResult(1, False, 1, 0.1, "", _tb("NameError: x"))],
        )
        outcome = ExecutionOutcome.from_report(report, "block-1.py")
        self.assertTrue(outcome.repairable)
        self.assertEqual(len(outcome.failing), 1)

    def test_not_repairable_when_all_pass(self) -> None:
        report = ExecutionReport(
            ran=True,
            reason="executed",
            blocks=[CodeBlockResult(1, True, 0, 0.1, "", "")],
        )
        self.assertFalse(ExecutionOutcome.from_report(report).repairable)

    def test_not_repairable_when_only_environmental(self) -> None:
        env = "httpx.ConnectError: name not known\n"
        report = ExecutionReport(
            ran=True,
            reason="executed",
            blocks=[CodeBlockResult(1, False, 1, 0.1, "", env)],
        )
        outcome = ExecutionOutcome.from_report(report, "block-1.py")
        self.assertFalse(outcome.repairable)

    def test_not_repairable_when_did_not_run(self) -> None:
        report = ExecutionReport(ran=False, reason="docker unavailable", blocks=[])
        self.assertFalse(ExecutionOutcome.from_report(report).repairable)


class RepairFeedbackTests(unittest.TestCase):
    def test_feedback_names_the_block_and_keeps_scope(self) -> None:
        report = ExecutionReport(
            ran=True,
            reason="executed",
            blocks=[
                CodeBlockResult(1, False, 1, 0.1, "", _tb("ZeroDivisionError: by zero", body="1/0"))
            ],
        )
        outcome = ExecutionOutcome.from_report(report, "block-1.py")
        feedback = build_repair_feedback(outcome)
        self.assertIn("block 1", feedback)
        self.assertIn("ZeroDivisionError", feedback)
        self.assertIn("COMPLETE answer", feedback)  # mirrors existing retry tone


class HostExecutorEndToEndTest(unittest.TestCase):
    """Exercises the host executor + classifier against a real stdlib venv."""

    def test_real_run_classifies_user_fault(self) -> None:
        # A pool whose python_for_specs returns the current interpreter, so no
        # venv build is needed to prove the executor + classifier path.
        class _StdlibPool:
            def python_for_specs(self, specs, *, wait=True):
                return Path(sys.executable)

        executor = host_venv_executor_factory(_StdlibPool())
        report = executor(["print('ok')\n1/0\n"], [])
        self.assertTrue(report.ran)
        self.assertEqual(len(report.blocks), 1)
        self.assertFalse(report.blocks[0].ok)
        outcome = ExecutionOutcome.from_report(report, "block.py")
        # A ZeroDivisionError in the generated block file is a user-code fault.
        self.assertIs(outcome.classifications[0].classification, TracebackClass.USER_CODE_FAULT)

    def test_real_run_passing_block(self) -> None:
        class _StdlibPool:
            def python_for_specs(self, specs, *, wait=True):
                return Path(sys.executable)

        executor = host_venv_executor_factory(_StdlibPool())
        report = executor(["print('hello from generated code')\n"], [])
        self.assertTrue(report.all_passed)


if __name__ == "__main__":
    unittest.main()
