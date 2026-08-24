import subprocess
import unittest
from pathlib import Path
from unittest import mock

from cortheon.sandbox import (
    CodeBlockResult,
    ExecutionReport,
    docker_install_command,
    docker_run_command,
    execute_answer_code,
    run_sandboxed_install_import_test,
)


class SandboxCommandTests(unittest.TestCase):
    def test_install_command_keeps_network_but_caps_resources(self) -> None:
        command = docker_install_command("python:3.12-slim", Path("/scratch/pkgs"), "rich==15.0.0")

        self.assertIn("--rm", command)
        self.assertIn("--pids-limit", command)
        self.assertIn("--memory", command)
        self.assertNotIn("--network", command)
        self.assertIn("/scratch/pkgs:/pkgs", command)
        self.assertIn("--target", command)
        self.assertIn("rich==15.0.0", command)

    def test_run_command_disables_network_and_mounts_packages_read_only(self) -> None:
        command = docker_run_command(
            "python:3.12-slim",
            Path("/scratch/pkgs"),
            Path("/scratch/work"),
            ["python", "/work/example.py"],
            network="none",
        )

        network_index = command.index("--network")
        self.assertEqual(command[network_index + 1], "none")
        self.assertIn("/scratch/pkgs:/pkgs:ro", command)
        self.assertIn("/scratch/work:/work", command)
        self.assertIn("-w", command)
        self.assertIn("--pids-limit", command)
        self.assertEqual(command[-2:], ["python", "/work/example.py"])


class SandboxUnavailableTests(unittest.TestCase):
    def test_missing_docker_fails_honestly_without_host_fallback(self) -> None:
        with mock.patch("cortheon.sandbox.shutil.which", return_value=None):
            result, evidence = run_sandboxed_install_import_test("rich", "15.0.0")

        self.assertFalse(result.install_ran)
        self.assertEqual(result.source, "docker_sandbox")
        self.assertEqual(evidence[0].support.value, "failed")
        self.assertIn("Docker is unavailable", evidence[0].claim)
        self.assertIn("No fallback to host execution", evidence[0].claim)


class FakeDockerRunner:
    """Simulates docker subprocess calls: install ok, import ok, one passing
    and one failing example."""

    def __init__(self, import_times_out: bool = False) -> None:
        self.import_times_out = import_times_out
        self.calls: list[list[str]] = []
        self.example_runs = 0

    def __call__(self, command, capture_output=True, text=True, timeout=None):
        self.calls.append(list(command))
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "pip" in command:
            return subprocess.CompletedProcess(command, 0, "installed", "")
        if "-c" in command:
            if self.import_times_out:
                raise subprocess.TimeoutExpired(command, timeout or 1)
            return subprocess.CompletedProcess(command, 0, "import-ok", "")
        if "/work/example.py" in command:
            self.example_runs += 1
            if self.example_runs == 1:
                return subprocess.CompletedProcess(command, 0, "hello", "")
            return subprocess.CompletedProcess(command, 3, "", "boom")
        raise AssertionError(f"unexpected docker command: {command}")


class SandboxFlowTests(unittest.TestCase):
    def test_full_flow_records_sandboxed_results_and_evidence(self) -> None:
        runner = FakeDockerRunner()
        with mock.patch("cortheon.sandbox.subprocess.run", side_effect=runner):
            result, evidence = run_sandboxed_install_import_test(
                "rich",
                "15.0.0",
                examples=["print('one')", "print('two')"],
            )

        self.assertTrue(result.install_ok)
        self.assertTrue(result.import_ok)
        self.assertEqual(result.source, "docker_sandbox")
        self.assertEqual(len(result.example_results), 2)
        self.assertTrue(result.example_results[0].ok)
        self.assertFalse(result.example_results[1].ok)
        self.assertEqual(evidence[0].support.value, "verified")
        self.assertIn("Docker sandbox", evidence[0].claim)
        self.assertEqual(evidence[1].support.value, "failed")
        self.assertIn("network=none", evidence[1].claim)
        # The import phase must run with the network disabled.
        import_call = next(call for call in runner.calls if "-c" in call)
        self.assertEqual(import_call[import_call.index("--network") + 1], "none")

    def test_import_timeout_kills_container_and_fails(self) -> None:
        runner = FakeDockerRunner(import_times_out=True)
        with mock.patch("cortheon.sandbox.subprocess.run", side_effect=runner):
            result, evidence = run_sandboxed_install_import_test("rich", "15.0.0")

        self.assertTrue(result.install_ok)
        self.assertFalse(result.import_ok)
        self.assertEqual(evidence[0].support.value, "failed")
        rm_calls = [call for call in runner.calls if call[:3] == ["docker", "rm", "-f"]]
        self.assertEqual(len(rm_calls), 1)
        self.assertTrue(rm_calls[0][3].startswith("cortheon-"))


class ExecuteAnswerCodeTests(unittest.TestCase):
    """The execution rung's offline-testable contracts.

    The full path needs Docker; these cover the two preconditions (no blocks,
    no docker) that must report honestly rather than fake a result, plus the
    report's all_passed semantics.
    """

    def test_no_blocks_reports_not_ran(self) -> None:
        report = execute_answer_code([], ["httpx"])
        self.assertFalse(report.ran)
        self.assertEqual(report.reason, "no code blocks")
        self.assertFalse(report.all_passed)

    def test_docker_unavailable_reported_honestly(self) -> None:
        with mock.patch("cortheon.sandbox.docker_available", return_value=False):
            report = execute_answer_code(["import httpx\nhttpx.get('x')"], ["httpx"])
        self.assertFalse(report.ran)
        self.assertEqual(report.reason, "docker unavailable")
        # Honest: it did not fake a pass or a fail. all_passed is False because
        # nothing ran, which the proxy turns into a 'skipped' disclosure.
        self.assertFalse(report.all_passed)

    def test_all_passed_true_only_when_ran_and_every_block_ok(self) -> None:
        report = ExecutionReport(
            ran=True,
            reason="executed",
            blocks=[
                CodeBlockResult(1, True, 0, 0.1, "", ""),
                CodeBlockResult(2, True, 0, 0.1, "", ""),
            ],
        )
        self.assertTrue(report.all_passed)
        report.blocks[1].ok = False
        self.assertFalse(report.all_passed)


if __name__ == "__main__":
    unittest.main()
