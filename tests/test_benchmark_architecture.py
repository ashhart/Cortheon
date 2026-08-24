"""Architecture guards for the cognitive_benchmark decomposition.

Round 5 moved the 4,880-line god file into the repository-only
``cortheon.benchmark_core`` subpackage behind a stable facade. These checks
pin the split's structural invariants: size budgets, single-owner
definitions, the re-export surface, and the facade-level monkeypatch
semantics that tests depend on.
"""

import ast
import sys
from argparse import Namespace
from pathlib import Path
from unittest import mock

import cortheon.benchmark_core.runner_local as runner_local
from cortheon import cognitive_benchmark as facade
from cortheon.benchmark_core import cli, fixtures_research

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "src" / "cortheon" / "benchmark_core"
FACADE = ROOT / "src" / "cortheon" / "cognitive_benchmark.py"


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_facade_is_a_small_stable_surface() -> None:
    assert len(FACADE.read_text(encoding="utf-8").splitlines()) <= 250
    # Every facade name must be a re-export: the facade owns no definitions.
    assert not _top_level_definitions(FACADE) - {"__all__"}


def test_core_modules_respect_size_and_single_ownership() -> None:
    modules = sorted(CORE.glob("*.py"))
    assert len(modules) >= 15
    owners: dict[str, str] = {}
    for path in modules:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path
        for name in _top_level_definitions(path):
            assert name not in owners, (name, owners[name], path.name)
            owners[name] = path.name


def test_facade_reexports_the_core_implementation() -> None:
    for name in facade.__all__:
        assert hasattr(facade, name), name
    assert facade.run_job is runner_local.run_job
    assert facade.main is cli.main
    assert facade.discover_research_cases is fixtures_research.discover_research_cases


def test_benchmark_core_stays_repository_only() -> None:
    assert "benchmark_core" not in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "benchmark_core" not in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")


def test_module_object_patches_propagate_through_shared_modules() -> None:
    # The facade and the implementation import the same cached module
    # objects, so patching attributes on them (subprocess.run, urlopen)
    # reaches the implementation without indirection.
    assert facade.subprocess is sys.modules["subprocess"]
    import cortheon.benchmark_core.execution_provenance as execution_provenance

    assert facade.subprocess is execution_provenance.subprocess
    import cortheon.benchmark_core.health as health

    assert facade.urllib.request is health.urllib.request


def test_facade_level_run_job_patch_drives_the_retry_path() -> None:
    failed = facade.RunResult(
        case_id="case-arch",
        repeat=0,
        condition="baseline",
        expected=True,
        final_text="",
        delivered=False,
        correct=False,
        latency_seconds=0.0,
        tokens=0,
        tool_calls=0,
        tool_errors=0,
        timed_out=False,
        process_error="opencode returned no assistant answer",
        expected_verdict="allow",
        failure_owner="candidate",
    )

    probe = mock.Mock(side_effect=[ValueError("down"), None])
    with mock.patch("cortheon.cognitive_benchmark.run_job") as run:
        out = facade._retry_after_infrastructure_death(
            Namespace(base_url="", api_key="", model_id="m"),
            mock.Mock(case_id="case-arch"),
            0,
            "baseline",
            failed,
            probe=probe,
            sleep=lambda _seconds: None,
        )
    assert out is run.return_value
    assert run.call_count == 1
    assert probe.call_count == 2


def test_facade_level_pypi_patch_drives_research_discovery() -> None:
    with mock.patch("cortheon.cognitive_benchmark._latest_pypi_release", return_value="9.9.9"):
        cases = facade.discover_research_cases(count=2, seed=7)
    assert len(cases) == 2
    assert all(case.expected == "9.9.9" for case in cases)


def test_health_probes_resolve_through_the_facade() -> None:
    with (
        mock.patch(
            "cortheon.cognitive_benchmark._model_endpoint_health",
            side_effect=ValueError("down"),
        ) as probe,
        mock.patch("cortheon.cognitive_benchmark.run_job", side_effect=ValueError("unused")) as run,
    ):
        result = facade.RunResult(
            case_id="c",
            repeat=0,
            condition="baseline",
            expected=True,
            final_text="",
            delivered=False,
            correct=False,
            latency_seconds=0.0,
            tokens=0,
            tool_calls=0,
            tool_errors=0,
            timed_out=False,
            process_error="opencode exited 137",
        )
        out = facade._retry_after_infrastructure_death(
            Namespace(base_url="", api_key="", model_id="m"),
            mock.Mock(case_id="c"),
            0,
            "baseline",
            result,
            sleep=lambda _seconds: None,
            recovery_attempts=1,
        )
    assert out is result
    assert probe.call_count == 2
    assert run.call_count == 0
