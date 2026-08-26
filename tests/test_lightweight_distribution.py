import ast
import json
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).parents[1]
# Uniform with tests/test_wheel_equivalence.py and
# tests/test_opencode_architecture.py; that module carries the
# measurement and the justification for both numbers.
WHEEL_CAP = 300_000
SDIST_CAP = 300_000
SHIPPED_PYTHON_MODULES = {
    "cortheon/__init__.py",
    "cortheon/cognitive_cli.py",
    "cortheon/cognitive_graph.py",
    "cortheon/cognitive_hooks.py",
    "cortheon/cognitive_http.py",
    "cortheon/cognitive_install.py",
    "cortheon/cognitive_mcp.py",
    "cortheon/cognitive_program.py",
    "cortheon/cognitive_protocol.py",
    "cortheon/cognitive_repair.py",
    "cortheon/cognitive_runtime.py",
    "cortheon/omp_host.py",
    "cortheon/sanitize.py",
    *(
        f"cortheon/cognitive_core/{path.name}"
        for path in sorted((ROOT / "src/cortheon/cognitive_core").glob("*.py"))
        if path.name != "__init__.py"
    ),
    *(
        f"cortheon/cognitive_cli_core/{path.name}"
        for path in sorted((ROOT / "src/cortheon/cognitive_cli_core").glob("*.py"))
        if path.name != "__init__.py"
    ),
    *(
        f"cortheon/cognitive_install_core/{path.name}"
        for path in sorted((ROOT / "src/cortheon/cognitive_install_core").glob("*.py"))
        if path.name != "__init__.py"
    ),
    *(
        f"cortheon/cognitive_hooks_core/{path.name}"
        for path in sorted((ROOT / "src/cortheon/cognitive_hooks_core").glob("*.py"))
        if path.name != "__init__.py"
    ),
    *(
        f"cortheon/cognitive_mcp_core/{path.name}"
        for path in sorted((ROOT / "src/cortheon/cognitive_mcp_core").glob("*.py"))
        if path.name != "__init__.py"
    ),
}
# Codex copies the plugin directory into its own cache and executes the
# facade from there, so every module the facade imports has to be installed
# beside it. Derived from the source directory: a new hook module is shipped
# or the wheel assertion fails.
SHIPPED_HOOK_MODULES = {
    f"cortheon/codex_plugins/cortheon/hooks/{path.name}"
    for path in sorted((ROOT / "src/cortheon/codex_plugins/cortheon/hooks").glob("*.py"))
}
SHIPPED_ADAPTER_MODULES = {
    "cortheon/opencode_plugin.js",
}
SHIPPED_ADAPTER_SOURCE_MODULES = {
    "cortheon/opencode_plugin.js",
    *(
        f"cortheon/opencode_core/{path.name}"
        for path in sorted((ROOT / "src/cortheon/opencode_core").glob("*.js"))
    ),
}


def test_product_has_no_runtime_dependencies_and_only_two_commands() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert configuration["project"]["dependencies"] == []
    assert configuration["project"]["scripts"] == {
        "cortheon": "cortheon.cognitive_cli:main",
        "cortheon-mcp": "cortheon.cognitive_mcp:main",
    }


def test_cli_import_keeps_operator_only_modules_lazy() -> None:
    script = (
        "import json,sys; import cortheon.cognitive_cli; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name in {'pathlib','subprocess','cortheon.cognitive_install'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert json.loads(completed.stdout) == []


def test_wheel_contains_only_the_deployable_runtime(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    wheel = next(wheel_dir.glob("cortheon-*.whl"))

    assert wheel.stat().st_size <= WHEEL_CAP
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    shipped_python = {
        member for member in members if member.startswith("cortheon/") and member.endswith(".py")
    }
    assert len(SHIPPED_HOOK_MODULES) == 6
    assert shipped_python == {*SHIPPED_PYTHON_MODULES, *SHIPPED_HOOK_MODULES}
    shipped_js = {
        member for member in members if member.startswith("cortheon/") and member.endswith(".js")
    }
    assert shipped_js == SHIPPED_ADAPTER_MODULES
    assert "cortheon/cognitive_runtime.py" in shipped_python
    assert "cortheon/cognitive_core/runtime.py" in shipped_python
    assert "cortheon/cognitive_core/models.py" in shipped_python
    assert not any(
        member.endswith(
            (
                "/benchmark.py",
                "/parity.py",
                "/cognitive_benchmark.py",
            )
        )
        for member in shipped_python
    )
    assert "cortheon/benchmark.py" not in members
    assert not any("/benchmark_core/" in member for member in members)
    # The round-8 parity benchmark split stays repository-only with its facade.
    assert not any("/parity_benchmark_core/" in member for member in members)
    # As does the release-contract split that replaced the parity.py god file.
    assert not any("/parity_gates/" in member for member in members)
    # And the pack-issuer split; it holds evaluator secret handling.
    assert not any("/parity_pack_core/" in member for member in members)

    entry_points = next(name for name in members if name.endswith("entry_points.txt"))
    with zipfile.ZipFile(wheel) as archive:
        scripts = {
            line.split("=", 1)[0].strip()
            for line in archive.read(entry_points).decode("utf-8").splitlines()
            if line.startswith(("cortheon", "cortheon-mcp")) and "=" in line
        }
    assert scripts == {"cortheon", "cortheon-mcp"}

    install_dir = tmp_path / "install"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(install_dir),
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json;"
                "from cortheon.cognitive_protocol import protocol_capabilities;"
                "from cortheon.cognitive_runtime import CognitiveRuntime;"
                "result=CognitiveRuntime().start('Inspect src/app.py for the bug',effort='quick');"
                "print(json.dumps({'capabilities':protocol_capabilities(),"
                "'graph':result['context']['cognitive_graph'],"
                "'program':result['cognition']['program'],"
                "'selection':result['cognition']['evidence_target']['selection']}))"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(install_dir)},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    installed = json.loads(completed.stdout)
    assert installed["capabilities"]["storage"] == "memory_only"
    assert installed["graph"]["digest"].startswith("cg_")
    assert installed["program"]["program_id"].startswith("cp_")
    assert installed["selection"]["action_id"] == "req1"


def test_source_archive_uses_the_same_runtime_allowlist(tmp_path: Path) -> None:
    distribution_dir = tmp_path / "sdist"
    distribution_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "sdist",
            "--dist-dir",
            str(distribution_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    archive_path = next(distribution_dir.glob("cortheon-*.tar.gz"))

    assert archive_path.stat().st_size <= SDIST_CAP
    with tarfile.open(archive_path, "r:gz") as archive:
        entries = archive.getmembers()
        members = {member.name for member in entries}

        def read(relative: str) -> str:
            handle = archive.extractfile(f"cortheon-0.1.0/{relative}")
            assert handle is not None, relative
            return handle.read().decode("utf-8")

        # The build-time modules ship compacted like the runtime ones.
        # Compaction is idempotent, so a compacted module equals its own
        # round trip; an un-compacted one still carries comments and layout.
        for build_file in (
            "setup.py",
            "build_backend.py",
            "build_support/lean_source.py",
            "build_support/pi_bundle.py",
        ):
            text = read(build_file)
            assert text == ast.unparse(ast.parse(text)) + "\n", f"{build_file} ships un-compacted"

        # egg_info derives SOURCES.txt from the repository, so the manifest is
        # rewritten to describe this archive instead of advertising the tests
        # and repository-only modules the allowlist drops.
        listed = read("src/cortheon.egg-info/SOURCES.txt").split()
    assert set(listed) == {member.name.split("/", 1)[1] for member in entries if member.isfile()}
    assert not any(member.endswith("/cortheon/benchmark.py") for member in members)
    assert not any("/benchmark_core/" in member for member in members)
    assert not any("/parity_benchmark_core/" in member for member in members)
    assert not any("/parity_gates/" in member for member in members)
    assert not any("/parity_pack_core/" in member for member in members)
    assert not any("/tests/" in member for member in members)
    sdist_js = {
        member[len("cortheon-0.1.0/src/") :]
        for member in members
        if member.startswith("cortheon-0.1.0/src/cortheon/") and member.endswith(".js")
    }
    assert sdist_js == SHIPPED_ADAPTER_SOURCE_MODULES


# setuptools writes these into the release tree after the template has already
# chosen what to copy in, so no MANIFEST.in directive can account for them.
GENERATED_SDIST_FILES = frozenset({"PKG-INFO", "setup.cfg"})
GENERATED_SDIST_PREFIX = "src/cortheon.egg-info/"
MANIFEST_DIRECTIVES = frozenset({"include", "recursive-include", "exclude", "global-exclude"})


def _declared_by_manifest() -> set[str]:
    """Resolve MANIFEST.in against this checkout, directive by directive.

    Only the four directives the template actually uses are understood: an
    unrecognised one fails the test rather than being quietly ignored, which
    is what would let the declaration drift away from the archive again.
    """

    declared: set[str] = set()
    for raw in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        directive, _, remainder = line.partition(" ")
        arguments = remainder.split()
        assert directive in MANIFEST_DIRECTIVES, f"unhandled directive {directive!r}"
        if directive == "include":
            for argument in arguments:
                assert (ROOT / argument).is_file(), f"MANIFEST.in includes missing {argument}"
                declared.add(argument)
        elif directive == "recursive-include":
            directory, patterns = arguments[0], arguments[1:]
            for pattern in patterns:
                declared.update(
                    path.relative_to(ROOT).as_posix()
                    for path in (ROOT / directory).rglob(pattern)
                    if path.is_file()
                )
        elif directive == "exclude":
            for argument in arguments:
                assert (ROOT / argument).is_file(), f"MANIFEST.in excludes missing {argument}"
                declared.discard(argument)
        else:
            declared = {
                name
                for name in declared
                if not any(
                    fnmatch(part, pattern) for part in Path(name).parts for pattern in arguments
                )
            }
    return declared


def test_manifest_declares_exactly_what_a_source_archive_ships(tmp_path: Path) -> None:
    """The template and the allowlist in setup.py have to agree.

    MANIFEST.in decides what setup.py is offered and setup.py decides what
    survives, so a name in one and not the other is a claim the artifact does
    not honour: either a declared file never ships, or a shipped file was
    never declared."""

    distribution_dir = tmp_path / "sdist"
    distribution_dir.mkdir()
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(distribution_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    with tarfile.open(next(distribution_dir.glob("cortheon-*.tar.gz"))) as archive:
        shipped = {
            member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()
        }

    generated = {
        name
        for name in shipped
        if name in GENERATED_SDIST_FILES or name.startswith(GENERATED_SDIST_PREFIX)
    }
    assert generated >= GENERATED_SDIST_FILES
    assert shipped - generated == _declared_by_manifest()
