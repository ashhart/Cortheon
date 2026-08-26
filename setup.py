"""Build only the deployable Cortheon runtime.

The repository contains benchmark and experimental modules, but an installed
MCP/plugin runtime should not carry them. Keeping the allowlist here makes the
shipping boundary explicit and independently testable.
"""

import shutil
from pathlib import Path
from typing import Any, cast

from setuptools import setup
from setuptools.command.build import build
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist

from build_support.lean_commands import LeanWheel, minify_json, repack_sdist
from build_support.lean_source import (
    compact_host_adapter,
    compact_host_source,
    compact_python_module,
    drop_typescript_declarations,
)
from build_support.lean_stamp import pin_source_date_epoch
from build_support.lean_workspace import open_workspace
from build_support.opencode_bundle import bundle_opencode_plugin
from build_support.pi_bundle import bundle_pi_extension, verify_pi_sources

# Fixed before any command runs: bdist_wheel reads SOURCE_DATE_EPOCH while it
# writes each member, so it has to be settled by import time.
SOURCE_DATE_EPOCH = pin_source_date_epoch()
PROJECT_ROOT = Path(__file__).parent
# setuptools removes os.link while a release tree is built, so distutils
# copies each file instead of hard-linking it to the repository source this
# build then rewrites in place. That helper is not part of the published
# command surface, so it is reached through a reference typed as the dynamic
# object it is -- the same way build_support/lean_compress.py reaches
# zipfile's compressor seam.
_SDIST: Any = sdist

RUNTIME_MODULES = frozenset(
    {
        "__init__",
        "cognitive_cli",
        "cognitive_graph",
        "cognitive_hooks",
        "cognitive_http",
        "cognitive_install",
        "cognitive_mcp",
        "cognitive_program",
        "cognitive_protocol",
        "cognitive_repair",
        "cognitive_runtime",
        "omp_host",
        "sanitize",
    }
)
COGNITIVE_CORE_MODULES = frozenset(
    {
        "__init__",
        "adaptive_stopping",
        "aggregate_alignment",
        "alignment",
        "claim_verification",
        "claims",
        "diffs",
        "frontier_policy",
        "models",
        "plan_joins",
        "profiles",
        "receipts",
        "requirements",
        "research_gaps",
        "runtime",
        "runtime_brief",
        "runtime_completion",
        "runtime_context",
        "runtime_discovery",
        "runtime_failed_verification",
        "runtime_frontier",
        "runtime_hypotheses",
        "runtime_lifecycle",
        "runtime_observations",
        "runtime_recommendation",
        "runtime_request_flow",
        "runtime_requests",
        "runtime_state",
        "runtime_verification",
        "semantic_graph",
        "semantic_join",
        "tasks",
        "text",
        "uncertainty_visibility",
    }
)
COGNITIVE_MCP_CORE_MODULES = frozenset(
    {"__init__", "arguments", "compat", "protocol", "server", "stdio", "tools"}
)
OMP_CORE_MODULES = frozenset({"__init__", "web"})
COGNITIVE_CLI_CORE_MODULES = frozenset(
    {
        "__init__",
        "conformance",
        "diagnostics",
        "dispatch",
        "operations",
        "parser",
    }
)
COGNITIVE_INSTALL_CORE_MODULES = frozenset(
    {"__init__", "config", "hosts", "lifecycle", "model", "omp"}
)
COGNITIVE_HOOKS_CORE_MODULES = frozenset(
    {
        "__init__",
        "automatic",
        "host_tools",
        "lifecycle",
        "observations",
        "patch_loop",
        "receipts",
        "registration",
        "responses",
        "state",
        "tracker",
        "tracker_base",
    }
)
# Adapter implementation modules shipped alongside the facade. Globbing the
# directory keeps the sdist allowlist complete without hand-maintaining names.
_ADAPTER_CORE_SOURCES = sorted((PROJECT_ROOT / "src" / "cortheon" / "opencode_core").glob("*.js"))
_PI_CORE_SOURCES = sorted((PROJECT_ROOT / "src" / "cortheon" / "pi_core").glob("*.ts"))
# The Codex plugin ships whole, minus the two files an installed plugin never
# runs: the console-script shim the wheel's entry point replaces, and the
# non-Codex agent profile.
PACKAGE_ROOT = "src/cortheon/"
CODEX_PLUGIN_ROOT = "src/cortheon/codex_plugins/cortheon/"
CODEX_PLUGIN_EXCLUDED = (
    "src/cortheon/codex_plugins/cortheon/scripts/cortheon-mcp",
    "src/cortheon/codex_plugins/cortheon/skills/cortheon-runtime/agents/openai.yaml",
)
RUNTIME_SOURCE_FILES = frozenset(
    {
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "build_backend.py",
        "src/cortheon/opencode_plugin.js",
        *(f"src/cortheon/opencode_core/{path.name}" for path in _ADAPTER_CORE_SOURCES),
        "src/cortheon/pi_extension.ts",
        *(f"src/cortheon/pi_core/{path.name}" for path in _PI_CORE_SOURCES),
        "src/cortheon/omp_skill/cortheon-runtime/SKILL.md",
        *(
            f"build_support/{path.name}"
            for path in sorted((PROJECT_ROOT / "build_support").glob("*.py"))
        ),
        *(f"src/cortheon/{module}.py" for module in RUNTIME_MODULES),
        *(f"src/cortheon/cognitive_core/{module}.py" for module in COGNITIVE_CORE_MODULES),
        *(f"src/cortheon/cognitive_cli_core/{module}.py" for module in COGNITIVE_CLI_CORE_MODULES),
        *(
            f"src/cortheon/cognitive_install_core/{module}.py"
            for module in COGNITIVE_INSTALL_CORE_MODULES
        ),
        *(
            f"src/cortheon/cognitive_hooks_core/{module}.py"
            for module in COGNITIVE_HOOKS_CORE_MODULES
        ),
        *(f"src/cortheon/cognitive_mcp_core/{m}.py" for m in COGNITIVE_MCP_CORE_MODULES),
        *(f"src/cortheon/omp_core/{module}.py" for module in OMP_CORE_MODULES),
    }
)


_WORKSPACE: Path | None = None


def _workspace() -> Path:
    """Return the staging directory shared by this invocation's commands."""

    # One per execution of this file, which is one per build: every PEP 517
    # hook re-executes setup.py, and so does every setup.py command line. So
    # ``build``, ``bdist_wheel``, ``egg_info``, and ``sdist`` within one
    # invocation agree on where they are working, and two invocations in the
    # same checkout never do.
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = open_workspace(PROJECT_ROOT)
    return _WORKSPACE


class LeanEggInfo(egg_info):
    """Write the generated metadata where only this build can see it."""

    # ``src/cortheon.egg-info`` is one directory shared by every build in the
    # checkout, and its files are copied verbatim into each wheel's dist-info.
    # Two builds writing it at once agree on the content but not on the
    # moment, so one can copy a file the other has truncated and not yet
    # refilled. A per-build location removes the overlap entirely.

    def finalize_options(self) -> None:
        # Only a default: dist_info and editable_wheel set egg_base
        # themselves, and an explicit --egg-base still wins.
        if self.egg_base is None:
            self.egg_base = str(_workspace())
        super().finalize_options()


class LeanBuildPy(build_py):
    """Exclude repository-only experiments from wheels and installations."""

    def find_package_modules(
        self,
        package: str,
        package_dir: str,
    ) -> list[tuple[str, str, str]]:
        modules = super().find_package_modules(package, package_dir)
        if package == "cortheon":
            allowed = RUNTIME_MODULES
        elif package == "cortheon.cognitive_core":
            allowed = COGNITIVE_CORE_MODULES
        elif package == "cortheon.cognitive_cli_core":
            allowed = COGNITIVE_CLI_CORE_MODULES
        elif package == "cortheon.cognitive_install_core":
            allowed = COGNITIVE_INSTALL_CORE_MODULES
        elif package == "cortheon.cognitive_hooks_core":
            allowed = COGNITIVE_HOOKS_CORE_MODULES
        elif package == "cortheon.cognitive_mcp_core":
            allowed = COGNITIVE_MCP_CORE_MODULES
        elif package == "cortheon.omp_core":
            allowed = OMP_CORE_MODULES
        else:
            return []
        return [module for module in modules if module[1] in allowed]


class LeanBuild(build):
    """Isolate this invocation's build output, then finish the lean packaging."""

    # ``build_lib``, ``build_temp``, and the ``bdist_base`` that bdist_wheel
    # stages the wheel in are all derived from ``build_base``, so pointing
    # that one option at this build's workspace moves every intermediate
    # inside it. Bundling happens only here: the copied facade is replaced by
    # the deterministic bundle, ``pi_core`` is removed from this output only,
    # and artifacts are compacted.

    def finalize_options(self) -> None:
        self.build_base = str(_workspace())
        super().finalize_options()

    def run(self) -> None:
        super().run()
        package_output = Path(self.build_lib) / "cortheon"
        bundle_pi_extension(
            package_output / "pi_extension.ts",
            package_output / "pi_core",
            package_output / "pi_extension.ts",
        )
        shutil.rmtree(package_output / "pi_core")
        bundle_opencode_plugin(
            package_output / "opencode_plugin.js",
            package_output / "opencode_core",
            package_output / "opencode_plugin.js",
        )
        shutil.rmtree(package_output / "opencode_core")
        drop_typescript_declarations(package_output / "pi_extension.ts")
        for path in package_output.rglob("*.json"):
            minify_json(path)
        for path in package_output.rglob("*.py"):
            relative = path.relative_to(package_output)
            compact_python_module(
                path,
                strip_private_metadata=True,
                private_module=(
                    any(part.endswith("_core") for part in relative.parts[:-1])
                    or ("hooks" in relative.parts and relative.name.startswith("hook_"))
                ),
            )
        for path in package_output.glob("*_core/__init__.py"):
            path.unlink()
        for suffix in ("*.js", "*.ts"):
            for path in package_output.rglob(suffix):
                compact_host_adapter(path)
        for relative in CODEX_PLUGIN_EXCLUDED:
            package_output.joinpath(relative.removeprefix(PACKAGE_ROOT)).unlink(missing_ok=True)


class LeanSdist(sdist):
    """Apply the runtime allowlist to source archives as well as wheels."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        filtered = [
            path
            for path in files
            if (path in RUNTIME_SOURCE_FILES or path.startswith(CODEX_PLUGIN_ROOT))
            and path not in CODEX_PLUGIN_EXCLUDED
        ]
        super().make_release_tree(base_dir, filtered)
        release_root = Path(base_dir)
        for path in release_root.joinpath("src", "cortheon").rglob("*.py"):
            relative = path.relative_to(release_root / "src" / "cortheon")
            compact_python_module(
                path,
                strip_private_metadata=True,
                private_module=(
                    any(part.endswith("_core") for part in relative.parts[:-1])
                    or ("hooks" in relative.parts and relative.name.startswith("hook_"))
                ),
            )
        for path in release_root.joinpath("src", "cortheon").rglob("*.js"):
            compact_host_source(path)
        pi_root = release_root / "src" / "cortheon"
        verify_pi_sources(pi_root / "pi_extension.ts", pi_root / "pi_core")
        compact_host_source(pi_root / "pi_extension.ts")
        for path in pi_root.joinpath("pi_core").glob("*.ts"):
            compact_host_source(path)
        # The build-time modules ship so a source build reproduces, and they
        # are executed rather than read, so the same formatting-only pass
        # applies to them. setuptools removes ``os.link`` for the whole
        # release-tree build, so these are copies and the repository sources
        # they came from are untouched.
        for path in (
            release_root / "setup.py",
            release_root / "build_backend.py",
            *sorted(release_root.joinpath("build_support").glob("*.py")),
        ):
            compact_python_module(
                path,
                strip_private_metadata=True,
                private_module=True,
                strip_all_callable_metadata=True,
            )
        # The packers accept only the pinned raw checkout or this exact
        # deterministic compacted source closure when rebuilding a wheel.
        self._ship_generated_metadata(release_root)

    def _ship_generated_metadata(self, release_root: Path) -> None:
        """Place this build's own egg-info in the archive and describe it."""

        # egg_info writes to the workspace now, so its output is copied in
        # from there rather than picked up from the checkout. The SOURCES.txt
        # it generated lists the repository, including the tests and
        # repository-only modules this archive deliberately drops, so it is
        # rewritten from the finished release tree: the shipped manifest
        # describes the artifact that carries it. Nothing reads it while
        # building from the sdist, because egg_info regenerates SOURCES.txt.
        generated = Path(cast(egg_info, self.get_finalized_command("egg_info")).egg_info)
        shipped = release_root / "src" / generated.name
        shipped.mkdir(parents=True, exist_ok=True)
        for path in sorted(generated.iterdir()):
            if path.is_file():
                shutil.copy2(path, shipped / path.name)
        listed = sorted(
            path.relative_to(release_root).as_posix()
            for path in release_root.rglob("*")
            if path.is_file()
        )
        shipped.joinpath("SOURCES.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")

    def make_distribution(self) -> None:
        """Stage the release tree inside this build's own workspace."""

        # distutils builds ``cortheon-0.1.0/`` beside setup.py and deletes it
        # afterwards, so two builds in one checkout share that one name: the
        # first to finish removes the tree the second is still archiving.
        # Rooting the archive at a workspace directory instead removes the
        # collision without changing what it contains, because the release
        # name stays the directory every member sits under.
        full_name = self.distribution.get_fullname()
        staging = _workspace() / "sdist"
        staging.mkdir(parents=True, exist_ok=True)
        release_tree = staging / full_name
        # Symmetric with LeanWheel: each finished archive is rewritten once
        # it is complete, without touching what it contains.
        with _SDIST._remove_os_link():
            self.make_release_tree(str(release_tree), self.filelist.files)
        formats = list(self.formats)
        if "tar" in formats:  # An uncompressed tar would overwrite its own inputs.
            formats.append(formats.pop(formats.index("tar")))
        self.archive_files = []
        for archive_format in formats:
            archive = self.make_archive(
                str(Path(self.dist_dir) / full_name),
                archive_format,
                root_dir=str(staging),
                base_dir=full_name,
                owner=self.owner,
                group=self.group,
            )
            if archive.endswith(".tar.gz"):
                repack_sdist(Path(archive), SOURCE_DATE_EPOCH)
            self.archive_files.append(archive)
            self.distribution.dist_files.append(("sdist", "", archive))
        if not self.keep_temp:
            shutil.rmtree(release_tree)


setup(
    cmdclass={
        "build": LeanBuild,
        "build_py": LeanBuildPy,
        "egg_info": LeanEggInfo,
        "sdist": LeanSdist,
        "bdist_wheel": LeanWheel,
    }
)
