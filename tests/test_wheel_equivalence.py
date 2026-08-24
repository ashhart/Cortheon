"""Installed-wheel equivalence regressions.

Build the real wheel through the packaging pipeline and prove the installed
artifact is behaviorally identical to source mode for the runtime's
dataclasses, that repeated builds over an existing wheel name still get
fully recompressed, and that both wheel and sdist verify against their
RECORD manifests.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from source_checkout_support import copy_source_tree

from build_support.lean_commands import recompress_wheel

ROOT = Path(__file__).parents[1]
# Cap justification, shared by tests/test_lightweight_distribution.py and
# tests/test_opencode_architecture.py so one measurement governs every gate.
# Restoring full public API correctness (docstrings, function annotations,
# and intact imports in shipped modules) put the wheel past the old 200,000
# cap that only stripped-metadata artifacts could meet. Measured on the
# hook-split tree, where cognitive_hooks became a 92-line facade over the
# twelve-module cognitive_hooks_core subpackage: wheel 229,376 bytes, sdist
# 211,564 bytes. The caps sit about 10 KB and 8 KB above those figures,
# which covers build-to-build variance and ordinary runtime growth but not
# a new payload; anything larger is a regression, not a rounding error.
# Re-measured after the round-24 bounded-completion capability (the
# evidence-sufficiency forced answer and the repeated-withhold terminal
# state added the causal_answer module and the budget guards): wheel
# 232,983 bytes, sdist 220,700 bytes. The sdist cap moved to keep its
# ~1.3 KB headroom; the wheel kept its old cap and headroom.
# Re-measured after the source-level slimming pass that restored the
# written contract cap (condensed comments across the pi_core TypeScript
# modules with the pinned hash table refreshed): wheel 233,777 bytes in
# every build, sdist 218,219-218,261 bytes across three builds.
# SDIST_CAP is back to the contracted 220_000, holding ~1.7 KB of
# headroom; the wheel cap and headroom are unchanged.
# Re-measured after the Codex hook split: the wheel ships the complete hook
# modules rather than the facade alone, because Codex copies the
# plugin directory into its cache and runs the facade from there. Adding
# them to the plain level-9 build measured 243,988 bytes, over the cap, so
# both artifacts moved to searched DEFLATE and gzip streams (the same
# standard formats, produced by the build-only compressor in
# build_support/lean_compress.py). Measured across three builds each:
# wheel 237,840 bytes every time, sdist 212,432-212,511 bytes. Neither cap
# moves; the wheel keeps ~2.1 KB of headroom and the sdist ~7.5 KB.
# Re-measured after the build-isolation and reproducibility pass: every
# intermediate moved into a per-build workspace, archive timestamps and
# ownership are pinned through SOURCE_DATE_EPOCH, and the source archive
# gained build_support/lean_stamp.py and build_support/lean_workspace.py.
# Across three builds of each and a wheel rebuilt from the sdist: wheel
# 238,015 bytes and sdist 214,809 bytes, now byte-identical rather than
# merely close, so the recorded figures are exact and not a range. Neither
# cap moves; the wheel keeps ~2.0 KB of headroom and the sdist ~5.2 KB.
# Re-measured after adding Pi web evidence and shipping the split CLI and
# installer packages. Release compaction now omits only private implementation
# metadata; source keeps it, while installed public API metadata remains exact.
# Final installed-Codex runtime ownership, fingerprint checks, clean replacement,
# public-only metadata compaction, and truthful host documentation produced
# byte-identical three-build artifacts: wheel 239,641 bytes and sdist 214,491
# bytes. The caps remain fixed.
# Stage 6 retained the caps after adding reasoning-record binding and
# content-free release records. Local-only annotations are removed from built
# code, SPDX metadata avoids embedding a second license body, and concise
# public facades avoid duplicated export inventories. Settled measurements:
# wheel 239,739 bytes and sdist 219,775 bytes.
WHEEL_CAP = 240_000
SDIST_CAP = 220_000

PROBE = """
import dataclasses, json, sys
from cortheon.cognitive_core import models
from cortheon.cognitive_hooks import HookTurn

report = {}
for name in ("Hypothesis", "Observation", "EvidenceRequest", "Investigation"):
    cls = getattr(models, name)
    report[f"models.{name}"] = {
        "fields": [f.name for f in dataclasses.fields(cls)],
        "defaults": [
            f.name
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
        ],
        "annotations": dict(cls.__annotations__),
    }
report["hooks.HookTurn"] = {
    "fields": [f.name for f in dataclasses.fields(HookTurn)],
    "annotations": dict(HookTurn.__annotations__),
}
turn = HookTurn(host_session_hash="h0")
turn.goal_hash = "g0"
turn.pending_request = {"request_id": "r1"}
report["hook_turn_instance"] = dataclasses.asdict(turn)
hypothesis = models.Hypothesis(
    hypothesis_id="hyp-1",
    statement="s",
    falsification_test="f",
)
report["hypothesis_instance"] = dataclasses.asdict(hypothesis)
print(json.dumps(report, sort_keys=True, default=str))
"""


def _build_wheel(dist_dir: Path) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist_dir),
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return next(dist_dir.glob("cortheon-*.whl"))


def _install(wheel: Path, target: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return target


def _probe(python_path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        env={**os.environ, "PYTHONPATH": str(python_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return completed.stdout


def test_installed_wheel_dataclasses_match_source(tmp_path: Path) -> None:
    """The wheel keeps class annotations, so representative dataclasses
    (HookTurn and the cognitive_core models) construct with the same fields,
    defaults, and values as source mode."""
    wheel = _build_wheel(tmp_path / "wheel")
    install_dir = _install(wheel, tmp_path / "install")
    assert _probe(install_dir) == _probe(ROOT / "src")


API_METADATA_PROBE = """
import inspect, json, sys
import cortheon
import cortheon.cognitive_runtime as runtime
import cortheon.cognitive_hooks as hooks
import cortheon.cognitive_core.models as models

def describe(obj):
    return {
        "doc": inspect.getdoc(obj),
        "signature": None
        if not callable(obj)
        else str(inspect.signature(obj)),
        "annotations": dict(getattr(obj, "__annotations__", {})),
    }

report = {
    "cortheon.__doc__": cortheon.__doc__,
    "runtime_module.__doc__": runtime.__doc__,
    "CognitiveRuntime": describe(runtime.CognitiveRuntime),
    "CognitiveRuntime.start": describe(runtime.CognitiveRuntime.start),
    "HookTurn": describe(hooks.HookTurn),
    "models.Hypothesis": describe(models.Hypothesis),
}
print(json.dumps(report, sort_keys=True, default=str))
"""


def test_installed_wheel_public_api_metadata_matches_source(tmp_path: Path) -> None:
    """Docstrings, signatures, and annotations survive the build transformer:
    help()/__doc__/inspect.signature/__annotations__ on the installed wheel
    are identical to repository source."""
    wheel = _build_wheel(tmp_path / "wheel")
    install_dir = _install(wheel, tmp_path / "install")

    def probe(python_path: Path) -> str:
        completed = subprocess.run(
            [sys.executable, "-c", API_METADATA_PROBE],
            env={**os.environ, "PYTHONPATH": str(python_path)},
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return completed.stdout

    assert probe(install_dir) == probe(ROOT / "src")


def _canonical_level9(wheel: Path, output: Path) -> None:
    """Independently recompress a wheel's members at plain DEFLATE level 9."""
    with (
        zipfile.ZipFile(wheel, "r") as source,
        zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target,
    ):
        for info in source.infolist():
            target.writestr(info, source.read(info.filename), compresslevel=9)


def test_repeated_build_recompresses_replacement(tmp_path: Path) -> None:
    """Rebuilding when the output directory already holds a wheel with the
    same name must still recompress the replacement: the final artifact is
    byte-identical to an independent level-9 rewrite and under the cap."""
    dist_dir = tmp_path / "dist"
    first = _build_wheel(dist_dir)
    name = first.name

    # Replace the good wheel with a worse-compressed one under the same
    # filename, as an older artifact of the same version would leave behind.
    bloated = dist_dir / f"bloated-{name}"
    with (
        zipfile.ZipFile(first, "r") as source,
        zipfile.ZipFile(bloated, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            target.writestr(info, source.read(info.filename), compresslevel=1)
    bloated_size = bloated.stat().st_size
    first_size = first.stat().st_size
    first.unlink()
    bloated.rename(dist_dir / name)
    assert bloated_size > first_size

    rebuilt = _build_wheel(dist_dir)
    assert rebuilt.name == name
    assert rebuilt.stat().st_size <= WHEEL_CAP
    # The replacement must be recompressed, not merely left at bdist
    # defaults: it is smaller than the bloated artifact and byte-identical
    # to an independent rewrite of its own members, which also makes the
    # search deterministic rather than merely repeatable in one process.
    assert rebuilt.stat().st_size < bloated_size
    canonical = dist_dir / "canonical.whl"
    shutil.copy2(rebuilt, canonical)
    recompress_wheel(canonical)
    assert rebuilt.read_bytes() == canonical.read_bytes()
    # And the searched streams are what keep it inside the cap: a plain
    # level-9 rewrite of the very same members is strictly larger.
    level9 = dist_dir / "level9.whl"
    _canonical_level9(rebuilt, level9)
    assert level9.stat().st_size > rebuilt.stat().st_size


def _verify_record(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        record_name = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        rows = {
            row[0]: row
            for row in csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
            if row
        }
        for name in archive.namelist():
            row = rows.get(name)
            assert row is not None, f"{name} missing from RECORD"
            data = archive.read(name)
            if name == record_name:
                continue
            digest, size = row[1], row[2]
            expected = (
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            )
            assert digest == expected, f"{name} hash mismatch"
            assert int(size) == len(data), f"{name} size mismatch"


def test_record_hashes_and_isolated_builds(tmp_path: Path) -> None:
    """PEP 517 wheel and sdist builds from a copied source tree with no .git,
    a wheel built from that sdist, and RECORD verification of both wheels."""
    # Staged by walking the filesystem, not by asking git: a release is built
    # from an extracted archive or a container COPY at least as often as from
    # a clone, and a suite that shells out to git would pass in the one case
    # and never exercise the other.
    checkout = copy_source_tree(ROOT, tmp_path / "checkout")
    assert not (checkout / ".git").exists()

    wheel_dir = tmp_path / "wheels"
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
            str(checkout),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    clean_wheel = next(wheel_dir.glob("cortheon-*.whl"))
    assert clean_wheel.stat().st_size <= WHEEL_CAP
    _verify_record(clean_wheel)

    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    # Invoke the build_sdist hook of the in-tree backend declared in
    # pyproject.toml (build_backend + backend-path), in a fresh process
    # rooted at the clean checkout.
    hook = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import build_backend, sys;print(build_backend.build_sdist(sys.argv[1]))"),
            str(sdist_dir),
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    archive = sdist_dir / hook.stdout.strip().splitlines()[-1]
    assert archive.exists()
    assert archive.stat().st_size <= SDIST_CAP

    from_sdist_dir = tmp_path / "from-sdist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(from_sdist_dir),
            str(archive),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    sdist_wheel = next(from_sdist_dir.glob("cortheon-*.whl"))
    assert sdist_wheel.stat().st_size <= WHEEL_CAP
    _verify_record(sdist_wheel)
