"""What a build has to survive: a shared checkout, no clone, and a clock.

Every check here drives the real packaging pipeline in subprocesses, because
the failures it covers only appear in a whole build: two invocations
trampling one staging directory, residue nobody removes, an artifact whose
bytes record when it was built rather than what it contains, and a source
tree that was never a git checkout.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from source_checkout_support import copy_source_tree

from build_support.lean_stamp import DEFAULT_EPOCH
from build_support.lean_workspace import (
    STALE_AFTER_SECONDS,
    WORKSPACE_PREFIX,
    discard_workspace,
    open_workspace,
)

ROOT = Path(__file__).parents[1]
# Uniform with tests/test_wheel_equivalence.py, which carries the measurement
# and the justification for both numbers.
WHEEL_CAP = 247_200
SDIST_CAP = 227_150
SDIST_HOOK = "import build_backend, sys;print(build_backend.build_sdist(sys.argv[1]))"
# A second fixed instant, unrelated to the build's default, so honouring the
# caller's SOURCE_DATE_EPOCH is distinguishable from ignoring it.
CALLER_EPOCH = 1_600_000_000


@pytest.fixture(scope="module")
def checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A source tree staged the way a release arrives: no .git, no build/.

    Builds run here rather than in the repository so ``build/`` belongs to
    this module alone and residue can be asserted on exactly.
    """

    return copy_source_tree(ROOT, tmp_path_factory.mktemp("staged") / "cortheon")


def _run(
    command: list[str],
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=None if environment is None else {**os.environ, **environment},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if check:
        assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed


def _wheel_command(target: Path, dist_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(dist_dir),
        str(target),
    ]


def _build_wheel(
    target: Path, dist_dir: Path, environment: Mapping[str, str] | None = None
) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _wheel_command(target, dist_dir), target if target.is_dir() else target.parent, environment
    )
    return next(dist_dir.glob("cortheon-*.whl"))


def _build_sdist(
    source: Path, dist_dir: Path, environment: Mapping[str, str] | None = None
) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    completed = _run([sys.executable, "-c", SDIST_HOOK, str(dist_dir)], source, environment)
    return dist_dir / completed.stdout.strip().splitlines()[-1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspaces(root: Path) -> set[str]:
    return {path.name for path in (root / "build").glob(f"{WORKSPACE_PREFIX}*")}


def test_concurrent_builds_in_one_checkout_agree_and_leave_no_residue(
    checkout: Path, tmp_path: Path
) -> None:
    """Four builds at once in one checkout, and nothing left behind.

    Before every intermediate moved into a per-build workspace, these shared
    ``build/lib``, ``build/bdist.<plat>``, the generated egg-info, and the
    ``cortheon-0.1.0/`` release tree: one build deleted what another was
    still reading. Agreeing on every byte is the evidence that no longer
    happens, and the workspace count is the evidence none of it is kept."""

    before = _workspaces(checkout)
    with ThreadPoolExecutor(max_workers=4) as pool:
        running = [
            pool.submit(_build_wheel, checkout, tmp_path / "wheel-a"),
            pool.submit(_build_wheel, checkout, tmp_path / "wheel-b"),
            pool.submit(_build_sdist, checkout, tmp_path / "sdist-a"),
            pool.submit(_build_sdist, checkout, tmp_path / "sdist-b"),
        ]
        wheel_a, wheel_b, sdist_a, sdist_b = (future.result() for future in running)

    assert _digest(wheel_a) == _digest(wheel_b)
    assert _digest(sdist_a) == _digest(sdist_b)
    assert wheel_a.stat().st_size <= WHEEL_CAP
    assert sdist_a.stat().st_size <= SDIST_CAP
    assert _workspaces(checkout) == before
    # The release tree is staged inside a workspace now, so the directory
    # distutils used to build beside setup.py never appears at all.
    assert not (checkout / "cortheon-0.1.0").exists()


def test_repeated_builds_of_one_source_tree_are_byte_identical(
    checkout: Path, tmp_path: Path
) -> None:
    """Three wheels, three source archives, and a wheel built from one of
    those archives: every artifact of a kind has to be the same bytes, or the
    build is recording something other than its own sources."""

    wheels = [_build_wheel(checkout, tmp_path / f"wheel-{index}") for index in range(3)]
    sdists = [_build_sdist(checkout, tmp_path / f"sdist-{index}") for index in range(3)]

    assert len({_digest(wheel) for wheel in wheels}) == 1
    assert len({_digest(sdist) for sdist in sdists}) == 1
    from_sdist = _build_wheel(sdists[0], tmp_path / "from-sdist")
    assert _digest(from_sdist) == _digest(wheels[0])


def test_the_default_epoch_dates_every_member_of_both_artifacts(
    checkout: Path, tmp_path: Path
) -> None:
    """Nothing in either artifact carries the moment it was built."""

    expected = time.gmtime(DEFAULT_EPOCH)[0:6]
    with zipfile.ZipFile(_build_wheel(checkout, tmp_path / "wheel")) as wheel:
        assert {info.date_time for info in wheel.infolist()} == {expected}
    with tarfile.open(_build_sdist(checkout, tmp_path / "sdist")) as sdist:
        members = sdist.getmembers()
    assert {member.mtime for member in members} == {DEFAULT_EPOCH}
    # Ownership records the account that ran the build, so it is pinned too.
    assert {(member.uid, member.gid) for member in members} == {(0, 0)}
    assert {(member.uname, member.gname) for member in members} == {("", "")}
    assert {member.mode for member in members} == {0o644, 0o755}


def test_a_caller_supplied_epoch_wins_and_is_still_reproducible(
    checkout: Path, tmp_path: Path
) -> None:
    """SOURCE_DATE_EPOCH is honoured, not merely tolerated: a release that
    pins its own instant gets it, and gets the same bytes every time."""

    environment = {"SOURCE_DATE_EPOCH": str(CALLER_EPOCH)}
    first = _build_wheel(checkout, tmp_path / "pinned-first", environment)
    default = _digest(_build_wheel(checkout, tmp_path / "default"))
    second = _build_wheel(checkout, tmp_path / "pinned-second", environment)

    assert _digest(first) == _digest(second) != default
    with zipfile.ZipFile(first) as wheel:
        assert {info.date_time for info in wheel.infolist()} == {time.gmtime(CALLER_EPOCH)[0:6]}
    with tarfile.open(_build_sdist(checkout, tmp_path / "pinned-sdist", environment)) as sdist:
        assert {member.mtime for member in sdist.getmembers()} == {CALLER_EPOCH}


def test_an_unreadable_epoch_stops_the_build_rather_than_dating_it_now(
    checkout: Path, tmp_path: Path
) -> None:
    """A malformed SOURCE_DATE_EPOCH is a caller error. Falling back to the
    wall clock would answer it with an artifact that silently is not the one
    the caller asked for."""

    failed = _run(
        _wheel_command(checkout, tmp_path / "broken"),
        checkout,
        {"SOURCE_DATE_EPOCH": "yesterday"},
        check=False,
    )

    assert failed.returncode != 0
    assert "SOURCE_DATE_EPOCH" in failed.stdout + failed.stderr
    assert not list((tmp_path / "broken").glob("cortheon-*.whl"))


def test_builds_need_no_version_control_data(checkout: Path, tmp_path: Path) -> None:
    """The staged tree has no .git at all, and both artifacts still build,
    stay inside their caps, and carry the modules a clone produces."""

    assert not (checkout / ".git").exists()
    wheel = _build_wheel(checkout, tmp_path / "wheel")
    sdist = _build_sdist(checkout, tmp_path / "sdist")

    assert wheel.stat().st_size <= WHEEL_CAP
    assert sdist.stat().st_size <= SDIST_CAP
    with zipfile.ZipFile(wheel) as archive:
        hooks = {
            name.rsplit("/", 1)[1]
            for name in archive.namelist()
            if name.startswith("cortheon/codex_plugins/cortheon/hooks/")
        }
    assert len({name for name in hooks if name.endswith(".py")}) == 6
    with tarfile.open(sdist) as archive:
        assert any(member.name.endswith("/setup.py") for member in archive.getmembers())


def test_a_source_build_never_rewrites_the_tree_it_reads(checkout: Path, tmp_path: Path) -> None:
    """distutils hard-links a release tree to the sources it copies whenever
    the platform allows it, and this build compacts what it stages -- so one
    hard link would rewrite the checkout itself. Nothing under the tree may
    change, and the staged archive still has to come out compacted."""

    def snapshot() -> dict[str, str]:
        return {
            str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(checkout.rglob("*"))
            if path.is_file() and not path.is_symlink() and "build" not in path.parts
        }

    before = snapshot()
    archive = _build_sdist(checkout, tmp_path / "sdist")

    assert snapshot() == before
    with tarfile.open(archive) as staged:
        handle = staged.extractfile("cortheon-0.1.0/setup.py")
        assert handle is not None
        shipped = handle.read().decode("utf-8")
    assert "\n# " not in shipped, "the staged copy was not compacted"


def test_an_abandoned_workspace_is_reclaimed_and_a_live_one_is_not(tmp_path: Path) -> None:
    """A build killed outright cannot run its own cleanup, so the next build
    in that checkout collects what it left -- but only once it is far older
    than any build could still be using it."""

    root = tmp_path / "project"
    (root / "build").mkdir(parents=True)
    abandoned = open_workspace(root)
    (abandoned / "leftover").write_bytes(b"x" * 1024)
    os.utime(abandoned, (0, time.time() - STALE_AFTER_SECONDS - 60))
    live = open_workspace(root)

    fresh = open_workspace(root)
    assert not abandoned.exists()
    assert live.exists(), "a workspace a running build could still own was removed"
    assert fresh.exists() and not list(fresh.iterdir())
    assert fresh != live

    discard_workspace(fresh)
    discard_workspace(live)
    assert not fresh.exists() and not live.exists()
    # Removing an already-removed workspace is what a second cleanup does.
    discard_workspace(fresh)
