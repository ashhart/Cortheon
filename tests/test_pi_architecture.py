"""Architecture and fail-closed mutations for the Pi adapter bundle.

The packer in ``build_support.pi_bundle`` is an exact-source packer: every
input byte is hash-pinned and the module graph is verified (external
allowlist, resolved imports, single owner per symbol, reachability,
acyclicity).  These tests pin the inventory, the graph facts, the shared
state owner, and the loop budget, and prove that every drift fails closed.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

import build_support.pi_bundle as pi_bundle
from build_support.lean_source import compact_host_source

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "pi_extension.ts"
CORE_DIR = ROOT / "src" / "cortheon" / "pi_core"
FILE_CAP = 500
STATE_MODULE = "state"


def _modules() -> dict[str, str]:
    sources = {pi_bundle.PI_FACADE_NAME: FACADE.read_text(encoding="utf-8")}
    for path in sorted(CORE_DIR.glob("*.ts")):
        sources[path.stem] = path.read_text(encoding="utf-8")
    return sources


def _scanned(sources: dict[str, str]) -> dict[str, pi_bundle._ScannedModule]:
    return {name: pi_bundle._scan_module(name, text) for name, text in sources.items()}


def test_pinned_hashes_match_the_reviewed_sources() -> None:
    for name, digest in pi_bundle.PI_SOURCE_SHA256.items():
        path = FACADE if name == pi_bundle.PI_FACADE_NAME else CORE_DIR / f"{name}.ts"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_module_inventory_is_exact() -> None:
    on_disk = {path.stem for path in CORE_DIR.glob("*.ts")}
    pinned = set(pi_bundle.PI_SOURCE_SHA256) - {pi_bundle.PI_FACADE_NAME}
    assert on_disk == pinned
    assert pi_bundle.PI_SOURCE_SHA256[pi_bundle.PI_FACADE_NAME]
    # Every module file is within the line cap.
    for path in [FACADE, *CORE_DIR.glob("*.ts")]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= FILE_CAP, path


def test_facade_has_default_export_and_only_it() -> None:
    scanned = _scanned(_modules())
    facade = scanned[pi_bundle.PI_FACADE_NAME]
    assert facade.has_default_export
    assert not any(
        module.has_default_export
        for name, module in scanned.items()
        if name != pi_bundle.PI_FACADE_NAME
    )


def test_single_owner_per_top_level_symbol() -> None:
    scanned = _scanned(_modules())
    owners: dict[str, str] = {}
    for name, module in sorted(scanned.items()):
        for symbol in module.declared:
            previous = owners.setdefault(symbol, name)
            assert previous == name, f"{symbol} declared in {previous} and {name}"


def test_shared_state_has_exactly_one_owner() -> None:
    scanned = _scanned(_modules())
    state = scanned[STATE_MODULE]
    accessors = {
        "getActive",
        "setActive",
        "abandonActive",
        "isEnabled",
        "setEnabled",
        "stopHeartbeat",
    }
    assert accessors <= set(state.exported)
    for name, module in scanned.items():
        if name == STATE_MODULE:
            continue
        assert not (accessors & module.declared), f"{name} redefines shared-state accessors"


def test_graph_is_acyclic_and_fully_reachable() -> None:
    sources = _modules()
    scanned = _scanned(sources)
    facade = scanned.pop(pi_bundle.PI_FACADE_NAME)
    graph = {name: set(module.local_imports) for name, module in scanned.items()}
    remaining = dict(graph)
    while remaining:
        ready = [e for e, deps in remaining.items() if not (deps & set(remaining))]
        assert ready, f"cyclic graph among {sorted(remaining)}"
        for entry in ready:
            del remaining[entry]
    reachable: set[str] = set()
    frontier = set(facade.local_imports)
    while frontier:
        entry = frontier.pop()
        if entry in reachable:
            continue
        reachable.add(entry)
        frontier |= graph[entry]
    assert reachable == set(scanned)


def test_loop_budget_is_exactly_one_automatic_continuation() -> None:
    text = (CORE_DIR / "protocol.ts").read_text(encoding="utf-8")
    match = re.search(r"MAX_AUTOMATIC_CONTINUATIONS\s*=\s*(\d+)", text)
    assert match is not None
    assert int(match.group(1)) == 1


def test_one_unified_continuation_budget_no_separate_answer_only_allowance() -> None:
    """The shipped adapter has exactly ONE continuation budget: no separate
    answer-only counter, and every follow-up grant — repair or answer-only —
    increments the same shared `automaticContinuations` counter gated by the
    same MAX_AUTOMATIC_CONTINUATIONS constant."""
    for path in [FACADE, *CORE_DIR.glob("*.ts")]:
        text = path.read_text(encoding="utf-8")
        assert "MAX_ANSWER_ONLY_CONTINUATIONS" not in text, path
        assert "answerOnlyContinuations" not in text, path
        assert "consumeAnswerOnlyContinuation" not in text, path
    session = (CORE_DIR / "session_events.ts").read_text(encoding="utf-8")
    grants = session.count("automaticContinuations += 1")
    assert grants == 2, "exactly two grant sites (repair + answer-only)"
    gates = session.count("automaticContinuations < MAX_AUTOMATIC_CONTINUATIONS")
    assert gates == 1, "one shared budget gate for the answer-only grant"
    assert "MAX_AUTOMATIC_CONTINUATIONS ||" in session or (
        "active.automaticContinuations >= MAX_AUTOMATIC_CONTINUATIONS" in session
    ), "the repair cap arm shares the same budget"


def _copy_sources(target: Path, overrides: dict[str, str] | None = None) -> Path:
    """Copy the real sources (with optional per-module overrides) and return
    a hash table that legitimately pins the copied bytes."""
    target.mkdir(parents=True)
    table: dict[str, str] = {}
    for name, text in _modules().items():
        if overrides and name in overrides:
            text = overrides[name]
        path = target / f"{name}.ts"
        path.write_text(text, encoding="utf-8")
        table[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return_path = target.parent / "table"
    del return_path
    return target


def _bundle_with(
    tmp_path: Path,
    overrides: dict[str, str] | None = None,
    *,
    update_hashes: bool = True,
    extra_files: dict[str, str] | None = None,
) -> None:
    core = tmp_path / "pi_core"
    core.mkdir(parents=True)
    facade = tmp_path / "pi_extension.ts"
    table: dict[str, str] = {}
    for name, text in _modules().items():
        body = (overrides or {}).get(name, text)
        if name == pi_bundle.PI_FACADE_NAME:
            facade.write_text(body, encoding="utf-8")
            table[name] = hashlib.sha256(facade.read_bytes()).hexdigest()
        else:
            path = core / f"{name}.ts"
            path.write_text(body, encoding="utf-8")
            table[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, text in (extra_files or {}).items():
        path = core / f"{name}.ts"
        path.write_text(text, encoding="utf-8")
        table[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not update_hashes:
        table = dict(pi_bundle.PI_SOURCE_SHA256)
    original = pi_bundle.PI_SOURCE_SHA256
    pi_bundle.PI_SOURCE_SHA256 = table
    try:
        pi_bundle.bundle_pi_extension(facade, core, tmp_path / "bundled.ts")
    finally:
        pi_bundle.PI_SOURCE_SHA256 = original


def test_bundle_over_the_real_sources_succeeds(tmp_path: Path) -> None:
    _bundle_with(tmp_path)
    assert (tmp_path / "bundled.ts").exists()


def test_compacted_source_closure_is_separately_pinned(tmp_path: Path) -> None:
    facade = tmp_path / "pi_extension.ts"
    core = tmp_path / "pi_core"
    shutil.copy2(FACADE, facade)
    shutil.copytree(CORE_DIR, core)
    compact_host_source(facade)
    for path in core.glob("*.ts"):
        compact_host_source(path)
    sources = pi_bundle.verify_pi_sources(facade, core)
    assert set(sources) == set(pi_bundle.PI_SOURCE_SHA256)


def test_changed_source_without_hash_update_fails_closed(tmp_path: Path) -> None:
    overrides = {"protocol": "export const MAX_AUTOMATIC_CONTINUATIONS = 9;\n"}
    with pytest.raises(SystemExit):
        _bundle_with(tmp_path, overrides, update_hashes=False)


def test_missing_module_fails_closed(tmp_path: Path) -> None:
    core = tmp_path / "pi_core"
    core.mkdir()
    facade = tmp_path / "pi_extension.ts"
    shutil.copy2(FACADE, facade)
    for path in sorted(CORE_DIR.glob("*.ts")):
        if path.stem != "actions":
            shutil.copy2(path, core / path.name)
    with pytest.raises(SystemExit, match=r"module set mismatch|no pinned hash"):
        pi_bundle.bundle_pi_extension(facade, core, tmp_path / "bundled.ts")


def test_extra_module_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        SystemExit,
        # "unreachable modules" names the exact rogue module, which is more
        # specific than the set/hash mismatch it graduated from.
        match=r"module set mismatch|no pinned hash|unreachable modules",
    ):
        _bundle_with(tmp_path, extra_files={"rogue": "export const rogue = 1;\n"})


def test_unknown_external_specifier_fails_closed(tmp_path: Path) -> None:
    overrides = {
        "merge": (
            'import { hostname } from "node:os";\n'
            + (CORE_DIR / "merge.ts").read_text(encoding="utf-8")
        )
    }
    with pytest.raises(SystemExit, match="unallowlisted specifier"):
        _bundle_with(tmp_path, overrides)


def test_unresolved_local_import_fails_closed(tmp_path: Path) -> None:
    text = (CORE_DIR / "certify.ts").read_text(encoding="utf-8")
    overrides = {"certify": text.replace('"./protocol.ts"', '"./missing.ts"')}
    with pytest.raises(SystemExit, match="unresolved module"):
        _bundle_with(tmp_path, overrides)


def test_duplicate_symbol_owner_fails_closed(tmp_path: Path) -> None:
    text = (CORE_DIR / "actions.ts").read_text(encoding="utf-8")
    overrides = {"actions": "export function getActive() {\n\treturn undefined;\n}\n" + text}
    with pytest.raises(SystemExit, match="declared in both"):
        _bundle_with(tmp_path, overrides)


def test_cyclic_graph_fails_closed(tmp_path: Path) -> None:
    # protocol is imported by state; add the reverse edge to force a cycle.
    state_text = (CORE_DIR / "state.ts").read_text(encoding="utf-8")
    overrides = {
        "protocol": (
            'import { getActive } from "./state.ts";\n'
            + (CORE_DIR / "protocol.ts").read_text(encoding="utf-8")
        )
    }
    del state_text
    with pytest.raises(SystemExit, match="cyclic graph"):
        _bundle_with(tmp_path, overrides)
