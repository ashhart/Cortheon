"""Deterministic exact-source bundler for the OpenCode adapter."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

FACADE = "opencode_plugin"
SOURCE_SHA256 = {
    "opencode_plugin": "845e6a76005d1285e8f1715a86c9bc9e8cfe1f62fb188850745680ac30b04370",
    "auto_evidence": "386c57c537e2b93628a5a748ff6792760953f12bf2bc52d2cc52c7f3f8baede7",
    "auto_research": "a1f162475caf89a58f352896235f3df33199bfa7aa3be6e623fe804caac649be",
    "completion": "570740b416d4f81095f50108f460d9fce4ac4a01b47d5bfff129e81b6ba9ba15",
    "evidence": "37bfc79ac68b976940c28df4a425da00c1d137d9c84b9e1478114cc6cd70a262",
    "evaluator_control": "a06cfdcb305b3d4bed10844ddc2e8889263c30e34ca3079ca57e43145e0939d5",
    "hook_conversation": "51435ed4a3cc2f03c42ae168d9c00548679de762044fc9adc803e0ec3074ea1d",
    "hook_output": "799ada5cf88fe3e846839ac853720b7cc2880e2c60d6616304982480d9a60e57",
    "hook_tool_before": "c21c030aa706092944b7add7ea674730f8ad929da6077c7cc16809706c6ccb1e",
    "hooks": "d53ad2f3421e4ac0b16c2095023ac494b30fafadf3a7b87728496357cc3b04eb",
    "host_access": "c757ad263f62f949a2f314f12b562055f473104923ca82f02fb32f3319694032",
    "investigation": "473d4627ad76dc4e259672295456dfb3364eb741549a0a25c087f01a7b2b3d47",
    "joins": "61544f3d9e8a380432a5c246b8eb233a77f324307aa8a930c52ca56bd7ae147d",
    "plans": "f842093a03f886023d848cb7f780a97c05dad822679eabebdc4af46621e2b9f9",
    "repair_derive": "3ce737735d49e6da8c03ea000a2b4169ea562892c0b72581f8e1aeb0e2f4e9b3",
    "repair_exec": "04a5e32f736e8bcabbbe9f180d8b58782ddffae13acbfeb33bb59a9379f65a4c",
    "state": "1a2036fa42f01de53ca088d41f175e6890d2c1638269940dcf0e7729eeb8b4c9",
    "state_merge": "676b8c591787796bb03d10c75464fb66dbeea32b98bb1d2558faef6b64c673d4",
}
COMPACTED_CLOSURE_SHA256 = "e8693b791b06c5dcf15f25581d2a48fc3a5ce4b60f37cd2440661415640af841"

_IMPORT = re.compile(r'^import\s+.*?\s+from\s+"(?P<spec>[^"]+)"\s*\n', re.MULTILINE | re.DOTALL)
_EXPORT_LIST = re.compile(r"^export\s*\{.*?\}\s*\n?", re.MULTILINE | re.DOTALL)
_DECLARATION = re.compile(
    r"^(?:const|let|class|function|async function)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


def _fail(message: str) -> SystemExit:
    return SystemExit(f"opencode bundler: {message}")


def _read_sources(facade: Path, core: Path) -> dict[str, str]:
    paths = {FACADE: facade, **{path.stem: path for path in sorted(core.glob("*.js"))}}
    if set(paths) != set(SOURCE_SHA256):
        raise _fail(
            f"module set mismatch: missing {sorted(set(SOURCE_SHA256) - set(paths))}, "
            f"unexpected {sorted(set(paths) - set(SOURCE_SHA256))}"
        )
    sources = {}
    closure = hashlib.sha256()
    raw_matches = True
    for name, path in paths.items():
        raw = path.read_bytes()
        raw_matches &= hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[name]
        closure.update(name.encode())
        closure.update(b"\0")
        closure.update(raw)
        closure.update(b"\0")
        sources[name] = raw.decode("utf-8")
    if not raw_matches and closure.hexdigest() != COMPACTED_CLOSURE_SHA256:
        raise _fail("sources do not match the reviewed raw or compacted SHA-256 closure")
    return sources


def _imports(name: str, source: str, modules: set[str]) -> tuple[set[str], list[str]]:
    local: set[str] = set()
    external: list[str] = []
    for match in _IMPORT.finditer(source):
        spec = match.group("spec")
        if spec.startswith("./"):
            target = spec.removeprefix("./").removesuffix(".js")
            if target.startswith("opencode_core/"):
                if name != FACADE:
                    raise _fail(f"{name}: qualified core import {spec!r}")
                target = target.removeprefix("opencode_core/")
            if target not in modules:
                raise _fail(f"{name}: unresolved import {spec!r}")
            local.add(target)
        else:
            if spec not in {"node:crypto", "node:fs"}:
                raise _fail(f"{name}: external import {spec!r} is not allowed")
            external.append(match.group(0).strip())
    return local, external


def bundle_opencode_plugin(facade: Path, core: Path, destination: Path) -> None:
    """Inline the pinned acyclic module graph into the shipped facade."""

    sources = _read_sources(facade, core)
    modules = set(sources) - {FACADE}
    imports = {name: _imports(name, source, modules) for name, source in sources.items()}
    graph = {name: imports[name][0] for name in modules}
    ordered: list[str] = []
    remaining = dict(graph)
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps & set(remaining))
        if not ready:
            raise _fail(f"cyclic graph among {sorted(remaining)}")
        ordered.extend(ready)
        for name in ready:
            del remaining[name]
    reachable: set[str] = set()
    frontier = set(imports[FACADE][0])
    while frontier:
        name = frontier.pop()
        if name not in reachable:
            reachable.add(name)
            frontier |= graph[name]
    if reachable != modules:
        raise _fail(f"unreachable modules {sorted(modules - reachable)}")
    owners: dict[str, str] = {}
    for name, source in sources.items():
        for symbol in _DECLARATION.findall(source):
            if symbol in owners and owners[symbol] != name:
                raise _fail(f"symbol {symbol!r} declared in {owners[symbol]} and {name}")
            owners[symbol] = name
    external = sorted({item for _, items in imports.values() for item in items})
    if len(external) != 2:
        raise _fail(f"expected two exact external imports, got {external!r}")

    def body(name: str) -> str:
        stripped = _IMPORT.sub("", sources[name])
        return stripped if name == FACADE else _EXPORT_LIST.sub("", stripped)

    destination.write_text(
        "\n\n".join([*external, *(body(name) for name in ordered), body(FACADE)]),
        encoding="utf-8",
    )
