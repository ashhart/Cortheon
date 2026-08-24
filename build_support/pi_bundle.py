"""Exact-source packer for the Cortheon Pi adapter.

Deliberately not a TypeScript parser.  The facade and every pi_core module
are pinned by SHA-256 below; the packer refuses to run over any byte that
differs from the reviewed source.  Over those known bytes the transform is
structural: drop relative imports, strip ``export`` prefixes, and emit one
merged external-import header in deterministic order, in a form both
strict TypeScript and Node's type-stripping loader accept.

Symbol-level invariants are still enforced fail-closed over the pinned
bytes: the external allowlist, resolved local imports, one owner per
top-level symbol, full facade reachability, and an acyclic module graph.
Any source edit fails the build until the hash table is deliberately
updated as part of review.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

PI_FACADE_NAME = "pi_extension"

# SHA-256 over the raw bytes of each final reviewed source, keyed by module
# name. The facade is "pi_extension"; every other key is a pi_core module stem.
PI_SOURCE_SHA256: dict[str, str] = {
    "pi_extension": "e8a96e60ac6b26a7bf5ca055ecce5e29ec4c9bd4485ef0fec279d5e13fddf268",
    "actions": "7a0ef3d363fe2674d3830e72e663dbc329594fcd73e7cb5841787b5fde161306",
    "budget": "c511ba97b1408c17e112aba019c653fd61061ce5f9159490ddc8d8a0733558c1",
    "certify_mutation": "dc6c16b1b94f4a32bc43a2826729fc30b7df761557703bcd03a3763a7ad02df7",
    "certify": "b19ffd130857e550ef79a19dcdb163d3ea793e6847692252ae631d072cf716e9",
    "commands": "8662abad41e5fad5f7d63cf9df39a31884d7bd13c5bd3b17fb9752b9463212ce",
    "derive": "26ec9f18ae23d157cd1d431a5459438c8d9236a2797aae1cbcb5dda29d741bd9",
    "evidence_ledger": "284887b5276ebb220c9d2cbdbc7a2ac887985fe1d594b84e7b9d7c80772b1cc9",
    "evidence_mapping": "2d061df8294b5734657024da05dceac73b602a4f9e2e2d7c81a4079aeae12178",
    "grounding": "bb51114d64c1301c119418a2528c06aae87894b282d0767b49b3342859aa3912",
    "text": "aaa54de2de8012800e4a32ab186756f7923a94d2018c2781630d5b5b0b984ad0",
    "events": "8c2ea7257c96a39275fa9e0a9ab482740299af5510853121a79eb86c85c0d1db",
    "host_evidence": "1e33beb163f351b6067e0fa8a5c7fd99062eb17d841e5089c8953beb092cc899",
    "merge": "46f7a5a8a27080f61c888f761c8ff58f003db40d667ba9442d70bf168aa24d21",
    "protocol": "918276eac8fb12e6c981cb25e13299bed09310520f1c311cdcf1f46106060305",
    "causal_complete": "e60e0e397289953cc66ffc5e93eea298364efe1a11dc85c8032af2916155caa9",
    "candidate_capture": "385195e9f743ac4b2c3da855e2cb7c506603245f13901079fed306d697be1a39",
    "repair": "f3336ea65ec622293afaaa62ec903ba246ae6d2dd71f14d33bc9456e26e300f0",
    "runtime": "82e3ea95888e99b0249eec86cd26b0c623ab7551e3ffe63cb65b6349ce898718",
    "satisfy": "8f264d5a627c89c9c51ca0cbcfb402fe78894b60d81a407809641e04e6a73fd4",
    "session_events": "f7e147e2f1255892f63d578c78f887dfa4b72b98cdd6c12d5c5c5b4f6a78ac23",
    "state": "38c779efa10037df11cf97d3d5576f344457499d96356254c6e7d0e6e39e7d8b",
    "task_analysis": "ca7ab38d80d5a8d7a554445d7a74096249ad08fe19b7e9e1b34ff17c2b00d39c",
    "observe_claim": "1924150be30ca2d67c374cfaf19485bd20eea6b49a84e52c25aefd89d983d581",
    "tool_events": "ad88e791935be85051cf4eb9095073656989507dcdf27164ae507310e9c00b4f",
    "observe_tools": "bad63e71939bd5df88ca4e431254c1eba7e7257d2ae3eda96eaefb0892edaf4b",
    "web_evidence": "dc205825714ddf8c728ffa1645913e053c42b1fc17e51c50c74f861f552cfab9",
    "terminal": "7705a5b745c701dcf4a6b9003c68466006b8ba70da6fe219fd51f180195fc1df",
    "causal_answer": "3cc6ba2a7b8f4479ee194dedf7d9df18c33b6e4d73cdda14c8b6c04b77ee34bc",
}
PI_COMPACTED_CLOSURE_SHA256 = "b528fb49e5d406d2e75bae614283e4f18ea6c399c57fb9bdc49533f4bdfd7f18"

# Exact external specifiers this adapter may import from, as
# spec -> (value symbols, type symbols, default binding or None).
PI_EXTERNAL_ALLOWLIST: dict[str, tuple[frozenset[str], frozenset[str], str | None]] = {
    "node:child_process": (frozenset({"spawn"}), frozenset(), None),
    "node:crypto": (frozenset({"createHash"}), frozenset(), None),
    "node:fs": (frozenset({"closeSync", "readFileSync"}), frozenset(), None),
    "node:fs/promises": (frozenset({"realpath"}), frozenset(), None),
    "node:path": (frozenset(), frozenset(), "path"),
    "@earendil-works/pi-ai": (frozenset({"uuidv7"}), frozenset({"Usage"}), None),
    "@earendil-works/pi-ai/compat": (frozenset({"complete"}), frozenset(), None),
    "@earendil-works/pi-coding-agent": (
        frozenset({"createFindTool", "createGrepTool", "createReadTool"}),
        frozenset(
            {
                "ExtensionAPI",
                "ExtensionContext",
                "ToolExecutionStartEvent",
                "ToolResultEvent",
            }
        ),
        None,
    ),
}

_IMPORT_STATEMENT = re.compile(r'^import\s+(type\s+)?(?P<rest>.+?)\s+from\s+"(?P<spec>[^"]+)";$')
_DEFAULT_BINDING = re.compile(r"^(?P<name>[A-Za-z_$][\w$]*)$")
_NAMED_BINDING = re.compile(r"^\{(?P<names>[^{}]+)\}$")
_IMPORT_ITEM = re.compile(r"^(?:type\s+)?(?P<name>[A-Za-z_$][\w$]*)$")
_EXPORT_DECLARATION = re.compile(
    r"^export\s+(?:async\s+)?(?P<kind>function|const|let|interface|type)\b"
)
_EXPORTED_NAME = re.compile(
    r"^export\s+(?:async\s+)?(?:function|const|let|interface|type)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)"
)
_TYPE_KINDS = frozenset({"interface", "type"})
_DECLARATION_HEAD = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?P<kind>function|const|let|interface|type)\b"
    r"(?:\s+(?P<name>[A-Za-z_$][\w$]*))?"
)


def _fail(message: str) -> SystemExit:
    return SystemExit(f"pi bundler: {message}")


class _ScannedModule:
    def __init__(self, name: str) -> None:
        self.name = name
        self.external_imports: list[tuple[str, list[tuple[str, str]], str | None]] = []
        self.local_imports: dict[str, dict[str, set[str]]] = {}
        self.exported: dict[str, str] = {}
        self.declared: set[str] = set()
        self.body: list[str] = []
        self.has_default_export = False


def _parse_import_items(module: str, clause_type_only: bool, names: str) -> list[tuple[str, str]]:
    items = [item.strip() for item in names.split(",") if item.strip()]
    matches = [(item, _IMPORT_ITEM.match(item)) for item in items]
    for item, match in matches:
        if match is None:
            raise _fail(f"{module}: unsupported import item {item!r}")
    if not matches:
        raise _fail(f"{module}: empty named import")
    return [
        (match.group("name"), "type" if clause_type_only or item.startswith("type ") else "value")
        for item, match in matches
        if match is not None
    ]


def _local_target(module: str, spec: str, facade: bool) -> str:
    if not spec.startswith("./") or not spec.endswith(".ts"):
        raise _fail(f"{module}: unsupported relative specifier {spec!r}")
    target = spec[2:-3]
    if target.startswith("pi_core/"):
        if not facade:
            raise _fail(f"{module}: pi_core-qualified specifier outside facade: {spec!r}")
        target = target[len("pi_core/") :]
    if not target or "/" in target:
        raise _fail(f"{module}: unsupported relative specifier {spec!r}")
    return target


def _scan_module(name: str, text: str) -> _ScannedModule:
    """Scan one pinned module's import header and top-level declarations.

    Hash-verified bytes, so this only extracts structural facts.
    """
    facade = name == PI_FACADE_NAME
    module = _ScannedModule(name)
    lines = text.splitlines()
    index = 0
    imports_done = False
    in_block_comment = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        index += 1
        if not imports_done and in_block_comment:
            # Inside a /* */ header comment: keep it for the body but keep
            # scanning for imports that follow it.
            module.body.append(line)
            if "*/" in stripped:
                in_block_comment = False
            continue
        if not imports_done and stripped.startswith("/*"):
            module.body.append(line)
            if "*/" not in stripped:
                in_block_comment = True
            continue
        if not imports_done and (not stripped or stripped.startswith("//")):
            continue
        if not imports_done and (stripped.startswith("import ") or stripped == "import"):
            statement = [stripped]
            while not statement[-1].rstrip().endswith(";"):
                if index >= len(lines):
                    raise _fail(f"{name}: unterminated import statement")
                statement.append(lines[index].strip())
                index += 1
            joined = " ".join(statement)
            match = _IMPORT_STATEMENT.match(joined)
            if match is None:
                raise _fail(f"{name}: unsupported import statement: {joined!r}")
            clause_type_only = match.group(1) is not None
            rest, spec = match.group("rest"), match.group("spec")
            if spec.startswith("./"):
                target = _local_target(name, spec, facade)
                named = _NAMED_BINDING.match(rest.strip())
                if named is None:
                    raise _fail(f"{name}: unsupported local import form: {joined!r}")
                kinds = module.local_imports.setdefault(target, {"value": set(), "type": set()})
                for symbol, kind in _parse_import_items(
                    name, clause_type_only, named.group("names")
                ):
                    kinds[kind].add(symbol)
                continue
            default = _DEFAULT_BINDING.match(rest.strip())
            if default is not None:
                module.external_imports.append((spec, [], default.group("name")))
                continue
            named = _NAMED_BINDING.match(rest.strip())
            if named is None:
                raise _fail(f"{name}: unsupported external import form: {joined!r}")
            module.external_imports.append(
                (spec, _parse_import_items(name, clause_type_only, named.group("names")), None)
            )
            continue
        imports_done = True
        if stripped.startswith("export default"):
            if not facade:
                raise _fail(f"{name}: default export outside the facade")
            module.has_default_export = True
            module.body.append(line)
            continue
        if stripped.startswith("export "):
            declaration = _EXPORT_DECLARATION.match(stripped)
            if declaration is None:
                raise _fail(f"{name}: unsupported export form: {stripped!r}")
            declared = _EXPORTED_NAME.match(stripped)
            if declared is not None:
                declared_name = declared.group("name")
                module.declared.add(declared_name)
                module.exported[declared_name] = (
                    "type" if declaration.group("kind") in _TYPE_KINDS else "value"
                )
            module.body.append(line[: len(line) - len(line.lstrip())] + stripped[len("export ") :])
            continue
        if not line[:1].isspace():
            declaration = _DECLARATION_HEAD.match(stripped)
            if declaration is not None and declaration.group("name"):
                module.declared.add(declaration.group("name"))
        module.body.append(line)
    return module


def verify_pi_sources(facade: Path, core_dir: Path) -> dict[str, str]:
    """Hash-verify the facade and every pi_core module, fail-closed.

    Extra, missing, or drifted modules abort the build.
    """
    paths = {
        PI_FACADE_NAME: facade,
        **{path.stem: path for path in sorted(core_dir.glob("*.ts"))},
    }
    expected = set(PI_SOURCE_SHA256)
    actual = set(paths)
    if actual != expected:
        raise _fail(
            f"module set mismatch: missing {sorted(expected - actual)}, "
            f"unexpected {sorted(actual - expected)}"
        )
    sources: dict[str, str] = {}
    closure = hashlib.sha256()
    raw_matches = True
    for name, path in paths.items():
        raw = path.read_bytes()
        raw_matches &= hashlib.sha256(raw).hexdigest() == PI_SOURCE_SHA256[name]
        closure.update(name.encode())
        closure.update(b"\0")
        closure.update(raw)
        closure.update(b"\0")
        sources[name] = raw.decode("utf-8")
    if not raw_matches and closure.hexdigest() != PI_COMPACTED_CLOSURE_SHA256:
        raise _fail("sources do not match the reviewed raw or compacted SHA-256 closure")
    return sources


def bundle_pi_extension(facade: Path, core_dir: Path, destination: Path) -> None:
    """Deterministically inline the pinned pi_core graph into one artifact."""
    sources = verify_pi_sources(facade, core_dir)
    facade_module = _scan_module(PI_FACADE_NAME, sources[PI_FACADE_NAME])
    if not facade_module.has_default_export:
        raise _fail("facade must have a default export")
    modules = {
        name: _scan_module(name, sources[name])
        for name in sorted(sources)
        if name != PI_FACADE_NAME
    }
    if not modules:
        raise _fail(f"no modules found under {core_dir}")

    owners: dict[str, str] = {}
    for name, module in [*sorted(modules.items()), (PI_FACADE_NAME, facade_module)]:
        for symbol in module.declared:
            previous = owners.setdefault(symbol, name)
            if previous != name:
                raise _fail(f"symbol {symbol!r} declared in both {previous} and {name}")

    external: dict[str, dict[str, set[str]]] = {}
    default_names: dict[str, str] = {}
    for name, module in [*sorted(modules.items()), (PI_FACADE_NAME, facade_module)]:
        for spec, names, default in module.external_imports:
            if spec not in PI_EXTERNAL_ALLOWLIST:
                raise _fail(f"{name}: unallowlisted specifier {spec!r}")
            value_allow, type_allow, default_allow = PI_EXTERNAL_ALLOWLIST[spec]
            if default is not None:
                if default_allow is None or default != default_allow:
                    raise _fail(f"{name}: default import {default!r} from {spec!r}")
                default_names.setdefault(spec, default)
                continue
            merged = external.setdefault(spec, {"value": set(), "type": set()})
            for symbol, kind in names:
                if symbol not in (value_allow if kind == "value" else type_allow):
                    raise _fail(
                        f"{name}: {kind} import {symbol!r} from {spec!r} is not allowlisted"
                    )
                merged[kind].add(symbol)

    def resolve(name: str, target: str, symbols: dict[str, set[str]]) -> None:
        if target not in modules:
            raise _fail(f"{name}: unresolved module {target!r}")
        for kind in ("value", "type"):
            for symbol in symbols[kind]:
                exported = modules[target].exported.get(symbol)
                if exported is None:
                    raise _fail(f"{name}: {target!r} does not export {symbol!r}")
                if exported != kind:
                    raise _fail(f"{name}: {kind} import of {exported} export {target}.{symbol}")

    for name, module in sorted(modules.items()):
        for target, symbols in module.local_imports.items():
            resolve(name, target, symbols)
    for target, symbols in facade_module.local_imports.items():
        resolve(PI_FACADE_NAME, target, symbols)

    graph = {name: set(module.local_imports) for name, module in modules.items()}
    ordered: list[str] = []
    remaining = dict(graph)
    while remaining:
        ready = sorted(entry for entry, deps in remaining.items() if not (deps & set(remaining)))
        if not ready:
            raise _fail(f"cyclic graph among {sorted(remaining)}")
        for entry in ready:
            ordered.append(entry)
            del remaining[entry]

    reachable: set[str] = set()
    frontier = set(facade_module.local_imports)
    while frontier:
        entry = frontier.pop()
        if entry in reachable:
            continue
        reachable.add(entry)
        frontier |= graph[entry]
    unreachable = set(modules) - reachable
    if unreachable:
        raise _fail(f"unreachable modules {sorted(unreachable)}")

    header: list[str] = []
    imported_names: set[str] = set(owners) | set(default_names.values())
    for spec in sorted(external):
        merged = external[spec]
        for symbol in merged["value"] | merged["type"]:
            if symbol in imported_names:
                raise _fail(f"imported name {symbol!r} collides with another top-level symbol")
            imported_names.add(symbol)
        symbols = [f"type {name}" for name in sorted(merged["type"] - merged["value"])]
        symbols = sorted(merged["value"]) + symbols
        if merged["value"]:
            header.append(f'import {{ {", ".join(symbols)} }} from "{spec}";')
        else:
            header.append(f'import type {{ {", ".join(symbols)} }} from "{spec}";')
    header.extend(f'import {default_names[spec]} from "{spec}";' for spec in sorted(default_names))

    sections = [
        "\n".join(section)
        for section in [header] + [modules[name].body for name in ordered] + [facade_module.body]
    ]
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
