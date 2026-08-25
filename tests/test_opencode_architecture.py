"""Architecture guard for the decomposed OpenCode adapter.

The facade must stay a thin, stable entry point; implementation lives in
opencode_core modules that each own one responsibility, stay under the size
budget, and remain individually importable. Distribution artifacts must ship
the exact reviewed source graph in the sdist and one equivalent bundle in the wheel.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from build_support.opencode_bundle import SOURCE_SHA256, bundle_opencode_plugin
from cortheon.qualification_core.conditions import AVAILABLE_CONDITIONS, execution_profile

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "opencode_plugin.js"
CORE = ROOT / "src" / "cortheon" / "opencode_core"
# Uniform with tests/test_wheel_equivalence.py and
# tests/test_lightweight_distribution.py; that module carries the
# measurement and the justification for both numbers.
WHEEL_CAP = 250_000
SDIST_CAP = 230_000


def _adapter_modules() -> list[Path]:
    return sorted(CORE.glob("*.js"))


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))


def test_facade_is_a_thin_stable_entry_point() -> None:
    assert FACADE.read_text().count("\n") <= 250
    assert 'from "./opencode_core/' in FACADE.read_text()


def test_every_core_module_stays_under_the_size_budget() -> None:
    for module in _adapter_modules():
        assert module.read_text().count("\n") <= 500, module.name


def test_no_duplicated_top_level_definitions() -> None:
    owned: Counter[str] = Counter()
    for path in [FACADE, *_adapter_modules()]:
        for line in _strip_comments(path.read_text()).splitlines():
            declaration = re.match(
                r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
                line,
            ) or re.match(r"^(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=", line)
            if declaration:
                owned[declaration.group(1)] += 1
    duplicated = [name for name, count in owned.items() if count > 1]
    assert duplicated == [], f"definitions owned by more than one module: {duplicated}"


def test_facade_exports_resolve_and_operators_are_intact() -> None:
    script = (
        'import { pathToFileURL } from "node:url";\n'
        "const mod = await import(pathToFileURL(process.argv[1]).href);\n"
        "console.log(JSON.stringify({\n"
        "  plugin: typeof mod.CortheonPlugin,\n"
        "  operators: Object.keys(mod.cortheonOperators || {}).sort(),\n"
        "}));"
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(FACADE)],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["plugin"] == "function"
    assert payload["operators"] == [
        "deriveDiagnosticConclusion",
        "deriveExactMatchMismatchInference",
        "deriveKeyedCollisionInference",
        "numericJoin",
    ]


def test_every_module_is_importable_and_relative_imports_resolve() -> None:
    modules = [FACADE, *_adapter_modules()]
    for module in modules:
        for specifier in re.findall(r'from\s+"(\./[^"]+)"', module.read_text()):
            assert (module.parent / specifier).is_file(), (
                f"{module.name}: unresolved import {specifier}"
            )
    for module in modules:
        subprocess.run(
            [
                "node",
                "--input-type=module",
                "-e",
                (
                    'import { pathToFileURL } from "node:url";\n'
                    "await import(pathToFileURL(process.argv[1]).href);\n"
                    "console.log('ok');"
                ),
                str(module),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )


def test_session_state_and_singleflight_survive_repeated_plugin_instances() -> None:
    script = r"""
import { pathToFileURL } from "node:url"

const plugin = await import(pathToFileURL(process.argv[1]).href + "?arch=1")
let starts = 0
globalThis.fetch = async (url) => {
  const path = new URL(String(url)).pathname
  if (path.endsWith("/v1/start")) {
    starts += 1
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        answer: "shared-certified-answer",
        session: { session_id: "vx_arch", deliverable: "code_understanding" },
        context: { goal: "Check the shared state." },
      }),
    }
  }
  throw new Error(`unexpected runtime call ${path}`)
}
const client = {
  app: { log: async () => {} },
  session: {
    messages: async () => ({
      data: [
        {
          info: { role: "user" },
          parts: [{ type: "text", text: "Check the shared state." }],
        },
      ],
    }),
  },
}
const hooksA = await plugin.CortheonPlugin({ client, directory: process.cwd() })
const outputA = { system: [] }
await hooksA["experimental.chat.system.transform"]({ sessionID: "arch" }, outputA)
if (!String(outputA.system[0]).includes("CORTHEON_CERTIFIED:")) {
  throw new Error("first instance did not certify")
}
const hooksB = await plugin.CortheonPlugin({ client, directory: process.cwd() })
const outputB = { system: [] }
await hooksB["experimental.chat.system.transform"]({ sessionID: "arch" }, outputB)
if (!String(outputB.system[0]).includes("shared-certified-answer")) {
  throw new Error("second instance did not observe the certified answer")
}
if (starts !== 1) {
  throw new Error(`singleton maps leaked across instances: ${starts} starts`)
}
console.log("ok")
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(FACADE)],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert completed.stdout.strip() == "ok"


def _expected_adapter_wheel_members() -> set[str]:
    return {"cortheon/opencode_plugin.js"}


def _expected_adapter_source_members() -> set[str]:
    return _expected_adapter_wheel_members() | {
        f"cortheon/opencode_core/{module.name}" for module in _adapter_modules()
    }


def test_bundle_is_pinned_deterministic_and_leaves_checkout_untouched(tmp_path) -> None:
    expected = {"opencode_plugin", *(module.stem for module in _adapter_modules())}
    assert set(SOURCE_SHA256) == expected
    before = {path: path.read_bytes() for path in [FACADE, *_adapter_modules()]}
    first = tmp_path / "first.js"
    second = tmp_path / "second.js"
    bundle_opencode_plugin(FACADE, CORE, first)
    bundle_opencode_plugin(FACADE, CORE, second)
    assert first.read_bytes() == second.read_bytes()
    assert before == {path: path.read_bytes() for path in before}


def test_bundle_fails_closed_on_source_or_membership_drift(tmp_path) -> None:
    facade = tmp_path / FACADE.name
    core = tmp_path / "opencode_core"
    facade.write_bytes(FACADE.read_bytes())
    core.mkdir()
    for module in _adapter_modules():
        (core / module.name).write_bytes(module.read_bytes())
    (core / "state.js").write_text("// drift\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"reviewed .*SHA-256"):
        bundle_opencode_plugin(facade, core, tmp_path / "bundle.js")
    (core / "state.js").write_bytes((CORE / "state.js").read_bytes())
    (core / "extra.js").write_text("export const extra = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="module set mismatch"):
        bundle_opencode_plugin(facade, core, tmp_path / "bundle.js")


def test_bundle_matches_source_exports_hooks_and_profile_scrubbing(tmp_path) -> None:
    bundle = tmp_path / "opencode_plugin.js"
    (tmp_path / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
    bundle_opencode_plugin(FACADE, CORE, bundle)
    script = r"""
      import {pathToFileURL} from 'node:url';
      const module = await import(pathToFileURL(process.argv[1]).href);
      globalThis.fetch = async (url) => {
        const path = new URL(String(url)).pathname;
        if (!path.endsWith('/v1/start')) throw new Error(`unexpected ${path}`);
        return {ok:true,status:200,json:async()=>({
          status:'complete',answer:'certified',
          session:{session_id:'vx_diff',deliverable:'code_understanding'},
          context:{goal:'Inspect the project.'},
        })};
      };
      const client = {app:{log:async()=>{}},session:{messages:async()=>({data:[{
        info:{role:'user'},parts:[{type:'text',text:'Inspect the project.'}],
      }]})}};
      const hooks = await module.CortheonPlugin({client,directory:process.cwd(),$:async()=>({})});
      const prompt = {system:['base']};
      await hooks['experimental.chat.system.transform']({sessionID:'diff'},prompt);
      const tool = {args:{filePath:'other.py'}};
      let toolError = null;
      try { await hooks['tool.execute.before']({sessionID:'diff',tool:'edit'},tool); }
      catch (error) { toolError = String(error.message || error); }
      const controls = Object.keys(process.env).filter((key) => [
        'CORTHEON_EVALUATOR_PROFILE','CORTHEON_COGNITIVE_TOKEN',
        'CORTHEON_EVALUATOR_MAX_STEPS','CORTHEON_AUTO_ENABLE',
        'CORTHEON_BENCHMARK_CAPTURE_CANDIDATE','CORTHEON_MAX_HOST_TOOL_CALLS',
      ].includes(key));
      console.log(JSON.stringify({
        exports:Object.keys(module).sort(),hooks:Object.keys(hooks).sort(),controls,
        operators:Object.keys(module.cortheonOperators).sort(),
        behavior:{prompt:prompt.system,tool,toolError},
      }));
    """

    def probe(path: Path, condition: str) -> dict:
        profile = execution_profile(condition, "a" * 64)
        profile["nonce"] = "3" * 32
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", script, str(path)],
            env={
                **os.environ,
                "CORTHEON_EVALUATOR_PROFILE": json.dumps(profile),
                "CORTHEON_COGNITIVE_TOKEN": "hidden",
                "CORTHEON_EVALUATOR_MAX_STEPS": "4",
                "CORTHEON_AUTO_ENABLE": "1",
                "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1",
                "CORTHEON_MAX_HOST_TOOL_CALLS": "12",
            },
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return json.loads(completed.stdout)

    for condition in AVAILABLE_CONDITIONS:
        source = probe(FACADE, condition)
        assert source == probe(bundle, condition), condition
        assert source["controls"] == []


def test_built_artifacts_ship_exactly_the_required_adapter_modules(
    tmp_path: Path,
) -> None:
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
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    wheel = next(wheel_dir.glob("cortheon-*.whl"))
    assert wheel.stat().st_size <= WHEEL_CAP
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    shipped_js = {
        member for member in members if member.startswith("cortheon/") and member.endswith(".js")
    }
    assert shipped_js == _expected_adapter_wheel_members()
    shipped_python = {
        member for member in members if member.startswith("cortheon/") and member.endswith(".py")
    }
    assert not any(
        re.fullmatch(r"cortheon/(benchmark|parity|research|proxy)\.py", member)
        or "/benchmark_core/" in member
        or "/parity_campaign/" in member
        for member in shipped_python
    )

    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(sdist_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    archive_path = next(sdist_dir.glob("cortheon-*.tar.gz"))
    assert archive_path.stat().st_size <= SDIST_CAP
    with tarfile.open(archive_path, "r:gz") as archive:
        sdist_members = {member.name for member in archive.getmembers()}
    sdist_js = {
        member.rsplit("/", 1)[0][len("cortheon-0.1.0/src/") :] + "/" + member.rsplit("/", 1)[1]
        for member in sdist_members
        if "/src/cortheon/" in member and member.endswith(".js")
    }
    assert sdist_js == _expected_adapter_source_members()


def test_facade_imports_from_an_installed_wheel(tmp_path: Path) -> None:
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
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    wheel = next(wheel_dir.glob("cortheon-*.whl"))
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
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    installed_facade = install_dir / "cortheon" / "opencode_plugin.js"
    assert installed_facade.is_file()
    script = (
        'import { pathToFileURL } from "node:url";\n'
        "const mod = await import(pathToFileURL(process.argv[1]).href);\n"
        "console.log(typeof mod.CortheonPlugin);"
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(installed_facade)],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert completed.stdout.strip() == "function"
