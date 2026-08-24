from __future__ import annotations

import subprocess
from pathlib import Path


def test_opencode_adapter_singleflights_concurrent_session_start():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    script = r"""
import { pathToFileURL } from "node:url"

const module = await import(pathToFileURL(process.argv[1]).href + "?singleflight=1")
let starts = 0
globalThis.fetch = async (url) => {
  if (String(url).endsWith("/v1/start")) starts += 1
  await new Promise((resolve) => setTimeout(resolve, 15))
  return {
    ok: true,
    status: 200,
    json: async () => ({
      status: "active",
      session: {
        session_id: "vx_test",
        deliverable: "general_answer",
      },
      context: { goal: "Inspect the project." },
    }),
  }
}
const client = {
  session: {
    messages: async () => ({
      data: [{
        info: { role: "user" },
        parts: [{ type: "text", text: "Inspect the project." }],
      }],
    }),
  },
  app: { log: async () => ({}) },
}
const hooks = await module.CortheonPlugin({
  client,
  directory: "/tmp",
  $: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
})
await Promise.all(
  Array.from({ length: 12 }, () => {
    const output = { system: [] }
    return hooks["experimental.chat.system.transform"](
      { sessionID: "host_session" },
      output,
    )
  }),
)
if (starts !== 1) {
  throw new Error(`expected one Cortheon start, observed ${starts}`)
}
await hooks.event({
  event: {
    type: "session.idle",
    properties: { sessionID: "host_session" },
  },
})
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_session" },
  { system: [] },
)
if (starts !== 1) {
  throw new Error(`idle continuation restarted Cortheon: ${starts}`)
}
"""

    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(plugin)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _opencode_adapter_corpus() -> str:
    """Facade plus every opencode_core module, as one source corpus.

    The implementation moved into focused modules; assertions must keep
    holding over the whole relocated corpus, not the facade alone.
    """

    root = Path(__file__).parents[1] / "src" / "cortheon"
    parts = [(root / "opencode_plugin.js").read_text()]
    parts.extend(path.read_text() for path in sorted((root / "opencode_core").glob("*.js")))
    return "\n".join(parts)


def test_opencode_adapter_prepends_system_guidance():
    source = _opencode_adapter_corpus()

    assert "output.system.push(" not in source
    assert "function prependSystemGuidance(" in source
    assert "output.system[0] = `${guidance}\\n\\n${output.system[0]}`" in source
    assert source.count("prependSystemGuidance(output, protocol)") == 2
    assert "CORTHEON_CERTIFIED: Return the following certified answer exactly" in source
    assert "const postCertificationReadTools = new Set([" in source
    assert "if (postCertificationReadTools.has(input.tool)) return" in source
    assert "already certified this patch" not in source
