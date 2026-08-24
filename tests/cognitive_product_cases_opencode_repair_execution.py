from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def test_opencode_adapter_executes_only_a_host_verified_bounded_repair():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        implementation = workspace / "calculator.py"
        test_file = workspace / "test_calculator.py"
        implementation.write_text(
            "def add(left: int, right: int) -> int:\n    return left - right\n"
        )
        test_file.write_text(
            "from calculator import add\n\n\n"
            "def test_adds_two_numbers() -> None:\n"
            "    assert add(2, 3) == 5\n"
        )
        script = r"""
import { pathToFileURL } from "node:url"
import { readFile } from "node:fs/promises"
import { spawnSync } from "node:child_process"

const module = await import(pathToFileURL(process.argv[1]).href + "?repair=1")
const directory = process.argv[2]
let repairExecutions = 0
let testExecutions = 0
let submitted
let passiveEvidenceAccepted = false
globalThis.fetch = async (url, options) => {
  const path = new URL(String(url)).pathname
  const body = options?.body ? JSON.parse(options.body) : {}
  if (path.endsWith("/v1/start")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_repair",
          deliverable: "code_change",
        },
        context: { goal: body.goal },
        next_action: {
          request: {
            request_id: "req_read",
            capability: "read_many",
            query: "Read the implementation and protected test.",
            parameters: {
              paths: ["calculator.py", "test_calculator.py"],
              symbols: ["add"],
            },
          },
        },
      }),
    }
  }
  if (path.endsWith("/v1/observe") && body.request_id === "req_read") {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_repair",
          deliverable: "code_change",
        },
        accepted_evidence_ids: ["ev_read_impl", "ev_read_test"],
      }),
    }
  }
  if (path.endsWith("/v1/observe")) {
    const kinds = body.observations.map((item) => item.kind).sort()
    if (kinds.join(",") !== "diff,test") {
      throw new Error(`expected diff and test evidence, got ${kinds}`)
    }
    const diff = body.observations.find((item) => item.kind === "diff")
    if (
      !diff.content.includes("--- a/calculator.py") ||
      !diff.content.includes("+++ b/calculator.py")
    ) {
      throw new Error(`diff lacks a verifier-readable path: ${diff.content}`)
    }
    passiveEvidenceAccepted = true
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_repair",
          deliverable: "code_change",
        },
        accepted_evidence_ids: ["ev_diff", "ev_test"],
      }),
    }
  }
  if (path.endsWith("/v1/complete")) {
    submitted = body
    if (!passiveEvidenceAccepted) {
      throw new Error("completion raced ahead of accepted diff/test evidence")
    }
    if (!body.answer.includes("Host verification passed")) {
      throw new Error(`uncertified completion wording: ${body.answer}`)
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        session_id: "vx_repair",
        answer: body.answer,
      }),
    }
  }
  throw new Error(`unexpected URL ${url}`)
}
const client = {
  session: {
    messages: async () => ({
      data: [{
        info: { role: "user" },
        parts: [{
          type: "text",
          text: (
            "Fix add in calculator.py so test_calculator.py passes. " +
            "Do not change the test. Run python3 -m pytest -q " +
            "test_calculator.py after the edit and report the verified result."
          ),
        }],
      }],
    }),
  },
  file: {
    read: async ({ query }) => ({
      data: {
        type: "text",
        content: await readFile(`${directory}/${query.path}`, "utf8"),
      },
    }),
  },
  app: { log: async () => ({}) },
}
const hostShell = (literals, ...values) => {
  let cwd = directory
  const chain = {
    cwd(value) {
      cwd = value
      return chain
    },
    quiet() {
      return chain
    },
    nothrow() {
      return chain
    },
    then(resolve, reject) {
      try {
        let completed
        if (literals[0].includes("python3 -I -c")) {
          repairExecutions += 1
          completed = spawnSync(
            "python3",
            ["-I", "-c", values[0], values[1], values[2], values[3]],
            { cwd, encoding: "utf8" },
          )
        } else if (literals[0].includes("/bin/sh -lc")) {
          testExecutions += 1
          completed = spawnSync(
            "/bin/sh",
            ["-lc", values[0]],
            { cwd, encoding: "utf8" },
          )
        } else {
          throw new Error(`unexpected host command: ${literals.join("{}")}`)
        }
        resolve({
          exitCode: completed.status,
          stdout: completed.stdout,
          stderr: completed.stderr,
        })
      } catch (error) {
        reject(error)
      }
    },
  }
  return chain
}
const hooks = await module.CortheonPlugin({ client, directory, $: hostShell })
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_repair" },
  { system: [] },
)
const changed = await readFile(`${directory}/calculator.py`, "utf8")
if (!changed.includes("return left + right")) {
  throw new Error(`bounded repair was not applied: ${changed}`)
}
const protectedTest = await readFile(`${directory}/test_calculator.py`, "utf8")
if (!protectedTest.includes("assert add(2, 3) == 5")) {
  throw new Error("the protected test was changed")
}
if (repairExecutions !== 1 || testExecutions !== 1) {
  throw new Error(
    `expected one repair and one test, got ${repairExecutions}/${testExecutions}`,
  )
}
if (!submitted) throw new Error("the verified patch was not certified")
const completion = { text: "uncertified tiny-model draft" }
await hooks["experimental.text.complete"](
  {
    sessionID: "host_repair",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (completion.text !== submitted.answer) {
  throw new Error(`certified patch answer was not released: ${completion.text}`)
}
"""

        completed = subprocess.run(
            [
                "node",
                "--input-type=module",
                "-e",
                script,
                str(plugin),
                str(workspace),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
