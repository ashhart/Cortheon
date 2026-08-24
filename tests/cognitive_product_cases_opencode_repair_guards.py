from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["failing_test", "symlink_target"])
def test_opencode_bounded_repair_fails_closed_and_preserves_files(scenario):
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        implementation = workspace / "calculator.py"
        owned = workspace / "owned.py"
        original = "def add(left: int, right: int) -> int:\n    return left - right\n"
        if scenario == "symlink_target":
            owned.write_text(original)
            implementation.symlink_to(owned.name)
        else:
            implementation.write_text(original)
        test_file = workspace / "test_calculator.py"
        test_file.write_text(
            "from calculator import add\n\n\n"
            "def test_adds_two_numbers() -> None:\n"
            "    assert add(2, 3) == 5\n"
            + (
                "    assert False, 'force rollback after the candidate edit'\n"
                if scenario == "failing_test"
                else ""
            )
        )
        script = r"""
import { pathToFileURL } from "node:url"
import { readFile } from "node:fs/promises"
import { spawnSync } from "node:child_process"

const module = await import(pathToFileURL(process.argv[1]).href + "?guards=1")
const directory = process.argv[2]
const scenario = process.argv[3]
let repairExecutions = 0
let testExecutions = 0
let completionAttempts = 0
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
          session_id: "vx_guarded_repair",
          deliverable: "code_change",
        },
        context: { goal: body.goal },
        next_action: {
          request: {
            request_id: "req_read",
            capability: "read_many",
            query: "Read implementation and protected test.",
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
          session_id: "vx_guarded_repair",
          deliverable: "code_change",
        },
        accepted_evidence_ids: ["ev_impl", "ev_test"],
      }),
    }
  }
  if (path.endsWith("/v1/observe")) {
    throw new Error("a rejected repair must not submit diff/test evidence")
  }
  if (path.endsWith("/v1/complete")) {
    completionAttempts += 1
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_guarded_repair",
          deliverable: "code_change",
        },
        next_action: {
          request: {
            request_id: "req_verified_patch",
            capability: "diff",
            query: "Provide a verified implementation diff and passing test.",
            parameters: {},
          },
        },
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
  { sessionID: "host_guarded_repair" },
  { system: [] },
)
const target =
  scenario === "symlink_target" ? `${directory}/owned.py` : `${directory}/calculator.py`
const after = await readFile(target, "utf8")
if (!after.includes("return left - right")) {
  throw new Error(`rejected repair was not restored exactly: ${after}`)
}
const protectedTest = await readFile(`${directory}/test_calculator.py`, "utf8")
if (!protectedTest.includes("assert add(2, 3) == 5")) {
  throw new Error("protected test changed")
}
if (scenario === "failing_test") {
  if (repairExecutions !== 2 || testExecutions !== 1) {
    throw new Error(
      `expected edit, test, rollback; got ${repairExecutions}/${testExecutions}`,
    )
  }
} else if (repairExecutions !== 1 || testExecutions !== 0) {
  throw new Error(
    `symlink repair was not refused before testing: ` +
      `${repairExecutions}/${testExecutions}`,
  )
}
if (completionAttempts !== 0) {
  throw new Error("repair path attempted completion before the model response")
}
const completion = { text: "claim the patch succeeded" }
await hooks["experimental.text.complete"](
  {
    sessionID: "host_guarded_repair",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (completionAttempts !== 1) {
  throw new Error("the runtime gate did not inspect the unsupported model claim")
}
if (!completion.text.startsWith("[Cortheon withheld")) {
  throw new Error(`unsupported completion escaped: ${completion.text}`)
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
                scenario,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
