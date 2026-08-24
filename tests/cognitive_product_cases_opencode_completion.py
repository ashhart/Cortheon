from __future__ import annotations

import subprocess
from pathlib import Path


def test_opencode_adapter_aborts_only_post_certification_compaction():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    script = r"""
import { pathToFileURL } from "node:url"

const module = await import(pathToFileURL(process.argv[1]).href + "?compaction=1")
let aborts = 0
globalThis.fetch = async (url) => {
  const path = new URL(String(url)).pathname
  if (path.endsWith("/v1/start")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_test",
          deliverable: "code_understanding",
        },
        context: { goal: "Check the import." },
        next_action: {
          request: {
            request_id: "req1",
            capability: "grep",
            query: (
              "Search pattern 'os' and path " +
              "'src/cortheon/verifier.py'."
            ),
          },
        },
      }),
    }
  }
  if (path.endsWith("/v1/observe")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_test",
          deliverable: "code_understanding",
        },
        accepted_evidence_ids: ["ev1"],
      }),
    }
  }
  if (path.endsWith("/v1/complete")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        session_id: "vx_test",
        answer: "Yes — import os",
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
            "Does src/cortheon/verifier.py import os? " +
            "Answer yes or no."
          ),
        }],
      }],
    }),
    abort: async () => {
      aborts += 1
      return { data: true }
    },
  },
  file: {
    read: async () => ({
      data: { type: "text", content: "import os\n" },
    }),
  },
  app: { log: async () => ({}) },
}
const hooks = await module.CortheonPlugin({
  client,
  directory: "/tmp",
  $: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
})
const system = { system: [] }
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_session" },
  system,
)
await hooks["experimental.session.compacting"](
  { sessionID: "host_session" },
  { context: [] },
)
if (aborts !== 0) throw new Error("aborted before certification")
const completion = { text: "The model draft." }
await hooks["experimental.text.complete"](
  {
    sessionID: "host_session",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (completion.text !== "Yes — import os") {
  throw new Error(`certified answer was not released: ${completion.text}`)
}
await hooks["experimental.session.compacting"](
  { sessionID: "host_session" },
  { context: [] },
)
if (aborts !== 1) {
  throw new Error(`expected one post-certification abort, observed ${aborts}`)
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


def test_opencode_adapter_releases_runtime_derived_semantic_rule():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    script = r"""
import { pathToFileURL } from "node:url"

const module = await import(pathToFileURL(process.argv[1]).href + "?semantic=1")
const documents = {
  "service_catalog.md": "Checkout is change class Coral and owned by Commerce.\n",
  "change_policy.md": (
    "Coral changes require approval from the Duty Security Officer.\n"
  ),
  "org_directory.md": "Duty Security Officer: Amara Okafor\n",
}
let submitted
globalThis.fetch = async (url, options) => {
  const path = new URL(String(url)).pathname
  if (path.endsWith("/v1/start")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_semantic",
          deliverable: "document_synthesis",
        },
        context: { goal: "Who approves the Checkout change, and why?" },
        next_action: {
          request: {
            request_id: "req_semantic",
            capability: "read_many",
            query: "Read all three named documents.",
            parameters: {
              paths: Object.keys(documents),
              operation: "semantic_join",
            },
          },
        },
      }),
    }
  }
  if (path.endsWith("/v1/observe")) {
    const observed = JSON.parse(options.body)
    const expectedRelations = [
      "Checkout is change class Coral",
      "Coral changes require approval from the Duty Security Officer",
      "Duty Security Officer: Amara Okafor",
    ]
    if (
      observed.observations.length !== expectedRelations.length ||
      !expectedRelations.every((relation, index) =>
        observed.observations[index].content.includes(`\n${relation}`),
      )
    ) {
      throw new Error(
        `semantic relations were altered before observation: ${options.body}`,
      )
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_semantic",
          deliverable: "document_synthesis",
        },
        context: {
          deterministic_derivations: [{
            operation: "semantic_rule",
            nodes: [
              "Checkout",
              "Coral",
              "Duty Security Officer",
              "Amara Okafor",
            ],
            sources: Object.keys(documents),
            premises: [
              {
                subject: "Checkout",
                relation: "classification",
                object: "Coral",
                source: "service_catalog.md",
              },
            ],
            exclude_unless_explicitly_negated: [
              "Product Director",
              "Communications Lead",
            ],
            confidence: "deterministic_conjunctive_rule",
          }],
        },
        cognition: {
          stage: "connect",
          task_frame: {
            requirements: [
              {
                requirement_id: "r1",
                statement: "Read the three documents",
                status: "covered",
              },
              {
                requirement_id: "r2",
                statement: "Explain why the approver is selected",
                status: "unresolved",
              },
            ],
          },
          reasoning_moves: [
            "Check every edge against its separate source before accepting the bridge.",
          ],
          derived_insights: [{
            statement: (
              "Checkout is connected to Amara Okafor through Coral, then " +
              "Duty Security Officer."
            ),
          }],
          decision_rule: "Challenge the derived bridge before finishing.",
        },
        accepted_evidence_ids: ["ev1", "ev2", "ev3"],
      }),
    }
  }
  if (path.endsWith("/v1/complete")) {
    submitted = JSON.parse(options.body)
    const required = [
      "Checkout",
      "Coral",
      "Duty Security Officer",
      "Amara Okafor",
    ]
    if (!required.every((node) => submitted.answer.includes(node))) {
      throw new Error(`incomplete semantic answer: ${submitted.answer}`)
    }
    if (
      submitted.answer.includes("Product Director") ||
      submitted.answer.includes("Communications Lead")
    ) {
      throw new Error(`unrelated semantic branch: ${submitted.answer}`)
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        session_id: "vx_semantic",
        answer: submitted.answer,
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
            "Read service_catalog.md, change_policy.md, and org_directory.md " +
            "as separate documents. Who approves the Checkout change, and why?"
          ),
        }],
      }],
    }),
  },
  file: {
    read: async ({ query }) => ({
      data: { type: "text", content: documents[query.path] },
    }),
  },
  app: { log: async () => ({}) },
}
const hooks = await module.CortheonPlugin({
  client,
  directory: "/tmp",
  $: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
})
const system = { system: [] }
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_semantic" },
  system,
)
const systemPrompt = system.system.join("\n")
if (!systemPrompt.includes("[CORTHEON_MODEL_CONTEXT_V1]")) {
  throw new Error(`model context was not injected: ${systemPrompt}`)
}
if (!systemPrompt.includes("beyond its weights")) {
  throw new Error(`model capability context was not injected: ${systemPrompt}`)
}
if (!systemPrompt.includes("Cortheon adaptive cognition: stage=connect")) {
  throw new Error(`adaptive cognition was not injected: ${systemPrompt}`)
}
if (!systemPrompt.includes("Candidate cross-source inference: Checkout")) {
  throw new Error(`derived insight was not injected: ${systemPrompt}`)
}
if (
  !systemPrompt.includes(
    "Unresolved task contract: r2: Explain why the approver is selected",
  ) ||
  systemPrompt.includes("Unresolved task contract: r1:")
) {
  throw new Error(`requirement contract was not bounded correctly: ${systemPrompt}`)
}
const completion = {
  text: "The tiny model omitted the intermediate role.",
}
await hooks["experimental.text.complete"](
  {
    sessionID: "host_semantic",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (!submitted) throw new Error("Cortheon completion was not attempted")
if (completion.text !== submitted.answer) {
  throw new Error(`certified semantic answer was not released: ${completion.text}`)
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
