from __future__ import annotations

import subprocess
from pathlib import Path


def test_opencode_adapter_preserves_each_read_in_ambiguity_completion():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    script = r"""
import { pathToFileURL } from "node:url"

const module = await import(pathToFileURL(process.argv[1]).href + "?ambiguity=1")
const documents = {
  "roster_a.md": "Roster A names Priya Nair as the Nimbus owner.\n",
  "roster_b.md": "Roster B names Elena Voss as the Nimbus owner.\n",
  "authority.md": (
    "Resolve ownership only with an effective date, supersession marker, or " +
    "higher authority level.\n"
  ),
}
let submitted
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
          session_id: "vx_ambiguity",
          deliverable: "document_synthesis",
        },
        context: {
          goal: (
            "Who currently owns Nimbus? Use the documents, but do not invent a " +
            "tie-break. State both live alternatives and the exact evidence needed."
          ),
        },
        next_action: {
          request: {
            request_id: "req_reads",
            capability: "read_many",
            query: "Read the three records.",
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
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_ambiguity",
          deliverable: "document_synthesis",
        },
        accepted_evidence_ids: ["ev1", "ev2", "ev3"],
      }),
    }
  }
  if (path.endsWith("/v1/complete")) {
    submitted = body
    for (const expected of [
      "Priya Nair",
      "Elena Voss",
      "effective date",
      "ambiguous",
      "clarification",
    ]) {
      if (!submitted.answer.toLowerCase().includes(expected.toLowerCase())) {
        throw new Error(`completion omitted ${expected}: ${submitted.answer}`)
      }
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        session_id: "vx_ambiguity",
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
            "Who currently owns Nimbus? Use roster_a.md, roster_b.md, and " +
            "authority.md, but do not invent a tie-break. State both live alternatives."
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
  $: () => {
    throw new Error("shell access is not expected")
  },
})
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_ambiguity" },
  { system: [] },
)
const completion = { text: "I cannot tell." }
await hooks["experimental.text.complete"](
  {
    sessionID: "host_ambiguity",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (!submitted || completion.text !== submitted.answer) {
  throw new Error(`certified ambiguity answer was not released: ${completion.text}`)
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


def test_opencode_adapter_discovers_then_reads_unnamed_documents():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    script = r"""
import { pathToFileURL } from "node:url"

const module = await import(pathToFileURL(process.argv[1]).href + "?discovery=1")
const documents = {
  "docs/product_registry.md": (
    "| Product | Runtime service |\n| --- | --- |\n| Order Console | Helios |\n"
  ),
  "docs/integration_matrix.md": (
    "| Service | Critical dataset |\n| --- | --- |\n| Helios | Ledger Aurora |\n"
  ),
  "docs/data_stewards.md": (
    "| Dataset | Current steward |\n| --- | --- |\n" +
    "| Ledger Aurora | Imani Brooks |\n"
  ),
  "README.md": "# Installation\nGeneral package instructions.\n",
}
let observations = 0
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
          session_id: "vx_discovery",
          deliverable: "document_synthesis",
        },
        context: {
          goal: (
            "Search across project documents and connect the evidence to identify " +
            "the steward of the critical dataset used by Order Console."
          ),
        },
        next_action: {
          request: {
            request_id: "req_discovery",
            capability: "search",
            query: "Find candidate project documents.",
            parameters: {
              operation: "document_discovery",
              max_candidates: 6,
            },
          },
        },
      }),
    }
  }
  if (path.endsWith("/v1/observe")) {
    observations += 1
    const body = JSON.parse(options.body)
    if (observations === 1) {
      const content = body.observations[0]?.content || ""
      if (
        body.request_id !== "req_discovery" ||
        !content.includes('"tool":"glob"') ||
        !content.includes("docs/product_registry.md")
      ) {
        throw new Error(`invalid discovery evidence: ${options.body}`)
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          status: "active",
          session: {
            session_id: "vx_discovery",
            deliverable: "document_synthesis",
          },
          context: { goal: "Find the Order Console dataset steward." },
          next_action: {
            request: {
              request_id: "req_reads",
              capability: "read_many",
              query: "Read the bounded candidates.",
              parameters: {
                paths: Object.keys(documents),
                operation: "semantic_join",
                discovered: true,
              },
            },
          },
        }),
      }
    }
    if (
      body.request_id !== "req_reads" ||
      body.observations.length !== Object.keys(documents).length ||
      !body.observations.every((item) => item.content.includes('"tool":"read"'))
    ) {
      throw new Error(`invalid discovered reads: ${options.body}`)
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_discovery",
          deliverable: "document_synthesis",
        },
        context: {
          deterministic_derivations: [{
            operation: "semantic_chain",
            nodes: [
              "Order Console",
              "Helios",
              "Ledger Aurora",
              "Imani Brooks",
            ],
            sources: [
              "docs/product_registry.md",
              "docs/integration_matrix.md",
              "docs/data_stewards.md",
            ],
            confidence: "deterministic_relational_match",
          }],
        },
        accepted_evidence_ids: ["ev2", "ev3", "ev4", "ev5"],
      }),
    }
  }
  if (path.endsWith("/v1/complete")) {
    submitted = JSON.parse(options.body)
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        session_id: "vx_discovery",
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
            "Search across project documents and connect the evidence to identify " +
            "the steward of the critical dataset used by Order Console."
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
const shellResult = {
  cwd() { return this },
  quiet() { return this },
  nothrow() {
    return Promise.resolve({
      exitCode: 0,
      stdout: Object.keys(documents).join("\n"),
      stderr: "",
    })
  },
}
const hooks = await module.CortheonPlugin({
  client,
  directory: "/tmp",
  $: () => shellResult,
})
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_discovery" },
  { system: [] },
)
if (observations !== 2) {
  throw new Error(`expected discovery and read observations, got ${observations}`)
}
const completion = { text: "The small model did not inspect the files." }
await hooks["experimental.text.complete"](
  {
    sessionID: "host_discovery",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (!submitted || !completion.text.includes("Imani Brooks")) {
  throw new Error(`discovered conclusion was not certified: ${completion.text}`)
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
