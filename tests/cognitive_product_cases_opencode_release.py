from __future__ import annotations

import subprocess
from pathlib import Path


def test_opencode_adapter_auto_fetches_and_certifies_current_release():
    plugin = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"
    script = r"""
import { pathToFileURL } from "node:url"

const module = await import(pathToFileURL(process.argv[1]).href + "?research=1")
process.env.CORTHEON_PLUGIN_DEBUG = "1"
const github = "https://github.com/example/widget/releases/latest"
const pypi = "https://pypi.org/project/widget/"
let hostFetches = 0
let submitted
globalThis.fetch = async (url, options) => {
  const target = String(url)
  if (target.startsWith(github)) {
    hostFetches += 1
    return new Response(
      (
        "<html><head><title>Release 9.9.1 · example/widget · GitHub</title>" +
        "</head><body><relative-time datetime=\"2026-07-26T09:30:00Z\">" +
        "26 Jul 2026</relative-time></body></html>"
      ),
      { status: 200, headers: { "content-type": "text/html" } },
    )
  }
  if (target.startsWith("https://pypi.org/rss/project/widget/releases.xml")) {
    hostFetches += 1
    return new Response(
      (
        "<rss><channel><item><title>9.9.1</title>" +
        "<pubDate>Sun, 26 Jul 2026 09:35:00 GMT</pubDate>" +
        "</item></channel></rss>"
      ),
      { status: 200, headers: { "content-type": "application/rss+xml" } },
    )
  }
  const path = new URL(target).pathname
  const body = options?.body ? JSON.parse(options.body) : {}
  if (path.endsWith("/v1/start")) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_research",
          deliverable: "research_answer",
        },
        context: { goal: body.goal },
        next_action: {
          request: {
            request_id: "req_research",
            capability: "search",
            query: "Check the two named current release sources.",
            parameters: {
              purpose: "contradiction_check",
              minimum_independent_origins: 2,
              require_primary_fetch: true,
              require_contradiction_check: true,
            },
          },
        },
      }),
    }
  }
  if (path.endsWith("/v1/observe") && body.request_id === "req_research") {
    if (
      body.observations.length !== 2 ||
      !body.observations.every(
        (item) =>
          item.kind === "web" &&
          item.purpose === "contradiction_check" &&
          item.retrieved_at,
      ) ||
      !body.observations.some((item) => item.published_at) ||
      body.observations.find((item) => item.url === pypi)?.published_at !==
        "2026-07-26"
    ) {
      throw new Error(`invalid contradiction evidence: ${options.body}`)
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_research",
          deliverable: "research_answer",
        },
        context: {
          deterministic_derivations: [{
            operation: "release_version",
            value: "9.9.1",
            sources: [github, pypi],
            independent_origins: 2,
            confidence: "independent_live_sources",
          }],
        },
        accepted_evidence_ids: ["ev_github", "ev_pypi"],
      }),
    }
  }
  if (path.endsWith("/v1/observe")) {
    if (
      body.observations.length !== 1 ||
      body.observations[0].purpose !== "primary_fetch"
    ) {
      throw new Error(`invalid primary evidence: ${options.body}`)
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "active",
        session: {
          session_id: "vx_research",
          deliverable: "research_answer",
        },
        context: {
          deterministic_derivations: [{
            operation: "release_version",
            value: "9.9.1",
            sources: [github, pypi],
            independent_origins: 2,
            confidence: "independent_live_sources",
          }],
        },
        accepted_evidence_ids: ["ev_primary"],
      }),
    }
  }
  if (path.endsWith("/v1/complete")) {
    submitted = body
    if (
      !body.answer.includes("9.9.1") ||
      !body.answer.includes(github) ||
      !body.answer.includes(pypi) ||
      !body.answer.includes("agree")
    ) {
      throw new Error(`incomplete research answer: ${body.answer}`)
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "complete",
        session_id: "vx_research",
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
            `Research the latest released widget version. Fetch ${github} and ` +
            `independently corroborate it with ${pypi}. Check freshness and ` +
            "contradictions. Include clickable URLs from both origins."
          ),
        }],
      }],
    }),
  },
  app: {
    log: async ({ body }) => {
      console.error(`[cortheon-test] ${body.message}`)
      return {}
    },
  },
}
const hooks = await module.CortheonPlugin({
  client,
  directory: "/tmp",
  $: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
})
await hooks["experimental.chat.system.transform"](
  { sessionID: "host_research" },
  { system: [] },
)
if (hostFetches !== 2) {
  throw new Error(`expected two bounded public fetches, got ${hostFetches}`)
}
if (!submitted) throw new Error("research evidence was not certified")
const completion = { text: "stale tiny-model guess" }
await hooks["experimental.text.complete"](
  {
    sessionID: "host_research",
    messageID: "message",
    partID: "part",
  },
  completion,
)
if (completion.text !== submitted.answer) {
  throw new Error(`certified research answer was not released: ${completion.text}`)
}
"""

    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(plugin)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
