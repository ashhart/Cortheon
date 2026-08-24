

import { createHash } from "node:crypto"
import {evaluatorControl} from "./evaluator_control.js"

const initialEnvironment = {
  profile: evaluatorControl.present
    ? evaluatorControl.value?.evaluation_profile
    : typeof process !== "undefined" && process.env.CORTHEON_EVALUATOR_PROFILE,
  token: evaluatorControl.present
    ? evaluatorControl.value?.cognitive_token || ""
    : typeof process !== "undefined" ? process.env.CORTHEON_COGNITIVE_TOKEN || "" : "",
  runtimeURL: typeof process !== "undefined" ? process.env.CORTHEON_RUNTIME_URL : undefined,
}
if (typeof process !== "undefined") {
  delete process.env.CORTHEON_EVALUATOR_PROFILE
  delete process.env.CORTHEON_COGNITIVE_TOKEN
  delete process.env.CORTHEON_EVALUATOR_MAX_STEPS
  delete process.env.CORTHEON_AUTO_ENABLE
  delete process.env.CORTHEON_BENCHMARK_CAPTURE_CANDIDATE
  delete process.env.CORTHEON_MAX_HOST_TOOL_CALLS
}

// Module-level singletons shared across every plugin instance created from
// this module graph. Keeping them here preserves the pre-split semantics
// where repeated imports of the facade observed one shared state set.
const investigations = new Map()
const investigationStarts = new Map()
const completionAttempts = new Map()
// Append-only record of certified answers per host session. Concurrent
// hooks can resume with a stale state snapshot and clobber the completed
// entry (lost update); a certification that the runtime already issued and
// discarded must survive that clobber, or a verified answer is silently
// downgraded to a withhold.
const certifiedAnswers = new Map()

const debugDump = (payload) => {
  try {
    fetch("http://127.0.0.1:8746/debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {})
  } catch {}
}

const sessionLocks = new Map()
async function withSessionLock(hostSessionID, task) {
  const previous = sessionLocks.get(hostSessionID) || Promise.resolve()
  const current = previous.then(() => task())
  sessionLocks.set(
    hostSessionID,
    current.then(
      () => {},
      () => {},
    ),
  )
  return current
}

const maxEvidenceCharacters = 1500
const hostEvidenceTools = new Set([
  "read",
  "grep",
  "glob",
  "bash",
  "webfetch",
  "websearch",
])
const postCertificationReadTools = new Set([
  "read",
  "grep",
  "glob",
  "webfetch",
  "websearch",
])
const workspaceMutationTools = new Set(["edit", "write", "patch", "apply_patch"])
const leaseSeconds = 30
const heartbeatIntervalMs = 10_000
const semanticStopwords = new Set([
  "about",
  "after",
  "also",
  "answer",
  "before",
  "between",
  "could",
  "current",
  "catalog",
  "directory",
  "document",
  "documents",
  "every",
  "from",
  "have",
  "into",
  "named",
  "only",
  "question",
  "read",
  "register",
  "roster",
  "should",
  "source",
  "that",
  "their",
  "then",
  "these",
  "they",
  "this",
  "through",
  "what",
  "when",
  "where",
  "which",
  "with",
  "would",
])

const modelContext = `
[CORTHEON_MODEL_CONTEXT_V1]
Cortheon is a lightweight reasoning runtime that gives this local model capabilities
beyond its weights. Use host tools to fetch current evidence, test explanations,
connect facts across sources, and verify work. The host runs tools; the model answers.
Follow Cortheon's current instruction, never invent evidence, and stop when released.
`.trim()

const protocol = `${modelContext}

Cortheon is the completion gate for this task.
1. Call cortheon_start once. Normally omit task_kind so Cortheon infers it.
2. Obey next_action. For a harness_tool action, ACTUALLY call the host read,
   grep, bash, web, or other named capability. capability="grep" means call
   the host grep tool, not read. Never invent tool output.
3. Call cortheon_observe with the exact request_id and a small excerpt. On the
   first observation omit supports/contradicts because hypothesis ids do not
   exist yet.
4. Prefer cortheon_complete: give compact public hypotheses resolved against
   real ev* evidence ids, evidence-linked claims, the exact answer, and the
   completion evidence ids. If Cortheon requests more evidence, use the host tool,
   observe it, and retry cortheon_complete.
5. An answer is deliverable only when cortheon_complete or cortheon_finish returns
   status="complete". Never emit an answer after a failed or pending gate.
`.trim()

function boundedHostOutput(value) {
  const text = String(value || "")
    .replace(/\u001b\[[0-9;]*m/g, "")
    .trim()
  if (text.length <= maxEvidenceCharacters) return text
  const headLength = Math.floor(maxEvidenceCharacters * 0.72)
  const tailLength = maxEvidenceCharacters - headLength
  return (
    text.slice(0, headLength).trimEnd() +
    "\n\n[CORTHEON BOUNDED OUTPUT: middle omitted]\n\n" +
    text.slice(-tailLength).trimStart()
  )
}

function focusedSessionDiff(diffs) {
  const sections = []
  for (const diff of Array.isArray(diffs) ? diffs.slice(0, 8) : []) {
    if (!diff || typeof diff.file !== "string") continue
    const before = String(diff.before || "").split("\n")
    const after = String(diff.after || "").split("\n")
    let prefix = 0
    while (
      prefix < before.length &&
      prefix < after.length &&
      before[prefix] === after[prefix]
    ) {
      prefix += 1
    }
    let suffix = 0
    while (
      suffix < before.length - prefix &&
      suffix < after.length - prefix &&
      before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
    ) {
      suffix += 1
    }
    const oldEnd = Math.max(prefix, before.length - suffix)
    const newEnd = Math.max(prefix, after.length - suffix)
    const oldLines = before.slice(prefix, oldEnd).slice(0, 12)
    const newLines = after.slice(prefix, newEnd).slice(0, 12)
    sections.push(
      [
        `${diff.file} (+${Number(diff.additions || 0)} -${Number(diff.deletions || 0)})`,
        `--- a/${diff.file}`,
        `+++ b/${diff.file}`,
        `@@ line ${prefix + 1} @@`,
        ...oldLines.map((line) => `- ${line}`),
        ...newLines.map((line) => `+ ${line}`),
      ].join("\n"),
    )
  }
  return boundedHostOutput(sections.join("\n\n"))
}

function focusedTextDiff(file, beforeText, afterText) {
  const before = String(beforeText || "").split("\n")
  const after = String(afterText || "").split("\n")
  if (before.length === after.length && before.every((line, i) => line === after[i])) {
    return ""
  }
  let prefix = 0
  while (
    prefix < before.length &&
    prefix < after.length &&
    before[prefix] === after[prefix]
  ) {
    prefix += 1
  }
  let suffix = 0
  while (
    suffix < before.length - prefix &&
    suffix < after.length - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix += 1
  }
  const oldEnd = Math.max(prefix, before.length - suffix)
  const newEnd = Math.max(prefix, after.length - suffix)
  return boundedHostOutput(
    [
      `${file}`,
      `--- a/${file}`,
      `+++ b/${file}`,
      `@@ line ${prefix + 1} @@`,
      ...before.slice(prefix, oldEnd).slice(0, 20).map((line) => `- ${line}`),
      ...after.slice(prefix, newEnd).slice(0, 20).map((line) => `+ ${line}`),
    ].join("\n"),
  )
}

export {
  adapterEvaluationProfile,
  prependSystemGuidance,
  decodePayload,
  decodeToolPayload,
  sessionID,
  pendingRequest,
  pendingEvidenceRequest,
  evidenceKind,
  evidenceStatus,
  evaluationProfile,
  permittedHostTool,
  safeHostArguments,
  investigations,
  investigationStarts,
  completionAttempts,
  certifiedAnswers,
  debugDump,
  withSessionLock,
  maxEvidenceCharacters,
  operatorEnabled,
  hostEvidenceTools,
  postCertificationReadTools,
  workspaceMutationTools,
  leaseSeconds,
  heartbeatIntervalMs,
  semanticStopwords,
  modelContext,
  protocol,
  boundedHostOutput,
  focusedSessionDiff,
  focusedTextDiff,
  initialEnvironment,
}

function prependSystemGuidance(output, guidance) {
  if (!Array.isArray(output?.system)) return
  if (typeof output.system[0] === "string" && output.system[0].trim()) {
    output.system[0] = `${guidance}\n\n${output.system[0]}`
    return
  }
  output.system.unshift(guidance)
}

function decodePayload(value) {
  if (value && typeof value === "object") {
    if (Array.isArray(value)) {
      for (const item of value) {
        const decoded = decodePayload(item)
        if (decoded && !Array.isArray(decoded)) return decoded
      }
      return undefined
    }
    if (Array.isArray(value.content)) {
      for (const item of value.content) {
        const decoded = decodePayload(item?.text)
        if (decoded) return decoded
      }
    }
    if (typeof value.text === "string") return decodePayload(value.text)
    if (typeof value.output === "string") return decodePayload(value.output)
    return value
  }
  if (typeof value !== "string" || !value.trim()) return undefined
  const text = value.trim()
  try {
    const value = JSON.parse(text)
    if (value && typeof value === "object") return value
  } catch {
    const start = text.indexOf("{")
    const end = text.lastIndexOf("}")
    if (start < 0 || end <= start) return undefined
    try {
      const embedded = JSON.parse(text.slice(start, end + 1))
      return embedded && typeof embedded === "object" ? embedded : undefined
    } catch {
      return undefined
    }
  }
  return undefined
}

function decodeToolPayload(output) {
  for (const candidate of [
    output?.structuredContent,
    output?.content,
    output?.output,
  ]) {
    const payload = decodePayload(candidate)
    if (
      payload &&
      (payload.session ||
        payload.session_id ||
        payload.status ||
        payload.next_action)
    ) {
      return payload
    }
  }
  return undefined
}

function sessionID(payload) {
  const nested = payload?.session?.session_id
  if (typeof nested === "string") return nested
  return typeof payload?.session_id === "string" ? payload.session_id : undefined
}

function pendingRequest(payload) {
  const request = payload?.next_action?.request
  return request && typeof request.request_id === "string"
    ? request.request_id
    : undefined
}

function pendingEvidenceRequest(payload) {
  const request = payload?.next_action?.request
  return request && typeof request.request_id === "string" ? request : undefined
}

function evidenceKind(tool) {
  if (["read", "grep", "glob"].includes(tool)) return "code"
  if (["webfetch", "websearch"].includes(tool)) return "web"
  return "command"
}

function evidenceStatus(tool) {
  return ["grep", "glob", "bash"].includes(tool) ? "verified" : "observed"
}

function permittedHostTool(tool, state) {
  const capability = state?.request?.capability
  if (!capability) return true
  if (capability === "grep") return tool === "grep"
  if (
    capability === "search" &&
    state.deliverable === "code_understanding"
  ) {
    return tool === "grep"
  }
  if (capability === "search" && state.deliverable === "research_answer") {
    return ["websearch", "webfetch"].includes(tool)
  }
  if (capability === "fetch" && state.deliverable === "research_answer") {
    return tool === "webfetch"
  }
  if (capability === "search") {
    return ["grep", "glob", "read", "websearch", "webfetch"].includes(tool)
  }
  if (capability === "read") return ["read", "grep"].includes(tool)
  if (capability === "test" || capability === "diff") return tool === "bash"
  return hostEvidenceTools.has(tool)
}

function safeHostArguments(tool, args) {
  if (tool === "grep") {
    return {
      pattern: String(args?.pattern || "").slice(0, 300),
      path: String(args?.path || args?.include || "").slice(0, 300),
    }
  }
  if (tool === "glob") {
    return {
      pattern: String(args?.pattern || "").slice(0, 300),
      path: String(args?.path || "").slice(0, 300),
    }
  }
  if (tool === "read") {
    return { filePath: String(args?.filePath || "").slice(0, 500) }
  }
  if (tool === "websearch") {
    return { query: String(args?.query || "").slice(0, 500) }
  }
  if (tool === "webfetch") {
    return { url: String(args?.url || "").slice(0, 1000) }
  }
  return {}
}
const operatorKeys = [
  "retrieval", "verification", "hypothesis_framing",
  "discriminating_evidence", "contradiction_revision",
  "cross_source_derivation", "adaptive_stopping",
]

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value && typeof value === "object") {
    return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`
  }
  return JSON.stringify(value)
}

function evaluationProfile() {
  const raw = initialEnvironment.profile
  if (!raw) return undefined
  try {
    const value = typeof raw === "string" ? JSON.parse(raw) : raw
    const config = value?.config
    const operators = config?.operators
    if (
      value?.schema_version !== 1 || config?.schema_version !== 1 || !operators ||
      operatorKeys.some((key) => typeof operators[key] !== "boolean") ||
      Object.keys(operators).length !== operatorKeys.length ||
      typeof config.intercepts_final !== "boolean" ||
      typeof config.cleanup_before_answer !== "boolean" ||
      config.hard_budgets_enforced !== true || config.sticky_terminal_safety !== true ||
      config.transport_failure_fails_open !== true ||
      !/^[0-9a-f]{64}$/.test(value.config_sha256) ||
      !/^[0-9a-f]{64}$/.test(value.implementation_sha256) ||
      !/^[0-9a-f]{32}$/.test(value.nonce)
    ) return undefined
    const digest = createHash("sha256").update(canonical(config)).digest("hex")
    if (digest !== value.config_sha256) return undefined
    return value
  } catch {
    return undefined
  }
}

function operatorEnabled(operator) {
  return evaluationProfile()?.config.operators[operator] ?? true
}

function adapterEvaluationProfile() {
  const profile = evaluationProfile()
  if (!profile) return undefined
  return {
    ...profile,
    adapter_receipt: {
      schema_version: 1,
      host: "opencode",
      control_transport: evaluatorControl.present ? "fd" : "env",
      config_sha256: profile.config_sha256,
      nonce: profile.nonce,
      operators: { ...profile.config.operators },
    },
  }
}
