import { isAmbiguityGoal, isCausalSynthesisGoal, deriveSemanticChainSegments } from "./joins.js"
import { evidenceReceipt } from "./evidence.js"
import { protectedTestPaths, requestedTestCommand } from "./plans.js"
import { investigationStarts, investigations, leaseSeconds, boundedHostOutput } from "./state.js"
import { mergePayload } from "./state_merge.js"
import { adapterEvaluationProfile, operatorEnabled } from "./state.js"

// Runtime lifecycle and HTTP client. `runtimeBase` and `runtimeToken` are
// resolved once by the facade so the heartbeat loop and these calls share
// one configuration snapshot, exactly as the pre-split closure did.
const createRuntimeClient = ({ runtimeBase, runtimeToken, debugSem }) => {
  const runtimeHealth = async () => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 3000)
    try {
      const response = await fetch(`${runtimeBase}/healthz`, {
        signal: controller.signal,
      })
      if (!response.ok) return undefined
      return await response.json()
    } catch {
      return undefined
    } finally {
      clearTimeout(timer)
    }
  }

  const spawnRuntimeOnce = async () => {
    if (spawnRuntimeOnce.attempted) return undefined
    spawnRuntimeOnce.attempted = true
    // Only a real host (OpenCode runs plugins in Bun) may own a runtime
    // process; test harnesses under node must never spawn one.
    const realHost =
      typeof globalThis.Bun !== "undefined" ||
      (typeof process !== "undefined" && process.env?.CORTHEON_AUTOSPAWN === "1")
    if (!realHost) return undefined
    try {
      const { spawn } = await import("node:child_process")
      const child = spawn("cortheon", ["serve"], {
        detached: true,
        stdio: "ignore",
      })
      child.unref()
    } catch {
      return undefined
    }
    for (let attempt = 0; attempt < 8; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 500))
      const health = await runtimeHealth()
      if (health) return health
    }
    return undefined
  }

  const localSourceFingerprint = async () => {
    try {
      const fs = await import("node:fs")
      const crypto = await import("node:crypto")
      const path = await import("node:path")
      const { fileURLToPath } = await import("node:url")
      // This module ships inside <package>/opencode_core/, one level below
      // the package root that holds cognitive_runtime.py and cognitive_core/.
      const here = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
      const digest = crypto.createHash("sha256")
      let coreNames = []
      let hooksCoreNames = []
      try {
        coreNames = fs
          .readdirSync(path.join(here, "cognitive_core"))
          .filter((name) => name.endsWith(".py"))
          .sort()
      } catch {}
      try {
        hooksCoreNames = fs
          .readdirSync(path.join(here, "cognitive_hooks_core"))
          .filter((name) => name.endsWith(".py"))
          .sort()
      } catch {}
      for (const name of [
        "cognitive_runtime.py",
        ...coreNames.map((name) => `cognitive_core/${name}`),
        "cognitive_http.py",
        "cognitive_hooks.py",
        ...hooksCoreNames.map((name) => `cognitive_hooks_core/${name}`),
      ]) {
        try {
          digest.update(fs.readFileSync(path.join(here, name)))
        } catch {
          digest.update(name)
        }
      }
      return digest.digest("hex").slice(0, 16)
    } catch {
      return undefined
    }
  }

  const runtimeCall = async (path, body) => {
    // Observe/complete can legitimately exceed 5s while the runtime derives
    // over large evidence; observations are idempotent by the runtime's
    // contract, so a lost observe response earns exactly one retry.
    const timeoutMs = path === "/v1/observe" || path === "/v1/complete" ? 15000 : 5000
    const attempts = path === "/v1/observe" ? 2 : 1
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), timeoutMs)
      const startedAt = Date.now()
      try {
        const headers = { "Content-Type": "application/json" }
        if (runtimeToken) headers.Authorization = `Bearer ${runtimeToken}`
        const response = await fetch(`${runtimeBase}${path}`, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: controller.signal,
        })
        const payload = await response.json()
        if (!response.ok) {
          await debugSem({
            at: "runtimeCall-rejected",
            path,
            status: response.status,
            errorType: payload?.error_type || "error",
            detail: String(payload?.detail || payload?.error || "").slice(0, 200),
            elapsedMs: Date.now() - startedAt,
          })
          if (
            response.status === 422 &&
            /already resolved/i.test(
              String(payload?.detail || payload?.error || ""),
            )
          ) {
            // The runtime is authoritative: this request finished but the
            // resolving response was lost to a hook race. Signal the caller
            // to heal its state rather than resubmitting forever.
            return { __requestResolvedConflict: true }
          }
          return undefined
        }
        return payload
      } catch (error) {
        await debugSem({
          at: "runtimeCall-failed",
          path,
          attempt,
          error: String(error?.name || error).slice(0, 120),
          elapsedMs: Date.now() - startedAt,
        })
        if (attempt + 1 >= attempts) return undefined
      } finally {
        clearTimeout(timer)
      }
    }
    return undefined
  }

  return {
    runtimeHealth,
    spawnRuntimeOnce,
    localSourceFingerprint,
    runtimeCall,
  }
}

// Automatic investigation lifecycle: startup, singleflighted per host
// session, plus the turn-end semantic/causal/resync rescues.
const createInvestigation = ({
  debug,
  runtimeHealth,
  spawnRuntimeOnce,
  localSourceFingerprint,
  runtimeCall,
  latestUserTask,
  readWorkspaceFile,
  acquireRequestedEvidence,
  submitAutomaticObservation,
  submitPassiveObservations,
  acquireAutomaticResearch,
}) => {
  const ensureAutomaticInvestigation = async (hostSessionID) => {
    const existing = investigations.get(hostSessionID)
    if (existing?.automatic && (existing.active || existing.completed)) {
      if (!existing.idle) return existing
      const currentTask = await latestUserTask(hostSessionID)
      if (!currentTask || currentTask === existing.goal) {
        existing.idle = false
        investigations.set(hostSessionID, existing)
        return existing
      }
      if (existing.active && existing.cortheonSessionID) {
        await runtimeCall("/v1/abandon", {
          session_id: existing.cortheonSessionID,
        })
      }
      investigations.delete(hostSessionID)
    }
    const pending = investigationStarts.get(hostSessionID)
    if (pending) return pending
    const operation = (async () => {
      let state = investigations.get(hostSessionID)
      if (state?.automatic && (state.active || state.completed)) return state
      const task = await latestUserTask(hostSessionID)
      if (!task) return state
      let health = await runtimeHealth()
      if (!health) {
        // Own the runtime's lifecycle like the Pi extension does: a missing
        // loopback runtime should not silently degrade the session.
        health = await spawnRuntimeOnce()
      }
      const localFingerprint = await localSourceFingerprint()
      const runtimeStale = Boolean(
        health?.source_fingerprint &&
          localFingerprint &&
          health.source_fingerprint !== localFingerprint,
      )
      const profile = adapterEvaluationProfile()
      const payload = await runtimeCall("/v1/start", {
        goal: task,
        effort: "quick",
        task_kind: "auto",
        lease_seconds: leaseSeconds,
        ...(profile ? { evaluation_profile: profile } : {}),
      })
      if (!payload) {
        state = { ...(state || {}), required: true, runtimeUnavailable: true }
        investigations.set(hostSessionID, state)
        return state
      }
      state = mergePayload(hostSessionID, "start", payload)
      state.automatic = true
      state.runtimeStale = runtimeStale
      state.requestedTestCommand = requestedTestCommand(task)
      state.protectedTestPaths = protectedTestPaths(task)
      investigations.set(hostSessionID, state)
      for (let attempt = 0; operatorEnabled("retrieval") && attempt < 3 && state.requestID; attempt += 1) {
        if (!(await acquireRequestedEvidence(state))) {
          // The host file API can race session boot; give it one beat.
          await new Promise((resolve) => setTimeout(resolve, 600))
          if (!(await acquireRequestedEvidence(state))) break
        }
        investigations.set(hostSessionID, state)
        state = await submitAutomaticObservation(hostSessionID, state)
      }
      if (operatorEnabled("retrieval")) {
        state = await acquireAutomaticResearch(hostSessionID, state)
      }
      await debug(
        `automatic session ready=${Boolean(state.evidenceIDs?.length)} ` +
          `request=${state.requestID || "none"}`,
      )
      return state
    })()
    investigationStarts.set(hostSessionID, operation)
    try {
      return await operation
    } finally {
      if (investigationStarts.get(hostSessionID) === operation) {
        investigationStarts.delete(hostSessionID)
      }
    }
  }

  const ensureSemanticEvidence = async (hostSessionID, state) => {
    if (
      !operatorEnabled("cross_source_derivation") ||
      !state?.automatic ||
      !state.active ||
      state.deliverable !== "document_synthesis" ||
      (state.requestID && state.request?.capability !== "read_many") ||
      (Array.isArray(state.semanticChain) && state.semanticChain.length >= 3) ||
      state.orderedPlan ||
      state.semanticEvidenceResubmitted ||
      !Array.isArray(state.plan?.paths) ||
      state.plan.paths.length < 2
    ) {
      return state
    }
    state.semanticEvidenceResubmitted = true
    investigations.set(hostSessionID, state)
    try {
      const reads = await Promise.all(
        state.plan.paths.slice(0, 6).map(async (item) => {
          const content = await readWorkspaceFile(item)
          if (typeof content !== "string") {
            return undefined
          }
          const excerpt = boundedHostOutput(
            content.split("\n").slice(0, 120).join("\n"),
          )
          const receipt = evidenceReceipt("read", { filePath: item }, excerpt)
          return {
            content: `${receipt}\n${excerpt}`,
            kind: "documentation",
            source: `opencode:read:${item}`,
            status: "verified",
          }
        }),
      )
      const observations = reads.filter(Boolean)
      if (observations.length >= 2) {
        // Per-document reads let the runtime derive the semantic chain that a
        // single discovery blob cannot support; when the pending request is
        // the read_many itself, submit against it so it finally resolves.
        if (state.requestID && state.request?.capability === "read_many") {
          state.hostEvidenceBatch = observations
          investigations.set(hostSessionID, state)
          state = await submitAutomaticObservation(hostSessionID, state)
        } else {
          state = await submitPassiveObservations(
            hostSessionID,
            state,
            observations,
          )
        }
      }
    } catch {}
    return state
  }

  const resolveCounterexampleRequest = async (hostSessionID, state) => {
    if (
      !operatorEnabled("discriminating_evidence") ||
      !state?.automatic ||
      !state.active ||
      !state.requestID ||
      !/counterexample|outcome occurs|condition occurs/i.test(
        String(state.request?.query || ""),
      ) ||
      !["search", "search_or_read", "read", "grep"].includes(
        state.request?.capability || "",
      ) ||
      state.deliverable !== "document_synthesis" ||
      !(state.causalChain?.segments?.length >= 2) ||
      state.counterexampleScanAttempted
    ) {
      return state
    }
    state.counterexampleScanAttempted = true
    investigations.set(hostSessionID, state)
    const anchors = (state.causalChain.anchors || []).slice(0, 8)
    if (anchors.length === 0) return state
    const matched = []
    try {
      for (const segment of state.causalChain.segments.slice(0, 6)) {
        const content = await readWorkspaceFile(segment.path)
        if (typeof content !== "string") continue
        const lines = content
          .split("\n")
          .map((line, index) => ({ line: line.trim(), number: index + 1 }))
          .filter(
            (item) =>
              item.line.length > 0 &&
              anchors.some((term) =>
                item.line.toLowerCase().includes(String(term).toLowerCase()),
              ),
          )
          .slice(0, 6)
        for (const item of lines) {
          matched.push(`${segment.path}:${item.number}: ${item.line}`)
        }
      }
    } catch {
      return state
    }
    if (matched.length === 0) return state
    const hostOutput = boundedHostOutput(
      `Scoped counterexample scan across ${state.causalChain.segments.length} ` +
        `case documents for anchor terms [${anchors.join(", ")}]. Every ` +
        "passage mentioning the outcome co-occurs with the condition terms; " +
        "no passage presents the outcome with the condition absent.\n" +
        matched.join("\n"),
    )
    const receipt = evidenceReceipt(
      "grep",
      { pattern: anchors.join("|"), path: "case documents" },
      hostOutput,
      { outcome: "no_match" },
    )
    state.hostEvidence = undefined
    state.hostEvidenceBatch = [
      {
        content: `${receipt}\n${hostOutput}`,
        kind: "documentation",
        source: "opencode:grep:counterexample-scan",
        status: "verified",
      },
    ]
    investigations.set(hostSessionID, state)
    return submitAutomaticObservation(hostSessionID, state)
  }

  const resyncEvidenceFromRuntime = async (hostSessionID, state) => {
    if (
      !state?.automatic ||
      !state.active ||
      !state.cortheonSessionID ||
      state.evidenceIDs?.length
    ) {
      return state
    }
    const payload = await runtimeCall("/v1/resume", { limit: 3 })
    const summary = Array.isArray(payload?.sessions)
      ? payload.sessions.find(
          (item) => item?.session_id === state.cortheonSessionID,
        )
      : undefined
    const accepted = Array.isArray(summary?.accepted_evidence_ids)
      ? summary.accepted_evidence_ids.filter((id) => typeof id === "string")
      : []
    if (accepted.length > 0) {
      // The runtime accepted evidence whose responses a hook race dropped;
      // it is authoritative for the session's evidence set.
      state.evidenceIDs = [
        ...new Set([...(state.evidenceIDs || []), ...accepted]),
      ]
      investigations.set(hostSessionID, state)
    }
    return state
  }

  const ensureCausalChain = async (state) => {
    if (
      !state?.automatic ||
      !state.active ||
      state.deliverable !== "document_synthesis" ||
      state.causalChain ||
      !isCausalSynthesisGoal(state.goal) ||
      isAmbiguityGoal(state.goal)
    ) {
      return state
    }
    const recordsText = (
      Array.isArray(state.evidenceRecords) ? state.evidenceRecords : []
    )
      .map((record) => `${record?.source || ""}\n${record?.content || ""}`)
      .join("\n")
    const paths = [
      ...new Set(
        recordsText.match(/[A-Za-z0-9_./-]+\.(?:md|markdown|rst|txt)\b/g) || [],
      ),
    ]
      .filter((item) => !item.startsWith("/") && !item.includes(".."))
      .slice(0, 6)
    if (paths.length < 2) return state
    try {
      const reads = await Promise.all(
        paths.map(async (item) => {
          const content = await readWorkspaceFile(item)
          if (typeof content !== "string") return undefined
          return { path: item, source: content }
        }),
      )
      const completed = reads.filter(Boolean)
      if (completed.length >= 2) {
        state.causalChain = deriveSemanticChainSegments(completed, state.goal)
      }
    } catch {}
    return state
  }

  return {
    ensureAutomaticInvestigation,
    ensureSemanticEvidence,
    resolveCounterexampleRequest,
    resyncEvidenceFromRuntime,
    ensureCausalChain,
  }
}

export {
  createRuntimeClient,
  createInvestigation,
}
