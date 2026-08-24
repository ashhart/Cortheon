import { duplicateTerminalLine, mutationTarget } from "./plans.js"
import { boundedHostOutput, focusedSessionDiff, focusedTextDiff } from "./state.js"
import { mergePayload } from "./state_merge.js"
import { investigations } from "./state.js"

// Read-only host session introspection: latest user task text and the
// session-scoped diff summary used as mutation evidence.
const createHostSession = ({ client, directory }) => {
  const latestUserTask = async (sessionID) => {
    try {
      const response = await client.session.messages({
        path: { id: sessionID },
        query: { directory, limit: 50 },
      })
      const messages = Array.isArray(response?.data)
        ? response.data
        : Array.isArray(response)
          ? response
          : []
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        const message = messages[index]
        if (message?.info?.role !== "user" || !Array.isArray(message.parts)) continue
        let text = message.parts
          .filter((part) => part?.type === "text" && !part.synthetic && !part.ignored)
          .map((part) => part.text)
          .join("\n")
          .trim()
        if (text.startsWith('"') && text.endsWith('"')) {
          try {
            const unquoted = JSON.parse(text)
            if (typeof unquoted === "string") text = unquoted.trim()
          } catch {
          }
        }
        if (text) return text.slice(0, 6000)
      }
    } catch {
    }
    return undefined
  }

  const sessionDiffEvidence = async (hostSessionID) => {
    try {
      const response = await client.session.diff({
        path: { id: hostSessionID },
        query: { directory },
      })
      const diffs = Array.isArray(response?.data)
        ? response.data
        : Array.isArray(response)
          ? response
          : []
      const content = focusedSessionDiff(diffs)
      if (!content) return undefined
      const receipt =
        "[CORTHEON_HOST_EVIDENCE] " +
        JSON.stringify({
          tool: "diff",
          executor: "session.diff",
          outcome: "changed",
          args: { files: diffs.length },
        })
      return {
        kind: "diff",
        content: `${receipt}\n${content}`,
        source: "opencode:session-diff",
        status: "observed",
      }
    } catch {
      return undefined
    }
  }

  return { latestUserTask, sessionDiffEvidence }
}

// Workspace file access and mutation capture through the host client, with
// a read-only shell fallback for the host file API's early-session races.
const createWorkspaceAccess = ({ client, directory, hostShell }) => {
  const readWorkspaceFile = async (path) => {
    try {
      const response = await client.file.read({ query: { directory, path } })
      const file = response?.data || response
      if (file?.type === "text" && typeof file.content === "string") {
        return file.content
      }
    } catch {}
    // The host file API intermittently fails early in a session; a plain
    // read-only shell read of the same workspace path is the fallback.
    if (typeof hostShell === "function") {
      try {
        const result = await hostShell`cat -- ${path}`
          .cwd(directory)
          .quiet()
          .nothrow()
        if (Number(result?.exitCode) === 0) {
          const text =
            typeof result?.stdout === "string"
              ? result.stdout
              : result?.stdout?.toString?.() || ""
          if (text && text.length <= 200_000) return text
        }
      } catch {}
    }
    return undefined
  }

  const captureMutationBefore = async (state, tool, args) => {
    const path = mutationTarget(args)
    state.pendingMutation = path ? { path } : undefined
    if (!path || !["edit", "write"].includes(tool)) return
    try {
      const response = await client.file.read({
        query: { directory, path },
      })
      const file = response?.data || response
      if (
        file?.type === "text" &&
        typeof file.content === "string" &&
        file.content.length <= 50_000
      ) {
        state.pendingMutation.before = file.content
      }
    } catch {
      if (tool === "write") state.pendingMutation.before = ""
    }
  }

  const captureMutationAfter = async (state, tool, args, output) => {
    const pending = state.pendingMutation
    state.pendingMutation = undefined
    const path = mutationTarget(args) || pending?.path || "workspace"
    let content = boundedHostOutput(
      output?.metadata?.diff || output?.metadata?.filediff?.patch || "",
    )
    if (!content && typeof pending?.before === "string") {
      let after
      if (tool === "write" && typeof args?.content === "string") {
        after = args.content
      } else {
        try {
          const response = await client.file.read({
            query: { directory, path },
          })
          const file = response?.data || response
          if (file?.type === "text" && typeof file.content === "string") {
            after = file.content
          }
        } catch {
        }
      }
      if (typeof after === "string") {
        content = focusedTextDiff(path, pending.before, after)
      }
    }
    if (!content) return
    state.mutationDiffs = [
      ...(Array.isArray(state.mutationDiffs) ? state.mutationDiffs : []),
      { path, tool, content },
    ].slice(-8)
  }

  const capturedMutationDiffEvidence = (state) => {
    const diffs = Array.isArray(state?.mutationDiffs)
      ? state.mutationDiffs
      : []
    if (diffs.length === 0) return undefined
    const paths = [...new Set(diffs.map((item) => item.path))].slice(0, 8)
    const receipt =
      "[CORTHEON_HOST_EVIDENCE] " +
      JSON.stringify({
        tool: "diff",
        executor: "mutation_hook",
        outcome: "changed",
        args: { paths },
      })
    return {
      kind: "diff",
      content:
        `${receipt}\n` +
        boundedHostOutput(
          diffs
            .map(
              (item, index) =>
                `Mutation ${index + 1} (${item.tool} ${item.path})\n${item.content}`,
            )
            .join("\n\n"),
        ),
      source: "opencode:mutation-diff",
      status: "observed",
    }
  }

  const patchHygieneIssue = async (state) => {
    const paths = [
      ...new Set(
        (Array.isArray(state?.mutationDiffs) ? state.mutationDiffs : [])
          .map((item) => item.path)
          .filter((path) => typeof path === "string" && path.endsWith(".py")),
      ),
    ].slice(0, 8)
    for (const path of paths) {
      try {
        const response = await client.file.read({
          query: { directory, path },
        })
        const file = response?.data || response
        if (file?.type !== "text" || typeof file.content !== "string") continue
        const issue = duplicateTerminalLine(file.content)
        if (issue) return `${path}: ${issue}.`
      } catch {
      }
    }
    return undefined
  }

  return {
    readWorkspaceFile,
    captureMutationBefore,
    captureMutationAfter,
    capturedMutationDiffEvidence,
    patchHygieneIssue,
  }
}

const observationFields = (item) => ({
  kind: item.kind,
  content: item.content,
  source: item.source,
  status: item.status,
  ...(item.url ? { url: item.url } : {}),
  ...(item.retrieved_at ? { retrieved_at: item.retrieved_at } : {}),
  ...(item.published_at ? { published_at: item.published_at } : {}),
  ...(item.purpose ? { purpose: item.purpose } : {}),
})

// Submits pending or passive observations to the runtime and merges the
// returned payload into the per-session investigation state.
const createObservationSubmitter = ({ runtimeCall }) => {
  const submitPassiveObservations = async (
    hostSessionID,
    state,
    observations,
  ) => {
    if (
      !state?.cortheonSessionID ||
      !Array.isArray(observations) ||
      observations.length === 0
    ) {
      return state
    }
    state.hostEvidenceBatch = observations
    investigations.set(hostSessionID, state)
    const payload = await runtimeCall("/v1/observe", {
      session_id: state.cortheonSessionID,
      observations: observations.map(observationFields),
    })
    if (!payload) return state
    const next = mergePayload(hostSessionID, "observe", payload)
    next.automatic = true
    investigations.set(hostSessionID, next)
    return next
  }

  const submitAutomaticObservation = async (hostSessionID, state) => {
    const observations = Array.isArray(state?.hostEvidenceBatch)
      ? state.hostEvidenceBatch
      : state?.hostEvidence
        ? [state.hostEvidence]
        : []
    if (
      observations.length === 0 ||
      !state.requestID ||
      !state.cortheonSessionID
    ) {
      return state
    }
    const payload = await runtimeCall("/v1/observe", {
      session_id: state.cortheonSessionID,
      request_id: state.requestID,
      observations: observations.map(observationFields),
    })
    if (payload?.__requestResolvedConflict) {
      // The tracked request already resolved server-side; clear the zombie
      // reference and land the evidence passively instead.
      state.requestID = undefined
      state.request = undefined
      investigations.set(hostSessionID, state)
      return submitPassiveObservations(hostSessionID, state, observations)
    }
    if (!payload) return state
    const next = mergePayload(hostSessionID, "observe", payload)
    next.automatic = true
    investigations.set(hostSessionID, next)
    return next
  }

  return { submitAutomaticObservation, submitPassiveObservations }
}

// Executes the task-requested test command through the host shell and
// records a verified receipt on success.
const createTestRunner = ({ hostShell, directory, debug }) => {
  const runRequestedTest = async (hostSessionID, state) => {
    const command = state?.requestedTestCommand
    if (!command || typeof hostShell !== "function") return undefined
    state.runningAutomaticTest = true
    investigations.set(hostSessionID, state)
    try {
      const result = await hostShell`/bin/sh -lc ${command}`
        .cwd(directory)
        .quiet()
        .nothrow()
      const exit = Number(result?.exitCode)
      if (!Number.isInteger(exit)) return undefined
      const cleanOutput = boundedHostOutput(
        [result?.stdout?.toString(), result?.stderr?.toString()]
          .filter(Boolean)
          .join("\n"),
      )
      if (exit !== 0) {
        return {
          passed: false,
          summary: cleanOutput || `Test command exited ${exit}.`,
        }
      }
      const receipt =
        "[CORTHEON_HOST_EVIDENCE] " +
        JSON.stringify({
          tool: "test",
          executor: "host_shell",
          outcome: "passed",
          args: { command },
        })
      return {
        passed: true,
        summary: cleanOutput,
        observation: {
          kind: "test",
          content:
            `${receipt}\nCommand: ${command}\nExit: 0\n` + cleanOutput,
          source: "opencode:host-shell:test",
          status: "verified",
        },
      }
    } catch (error) {
      await debug(
        `host test execution failed: ${String(error || "unknown").slice(0, 500)}`,
      )
      return undefined
    } finally {
      const latest = investigations.get(hostSessionID)
      if (latest) {
        latest.runningAutomaticTest = false
        investigations.set(hostSessionID, latest)
      }
    }
  }

  return { runRequestedTest }
}

export {
  createHostSession,
  createWorkspaceAccess,
  createObservationSubmitter,
  createTestRunner,
}
