import {
  hostEvidenceTools,
  postCertificationReadTools,
  workspaceMutationTools,
} from "./state.js"
import { goalCodePaths } from "./joins.js"
import { permittedHostTool } from "./state.js"
import { constrainRedundantTool, scopedPredicatePlan } from "./plans.js"
import { statementIsNegative } from "./evidence.js"
import { investigations } from "./state.js"
import { isTestCommand, testCommand } from "./plans.js"
import { evaluationProfile, operatorEnabled } from "./state.js"

// Pre-tool-execution hook: path repair, capability gating, repair binding,
// protected test paths, and Cortheon tool argument binding.
const createToolBeforeHook = ({
  debug,
  directory,
  latestUserTask,
  acquireRequestedEvidence,
  captureMutationBefore,
}) => ({
  "tool.execute.before": async (input, output) => {
    const state = investigations.get(input.sessionID)
    if (evaluationProfile() && !operatorEnabled("retrieval")) return
    if (state?.automatic && state.active && output.args) {
      // Small models hallucinate absolute roots (/testbed/, /repo/, /app/)
      // from training data; the host then denies the out-of-workspace path
      // and the investigation starves. Strip a fabricated leading root so
      // the call resolves inside the real workspace.
      for (const key of ["filePath", "path"]) {
        const raw = output.args[key]
        if (typeof raw !== "string" || raw.length === 0) continue
        if (raw.startsWith("/")) {
          const workspace = String(directory || "").replace(/\/+$/, "")
          if (
            workspace &&
            (raw === workspace || raw.startsWith(`${workspace}/`))
          ) {
            continue
          }
          const parts = raw.split("/").filter(Boolean)
          if (parts.length >= 2) {
            output.args[key] = parts.slice(1).join("/")
          } else if (parts.length === 1) {
            output.args[key] = parts[0]
          }
        }
        // Models also drop directory prefixes (points.py for
        // journey/points.py); resolve against the goal-named paths.
        const current = output.args[key]
        if (
          typeof current === "string" &&
          current.length > 0 &&
          !current.includes("/")
        ) {
          const resolved = goalCodePaths(state.goal).find((item) =>
            item.endsWith(`/${current}`),
          )
          if (resolved) output.args[key] = resolved
        }
      }
      if (input.tool === "webfetch") {
        const url = String(output.args?.url || "")
        const basename = url.split("/").pop()?.split("?")[0] || ""
        if (
          basename &&
          /\.[a-z]{2,4}$/i.test(basename) &&
          (String(state.goal || "").includes(basename) ||
            (Array.isArray(state.plan?.paths) &&
              state.plan.paths.some(
                (item) => item === basename || item.endsWith(`/${basename}`),
              )))
        ) {
          // The model invented a remote location for a workspace document.
          throw new Error(
            `${basename} is a local project document, not a remote page. ` +
              `Call read with filePath "${basename}" instead.`,
          )
        }
      }
    }
    if (
      operatorEnabled("retrieval") &&
      state?.automatic &&
      state.active &&
      state.deliverable === "research_answer" &&
      input.tool === "webfetch" &&
      typeof output.args?.url === "string"
    ) {
      try {
        const target = new URL(output.args.url)
        const project = target.pathname.match(/^\/project\/([^/]+)\/?$/)?.[1]
        if (target.hostname === "pypi.org" && project) {
          output.args.url =
            `https://pypi.org/rss/project/${encodeURIComponent(project)}/releases.xml`
          output.args.format = "html"
        }
      } catch {
      }
    }
    if (state?.automatic && state.completed) {
      if (postCertificationReadTools.has(input.tool)) return
      throw new Error(
        "Cortheon already certified this result. Stop calling tools and report the " +
          "host-verified completion.",
      )
    }
    if (state?.automatic && String(input.tool || "").startsWith("cortheon_")) {
      throw new Error(
        "Cortheon is managed automatically by this OpenCode adapter; use host tools only.",
      )
    }
    if (
      state?.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      state.requestedTestCommand &&
      !state.requestID &&
      state.evidenceIDs?.length &&
      state.repairPlan &&
      !state.multiMutationTask &&
      !state.testFailed
    ) {
      if (input.tool === "write") {
        throw new Error(
          `Cortheon derived a bounded one-line repair for ${state.repairPlan.path}. ` +
            "Use edit so the rest of the file remains unchanged.",
        )
      }
      if (input.tool === "edit") {
        output.args.filePath = state.repairPlan.path
        output.args.oldString = state.repairPlan.oldString
        output.args.newString = state.repairPlan.newString
        output.args.replaceAll = false
        await debug(
          `bound edit to assertion-derived repair for ${state.repairPlan.path}`,
        )
      }
    }
    if (
      state?.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      state.requestedTestCommand &&
      !state.requestID &&
      state.evidenceIDs?.length &&
      !workspaceMutationTools.has(input.tool)
    ) {
      throw new Error(
        state.testFailed
          ? "The last edit failed host verification. Revise the implementation with " +
              "edit or write; Cortheon will rerun the exact test automatically."
          : "Cortheon already completed the focused reads. Make the smallest " +
              "implementation change with edit or write; Cortheon owns test execution.",
      )
    }
    if (
      state?.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      workspaceMutationTools.has(input.tool) &&
      state.requestID &&
      !["diff", "test"].includes(state.request?.capability || "")
    ) {
      // A pending diff/test request is satisfied BY the mutation, so blocking
      // edit while it is pending deadlocks the change against its own proof.
      throw new Error(
        `Inspect the live code for Cortheon request ${state.requestID} before editing.`,
      )
    }
    if (
      state?.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      workspaceMutationTools.has(input.tool) &&
      Array.isArray(state.protectedTestPaths)
    ) {
      const target = String(
        output.args?.filePath || output.args?.path || "",
      )
      if (
        state.protectedTestPaths.some(
          (path) => target === path || target.endsWith(`/${path}`),
        )
      ) {
        throw new Error(
          `The user protected ${target} from modification; change implementation code only.`,
        )
      }
    }
    if (
      state?.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      state.mutated &&
      !state.latestPassingTest &&
      !state.testFailed
    ) {
      if (state.runningAutomaticTest && input.tool === "bash") return
      if (input.tool !== "bash") {
        throw new Error(
          "The implementation changed. Run the required relevant test now; unrelated " +
            "tools and further edits are blocked until a zero-exit test result exists.",
        )
      }
      if (state.requestedTestCommand) {
        output.args.command = state.requestedTestCommand
      }
      if (!isTestCommand(testCommand(output.args))) {
        throw new Error(
          "The implementation changed. The next command must be a relevant test.",
        )
      }
    }
    if (
      state?.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      workspaceMutationTools.has(input.tool)
    ) {
      await captureMutationBefore(state, input.tool, output.args)
      investigations.set(input.sessionID, state)
    }
    if (
      operatorEnabled("adaptive_stopping") &&
      state?.automatic &&
      state.active &&
      !state.requestID &&
      state.evidenceIDs?.length &&
      hostEvidenceTools.has(input.tool) &&
      (state.plan?.pattern ||
        ["sum", "semantic_join"].includes(state.plan?.operation))
    ) {
      const constrained = constrainRedundantTool(state, input.tool, output.args)
      if (constrained) {
        if (
          state.deliverable !== "document_synthesis" ||
          state.plan?.operation !== "semantic_join"
        ) {
          return
        }
        if (
          Number(state.redundantEvidenceCalls || 0) >=
          Math.min(3, state.plan.paths?.length || 1)
        ) {
          throw new Error(
            "Cortheon already supplied every focused document read. Answer now " +
              "without another tool call.",
          )
        }
        state.redundantEvidenceCalls =
          Number(state.redundantEvidenceCalls || 0) + 1
        investigations.set(input.sessionID, state)
        return
      }
      if (
        state.deliverable === "document_synthesis" &&
        state.plan?.operation === "semantic_join" &&
        ["read", "glob"].includes(input.tool) &&
        Number(state.redundantEvidenceCalls || 0) <
          Math.min(3, state.plan.paths?.length || 1)
      ) {
        state.redundantEvidenceCalls =
          Number(state.redundantEvidenceCalls || 0) + 1
        if (input.tool === "read" && state.plan.paths?.length) {
          const path =
            state.plan.paths[state.redundantEvidenceCalls - 1] ||
            state.plan.paths[0]
          output.args.filePath = path
          output.args.offset = 1
          output.args.limit = 240
        } else if (input.tool === "glob") {
          output.args.pattern = "**/*.{md,markdown,rst,txt}"
          delete output.args.path
        }
        investigations.set(input.sessionID, state)
        return
      }
      throw new Error(
        "Cortheon already acquired the complete scoped evidence for this deterministic " +
          "task; additional broad tool calls are blocked.",
      )
    }
    if (hostEvidenceTools.has(input.tool) && state?.active && state.requestID) {
      if (!permittedHostTool(input.tool, state)) {
        if (["read", "grep", "glob"].includes(input.tool)) {
          // Read-only inspection never blocks an investigation: run it and
          // harvest the receipt passively instead of against the pending
          // request, so it cannot masquerade as the requested capability.
          state.hostCallPassive = true
          state.hostCall = { tool: input.tool, args: { ...output.args } }
          investigations.set(input.sessionID, state)
        } else {
          throw new Error(
            `Cortheon request ${state.requestID} requires capability ` +
              `${state.request?.capability || "inspect"}. Call that exact host tool ` +
              `for: ${state.request?.query || state.goal || "the live task"}`,
          )
        }
      } else {
        state.hostCallPassive = false
        state.hostCall = { tool: input.tool, args: { ...output.args } }
        investigations.set(input.sessionID, state)
      }
    }
    if (
      workspaceMutationTools.has(input.tool) &&
      state?.active &&
      state.deliverable !== "code_change"
    ) {
      throw new Error(
        "Cortheon classified this as an understanding task, so workspace mutation is blocked.",
      )
    }
    if (input.tool === "cortheon_cortheon_start") {
      const task = await latestUserTask(input.sessionID)
      if (task) {
        output.args.goal = task
        output.args.task_kind = "auto"
        await debug(`bound cortheon_start to ${task.length} characters of live user input`)
      }
      return
    }
    if (input.tool === "cortheon_cortheon_complete") {
      if (!state?.active || !state.cortheonSessionID) {
        throw new Error("Cortheon has no active investigation to complete.")
      }
      const known = new Set(state.evidenceIDs || [])
      if (known.size === 0) {
        throw new Error("Cortheon completion requires accepted live evidence first.")
      }
      const bind = (items) =>
        Array.isArray(items)
          ? items.map((item) => {
              const selected = Array.isArray(item?.evidence_ids)
                ? item.evidence_ids.filter((id) => known.has(id))
                : []
              return {
                ...item,
                evidence_ids: selected.length > 0 ? selected : [...known],
              }
            })
          : items
      output.args.session_id = state.cortheonSessionID
      output.args.claims = bind(output.args.claims)
      output.args.hypotheses = bind(output.args.hypotheses)
      if (
        Array.isArray(output.args.hypotheses) &&
        scopedPredicatePlan(state.plan) &&
        (state.deterministicOutcome === "match" ||
          state.deterministicOutcome === "no_match")
      ) {
        const predicatePresent = state.deterministicOutcome === "match"
        output.args.hypotheses = output.args.hypotheses.map((item) => ({
          ...item,
          status:
            !statementIsNegative(item?.statement) === predicatePresent
              ? "supported"
              : "refuted",
        }))
      }
      const completionIDs = Array.isArray(output.args.completion_evidence_ids)
        ? output.args.completion_evidence_ids.filter((id) => known.has(id))
        : []
      output.args.completion_evidence_ids =
        completionIDs.length > 0 ? completionIDs : [...known]
      return
    }
    if (input.tool !== "cortheon_cortheon_observe") return
    if (!state?.active || !state.requestID) {
      throw new Error(
        "Cortheon has no pending evidence request. Start or follow the returned next action.",
      )
    }
    if (!state.hostEvidence) {
      const acquired = await acquireRequestedEvidence(state)
      if (!acquired) {
        throw new Error(
          `Cortheon request ${state.requestID} requires a real host tool first. ` +
            "Call read, grep, glob, bash, webfetch, or websearch; never invent evidence.",
        )
      }
      investigations.set(input.sessionID, state)
    }
    output.args.session_id = state.cortheonSessionID
    output.args.request_id = state.requestID
    output.args.observations = [
      {
        kind: state.hostEvidence.kind,
        content: state.hostEvidence.content,
        source: state.hostEvidence.source,
        status: state.hostEvidence.status,
      },
    ]
  },
})

export { createToolBeforeHook }
