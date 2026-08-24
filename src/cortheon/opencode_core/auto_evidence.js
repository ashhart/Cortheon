import { deriveDiagnosticConclusion } from "./plans.js"
import {
  isCausalSynthesisGoal,
} from "./joins.js"
import {
  deriveSemanticBridge,
  deriveSemanticChainSegments,
  semanticTerms,
  setIntersection,
} from "./joins.js"
import {
  plannedGrep,
  plannedProjectDiscovery,
  plannedReads,
} from "./plans.js"
import {
  deriveExactDocumentEdits,
  deriveSimpleRepairPlans,
} from "./repair_derive.js"
import { evidenceReceipt } from "./evidence.js"
import { boundedHostOutput } from "./state.js"

// Adapter-owned evidence acquisition for the runtime's startup requests:
// scoped grep, read_many joins, and bounded project document discovery.
const createAutoEvidence = ({
  client,
  directory,
  hostShell,
  debug,
  readWorkspaceFile,
}) => {
  const acquireRequestedEvidence = async (state) => {
    const grepPlan = plannedGrep(state?.request)
    const readPlan = plannedReads(state?.request)
    const discoveryPlan = plannedProjectDiscovery(state?.request)
    if (!grepPlan && !readPlan && !discoveryPlan) return false
    try {
      if (discoveryPlan) {
        if (typeof hostShell !== "function") return false
        const listing =
          discoveryPlan.operation === "code_discovery"
            ? await hostShell`rg --files -g '*.py' -g '*.js' -g '*.jsx' -g '*.ts' -g '*.tsx' -g '*.go' -g '*.rs' -g '*.java' -g '*.rb' -g '*.php' -g '*.swift'`
                .cwd(directory)
                .quiet()
                .nothrow()
            : await hostShell`rg --files -g '*.md' -g '*.markdown' -g '*.rst' -g '*.txt'`
                .cwd(directory)
                .quiet()
                .nothrow()
        if (Number(listing?.exitCode) > 1) return false
        const paths = [
          ...new Set(
            String(listing?.stdout || "")
              .split("\n")
              .map((item) => item.trim())
              .filter(
                (item) =>
                  item &&
                  item.length <= 240 &&
                  !item.startsWith("/") &&
                  !item.split("/").includes("..") &&
                  !/(?:^|\/)(?:\.git|\.cortheon|\.venv|build|dist|node_modules)(?:\/|$)/.test(
                    item,
                  ),
              ),
          ),
        ].slice(0, 80)
        const goalTerms = semanticTerms(state.goal)
        const inspected = await Promise.all(
          paths.map(async (path) => {
            const response = await client.file.read({
              query: { directory, path },
            })
            const file = response?.data || response
            if (file?.type !== "text" || typeof file.content !== "string") {
              return undefined
            }
            const lines = file.content
              .split("\n")
              .slice(0, 240)
              .map((line, index) => {
                const terms = semanticTerms(line)
                return {
                  line: line.trim().slice(0, 320),
                  lineNumber: index + 1,
                  score: setIntersection(goalTerms, terms).length,
                }
              })
              .filter((item) => item.line && !/^\s*#/.test(item.line))
              .sort(
                (left, right) =>
                  right.score - left.score || left.lineNumber - right.lineNumber,
              )
            const selected = lines.slice(0, 3)
            return {
              path,
              score: selected.reduce((sum, item) => sum + item.score, 0),
              excerpts: selected,
            }
          }),
        )
        let candidates = inspected
          .filter(Boolean)
          .sort(
            (left, right) =>
              right.score - left.score ||
              left.path.length - right.path.length ||
              left.path.localeCompare(right.path),
          )
          .slice(0, discoveryPlan.maximum)
        if (discoveryPlan.preferTests) {
          const isTest = (path) =>
            /(?:^|\/)(?:test[_-]|tests?\/)|(?:[_-]test|\.spec|\.test)\./i.test(path)
          const test = candidates.find((item) => isTest(item.path))
          const implementation = candidates.find((item) => !isTest(item.path))
          if (test && implementation) {
            candidates = [
              implementation,
              test,
              ...candidates.filter(
                (item) => item !== implementation && item !== test,
              ),
            ]
          }
        }
        const result =
          candidates.length === 0
            ? "No matching project documents were found."
            : candidates
                .flatMap((candidate) =>
                  candidate.excerpts.map(
                    (item) =>
                      `${candidate.path}:${item.lineNumber}: ${item.line}`,
                  ),
                )
                .join("\n")
        const outcome = candidates.length === 0 ? "no_match" : "match"
        const hostOutput = boundedHostOutput(result)
        const args = {
          pattern:
            discoveryPlan.operation === "code_discovery"
              ? "**/*.{py,js,jsx,ts,tsx,go,rs,java,rb,php,swift}"
              : "**/*.{md,markdown,rst,txt}",
        }
        const receipt = evidenceReceipt("glob", args, hostOutput, { outcome })
        state.hostCall = { tool: "glob", args }
        state.hostEvidence = {
          content: `${receipt}\n${hostOutput}`,
          kind: "documentation",
          source:
            discoveryPlan.operation === "code_discovery"
              ? "opencode:glob:project-code"
              : "opencode:glob:project-documents",
          status: "verified",
        }
        state.hostEvidenceBatch = undefined
        await debug(
          `discovered ${candidates.length} bounded document candidates for ${state.requestID}`,
        )
        return true
      }
      if (grepPlan) {
        const response = await client.file.read({
          query: { directory, path: grepPlan.path },
        })
        const file = response?.data || response
        if (file?.type !== "text" || typeof file.content !== "string") {
          return false
        }
        const escaped = grepPlan.pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
        const matcher = new RegExp(
          `(^|[^A-Za-z0-9_])${escaped}([^A-Za-z0-9_]|$)`,
        )
        const matches = file.content
          .split("\n")
          .map((line, index) => ({ line, lineNumber: index + 1 }))
          .filter((item) => matcher.test(item.line))
        const result =
          matches.length === 0
            ? `No matches found in ${grepPlan.path}.`
            : matches
                .slice(0, 20)
                .map(
                  (item) =>
                    `${grepPlan.path}:${item.lineNumber}: ${item.line.trim()}`,
                )
                .join("\n")
        const hostOutput = boundedHostOutput(result)
        const receipt = evidenceReceipt("grep", grepPlan, hostOutput)
        state.hostCall = { tool: "grep", args: grepPlan }
        state.hostEvidence = {
          content: `${receipt}\n${hostOutput}`,
          kind: "code",
          source: `opencode:grep:${grepPlan.path}`,
          status: "verified",
        }
        state.hostEvidenceBatch = undefined
        await debug(
          `acquired ${matches.length} scoped host grep matches for ${state.requestID}`,
        )
        return true
      }

      const symbolMatchers = readPlan.symbols.map((symbol) => {
        const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
        return new RegExp(`(^|[^A-Za-z0-9_])${escaped}([^A-Za-z0-9_]|$)`)
      })
      const reads = await Promise.all(
        readPlan.paths.map(async (path) => {
          const content = await readWorkspaceFile(path)
          if (typeof content !== "string") {
            return undefined
          }
          const file = { content }
          const matches = file.content
            .split("\n")
            .map((line, index) => ({ line, lineNumber: index + 1 }))
            .filter(
              (item) =>
                symbolMatchers.length === 0 ||
                symbolMatchers.some((matcher) => matcher.test(item.line)),
            )
          const result =
            matches.length === 0
              ? `No requested symbols found in ${path}.`
              : readPlan.operation === "semantic_join"
                ? matches
                    .slice(0, 120)
                    .map((item) => item.line.trim())
                    .join("\n")
                : matches
                    .slice(0, 20)
                    .map(
                      (item) => `${path}:${item.lineNumber}: ${item.line.trim()}`,
                    )
                    .join("\n")
          const hostOutput = boundedHostOutput(result)
          const receipt = evidenceReceipt(
            "read",
            { filePath: path },
            hostOutput,
          )
          return {
            path,
            source: file.content,
            observation: {
              content: `${receipt}\n${hostOutput}`,
              kind:
                readPlan.operation === "semantic_join"
                  ? "documentation"
                  : "code",
              source: `opencode:read:${path}`,
              status: "verified",
            },
          }
        }),
      )
      const completedReads = reads.filter(Boolean)
      if (completedReads.length !== readPlan.paths.length) return false
      const observations = completedReads.map((item) => item.observation)
      state.hostCall = { tool: "read_many", args: readPlan }
      state.hostEvidence = undefined
      state.hostEvidenceBatch = observations
      state.repairPlans = deriveSimpleRepairPlans(completedReads)
      state.repairPlan = state.repairPlans[0]
      state.documentEdits = deriveExactDocumentEdits(
        completedReads,
        state.goal,
      )
      state.diagnosticConclusion = deriveDiagnosticConclusion(
        completedReads,
        state.goal,
      )
      state.semanticBridge =
        readPlan.operation === "semantic_join"
          ? deriveSemanticBridge(completedReads, state.goal)
          : undefined
      state.causalChain = isCausalSynthesisGoal(state.goal)
        ? deriveSemanticChainSegments(completedReads, state.goal)
        : undefined
      await debug(
        `acquired ${observations.length} scoped host reads for ${state.requestID}; ` +
          `repair=${state.repairPlan?.functionName || "none"} ` +
          `semantic_bridge=${Boolean(state.semanticBridge)}`,
      )
      return true
    } catch {
      return false
    }
  }

  return { acquireRequestedEvidence }
}

export { createAutoEvidence }
