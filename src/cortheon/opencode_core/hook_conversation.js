import { modelContext, protocol } from "./state.js"
import { isAmbiguityGoal, isCausalSynthesisGoal, deriveKeyedCollisionInference } from "./joins.js"
import { numericJoin } from "./plans.js"
import { prependSystemGuidance } from "./state.js"
import { certifiedAnswers, investigations } from "./state.js"
import { boundedHostOutput } from "./state.js"
import { evaluationProfile, operatorEnabled } from "./state.js"

// The per-turn system prompt transform: ensures the automatic
// investigation, retries acquisition, drives deterministic completions, and
// injects certified answers or bounded guidance.
const createSystemTransformHook = ({
  runtimeBase,
  ensureAutomaticInvestigation,
  acquireRequestedEvidence,
  submitAutomaticObservation,
  ensureCausalChain,
  resolveCounterexampleRequest,
  attemptBoundedMultiRepair,
  attemptBoundedAutomaticRepair,
  certifyDeterministicResearch,
  resyncEvidenceFromRuntime,
  ensureSemanticEvidence,
  submitAutomaticCompletion,
  runtimeCall,
}) => ({
  "experimental.chat.system.transform": async (input, output) => {
    const hostSessionID = input.sessionID
    if (!hostSessionID) {
      prependSystemGuidance(output, protocol)
      return
    }
    let state = await ensureAutomaticInvestigation(hostSessionID)
    if (state?.runtimeUnavailable) {
      // Degradation must be visible: the session is running without
      // verification, and the model (and the user reading its output)
      // should know it.
      prependSystemGuidance(
        output,
        `[CORTHEON] Runtime unavailable at ${runtimeBase}; this session runs ` +
          "without evidence verification. Start `cortheon serve` to restore it.",
      )
    }
    if (!state?.automatic) {
      prependSystemGuidance(output, protocol)
      return
    }
    const profile = evaluationProfile()
    if (profile?.config.cleanup_before_answer) {
      if (state.evidenceSummary) {
        prependSystemGuidance(
          output,
          modelContext +
            "\n\nCORTHEON RETRIEVAL CONDITION: Use the bounded host evidence below. " +
            "Cortheon will not reason, verify, block, or rewrite the answer.\n\n" +
            boundedHostOutput(state.evidenceSummary),
        )
      } else {
        prependSystemGuidance(output, modelContext)
      }
      if (state.active && state.cortheonSessionID) {
        await runtimeCall("/v1/abandon", {
          session_id: state.cortheonSessionID,
        })
      }
      investigations.delete(hostSessionID)
      return
    }
    if (state.runtimeStale && !state.runtimeStaleAnnounced) {
      state.runtimeStaleAnnounced = true
      investigations.set(hostSessionID, state)
      prependSystemGuidance(
        output,
        "[CORTHEON] The runtime process is stale: its sources changed after " +
          "it started. Restart `cortheon serve` so the plugin and runtime match.",
      )
    }
    if (profile && !operatorEnabled("retrieval")) {
      prependSystemGuidance(output, modelContext)
      return
    }
    if (operatorEnabled("retrieval") && state.active && state.requestID) {
      // Acquisition at session start can fail transiently (workspace still
      // staging); retry the adapter-owned evidence request each turn so a
      // pending request cannot silently starve the whole investigation.
      if (await acquireRequestedEvidence(state)) {
        investigations.set(hostSessionID, state)
        state = await submitAutomaticObservation(hostSessionID, state)
      }
    }
    if (operatorEnabled("cross_source_derivation")) {
      state = await ensureCausalChain(state)
    }
    investigations.set(hostSessionID, state)
    if (operatorEnabled("discriminating_evidence")) {
      state = await resolveCounterexampleRequest(hostSessionID, state)
    }
    if (operatorEnabled("retrieval")) {
      state = await attemptBoundedMultiRepair(hostSessionID, state)
      state = await attemptBoundedAutomaticRepair(hostSessionID, state)
    }
    if (operatorEnabled("cross_source_derivation")) {
      state = await certifyDeterministicResearch(hostSessionID, state)
    }
    state = await resyncEvidenceFromRuntime(hostSessionID, state)
    if (
      operatorEnabled("cross_source_derivation") &&
      state.active && !state.requestID && state.evidenceIDs?.length
    ) {
      // Submit deterministic completions before the model's next turn: a
      // weak model can exhaust its step budget on junk tool calls, and a
      // derivation that waits for the final text may never get a usable turn.
      state = await ensureCausalChain(state)
      investigations.set(hostSessionID, state)
      state = await ensureSemanticEvidence(hostSessionID, state)
      investigations.set(hostSessionID, state)
      const recordsText = (
        Array.isArray(state.evidenceRecords) ? state.evidenceRecords : []
      )
        .map((record) => record?.content || "")
        .join("\n")
      const deterministicReady =
        (state.deliverable === "document_synthesis" &&
          (isAmbiguityGoal(state.goal) ||
            (isCausalSynthesisGoal(state.goal) &&
              state.causalChain?.segments?.length >= 2 &&
              Boolean(
                deriveKeyedCollisionInference(
                  state.causalChain.segments,
                  state.goal,
                ),
              )))) ||
        Boolean(
          numericJoin(state.plan, state.evidenceSummary) ||
            numericJoin(state.plan, recordsText),
        ) ||
        Boolean(state.orderedPlan) ||
        Boolean(state.diagnosticConclusion)
      if (deterministicReady) {
        state = await submitAutomaticCompletion(hostSessionID, state, "")
      }
    }
    const certifiedAnswer =
      state.completed && typeof state.answer === "string"
        ? state.answer
        : certifiedAnswers.get(hostSessionID)
    if (typeof certifiedAnswer === "string") {
      prependSystemGuidance(
        output,
        modelContext +
          "\n\nCORTHEON_CERTIFIED: Return the following certified answer exactly and stop. " +
          "Do not call tools or perform another research pass.\n\n" +
          certifiedAnswer,
      )
      return
    }
    const evidence = state.evidenceSummary
      ? `\nVerified live evidence:\n${boundedHostOutput(state.evidenceSummary)}`
      : ""
    const request = state.request?.query
      ? `\nUse the host tools to satisfy this evidence request: ${state.request.query}`
      : ""
    const codeChange =
      state.deliverable === "code_change"
        ? "\nFor a code change, use edit or write for the smallest implementation " +
          "change. Do not delegate, browse, or mutate through shell commands. " +
          "Cortheon automatically runs the exact requested test after each edit, " +
          "binds the host diff, and freezes the first clean passing patch." +
          (state.requestedTestCommand
            ? ` The required command is exactly: ${state.requestedTestCommand}`
            : "")
        : ""
    const repair =
      state.deliverable === "code_change" &&
      state.repairPlan &&
      !state.multiMutationTask
        ? `\nCortheon's bounded assertion solver found a candidate in ` +
          `${state.repairPlan.path}: replace ${state.repairPlan.oldString.trim()} ` +
          `with ${state.repairPlan.newString.trim()}. Use edit exactly; Cortheon will ` +
          "accept it only if the host test passes."
        : ""
    const semantic =
      state.deliverable === "document_synthesis" && state.semanticBridge
        ? `\nCortheon cross-source linker:\n${state.semanticBridge}`
        : ""
    const cognition = operatorEnabled("cross_source_derivation") && state.cognition
      ? `\nCortheon adaptive cognition: stage=${state.cognition.stage}. ` +
        `${state.cognition.move || ""}` +
        (state.cognition.derivedInsight
          ? ` Candidate cross-source inference: ${state.cognition.derivedInsight}`
          : "") +
        (state.cognition.unresolvedRequirements?.length
          ? ` Unresolved task contract: ${state.cognition.unresolvedRequirements.join("; ")}.`
          : "") +
        (state.cognition.decisionRule
          ? ` Decision rule: ${state.cognition.decisionRule}`
          : "")
      : ""
    const hypotheses = operatorEnabled("hypothesis_framing") &&
      Array.isArray(state.hypotheses) && state.hypotheses.length > 0
        ? "\nCortheon competing hypotheses:\n" +
          state.hypotheses
            .map(
              (item, index) =>
                `${index + 1}. ${item.statement} Falsification: ` +
                `${item.falsification_test}`,
            )
            .join("\n")
        : ""
    const reasonAction = operatorEnabled("discriminating_evidence") && state.reasonAction
      ? `\nCortheon reasoning action: ${state.reasonAction}`
      : ""
    const verificationGaps = operatorEnabled("contradiction_revision") && state.verificationGaps?.length
      ? `\nA previous completion was withheld for: ` +
        `${state.verificationGaps.join("; ")}. Correct these gaps once; do not ` +
        "repeat already accepted reads."
      : ""
    const finalReasoning =
      !state.requestID &&
      state.evidenceIDs?.length &&
      state.deliverable === "document_synthesis"
        ? "\nFINAL REASONING TURN: all focused host reads are already complete. Do not " +
          "call, announce, or plan another tool use. Answer now from the verified evidence. " +
          "For causal synthesis, state the observed clues, at least two genuinely different " +
          "hypotheses, the best-supported causal bridge, and a discriminating observation " +
          "that would falsify it. For ambiguity, name every viable interpretation, make no " +
          "unsupported choice, and ask the smallest clarification that separates them."
        : ""
    const research =
      state.deliverable === "research_answer"
        ? `\nCurrent date: ${new Date().toISOString().slice(0, 10)}. For research, ` +
          "obey each returned web request, prefer primary sources, preserve dates, " +
          "actively check for contradiction, and include clickable source URLs in " +
          "the final answer."
        : ""
    prependSystemGuidance(
      output,
      (
        modelContext +
        "\n\nCortheon is running automatically as the completion gate. Do not call Cortheon " +
        "MCP tools or invent observations. When verified live evidence is included " +
        "below and no further request is shown, the host has already satisfied the " +
        "evidence request: answer directly from that bounded evidence and do not read " +
        "the same files again. Cortheon will verify and release only a certified " +
        `result.${request}${codeChange}${repair}${semantic}${cognition}` +
        `${hypotheses}${reasonAction}${verificationGaps}${finalReasoning}` +
        `${research}${evidence}`
      ).trim(),
    )
  },
})

// Session lifecycle hooks: compaction termination after certification and
// cleanup on session deletion or error.
const createSessionLifecycleHooks = ({ debug, client, directory, runtimeCall }) => ({
  "experimental.session.compacting": async (input, output) => {
    const state = investigations.get(input.sessionID)
    if (
      !state?.automatic ||
      !state.completed ||
      typeof state.answer !== "string" ||
      state.terminationRequested
    ) {
      return
    }
    state.terminationRequested = true
    investigations.set(input.sessionID, state)
    let aborted = false
    try {
      const response = await client.session.abort({
        path: { id: input.sessionID },
        query: { directory },
      })
      aborted = response?.data === true || response === true
    } catch {
    }
    state.terminationSucceeded = aborted
    if (investigations.get(input.sessionID) === state) {
      investigations.set(input.sessionID, state)
    }
    output.context.push(
      aborted
        ? "Cortheon already certified the exact answer; host compaction was cancelled."
        : "Cortheon already certified the exact answer; do not alter it during compaction.",
    )
    await debug(
      `certified compaction termination requested session=${input.sessionID} ` +
        `aborted=${aborted}`,
    )
  },

  event: async ({ event }) => {
    if (event?.type === "session.idle") {
      const hostSessionID =
        event.properties?.sessionID || event.properties?.info?.id
      const state = investigations.get(hostSessionID)
      if (state?.automatic) {
        state.idle = true
        investigations.set(hostSessionID, state)
      }
      return
    }
    if (event?.type === "session.deleted" || event?.type === "session.error") {
      const hostSessionID =
        event.properties?.sessionID || event.properties?.info?.id
      const state = investigations.get(hostSessionID)
      if (state?.automatic && state.active && state.cortheonSessionID) {
      await runtimeCall("/v1/abandon", {
        session_id: state.cortheonSessionID,
      })
      }
      certifiedAnswers.delete(hostSessionID)
      investigations.delete(hostSessionID)
    }
  },
})

export {
  createSystemTransformHook,
  createSessionLifecycleHooks,
}
