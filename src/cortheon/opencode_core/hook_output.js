import { automaticCompletion } from "./completion.js"
import { hostEvidenceTools, workspaceMutationTools, certifiedAnswers, investigations, boundedHostOutput } from "./state.js"
import { decodeToolPayload, evidenceKind, evidenceStatus } from "./state.js"
import { evidenceReceipt, webEvidenceBatch } from "./evidence.js"
import { isTestCommand, testCommand, numericJoin } from "./plans.js"
import { mergePayload } from "./state_merge.js"
import { evaluationProfile, operatorEnabled } from "./state.js"

// Post-tool-execution hook: decodes Cortheon tool payloads, captures
// mutation diffs, binds host receipts to pending requests, and submits
// web provenance.
const createToolAfterHook = ({
  debug,
  captureMutationAfter,
  runRequestedTest,
  patchHygieneIssue,
  certifyCodeChange,
  submitAutomaticObservation,
  submitPassiveObservations,
}) => ({
  "tool.execute.after": async (input, output) => {
    const tool = String(input.tool || "")
    if (tool.startsWith("cortheon_")) {
      const payload = decodeToolPayload(output)
      const shape = Object.entries(output || {})
        .map(([key, value]) => `${key}:${typeof value}`)
        .join(",")
      await debug(
        `after ${tool}: output=${typeof output.output} decoded=${Boolean(payload)} ` +
          `session=${input.sessionID || "missing"} shape=${shape || "empty"}`,
      )
      if (!payload) return
      const operation = tool.replace("cortheon_cortheon_", "")
      const next = mergePayload(input.sessionID, operation, payload)
      await debug(
        `state ${tool}: request=${next.requestID || "none"} active=${Boolean(next.active)} ` +
          `complete=${Boolean(next.completed)} abandoned=${Boolean(next.abandoned)}`,
      )
      return
    }

    let state = investigations.get(input.sessionID)
    if (!state?.active) return
    if (
      state.automatic &&
      state.deliverable === "code_change" &&
      workspaceMutationTools.has(tool) &&
      typeof output.output === "string" &&
      output.output.trim() &&
      !/\b(?:error|failed|no changes to apply)\b/i.test(output.output)
    ) {
      state.mutated = true
      state.testFailed = false
      state.latestPassingTest = undefined
      await captureMutationAfter(state, tool, input.args, output)
      investigations.set(input.sessionID, state)
      if (operatorEnabled("retrieval") && state.requestedTestCommand) {
        const result = await runRequestedTest(input.sessionID, state)
        state = investigations.get(input.sessionID) || state
        state.runningAutomaticTest = false
        const hygieneIssue =
          result?.passed && result.observation
            ? await patchHygieneIssue(state)
            : undefined
        if (result?.passed && result.observation && !hygieneIssue) {
          state.latestPassingTest = result.observation
          state.testEverPassed = true
          state.testFailed = false
          output.output +=
            "\n\n[CORTHEON] Host-verified required test passed:\n" +
            result.summary
          investigations.set(input.sessionID, state)
          state = await certifyCodeChange(input.sessionID, state)
          if (state?.completed) {
            output.output +=
              "\n\n[CORTHEON] Diff and test are bound; completion is certified."
          }
        } else {
          state.latestPassingTest = undefined
          state.testFailed = true
          output.output +=
            "\n\n[CORTHEON] Required test did not pass; revise the implementation:\n" +
            (hygieneIssue
              ? `The test passed, but patch hygiene failed: ${hygieneIssue}`
              : result?.summary || "The host could not verify the test command.")
        }
        investigations.set(input.sessionID, state)
      }
    }
    if (
      state.automatic &&
      state.deliverable === "code_change" &&
      tool === "bash" &&
      isTestCommand(testCommand(input.args))
    ) {
      const command = testCommand(input.args)
      if (Number(output.metadata?.exit) === 0) {
        const receipt =
          "[CORTHEON_HOST_EVIDENCE] " +
          JSON.stringify({
            tool: "test",
            executor: "bash",
            outcome: "passed",
            args: { command: command.slice(0, 500) },
          })
        state.latestPassingTest = {
          kind: "test",
          content:
            `${receipt}\nCommand: ${command.slice(0, 500)}\nExit: 0\n` +
            boundedHostOutput(output.output),
          source: "opencode:bash:test",
          status: "verified",
        }
      } else {
        state.latestPassingTest = undefined
      }
      investigations.set(input.sessionID, state)
    }
    if (
      state.automatic &&
      ["websearch", "webfetch"].includes(tool) &&
      (state.deliverable === "research_answer" ||
        ["search", "fetch", "search_or_fetch"].includes(
          state.request?.capability || "",
        ))
    ) {
      const observations = webEvidenceBatch(tool, input.args, output.output, state)
      state.hostEvidence = undefined
      state.hostEvidenceBatch = observations
      investigations.set(input.sessionID, state)
      const next = operatorEnabled("retrieval") && state.requestID
        ? await submitAutomaticObservation(input.sessionID, state)
        : await submitPassiveObservations(input.sessionID, state, observations)
      if (operatorEnabled("retrieval") && typeof output.output === "string") {
        output.output =
          `${observations.map((item) => item.content).join("\n\n")}\n\n` +
          "[CORTHEON] Web provenance accepted; " +
          (next.request?.query
            ? `next evidence request: ${next.request.query}`
            : "synthesize with URL citations and report any material source conflict.")
      }
      return
    }
    if (
      evaluationProfile() &&
      !operatorEnabled("retrieval") &&
      operatorEnabled("verification") &&
      hostEvidenceTools.has(tool)
    ) {
      const hostOutput = boundedHostOutput(output.output)
      if (!hostOutput) return
      const passive = {
        content: `${evidenceReceipt(tool, input.args, hostOutput)}\n${hostOutput}`,
        kind: evidenceKind(tool),
        source: `opencode:${tool}`,
        status: evidenceStatus(tool),
      }
      await submitPassiveObservations(input.sessionID, state, [passive])
      return
    }
    if (!state.requestID) return
    if (!hostEvidenceTools.has(tool)) return
    const hostOutput = boundedHostOutput(output.output)
    if (!hostOutput) return
    const receipt = evidenceReceipt(tool, input.args, hostOutput)
    if (state.hostCallPassive && state.hostCall?.tool === tool) {
      state.hostCallPassive = false
      const passive = {
        content: `${receipt}\n${hostOutput}`,
        kind: evidenceKind(tool),
        source: `opencode:${tool}`,
        status: evidenceStatus(tool),
      }
      investigations.set(input.sessionID, state)
      if (state.automatic && state.cortheonSessionID) {
        await submitPassiveObservations(input.sessionID, state, [passive])
      }
      return
    }
    const requestedKind =
      state.request?.capability === "test"
        ? "test"
        : state.request?.capability === "diff"
          ? "diff"
          : evidenceKind(tool)
    const requestedStatus =
      requestedKind === "test"
        ? Number(output.metadata?.exit) === 0
          ? "verified"
          : "failed"
        : evidenceStatus(tool)
    state.hostEvidence = {
      content: `${receipt}\n${hostOutput}`,
      kind: requestedKind,
      source: `opencode:${tool}`,
      status: requestedStatus,
    }
    investigations.set(input.sessionID, state)
    if (state.automatic) {
      const next = await submitAutomaticObservation(input.sessionID, state)
      if (typeof output.output === "string") {
        output.output =
          `${receipt}\n${hostOutput}\n\n[CORTHEON] Live evidence accepted; ` +
          (next.request?.query
            ? `next evidence request: ${next.request.query}`
            : "continue to the answer.")
      }
      return
    }
    const instruction =
      `\n\n[CORTHEON HOST RECEIPT] This is live host-tool output for request ` +
      `${state.requestID}. Now call cortheon_observe with request_id=` +
      `${state.requestID} and only the smallest relevant excerpt. Do not answer yet.`
    if (typeof output.output === "string") {
      output.output = `${receipt}\n${hostOutput}${instruction}`
    }
  },
})

// Final text gate: last-chance rescues, deterministic completion release,
// and the certified/withheld/released-uncertified decision.
const createTextCompleteHook = ({
  acquireRequestedEvidence,
  submitAutomaticObservation,
  resyncEvidenceFromRuntime,
  ensureCausalChain,
  resolveCounterexampleRequest,
  ensureSemanticEvidence,
  attemptBoundedMultiRepair,
  acquireAutomaticResearch,
  certifyDeterministicResearch,
  runRequestedTest,
  patchHygieneIssue,
  certifyCodeChange,
  finalizeCodeChangeEvidence,
  submitAutomaticCompletion,
}) => ({
  "experimental.text.complete": async (input, output) => {
    let state = investigations.get(input.sessionID)
    if (!state) return
    if (!operatorEnabled("verification")) {
      investigations.delete(input.sessionID)
      return
    }

    const certified =
      state.completed && typeof state.answer === "string"
        ? state.answer
        : certifiedAnswers.get(input.sessionID)
    if (typeof certified === "string") {
      output.text = certified
      return
    }
    if (
      operatorEnabled("retrieval") &&
      state.automatic &&
      state.active &&
      state.requestID &&
      !state.evidenceIDs?.length
    ) {
      // A one-turn session can end with the startup acquisition race lost
      // and no later transform to retry it; this is the last chance.
      if (await acquireRequestedEvidence(state)) {
        investigations.set(input.sessionID, state)
        state = await submitAutomaticObservation(input.sessionID, state)
      }
    }
    state = await resyncEvidenceFromRuntime(input.sessionID, state)
    if (operatorEnabled("cross_source_derivation") && state.automatic && state.active) {
      state = await ensureCausalChain(state)
      investigations.set(input.sessionID, state)
    }
    if (operatorEnabled("discriminating_evidence") && state.automatic && state.active) {
      state = await resolveCounterexampleRequest(input.sessionID, state)
    }
    if (
      operatorEnabled("retrieval") &&
      operatorEnabled("cross_source_derivation") &&
      state.automatic &&
      state.active
    ) {
      // Turn-end rescues: give each deterministic mechanism a final chance
      // before the answer is judged, mirroring the code-change test rescue.
      state = await ensureSemanticEvidence(input.sessionID, state)
      if (
        state.deliverable === "research_answer" &&
        !state.automaticResearchAcquired &&
        !state.researchTurnEndRetry &&
        state.requestID &&
        ["contradiction_check", "primary_fetch"].includes(
          state.request?.parameters?.purpose || "",
        )
      ) {
        state.researchTurnEndRetry = true
        state.automaticResearchAttempted = false
        investigations.set(input.sessionID, state)
        state = await acquireAutomaticResearch(input.sessionID, state)
      }
      state = await certifyDeterministicResearch(input.sessionID, state)
    }
    if (
      operatorEnabled("retrieval") &&
      state.automatic &&
      state.active &&
      state.multiMutationTask &&
      !state.mutated &&
      !state.testFailed
    ) {
      state = await attemptBoundedMultiRepair(input.sessionID, state)
    }
    if (
      operatorEnabled("retrieval") &&
      state.automatic &&
      state.active &&
      state.deliverable === "code_change" &&
      state.mutated &&
      state.requestedTestCommand &&
      !state.latestPassingTest &&
      !state.testFailed
    ) {
      // The model edited but its step budget died before the requested test
      // ran; execute it now so real work is not lost to the turn limit.
      const result = await runRequestedTest(input.sessionID, state)
      state = investigations.get(input.sessionID) || state
      const hygieneIssue =
        result?.passed && result.observation
          ? await patchHygieneIssue(state)
          : undefined
      if (result?.passed && result.observation && !hygieneIssue) {
        state.latestPassingTest = result.observation
        state.testEverPassed = true
        state.testFailed = false
        investigations.set(input.sessionID, state)
        state = await certifyCodeChange(input.sessionID, state)
        if (state.completed && typeof state.answer === "string") {
          output.text = state.answer
          return
        }
      }
    }
    if (
      state.automatic &&
      state.active &&
      state.deliverable === "code_change"
    ) {
      state = await finalizeCodeChangeEvidence(input.sessionID, state)
    }
    if (
      state.automatic &&
      state.active &&
      state.cortheonSessionID &&
      state.evidenceIDs?.length &&
      output.text.trim()
    ) {
      state = await submitAutomaticCompletion(
        input.sessionID,
        state,
        output.text,
      )
      if (state.completed && typeof state.answer === "string") {
        output.text = state.answer
        return
      }
    }
    if (
      (state.required || state.active || state.abandoned) &&
      output.text.trim()
    ) {
      const gaps = state.verificationGaps?.length
        ? ` Unresolved gates: ${state.verificationGaps.join("; ")}.`
        : ""
      if (
        state.deliverable === "code_change" &&
        !state.latestPassingTest &&
        !(state.testEverPassed && !state.testFailed)
      ) {
        // A mutation whose requested test never passed (or that was rolled
        // back) must stay withheld: releasing "the patch succeeded" with a
        // caveat is still a false claim about the workspace.
        output.text =
          "[Cortheon withheld this output: no verified completion exists. " +
          `Continue with live host evidence or check the local Cortheon runtime.${gaps.replace(" Unresolved gates:", " Failed gates:")}]`
      } else {
        const supported =
          state.testEverPassed ||
          Boolean(state.releaseVersion?.sources?.length >= 2) ||
          Boolean(state.orderedPlan) ||
          Boolean(state.diagnosticConclusion) ||
          Boolean(state.semanticChain?.length >= 3) ||
          Boolean(state.causalChain?.segments?.length >= 2) ||
          Boolean(numericJoin(state.plan, state.evidenceSummary))
        const boilerplate =
          /maximum (?:number of |allowed )?steps|max(?:imum)? steps reached|completed the maximum number of steps/i.test(
            output.text.slice(0, 300),
          )
        if (supported && boilerplate) {
          // The model's final turn is a step-budget non-answer, but a
          // derivation stands ready: release the derived answer, never
          // the boilerplate.
          const derived = automaticCompletion(state, "")
          const answer = String(derived?.answer || "").trim()
          if (
            answer &&
            !/maximum (?:number of |allowed )?steps/i.test(answer.slice(0, 200))
          ) {
            output.text =
              "[Cortheon: released uncertified — completion verification did " +
              `not pass, so treat unverified statements accordingly.${gaps}]\n\n` +
              answer
          } else {
            output.text =
              "[Cortheon withheld this output: no verified completion exists. " +
              `Continue with live host evidence or check the local Cortheon runtime.${gaps.replace(" Unresolved gates:", " Failed gates:")}]`
          }
        } else if (supported) {
          // Release with an explicit caveat: an evidence-backed derivation
          // stands behind the draft even though certification is incomplete.
          output.text =
            "[Cortheon: released uncertified — completion verification did not " +
            `pass, so treat unverified statements accordingly.${gaps}]\n\n` +
            output.text
        } else {
          // No derivation supports the draft; releasing it would trade a
          // withhold for a false allow.
          output.text =
            "[Cortheon withheld this output: no verified completion exists. " +
            `Continue with live host evidence or check the local Cortheon runtime.${gaps.replace(" Unresolved gates:", " Failed gates:")}]`
        }
      }
    }
  },
})

export {
  createToolAfterHook,
  createTextCompleteHook,
}
