import {
  compactEvidence,
  mergeEvidenceRecords,
  receiptOutcome,
} from "./evidence.js"
import {
  pendingEvidenceRequest,
  pendingRequest,
  sessionID,
} from "./state.js"
import { plannedGrep, plannedReads } from "./plans.js"
import { investigations, certifiedAnswers } from "./state.js"
function mergePayload(hostSessionID, operation, payload) {
  const previous = investigations.get(hostSessionID) || {}
  const request = pendingEvidenceRequest(payload)
  const publicHypotheses = Array.isArray(payload?.context?.hypotheses)
    ? payload.context.hypotheses
        .filter(
          (item) =>
            item &&
            typeof item.statement === "string" &&
            typeof item.falsification_test === "string",
        )
        .slice(0, 5)
    : []
  const reasonAction =
    payload?.next_action?.type === "reason" &&
    typeof payload.next_action.instruction === "string"
      ? payload.next_action.instruction.trim()
      : undefined
  const verificationGaps = Array.isArray(payload?.verification?.gaps)
    ? payload.verification.gaps
        .filter((item) => typeof item === "string" && item.trim())
        .slice(0, 4)
    : []
  const cognition = payload?.cognition
  const cognitionMoves = Array.isArray(cognition?.reasoning_moves)
    ? cognition.reasoning_moves.filter(
        (item) => typeof item === "string" && item.trim(),
      )
    : []
  const cognitionInsight = Array.isArray(cognition?.derived_insights)
    ? cognition.derived_insights.find(
        (item) => typeof item?.statement === "string" && item.statement.trim(),
      )
    : undefined
  const taskRequirements = Array.isArray(cognition?.task_frame?.requirements)
    ? cognition.task_frame.requirements
    : []
  const mutationRequirementCount = taskRequirements.filter(
    (item) => item?.proof === "mutation",
  ).length
  const cognitionRequirements = taskRequirements
        .filter(
          (item) =>
            item &&
            item.status !== "covered" &&
            typeof item.statement === "string" &&
            item.statement.trim(),
        )
        .slice(0, 4)
        .map(
          (item) =>
            `${item.requirement_id || "requirement"}: ${item.statement.trim()} ` +
            `[${item.status || "unresolved"}]`,
        )
  const cognitionBrief =
    typeof cognition?.stage === "string" && cognition.stage.trim()
      ? {
          stage: cognition.stage.trim(),
          move: cognitionMoves[0]?.trim(),
          derivedInsight: cognitionInsight?.statement.trim(),
          unresolvedRequirements: cognitionRequirements,
          decisionRule:
            typeof cognition?.decision_rule === "string"
              ? cognition.decision_rule.trim()
              : undefined,
        }
      : undefined
  const derivations = Array.isArray(
    payload?.context?.deterministic_derivations,
  )
    ? payload.context.deterministic_derivations
    : []
  const release = derivations.find(
    (item) =>
      item?.operation === "release_version" &&
      typeof item?.value === "string" &&
      Array.isArray(item?.sources),
  )
  const semanticDerivation = derivations.find(
    (item) =>
      (
        (
          item?.operation === "semantic_chain" &&
          item?.confidence === "deterministic_relational_match"
        ) ||
        (
          item?.operation === "semantic_rule" &&
          item?.confidence === "deterministic_conjunctive_rule"
        )
      ) &&
      Array.isArray(item?.nodes) &&
      item.nodes.length >= 3 &&
      item.nodes.length <= 7 &&
      item.nodes.every(
        (node) =>
          typeof node === "string" &&
          node.trim().length > 0 &&
          node.trim().length <= 120,
      ) &&
      Array.isArray(item?.sources) &&
      item.sources.length === item.nodes.length - 1 &&
      item.sources.every(
        (source) =>
          typeof source === "string" &&
          source.trim().length > 0 &&
          source.trim().length <= 1_024,
      ) &&
      new Set(
        item.sources.map((source) => source.trim().toLowerCase()),
      ).size >= 2,
  )
  const orderedDerivation = derivations.find(
    (item) =>
      item?.operation === "ordered_plan" &&
      item?.confidence === "deterministic_constraint_order" &&
      Array.isArray(item?.nodes) &&
      item.nodes.length >= 3 &&
      item.nodes.length <= 8 &&
      item.nodes.every(
        (node) => typeof node === "string" && node.trim().length <= 120,
      ) &&
      item.owners &&
      typeof item.owners === "object" &&
      Array.isArray(item.sources) &&
      new Set(item.sources).size >= 2,
  )
  const orderedNodes = orderedDerivation?.nodes.map((node) => node.trim())
  const orderedChain = orderedNodes?.join(" → ")
  const next = {
    ...previous,
    required: true,
    active: !["complete", "abandoned", "disengaged"].includes(payload.status),
    cortheonSessionID: sessionID(payload) || previous.cortheonSessionID,
    requestID: pendingRequest(payload),
    request,
    plan: plannedGrep(request) || plannedReads(request) || previous.plan,
    deliverable: payload?.session?.deliverable || previous.deliverable,
    goal: payload?.context?.goal || previous.goal,
    releaseVersion: release
      ? {
          value: release.value,
          sources: release.sources.filter(
            (item) => typeof item === "string" && /^https?:\/\//.test(item),
          ),
        }
      : previous.releaseVersion,
    semanticChain: semanticDerivation
      ? semanticDerivation.nodes.map((node) => node.trim())
      : operation === "start"
        ? undefined
        : previous.semanticChain,
    orderedPlan: orderedDerivation
      ? {
          answer:
            "Safe evidence-bound order:\n" +
            orderedNodes
              .map(
                (node, index) =>
                  `${index + 1}. ${node}` +
                  (orderedDerivation.owners[node]
                    ? ` — owner: ${orderedDerivation.owners[node]}`
                    : ""),
              )
              .join("\n") +
            `\nDependency chain: ${orderedChain}.`,
          claim:
            `Accepted live documents establish this dependency order: ` +
            `${orderedChain}.`,
        }
      : previous.orderedPlan,
    hypotheses:
      publicHypotheses.length > 0
        ? publicHypotheses
        : operation === "start"
          ? []
          : previous.hypotheses || [],
    originatedHypotheses:
      publicHypotheses.some(
        (item) => item.origin === "substrate_abduction",
      )
        ? publicHypotheses.filter(
              (item) => item.origin === "substrate_abduction",
            )
        : previous.originatedHypotheses || [],
    reasonAction,
    verificationGaps,
    cognition: cognitionBrief || previous.cognition,
    multiMutationTask:
      previous.multiMutationTask || mutationRequirementCount > 1,
    mutationRequirementCount: Math.max(
      Number(previous.mutationRequirementCount || 0),
      mutationRequirementCount,
    ),
  }
  if (operation === "start") {
    next.hostEvidence = undefined
    next.hostEvidenceBatch = undefined
    next.evidenceRecords = []
  }
  if (operation === "observe") {
    const submitted = Array.isArray(previous.hostEvidenceBatch)
      ? previous.hostEvidenceBatch
      : previous.hostEvidence
        ? [previous.hostEvidence]
        : []
    next.deterministicOutcome =
      submitted
        .map((item) => receiptOutcome(item?.content))
        .find((item) => item === "match" || item === "no_match") ||
      previous.deterministicOutcome
    next.evidenceRecords = mergeEvidenceRecords(
      previous.evidenceRecords,
      submitted,
    )
    next.evidenceSummary =
      submitted.length > 0
        ? compactEvidence(next.evidenceRecords)
        : previous.evidenceSummary
    next.hostEvidence = undefined
    next.hostEvidenceBatch = undefined
    const accepted = Array.isArray(payload.accepted_evidence_ids)
      ? payload.accepted_evidence_ids.filter((item) => typeof item === "string")
      : []
    next.evidenceIDs = [
      ...new Set([...(previous.evidenceIDs || []), ...accepted]),
    ]
  }
  if (payload.status === "complete" && typeof payload.answer === "string") {
    next.active = false
    next.completed = true
    next.answer = payload.answer
    certifiedAnswers.set(hostSessionID, payload.answer)
  } else if (payload.status === "abandoned") {
    next.active = false
    next.completed = false
    next.abandoned = true
  }
  investigations.set(hostSessionID, next)
  return next
}
export { mergePayload }
