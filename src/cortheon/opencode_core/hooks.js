// Wiring-only assembly: constructs each focused factory in dependency
// order and returns the five host hooks. No logic lives here.
import { createAutoEvidence } from "./auto_evidence.js"
import { createAutoResearch } from "./auto_research.js"
import { createCertification } from "./completion.js"
import {
  createSessionLifecycleHooks,
  createSystemTransformHook,
} from "./hook_conversation.js"
import { createTextCompleteHook, createToolAfterHook } from "./hook_output.js"
import { createToolBeforeHook } from "./hook_tool_before.js"
import { createInvestigation, createRuntimeClient } from "./investigation.js"
import { createObservationSubmitter, createTestRunner } from "./host_access.js"
import { createRepairExecutor } from "./repair_exec.js"
import {
  createHostSession,
  createWorkspaceAccess,
} from "./host_access.js"

const createAdapterHooks = ({
  client,
  directory,
  hostShell,
  runtimeBase,
  runtimeToken,
  debug,
  debugSem,
}) => {
  const { runtimeHealth, spawnRuntimeOnce, localSourceFingerprint, runtimeCall } =
    createRuntimeClient({ runtimeBase, runtimeToken, debugSem })
  const { latestUserTask, sessionDiffEvidence } = createHostSession({
    client,
    directory,
  })
  const {
    readWorkspaceFile,
    captureMutationBefore,
    captureMutationAfter,
    capturedMutationDiffEvidence,
    patchHygieneIssue,
  } = createWorkspaceAccess({ client, directory, hostShell })
  const { runRequestedTest } = createTestRunner({
    hostShell,
    directory,
    debug,
  })
  const { submitAutomaticObservation, submitPassiveObservations } =
    createObservationSubmitter({ runtimeCall })
  const { acquireRequestedEvidence } = createAutoEvidence({
    client,
    directory,
    hostShell,
    debug,
    readWorkspaceFile,
  })
  const { acquireAutomaticResearch } = createAutoResearch({
    debug,
    submitAutomaticObservation,
    submitPassiveObservations,
  })
  const {
    finalizeCodeChangeEvidence,
    submitAutomaticCompletion,
    certifyCodeChange,
    certifyDeterministicResearch,
  } = createCertification({
    debug,
    runtimeCall,
    capturedMutationDiffEvidence,
    sessionDiffEvidence,
    submitAutomaticObservation,
    submitPassiveObservations,
  })
  const {
    ensureAutomaticInvestigation,
    ensureSemanticEvidence,
    resolveCounterexampleRequest,
    resyncEvidenceFromRuntime,
    ensureCausalChain,
  } = createInvestigation({
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
  })
  const { attemptBoundedMultiRepair, attemptBoundedAutomaticRepair } =
    createRepairExecutor({
      debug,
      client,
      directory,
      hostShell,
      readWorkspaceFile,
      runRequestedTest,
      patchHygieneIssue,
      certifyCodeChange,
    })

  return {
    ...createSystemTransformHook({
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
    }),
    ...createToolAfterHook({
      debug,
      captureMutationAfter,
      runRequestedTest,
      patchHygieneIssue,
      certifyCodeChange,
      submitAutomaticObservation,
      submitPassiveObservations,
    }),
    ...createToolBeforeHook({
      debug,
      directory,
      latestUserTask,
      acquireRequestedEvidence,
      captureMutationBefore,
    }),
    ...createTextCompleteHook({
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
    }),
    ...createSessionLifecycleHooks({
      debug,
      client,
      directory,
      runtimeCall,
    }),
  }
}

export { createAdapterHooks }
