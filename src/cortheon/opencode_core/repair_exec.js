import { goalCodePaths } from "./joins.js"
import { investigations } from "./state.js"
import {
  deriveExactDocumentEdits,
  deriveSimpleRepairPlans,
} from "./repair_derive.js"
import { focusedTextDiff } from "./state.js"

// Bounded automatic repairs: derives edit sets from goal-named files, applies
// them through a hardened single-occurrence replace, and rolls back unless
// the host test passes.
const createRepairExecutor = ({
  debug,
  client,
  directory,
  hostShell,
  readWorkspaceFile,
  runRequestedTest,
  patchHygieneIssue,
  certifyCodeChange,
}) => {
  const replaceHostFileText = async (path, oldString, newString) => {
    if (
      typeof hostShell !== "function" ||
      typeof path !== "string" ||
      path.length === 0 ||
      path.length > 500 ||
      path.startsWith("/") ||
      path.includes("\0") ||
      path.split("/").some((part) => !part || part === "." || part === "..") ||
      typeof oldString !== "string" ||
      oldString.length === 0 ||
      oldString.length > 4_000 ||
      typeof newString !== "string" ||
      newString.length > 4_000 ||
      oldString === newString
    ) {
      return false
    }
    const writer = [
      "from pathlib import Path",
      "import os, sys, tempfile",
      "root = Path.cwd().resolve()",
      "relative = Path(sys.argv[1])",
      "candidate = root / relative",
      "cursor = root",
      "for part in relative.parts:",
      "    cursor = cursor / part",
      "    if cursor.is_symlink(): raise SystemExit(21)",
      "target = candidate.resolve(strict=True)",
      "try: target.relative_to(root)",
      "except ValueError: raise SystemExit(22)",
      "if not target.is_file(): raise SystemExit(23)",
      "with target.open('r', encoding='utf-8', newline='') as stream:",
      "    content = stream.read()",
      "old, new = sys.argv[2], sys.argv[3]",
      "if len(content) > 50000 or content.count(old) != 1: raise SystemExit(24)",
      "updated = content.replace(old, new, 1)",
      "descriptor, temporary = tempfile.mkstemp(",
      "    dir=target.parent, prefix='.cortheon-repair-', text=True",
      ")",
      "try:",
      "    os.fchmod(descriptor, target.stat().st_mode)",
      "    with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:",
      "        stream.write(updated)",
      "        stream.flush()",
      "        os.fsync(stream.fileno())",
      "    os.replace(temporary, target)",
      "finally:",
      "    if os.path.exists(temporary): os.unlink(temporary)",
    ].join("\n")
    try {
      const result =
        await hostShell`python3 -I -c ${writer} ${path} ${oldString} ${newString}`
          .cwd(directory)
          .quiet()
          .nothrow()
      return Number(result?.exitCode) === 0
    } catch {
      return false
    }
  }

  const deriveMultiEditsFromGoal = async (state) => {
    const paths = [
      ...new Set(
        String(state?.goal || "").match(
          /[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|rb|php|swift|md|markdown|rst|txt)\b/g,
        ) || [],
      ),
    ]
      .filter((path) => !path.startsWith("/") && !path.includes(".."))
      .slice(0, 8)
    const protectedPaths = Array.isArray(state?.protectedTestPaths)
      ? state.protectedTestPaths
      : []
    const targets = paths.filter(
      (item) =>
        !protectedPaths.some(
          (test) => item === test || item.endsWith(`/${test}`),
        ),
    )
    const tests = paths.filter((item) =>
      protectedPaths.some((test) => item === test || item.endsWith(`/${test}`)),
    )
    if (targets.length === 0 || tests.length === 0) return undefined
    try {
      const reads = await Promise.all(
        [...targets, ...tests].map(async (item) => {
          const content = await readWorkspaceFile(item)
          if (typeof content !== "string") return undefined
          return { path: item, source: content }
        }),
      )
      if (reads.some((item) => !item)) return undefined
      return {
        repairPlans: deriveSimpleRepairPlans(reads) || [],
        documentEdits: deriveExactDocumentEdits(reads, state.goal) || [],
      }
    } catch {
      return undefined
    }
  }

  const attemptBoundedMultiRepair = async (hostSessionID, state) => {
    if (
      state?.automatic &&
      state.active &&
      state.multiMutationTask &&
      state.deliverable === "code_change" &&
      state.requestedTestCommand &&
      !state.automaticMultiRepairAttempted &&
      !state.testFailed &&
      !(Array.isArray(state.repairPlans) && state.repairPlans.length > 0) &&
      !(Array.isArray(state.documentEdits) && state.documentEdits.length > 0)
    ) {
      // Model-driven reads never derive the edit set; arm the transaction
      // from the goal-named implementation, document, and test files.
      const derived = await deriveMultiEditsFromGoal(state)
      if (derived) {
        state.repairPlans = derived.repairPlans
        state.documentEdits = derived.documentEdits
        investigations.set(hostSessionID, state)
      }
    }
    const edits = [
      ...(Array.isArray(state?.repairPlans) ? state.repairPlans : []),
      ...(Array.isArray(state?.documentEdits) ? state.documentEdits : []),
    ]
    const protectedPaths = Array.isArray(state?.protectedTestPaths)
      ? state.protectedTestPaths
      : []
    if (
      !state?.automatic ||
      !state.active ||
      !state.multiMutationTask ||
      state.deliverable !== "code_change" ||
      (state.requestID &&
        !["diff", "test"].includes(state.request?.capability || "")) ||
      !state.evidenceIDs?.length ||
      state.automaticMultiRepairAttempted ||
      state.testFailed ||
      !state.requestedTestCommand ||
      edits.length < state.mutationRequirementCount ||
      edits.length > 6 ||
      edits.some(
        (edit) =>
          !edit?.path ||
          !edit.oldString ||
          typeof edit.newString !== "string" ||
          protectedPaths.some(
            (path) => edit.path === path || edit.path.endsWith(`/${path}`),
          ),
      )
    ) {
      return state
    }
    state.automaticMultiRepairAttempted = true
    investigations.set(hostSessionID, state)
    const applied = []
    for (const edit of edits) {
      if (!(await replaceHostFileText(edit.path, edit.oldString, edit.newString))) {
        for (const previous of [...applied].reverse()) {
          await replaceHostFileText(
            previous.path,
            previous.newString,
            previous.oldString,
          )
        }
        state.testFailed = true
        state.automaticRepairFailure =
          `The bounded multi-edit transaction could not update ${edit.path}.`
        investigations.set(hostSessionID, state)
        return state
      }
      applied.push(edit)
    }
    state.mutated = true
    state.mutationDiffs = applied.map((edit) => ({
      path: edit.path,
      tool: "host_bounded_multi_edit",
      content: focusedTextDiff(edit.path, edit.oldString, edit.newString),
    }))
    state.latestPassingTest = undefined
    investigations.set(hostSessionID, state)
    const result = await runRequestedTest(hostSessionID, state)
    state = investigations.get(hostSessionID) || state
    const hygieneIssue =
      result?.passed && result.observation
        ? await patchHygieneIssue(state)
        : undefined
    if (!result?.passed || !result.observation || hygieneIssue) {
      let rolledBack = true
      for (const edit of [...applied].reverse()) {
        rolledBack =
          (await replaceHostFileText(edit.path, edit.newString, edit.oldString)) &&
          rolledBack
      }
      state.mutated = !rolledBack
      state.mutationDiffs = rolledBack ? [] : state.mutationDiffs
      state.latestPassingTest = undefined
      state.testFailed = true
      state.automaticRepairFailure =
        hygieneIssue ||
        result?.summary ||
        "The host could not verify the bounded multi-edit repair."
      investigations.set(hostSessionID, state)
      return state
    }
    state.latestPassingTest = result.observation
    state.testEverPassed = true
    state.testFailed = false
    investigations.set(hostSessionID, state)
    return certifyCodeChange(hostSessionID, state)
  }

  const deriveRepairPlansFromGoal = async (state) => {
    const paths = goalCodePaths(state?.goal)
    const protectedPaths = Array.isArray(state?.protectedTestPaths)
      ? state.protectedTestPaths
      : []
    const implPaths = paths.filter(
      (path) =>
        !protectedPaths.some(
          (test) => path === test || path.endsWith(`/${test}`),
        ),
    )
    const testPaths = paths.filter((path) =>
      protectedPaths.some((test) => path === test || path.endsWith(`/${test}`)),
    )
    if (implPaths.length === 0 || testPaths.length === 0) return undefined
    try {
      const reads = await Promise.all(
        [...implPaths, ...testPaths].map(async (path) => {
          const content = await readWorkspaceFile(path)
          if (typeof content !== "string") return undefined
          return { path, source: content }
        }),
      )
      if (reads.some((item) => !item)) return undefined
      return deriveSimpleRepairPlans(reads)
    } catch {
      return undefined
    }
  }

  const attemptBoundedAutomaticRepair = async (hostSessionID, state) => {
    let plan = state?.repairPlan
    const requestedPaths = Array.isArray(state?.plan?.paths)
      ? state.plan.paths
      : []
    const protectedPaths = Array.isArray(state?.protectedTestPaths)
      ? state.protectedTestPaths
      : []
    if (
      !state?.automatic ||
      !state.active ||
      state.deliverable !== "code_change" ||
      state.multiMutationTask ||
      // A pending diff/test request is satisfied by this repair's own
      // captured diff and host-run test, so it must not block the attempt.
      (state.requestID &&
        !["diff", "test"].includes(state.request?.capability || "")) ||
      !state.evidenceIDs?.length ||
      state.automaticRepairAttempted ||
      state.testFailed ||
      !state.requestedTestCommand
    ) {
      return state
    }
    if (!plan) {
      // Model-driven reads never derive repair plans, so arm the transaction
      // here from the goal-named implementation/test pair.
      const derived = await deriveRepairPlansFromGoal(state)
      if (derived?.length) {
        state.repairPlans = derived
        state.repairPlan = derived[0]
        plan = derived[0]
        investigations.set(hostSessionID, state)
      }
    }
    const allowedPaths =
      requestedPaths.length > 0 ? requestedPaths : goalCodePaths(state.goal)
    if (
      !plan ||
      !Number.isInteger(plan.examples) ||
      plan.examples < 1 ||
      !allowedPaths.includes(plan.path) ||
      protectedPaths.some(
        (path) => plan.path === path || plan.path.endsWith(`/${path}`),
      )
    ) {
      return state
    }
    state.automaticRepairAttempted = true
    investigations.set(hostSessionID, state)
    let before
    try {
      const response = await client.file.read({
        query: { directory, path: plan.path },
      })
      const file = response?.data || response
      if (
        file?.type !== "text" ||
        typeof file.content !== "string" ||
        file.content.length > 50_000
      ) {
        return state
      }
      before = file.content
    } catch {
      return state
    }
    if (
      before.split(plan.oldString).length !== 2 ||
      plan.path.startsWith("/") ||
      plan.path.includes("\0") ||
      plan.path.split("/").some((part) => !part || part === "." || part === "..")
    ) {
      return state
    }
    const after = before.replace(plan.oldString, plan.newString)
    if (
      !(await replaceHostFileText(
        plan.path,
        plan.oldString,
        plan.newString,
      ))
    ) {
      return state
    }
    let applied = false
    try {
      const response = await client.file.read({
        query: { directory, path: plan.path },
      })
      const file = response?.data || response
      applied =
        file?.type === "text" &&
        typeof file.content === "string" &&
        file.content === after
    } catch {
    }
    if (!applied) {
      await replaceHostFileText(plan.path, plan.newString, plan.oldString)
      return state
    }
    state.mutated = true
    state.mutationDiffs = [
      {
        path: plan.path,
        tool: "host_bounded_edit",
        content: focusedTextDiff(plan.path, before, after),
      },
    ]
    state.latestPassingTest = undefined
    investigations.set(hostSessionID, state)
    const result = await runRequestedTest(hostSessionID, state)
    state = investigations.get(hostSessionID) || state
    const hygieneIssue =
      result?.passed && result.observation
        ? await patchHygieneIssue(state)
        : undefined
    if (!result?.passed || !result.observation || hygieneIssue) {
      const rolledBack = await replaceHostFileText(
        plan.path,
        plan.newString,
        plan.oldString,
      )
      state.mutated = !rolledBack
      state.mutationDiffs = rolledBack ? [] : state.mutationDiffs
      state.latestPassingTest = undefined
      state.testFailed = true
      state.automaticRepairFailure =
        hygieneIssue ||
        result?.summary ||
        "The host could not verify the bounded repair."
      investigations.set(hostSessionID, state)
      await debug(
        `bounded repair rejected path=${plan.path} rolled_back=${rolledBack}`,
      )
      return state
    }
    state.latestPassingTest = result.observation
    state.testEverPassed = true
    state.testFailed = false
    investigations.set(hostSessionID, state)
    state = await certifyCodeChange(hostSessionID, state)
    await debug(
      `bounded repair certified path=${plan.path} ` +
        `complete=${Boolean(state?.completed)}`,
    )
    return state
  }

  return { attemptBoundedMultiRepair, attemptBoundedAutomaticRepair }
}

export { createRepairExecutor }
