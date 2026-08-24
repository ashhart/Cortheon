// Stable facade for the OpenCode adapter. The implementation lives in the
// focused opencode_core modules; host configuration continues to point at
// this file only, and the relative ESM imports resolve identically from a
// checkout and from an installed wheel.
import { createAdapterHooks } from "./opencode_core/hooks.js"
import {
  deriveExactMatchMismatchInference,
  deriveKeyedCollisionInference,
} from "./opencode_core/joins.js"
import {
  deriveDiagnosticConclusion,
  numericJoin,
} from "./opencode_core/plans.js"
import {
  heartbeatIntervalMs,
  initialEnvironment,
  investigations,
} from "./opencode_core/state.js"

// Pure derivation operators exported for direct testing.
const cortheonOperators = {
  deriveDiagnosticConclusion,
  deriveKeyedCollisionInference,
  deriveExactMatchMismatchInference,
  numericJoin,
}
export { cortheonOperators }


export const CortheonPlugin = async ({ client, directory, $: hostShell }) => {
  const runtimeBase = String(
    initialEnvironment.runtimeURL ||
      "http://127.0.0.1:8743",
  ).replace(/\/+$/, "")
  const runtimeToken = initialEnvironment.token

  const heartbeatTimer = setInterval(() => {
    for (const state of investigations.values()) {
      if (!state?.active || !state.cortheonSessionID) continue
      void fetch(`${runtimeBase}/v1/heartbeat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(runtimeToken
            ? { Authorization: `Bearer ${runtimeToken}` }
            : {}),
        },
        body: JSON.stringify({ session_id: state.cortheonSessionID }),
      }).catch(() => {})
    }
  }, heartbeatIntervalMs)
  heartbeatTimer.unref?.()

  const debug = async (message) => {
    if (typeof process === "undefined" || process.env.CORTHEON_PLUGIN_DEBUG !== "1") {
      return
    }
    try {
      await client.app.log({
        query: { directory },
        body: {
          service: "cortheon",
          level: "info",
          message,
        },
      })
    } catch {
    }
  }

  const debugSem = async (payload) => {
    if (typeof process === "undefined" || !process.env?.CORTHEON_JOIN_DEBUG) {
      return
    }
    try {
      const fs = await import("node:fs")
      fs.appendFileSync(
        process.env.CORTHEON_JOIN_DEBUG,
        JSON.stringify({ tag: "[DEBUG-sem]", ...payload }) + "\n",
      )
    } catch {}
  }

  return createAdapterHooks({
    client,
    directory,
    hostShell,
    runtimeBase,
    runtimeToken,
    debug,
    debugSem,
  })
}
