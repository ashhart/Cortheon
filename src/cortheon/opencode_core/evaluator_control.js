import {closeSync, readFileSync} from "node:fs"

const controlKeys = [
  "schema_version", "evaluation_profile", "cognitive_token",
  "evaluator_max_steps", "auto_enable", "benchmark_capture_candidate",
  "max_host_tool_calls",
]

function readEvaluatorControl() {
  const descriptorText = typeof process !== "undefined"
    ? process.env.CORTHEON_CONTROL_FD
    : undefined
  if (typeof process !== "undefined") delete process.env.CORTHEON_CONTROL_FD
  if (descriptorText === undefined) return {present: false}
  if (!/^[1-9]\d{0,6}$/.test(descriptorText)) return {present: true}
  const descriptor = Number(descriptorText)
  let raw
  try {
    raw = readFileSync(descriptor, {encoding: "utf8"})
  } catch {
    return {present: true}
  } finally {
    try { closeSync(descriptor) } catch {}
  }
  if (raw.length > 16_384) return {present: true}
  try {
    const value = JSON.parse(raw)
    if (
      !value || typeof value !== "object" || Array.isArray(value) ||
      Object.keys(value).sort().join("\0") !== [...controlKeys].sort().join("\0") ||
      value.schema_version !== 1 ||
      !(value.evaluation_profile === null ||
        (typeof value.evaluation_profile === "object" && !Array.isArray(value.evaluation_profile))) ||
      typeof value.cognitive_token !== "string" || value.cognitive_token.length > 4_096 ||
      !(value.evaluator_max_steps === null ||
        (Number.isInteger(value.evaluator_max_steps) &&
          value.evaluator_max_steps >= 1 && value.evaluator_max_steps <= 1_024)) ||
      typeof value.auto_enable !== "boolean" ||
      typeof value.benchmark_capture_candidate !== "boolean" ||
      !(Number.isInteger(value.max_host_tool_calls) &&
        value.max_host_tool_calls >= 1 && value.max_host_tool_calls <= 64)
    ) return {present: true}
    return {present: true, value}
  } catch {
    return {present: true}
  }
}

const evaluatorControl = readEvaluatorControl()
if (evaluatorControl.present && !evaluatorControl.value) {
  throw new Error("invalid evaluator control descriptor or payload")
}

export {evaluatorControl}
