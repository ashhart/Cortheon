import {closeSync, readFileSync} from "node:fs"

const descriptorText = process.env.CORTHEON_CONTROL_FD
delete process.env.CORTHEON_CONTROL_FD
if (!descriptorText || !/^[1-9]\d{0,6}$/.test(descriptorText)) {
  throw new Error("missing evaluator control descriptor")
}
const descriptor = Number(descriptorText)
let control
try {
  const raw = readFileSync(descriptor, {encoding: "utf8"})
  if (raw.length > 16_384) throw new Error("oversized evaluator control")
  control = JSON.parse(raw)
} finally {
  try { closeSync(descriptor) } catch {}
}
const validRuntimeURL = (value) => {
  try {
    const url = new URL(value)
    const port = Number(url.port)
    return (
      url.protocol === "http:" &&
      (url.hostname === "127.0.0.1" || url.hostname === "localhost") &&
      Number.isInteger(port) && port >= 1 && port <= 65_535 &&
      url.pathname === "/" && !url.search && !url.hash
    )
  } catch {
    return false
  }
}
if (
  !control || control.schema_version !== 1 ||
  typeof control.cognitive_token !== "string" ||
  control.cognitive_token.length < 32 || control.cognitive_token.length > 4_096 ||
  typeof control.runtime_url !== "string" || !validRuntimeURL(control.runtime_url)
) {
  throw new Error("invalid evaluator control payload")
}

const frozen = await import("./program.js")

export const CortheonPlugin = async (context) => {
  process.env.CORTHEON_COGNITIVE_TOKEN = control.cognitive_token
  process.env.CORTHEON_RUNTIME_URL = control.runtime_url
  try {
    return await frozen.CortheonPlugin(context)
  } finally {
    delete process.env.CORTHEON_COGNITIVE_TOKEN
    delete process.env.CORTHEON_RUNTIME_URL
  }
}
