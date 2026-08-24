

function plannedGrep(request) {
  if (request?.capability !== "grep" || typeof request.query !== "string") {
    return undefined
  }
  const match = request.query.match(
    /pattern '([^']{1,300})' and path '([^']{1,500})'/i,
  )
  if (!match) return undefined
  return { pattern: match[1], path: match[2] }
}

function plannedReads(request) {
  if (request?.capability !== "read_many") return undefined
  const paths = Array.isArray(request?.parameters?.paths)
    ? request.parameters.paths
        .filter((item) => typeof item === "string")
        .slice(0, 6)
    : []
  const symbols = Array.isArray(request?.parameters?.symbols)
    ? request.parameters.symbols
        .filter((item) => typeof item === "string")
        .slice(0, 12)
    : []
  const terms = Array.isArray(request?.parameters?.terms)
    ? request.parameters.terms
        .filter((item) => typeof item === "string")
        .slice(0, 12)
    : []
  const operation =
    typeof request?.parameters?.operation === "string"
      ? request.parameters.operation
      : undefined
  return paths.length >= 1
    ? { paths, symbols: [...new Set([...symbols, ...terms])], operation }
    : undefined
}

function plannedProjectDiscovery(request) {
  const operation = request?.parameters?.operation
  if (
    request?.capability !== "search" ||
    !["document_discovery", "code_discovery"].includes(operation)
  ) {
    return undefined
  }
  const maximum = Number(request?.parameters?.max_candidates)
  return {
    maximum:
      Number.isInteger(maximum) && maximum >= 2 && maximum <= 6
        ? maximum
        : 6,
    operation,
    preferTests: Boolean(request?.parameters?.prefer_tests),
  }
}

function scopedPredicatePlan(plan) {
  return (
    typeof plan?.path === "string" &&
    plan.path.length > 0 &&
    typeof plan?.pattern === "string" &&
    plan.pattern.length > 0
  )
}

function numericJoin(plan, evidence) {
  if (plan?.operation !== "sum" || !Array.isArray(plan.symbols)) {
    return undefined
  }
  const values = []
  for (const symbol of plan.symbols) {
    const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    const matcher = new RegExp(
      `\\b${escaped}\\b(?:\\s*:[^=\\n]+)?\\s*=\\s*` +
        "([-+]?(?:0[xX][0-9A-Fa-f][0-9A-Fa-f_]*|[0-9][0-9_,]*))",
      "g",
    )
    const found = [...String(evidence || "").matchAll(matcher)].map(
      (match) => match[1],
    )
    let distinct
    try {
      distinct = [
        ...new Set(
          found.map((token) =>
            BigInt(token.replaceAll("_", "").replaceAll(",", "")).toString(),
          ),
        ),
      ]
    } catch {
      return undefined
    }
    if (distinct.length !== 1) return undefined
    values.push({ symbol, value: BigInt(distinct[0]) })
  }
  if (values.length === 0) return undefined
  const total = values.reduce((sum, item) => sum + item.value, 0n)
  const operands = values.map((item) => item.value.toString()).join(" + ")
  const bindings = values
    .map((item) => `${item.symbol} = ${item.value.toString()}`)
    .join("; ")
  const result = total.toString()
  return {
    answer: `${bindings}. ${operands} = ${result}.`,
    claim: `The evidence-bound sum of ${plan.symbols.join(", ")} is ${result}.`,
  }
}

function constrainRedundantTool(state, tool, args) {
  if (tool === "read") {
    const requested = String(args?.filePath || "")
    const paths = Array.isArray(state.plan?.paths)
      ? state.plan.paths
      : state.plan?.path
        ? [state.plan.path]
        : []
    const path = paths.find(
      (item) => requested === item || requested.endsWith(`/${item}`),
    )
    if (!path) return false
    const escaped = path.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    const line = String(state.semanticBridge || state.evidenceSummary || "").match(
      new RegExp(`(?:^|\\n)${escaped}:(\\d+):`),
    )
    args.filePath = path
    if (line) {
      // Pinpoint a re-read around the already-evidenced line with real context.
      args.offset = Math.max(1, Number(line[1]) - 2)
      args.limit = 25
    } else {
      // No prior line marker: a 1-line clamp starves evidence acquisition, so
      // keep the read bounded but wide enough to expose real definitions.
      args.offset = Math.max(1, Number(args.offset) || 1)
      const requestedLimit = Number(args.limit)
      args.limit = Math.min(requestedLimit > 0 ? requestedLimit : 400, 400)
    }
    return true
  }
  if (tool === "grep" && state.plan?.pattern && state.plan?.path) {
    args.pattern = state.plan.pattern
    args.path = state.plan.path
    delete args.include
    return true
  }
  if (
    tool === "grep" &&
    Array.isArray(state.plan?.symbols) &&
    state.plan.symbols.length > 0
  ) {
    const requested = String(args?.pattern || "")
    const symbol = state.plan.symbols.find((item) => item === requested)
    const path = state.plan.paths?.find(
      (item) => args?.path === item || String(args?.path || "").endsWith(`/${item}`),
    )
    if (!symbol || !path) return false
    args.pattern = symbol
    args.path = path
    delete args.include
    return true
  }
  if (tool === "grep" && state.plan?.operation === "semantic_join") {
    const requested = String(args?.path || "")
    const path = state.plan.paths?.find(
      (item) => requested === item || requested.endsWith(`/${item}`),
    )
    const words = String(args?.pattern || "").match(/[A-Za-z0-9_-]+/g) || []
    if (!path || words.length === 0) return false
    args.pattern = words.slice(0, 8).join(".*")
    args.path = path
    delete args.include
    return true
  }
  return false
}

function mutationTarget(args) {
  const target = args?.filePath || args?.path
  return typeof target === "string" && target.trim()
    ? target.trim().slice(0, 1000)
    : undefined
}

function duplicateTerminalLine(content) {
  let pending
  const lines = String(content || "").split("\n")
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (!line.trim() || /^\s*#/.test(line)) continue
    const terminal = line.match(/^(\s*)(?:return|raise|break|continue)\b/)
    const indent = line.match(/^\s*/)?.[0].replaceAll("\t", "    ").length || 0
    if (pending && indent < pending.indent) pending = undefined
    if (
      pending &&
      indent === pending.indent &&
      terminal
    ) {
      return `lines ${pending.line} and ${index + 1} contain consecutive ` +
        "terminal statements at the same indentation"
    }
    if (pending && indent === pending.indent) pending = undefined
    if (terminal) pending = { indent, line: index + 1 }
  }
  return undefined
}

function testCommand(args) {
  return String(args?.command || args?.cmd || "").trim()
}

function isTestCommand(command) {
  return /(?:^|\s)(?:pytest|py\.test|python3?\s+-m\s+(?:pytest|unittest)|npm(?:\s+run)?\s+test|pnpm(?:\s+run)?\s+test|yarn\s+test|bun\s+test|cargo\s+test|go\s+test|dotnet\s+test|mvn(?:w)?\b[^\n]*\btest|gradle(?:w)?\b[^\n]*\btest)(?:\s|$)/i.test(
    command,
  )
}

function requestedTestCommand(task) {
  const match = String(task || "").match(
    /\brun\s+(.+?)(?=\s+after\b|\s+and\s+(?:report|verify|then)\b|$)/i,
  )
  if (!match) return undefined
  const command = match[1].replace(/^[`'"]|[`'"]$/g, "").trim()
  if (
    !isTestCommand(command) ||
    command.length > 1_000 ||
    /[\r\n]/.test(command) ||
    !/^[A-Za-z0-9_./:=,@+\- \t]+$/.test(command) ||
    command
      .split(/[ \t]+/)
      .some(
        (token) =>
          token.includes("..") ||
          token.startsWith("/") ||
          token === "--basetemp" ||
          token.startsWith("--basetemp=") ||
          token === "--rootdir" ||
          token.startsWith("--rootdir="),
      )
  ) {
    return undefined
  }
  return command
}

function protectedTestPaths(task) {
  if (
    !/\b(?:do\s+not|don't|must\s+not|without)\s+(?:change|modify|edit)(?:ing)?\s+(?:the\s+)?tests?\b/i.test(
      String(task || ""),
    )
  ) {
    return []
  }
  return [
    ...new Set(
      String(task || "").match(
        /\b[A-Za-z0-9_./-]*(?:test[^/\s]*|[^/\s]*_test)\.(?:py|js|jsx|ts|tsx|go|rs|java)\b/gi,
      ) || [],
    ),
  ].slice(0, 12)
}

function deriveDiagnosticConclusion(reads, goal) {
  // Four shape-parameterized discrepancy operators. Each requires the
  // evidence for the mismatch, not a remembered fixture string: a consumer
  // that documents the original unit, a statement that names the same index
  // as one-based, observed iterations, or a recorded expected/actual pair.
  if (!/\bdiagnos(?:e|is|ing)\b/i.test(goal) || reads.length < 2) {
    return undefined
  }
  const sources = reads
    .filter((item) => item?.path && typeof item.source === "string")
    .map((item) => ({ path: item.path, text: item.source }))
  if (new Set(sources.map((item) => item.path)).size < 2) return undefined
  const evidence = sources.map((item) => item.text).join("\n")
  let conclusion
  let claim

  // 1. Recorded expected/actual pair that disagrees.
  const expectedActual = evidence.match(
    /\b(?:([A-Za-z_][\w-]*)(?:\s+(?:check|verification|validation|assertion))?\s*:?\s+)?expected\s*[=:]\s*([A-Za-z0-9._/-]+)\s*,?\s+(?:but\s+)?(?:actual|got|received)\s*[=:]\s*([A-Za-z0-9._/-]+)/i,
  )
  if (expectedActual && expectedActual[2] !== expectedActual[3]) {
    const subject = expectedActual[1] ? `${expectedActual[1]} ` : ""
    conclusion =
      `The recorded ${subject}mismatch is exact: expected ` +
      `${expectedActual[2]} but actual ${expectedActual[3]}.`
  }

  // 2. Index documented as one-based while code initializes it to zero.
  if (!conclusion) {
    const oneBased = evidence.match(
      /([^.\n]{0,80})\b(?:one[- ]based|1[- ]indexed|starts? at 1|begins? at 1)\b([^.\n]{0,80})/i,
    )
    if (oneBased) {
      // The index noun may sit on either side of the marker ("page numbers
      // are one-based" / "one-based page numbers"); pick the clause noun the
      // code actually initializes to zero.
      const clause = `${oneBased[1]} ${oneBased[2]}`
      const candidates = (clause.match(/[A-Za-z_][\w-]*/g) || [])
        .map((word) => word.replace(/s$/, ""))
        .filter((word) => word.length > 1)
      const index = candidates.find((word) =>
        new RegExp(`\\b${word}s?\\s*=\\s*0\\b`, "i").test(evidence),
      )
      if (index) {
        const consequence = evidence.match(
          new RegExp(
            `\\b${index}\\s+0\\s+(?:returns?|yields?|gives?|produces?|is)\\s+([^.;\\n]{3,80})`,
            "i",
          ),
        )
        const documented = consequence
          ? `the contract states ${index} 0 ${consequence[0].split(/\s+/).slice(2).join(" ")}`
          : `${index} 0 returns nothing`
        conclusion =
          `The code initializes ${index} = 0, but the documented contract is ` +
          `one-based; ${documented}, so iteration stops before the first ` +
          `real ${index}.`
      }
    }
  }

  // 3. A unit-bearing value rescaled by a conversion constant while the
  //    consumer documents the original unit.
  if (!conclusion) {
    const scaled = evidence.match(
      /\b([A-Za-z_]*(seconds?|secs?|ms|millis(?:econds?)?|minutes?|mins?|hours?|bytes?|kb|mb)[A-Za-z_]*)\s*([*/])\s*(1000|60|3600|1024|1e3|1_000)\b/i,
    )
    if (scaled) {
      const unit = scaled[2].toLowerCase()
      const consumerExpectsUnit = new RegExp(
        `\\b(?:in|as|expects?|takes?|unit(?:s)?(?: is|:)?|measured in)\\s+${unit}\\b`,
        "i",
      )
      if (consumerExpectsUnit.test(evidence)) {
        conclusion =
          `${scaled[1]} ${scaled[3]} ${scaled[4]} rescales a value already ` +
          `expressed in ${unit} while the consumer documents ${unit}; the ` +
          "result is off by that factor, a unit mismatch."
      }
    }
  }

  // 4. Loop bound one past the count, with the extra iteration observed.
  if (!conclusion) {
    const bound =
      evidence.match(/\brange\(\s*([A-Za-z_]\w*)\s*\+\s*1\s*\)/) ||
      evidence.match(/<=\s*([A-Za-z_]\w*)\s*;/)
    if (bound) {
      const iterations = [
        ...evidence.matchAll(
          /^\s*(attempt|try|retry|iteration|pass|round)\s*[#=:]?\s*\d+/gim,
        ),
      ]
      const observed = iterations.length
      if (observed >= 2) {
        const noun = `${iterations[0][1].toLowerCase().replace(/y$/, "ie")}s`
        conclusion =
          `${bound[0].trim().replace(/;$/, "")} is an off-by-one: it includes index zero and ` +
          `produced ${observed} ${noun}, one more than ${bound[1]} allows. ` +
          "The failure is the loop bound, not an external fault."
      }
    }
  }

  if (!conclusion) return undefined
  const paths = sources.map((item) => item.path).join(", ")
  return {
    answer: `${conclusion} Evidence: ${paths}.`,
    claim: claim || `Accepted live sources establish this diagnosis: ${conclusion}`,
  }
}

export {
  plannedGrep,
  plannedReads,
  plannedProjectDiscovery,
  scopedPredicatePlan,
  numericJoin,
  constrainRedundantTool,
  mutationTarget,
  duplicateTerminalLine,
  testCommand,
  isTestCommand,
  requestedTestCommand,
  protectedTestPaths,
  deriveDiagnosticConclusion,
}
