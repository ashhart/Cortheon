function simpleLiteral(value) {
  const text = String(value || "").trim()
  if (text === "True") return true
  if (text === "False") return false
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)) return undefined
  const number = Number(text)
  return Number.isFinite(number) ? number : undefined
}

function evaluateSimpleExpression(expression, parameters, values) {
  if (
    typeof expression !== "string" ||
    !/^[A-Za-z0-9_.,()+\-*/%<>=!\s]+$/.test(expression) ||
    expression.includes("//")
  ) {
    return undefined
  }
  const allowed = new Set([
    ...parameters,
    "min",
    "max",
    "abs",
    "if",
    "else",
    "True",
    "False",
  ])
  const identifiers = expression.match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) || []
  if (identifiers.some((identifier) => !allowed.has(identifier))) {
    return undefined
  }
  let translated = expression
    .replace(/\bmin\s*\(/g, "Math.min(")
    .replace(/\bmax\s*\(/g, "Math.max(")
    .replace(/\babs\s*\(/g, "Math.abs(")
    .replace(/\bTrue\b/g, "true")
    .replace(/\bFalse\b/g, "false")
  const conditional = translated.match(/^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$/)
  if (conditional) {
    translated = `((${conditional[2]}) ? (${conditional[1]}) : (${conditional[3]}))`
  }
  try {
    const evaluator = Function(
      ...parameters,
      `"use strict"; return (${translated});`,
    )
    return evaluator(...values)
  } catch {
    return undefined
  }
}

function expressionDistance(left, right) {
  const length = Math.max(left.length, right.length)
  let distance = Math.abs(left.length - right.length)
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    if (left[index] !== right[index]) distance += 1
  }
  return distance + length / 10_000
}

function deriveSimpleRepairPlan(reads, selectedImplementation) {
  const implementation = selectedImplementation || reads.find(
    (item) =>
      item &&
      typeof item.path === "string" &&
      !/(?:^|\/)(?:test_|[^/]*_test\.)/.test(item.path) &&
      typeof item.source === "string",
  )
  if (!implementation) return undefined
  const lines = implementation.source.split("\n")
  let functionName
  let parameters
  let returnLine
  let returnExpression
  for (let index = 0; index < lines.length; index += 1) {
    const definition = lines[index].match(
      /^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)/,
    )
    if (!definition) continue
    const parsedParameters = definition[2]
      .split(",")
      .map((item) => item.split(/[=:]/, 1)[0].trim())
      .filter((item) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(item))
    for (let bodyIndex = index + 1; bodyIndex < lines.length; bodyIndex += 1) {
      if (/^\s*def\s+/.test(lines[bodyIndex])) break
      const returned = lines[bodyIndex].match(/^(\s+)return\s+([^#]+?)\s*$/)
      if (!returned) continue
      functionName = definition[1]
      parameters = parsedParameters
      returnLine = lines[bodyIndex]
      returnExpression = returned[2].trim()
      break
    }
    if (returnLine) break
  }
  if (
    !functionName ||
    !parameters?.length ||
    !returnLine ||
    !returnExpression
  ) {
    return undefined
  }

  const testSource = reads
    .filter((item) => item !== implementation && typeof item?.source === "string")
    .map((item) => item.source)
    .join("\n")
  const escapedName = functionName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  const assertion = new RegExp(
    `\\b${escapedName}\\s*\\(([^()]*)\\)\\s*(?:==|is)\\s*` +
      "([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)|True|False)",
    "g",
  )
  const examples = []
  for (const match of testSource.matchAll(assertion)) {
    const values = match[1].split(",").map(simpleLiteral)
    const expected = simpleLiteral(match[2])
    if (
      values.length === parameters.length &&
      values.every((item) => item !== undefined) &&
      expected !== undefined
    ) {
      examples.push({ values, expected })
    }
  }
  if (examples.length === 0) return undefined

  const candidates = new Set()
  for (const operator of [" + ", " - ", " * ", " / ", " % "]) {
    if (!returnExpression.includes(operator)) continue
    for (const replacement of [" + ", " - ", " * ", " / ", " % "]) {
      if (replacement !== operator) {
        candidates.add(returnExpression.replace(operator, replacement))
      }
    }
  }
  if (/\b(?:min|max)\s*\(/.test(returnExpression)) {
    candidates.add(
      returnExpression
        .replace(/\bmin\s*\(/g, "__CORTHEON_MAX__(")
        .replace(/\bmax\s*\(/g, "min(")
        .replace(/__CORTHEON_MAX__\(/g, "max("),
    )
  }
  const conditional = returnExpression.match(
    /^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$/,
  )
  if (conditional) {
    candidates.add(`${conditional[3]} if ${conditional[2]} else ${conditional[1]}`)
  }
  candidates.add(returnExpression.replace(/==\s*0\b/, "== 1"))
  candidates.add(returnExpression.replace(/==\s*1\b/, "== 0"))
  candidates.add(returnExpression.replace(/!=\s*0\b/, "!= 1"))
  candidates.add(returnExpression.replace(/!=\s*1\b/, "!= 0"))
  candidates.add(returnExpression.replace(/>=/, "<"))
  candidates.add(returnExpression.replace(/<=/, ">"))
  candidates.add(returnExpression.replace(/(?<![<])>(?!=)/, "<="))
  candidates.add(returnExpression.replace(/(?<![>])<(?!=)/, ">="))
  if (parameters.length === 2) {
    const [left, right] = parameters
    candidates.add(`${left} + ${right}`)
    candidates.add(`${left} - ${right}`)
    candidates.add(`${left} * ${right}`)
    candidates.add(`${left} + ${left} * ${right}`)
    candidates.add(`${left} - ${left} * ${right}`)
    candidates.add(`${left} * (1 + ${right})`)
    candidates.add(`${left} * (1 - ${right})`)
  }
  candidates.delete(returnExpression)
  candidates.delete("")

  const passing = [...candidates].filter((candidate) =>
    examples.every((example) => {
      const observed = evaluateSimpleExpression(
        candidate,
        parameters,
        example.values,
      )
      if (typeof example.expected === "boolean") {
        return observed === example.expected
      }
      return (
        typeof observed === "number" &&
        Math.abs(observed - example.expected) <= 1e-9
      )
    }),
  )
  if (passing.length === 0) return undefined
  const rootOperator = (expression) =>
    expression.match(
      new RegExp(`^\\s*${parameters[0]}\\s*([+*/%]|-)`),
    )?.[1]
  const originalRootOperator = rootOperator(returnExpression)
  passing.sort(
    (left, right) =>
      Number(
        Boolean(originalRootOperator) &&
          rootOperator(left) !== originalRootOperator,
      ) -
        Number(
          Boolean(originalRootOperator) &&
            rootOperator(right) !== originalRootOperator,
        ) ||
      expressionDistance(left, returnExpression) -
        expressionDistance(right, returnExpression),
  )
  const replacement = returnLine.replace(returnExpression, passing[0])
  return {
    path: implementation.path,
    oldString: returnLine,
    newString: replacement,
    functionName,
    examples: examples.length,
  }
}

function deriveSimpleRepairPlans(reads) {
  return reads
    .filter(
      (item) =>
        item &&
        typeof item.path === "string" &&
        !/(?:^|\/)(?:test_|[^/]*_test\.)/.test(item.path) &&
        typeof item.source === "string",
    )
    .map((item) => deriveSimpleRepairPlan(reads, item))
    .filter(Boolean)
}

function deriveExactDocumentEdits(reads, goal) {
  const edits = []
  const matcher =
    /\bupdate\s+([A-Za-z0-9_./-]+\.(?:md|markdown|rst|txt))\s+with\s+the\s+exact\s+sentence\s+['"]([^'"]{1,500})['"]/gi
  for (const match of String(goal || "").matchAll(matcher)) {
    const item = reads.find(
      (read) => read?.path === match[1] && typeof read.source === "string",
    )
    if (!item || item.source.includes(match[2])) continue
    edits.push({
      path: item.path,
      oldString: item.source,
      newString:
        item.source.replace(/\s*$/, "") +
        `\n\n${match[2]}\n`,
    })
  }
  return edits
}

export {
  simpleLiteral,
  evaluateSimpleExpression,
  expressionDistance,
  deriveSimpleRepairPlan,
  deriveSimpleRepairPlans,
  deriveExactDocumentEdits,
}
