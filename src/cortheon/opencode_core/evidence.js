import { safeHostArguments } from "./state.js"
import { boundedHostOutput, maxEvidenceCharacters } from "./state.js"

function evidenceReceipt(tool, args, output, metadata = {}) {
  let outcome = "result"
  if (tool === "grep" || tool === "websearch") {
    outcome =
      !output ||
      /\b(?:no (?:files|results?) found|no matches?(?: found)?|0 (?:matches?|results?))\b/i.test(
        output,
      )
        ? "no_match"
        : tool === "grep"
          ? "match"
          : "result"
  }
  return (
    "[CORTHEON_HOST_EVIDENCE] " +
    JSON.stringify({
      tool,
      outcome,
      args: safeHostArguments(tool, args),
      ...metadata,
    })
  )
}

function receiptOutcome(content) {
  const line = String(content || "")
    .split("\n")
    .find((item) => item.startsWith("[CORTHEON_HOST_EVIDENCE] "))
  if (!line) return undefined
  try {
    const value = JSON.parse(line.slice("[CORTHEON_HOST_EVIDENCE] ".length))
    return value?.outcome === "match" || value?.outcome === "no_match"
      ? value.outcome
      : undefined
  } catch {
    return undefined
  }
}

function statementIsNegative(value) {
  return /\b(?:no|not|never|without|absent|missing|doesn't|does\s+not|isn't|is\s+not)\b/i.test(
    String(value || ""),
  )
}

function completionClaim(value) {
  const withoutCitations = String(value || "")
    .replace(/\([^)]*\.(?:md|markdown|rst|txt):\d+[^)]*\)/gi, " ")
    .replace(/`[^`\n]*\.(?:md|markdown|rst|txt):\d+[^`\n]*`/gi, " ")
    .replace(/\b[\w./-]+\.(?:md|markdown|rst|txt):\d+:/gi, " ")
    .replace(
      /\b(hypothesis|alternative|interpretation|option)\s+\d+\b/gi,
      "$1",
    )
  return withoutCitations
    .split("\n")
    .map((line) =>
      line
        .replace(/^\s*(?:[-*+]|\d+[.)])\s+/, "")
        .replace(/[*_`~#]+/g, "")
        .trim(),
    )
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .slice(0, 1_900)
}

function evidenceFacts(value) {
  return completionClaim(
    String(value || "")
      .split("\n")
      .filter((line) => !line.startsWith("[CORTHEON_HOST_EVIDENCE]"))
      .join("\n"),
  )
}

function evidenceRecordContent(value) {
  // Evidence records feed deterministic derivations and model context, so
  // preserve identifiers and line structure verbatim; completionClaim's
  // markdown-strip belongs to claim normalization, not evidence storage
  // (it deleted every underscore, mangling MAX_INPUT_CHARS-style symbols).
  return String(value || "")
    .split("\n")
    .filter((line) => !line.startsWith("[CORTHEON_HOST_EVIDENCE]"))
    .join("\n")
    .replace(/\u001b\[[0-9;]*m/g, "")
    .trim()
    .slice(0, 1_200)
}

function receiptTool(content) {
  const line = String(content || "")
    .split("\n")
    .find((item) => item.startsWith("[CORTHEON_HOST_EVIDENCE] "))
  if (!line) return undefined
  try {
    const value = JSON.parse(line.slice("[CORTHEON_HOST_EVIDENCE] ".length))
    return typeof value?.tool === "string" ? value.tool : undefined
  } catch {
    return undefined
  }
}

function mergeEvidenceRecords(previous, submitted) {
  const records = new Map(
    (Array.isArray(previous) ? previous : []).map((item) => [
      item.source,
      item,
    ]),
  )
  for (const item of submitted) {
    if (!item || typeof item.content !== "string") continue
    const source =
      typeof item.source === "string" && item.source
        ? item.source
        : `host:${receiptTool(item.content) || "evidence"}`
    records.set(source, {
      source,
      tool: receiptTool(item.content),
      content: evidenceRecordContent(item.content),
    })
  }
  return [...records.values()].slice(-8)
}

function compactEvidence(records) {
  let selected = Array.isArray(records) ? records : []
  if (selected.some((item) => item.tool === "read")) {
    selected = selected.filter((item) => item.tool !== "glob")
  }
  if (selected.length === 0) return undefined
  const headings = selected.reduce(
    (total, item) => total + String(item.source || "").length + 4,
    0,
  )
  const perRecord = Math.max(
    120,
    Math.floor((maxEvidenceCharacters - headings) / selected.length),
  )
  return selected
    .map(
      (item) =>
        `[${item.source}]\n${String(item.content || "").slice(0, perRecord)}`,
    )
    .join("\n\n")
    .slice(0, maxEvidenceCharacters)
}

function sourceDate(value) {
  const text = String(value || "")
  const iso = text.match(/\b(20\d{2}-\d{2}-\d{2})\b/)
  if (iso) return iso[1]
  const named = text.match(
    /\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+20\d{2}\b/i,
  )
  const rfc = text.match(
    /\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+20\d{2}\b/i,
  )
  const candidate = named?.[0] || rfc?.[0]
  if (!candidate) return undefined
  const months = {
    jan: "01",
    feb: "02",
    mar: "03",
    apr: "04",
    may: "05",
    jun: "06",
    jul: "07",
    aug: "08",
    sep: "09",
    oct: "10",
    nov: "11",
    dec: "12",
  }
  const parts = named
    ? candidate.match(/^([A-Za-z]+)\s+(\d{1,2}),\s+(20\d{2})$/)
    : candidate.match(
        /^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})$/i,
      )
  if (!parts) return undefined
  const monthName = named ? parts[1] : parts[2]
  const day = named ? parts[2] : parts[1]
  const year = parts[3]
  const month = months[monthName.slice(0, 3).toLowerCase()]
  return month ? `${year}-${month}-${String(day).padStart(2, "0")}` : undefined
}

function httpUrls(value) {
  const matches = String(value || "").match(/https?:\/\/[^\s<>"')\]]+/g) || []
  return matches.map((item) => item.replace(/[.,;:!?}]+$/, ""))
}

function automaticResearchTargets(goal) {
  const targets = []
  const seen = new Set()
  for (const candidate of httpUrls(goal)) {
    let parsed
    try {
      parsed = new URL(candidate)
    } catch {
      continue
    }
    if (
      parsed.protocol !== "https:" ||
      parsed.username ||
      parsed.password ||
      parsed.port ||
      !["github.com", "pypi.org"].includes(parsed.hostname.toLowerCase())
    ) {
      continue
    }
    parsed.hash = ""
    const hostname = parsed.hostname.toLowerCase()
    let fetchUrl
    if (
      hostname === "github.com" &&
      /^\/[^/]+\/[^/]+\/releases\/latest\/?$/.test(parsed.pathname)
    ) {
      fetchUrl = parsed.href
    } else {
      const project = parsed.pathname.match(/^\/project\/([A-Za-z0-9._-]+)\/?$/)?.[1]
      if (hostname !== "pypi.org" || !project) continue
      fetchUrl =
        `https://pypi.org/rss/project/${encodeURIComponent(project)}/releases.xml`
    }
    const origin = parsed.origin.toLowerCase()
    if (seen.has(origin)) continue
    seen.add(origin)
    targets.push({
      sourceUrl: parsed.href,
      fetchUrl,
      hostname,
    })
    if (targets.length >= 3) break
  }
  return targets.length >= 2 ? targets : []
}

function cleanWebFragment(value) {
  return String(value || "")
    .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&(?:nbsp|#160);/gi, " ")
    .replace(/&middot;/gi, "·")
    .replace(/&amp;/gi, "&")
    .replace(/\s+/g, " ")
    .trim()
}

function focusedWebPassage(value) {
  const text = String(value || "")
  const fragments = []
  for (const item of text.matchAll(/<item\b[^>]*>([\s\S]*?)<\/item>/gi)) {
    const title = item[1].match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]
    const version = cleanWebFragment(title)
    if (!/^\d+\.\d+(?:\.\d+){0,3}$/.test(version)) continue
    fragments.push(`Release ${version}`)
    const published = item[1].match(
      /<pubDate[^>]*>([\s\S]*?)<\/pubDate>/i,
    )?.[1]
    if (published) fragments.push(`Published ${cleanWebFragment(published)}`)
    break
  }
  const patterns = [
    /<title[^>]*>([\s\S]{1,400}?)<\/title>/gi,
    /<h1[^>]*(?:package-header__name)?[^>]*>([\s\S]{1,400}?)<\/h1>/gi,
    /(?:latest|release(?:d)?|version)[^<\n]{0,160}v?\d+\.\d+(?:\.\d+){0,3}/gi,
    /(?:released\s+on|upload(?:ed)?\s+date)[^<\n]{0,120}20\d{2}[-/ ][A-Za-z0-9, -]{2,30}/gi,
    /<relative-time[^>]+datetime=["']([^"']+)["'][^>]*>/gi,
  ]
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) {
      const fragment = cleanWebFragment(match[1] || match[0])
      if (fragment && !fragments.includes(fragment)) fragments.push(fragment)
      if (fragments.length >= 8) break
    }
    if (fragments.length >= 8) break
  }
  if (fragments.length === 0) return boundedHostOutput(text)
  return boundedHostOutput(fragments.join("\n"))
}

function webEvidenceBatch(tool, args, output, state) {
  const text = String(output || "")
  const candidates =
    tool === "webfetch"
      ? typeof args?.url === "string"
        ? [args.url]
        : []
      : httpUrls(text)
  const byOrigin = new Map()
  for (const candidate of candidates) {
    try {
      const parsed = new URL(candidate)
      if (!["http:", "https:"].includes(parsed.protocol)) continue
      const origin = parsed.origin.toLowerCase()
      if (!byOrigin.has(origin)) byOrigin.set(origin, parsed.href)
    } catch {
    }
  }
  const retrievedAt = new Date().toISOString()
  const purpose =
    typeof state?.request?.parameters?.purpose === "string"
      ? state.request.parameters.purpose
      : "passive"
  const urls = [...byOrigin.values()].slice(0, 3)
  if (urls.length === 0) {
    const receipt = evidenceReceipt(tool, args, text, {
      purpose,
      retrieved_at: retrievedAt,
    })
    return [
      {
        content: `${receipt}\n${boundedHostOutput(text)}`,
        kind: "web",
        source: `opencode:${tool}:unattributed`,
        status: "observed",
        retrieved_at: retrievedAt,
        purpose,
      },
    ]
  }
  return urls.map((url) => {
    const position = text.indexOf(url)
    const excerpt =
      tool === "webfetch"
        ? focusedWebPassage(text)
        : position >= 0
        ? text.slice(Math.max(0, position - 300), position + url.length + 900)
        : text
    const publishedAt = sourceDate(excerpt)
    const receipt = evidenceReceipt(tool, args, excerpt, {
      purpose,
      retrieved_at: retrievedAt,
      source_url: url,
      ...(publishedAt ? { published_at: publishedAt } : {}),
    })
    return {
      content:
        `${receipt}\nSource URL: ${url}\nRetrieved: ${retrievedAt}\n` +
        boundedHostOutput(excerpt),
      kind: "web",
      source: url,
      status: "observed",
      url,
      retrieved_at: retrievedAt,
      published_at: publishedAt,
      purpose,
    }
  })
}

export {
  evidenceReceipt,
  receiptOutcome,
  statementIsNegative,
  completionClaim,
  evidenceFacts,
  evidenceRecordContent,
  receiptTool,
  mergeEvidenceRecords,
  compactEvidence,
  sourceDate,
  httpUrls,
  automaticResearchTargets,
  cleanWebFragment,
  focusedWebPassage,
  webEvidenceBatch,
}
