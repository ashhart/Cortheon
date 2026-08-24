import { investigations } from "./state.js"
import {
  automaticResearchTargets,
  focusedWebPassage,
  sourceDate,
} from "./evidence.js"
import { evidenceReceipt } from "./evidence.js"

// Bounded automatic research over user-cited public release sources.
const createAutoResearch = ({
  debug,
  submitAutomaticObservation,
  submitPassiveObservations,
}) => {
  const fetchBoundedPublicSource = async (target) => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 12_000)
    try {
      let current = target.fetchUrl
      let response
      for (let redirects = 0; redirects <= 3; redirects += 1) {
        response = await fetch(current, {
          method: "GET",
          headers: {
            Accept: "text/html, application/rss+xml, application/xml;q=0.9",
            "User-Agent": "Cortheon/0.1 bounded-research-adapter",
          },
          redirect: "manual",
          signal: controller.signal,
        })
        if (![301, 302, 303, 307, 308].includes(response.status)) break
        const location = response.headers.get("location")
        if (!location || redirects === 3) return undefined
        const redirected = new URL(location, current)
        if (
          redirected.protocol !== "https:" ||
          redirected.hostname.toLowerCase() !== target.hostname ||
          redirected.username ||
          redirected.password ||
          redirected.port
        ) {
          return undefined
        }
        current = redirected.href
      }
      if (!response?.ok) return undefined
      const contentType = String(response.headers.get("content-type") || "")
      if (!/(?:html|xml|rss|text)/i.test(contentType)) return undefined
      const decoder = new TextDecoder()
      const chunks = []
      let bytes = 0
      if (response.body?.getReader) {
        const reader = response.body.getReader()
        while (bytes < 1_000_000) {
          const { done, value } = await reader.read()
          if (done) break
          const remaining = 1_000_000 - bytes
          const bounded = value.byteLength > remaining
            ? value.slice(0, remaining)
            : value
          chunks.push(decoder.decode(bounded, { stream: true }))
          bytes += bounded.byteLength
          if (bounded.byteLength < value.byteLength) {
            await reader.cancel()
            break
          }
        }
        chunks.push(decoder.decode())
      } else {
        chunks.push((await response.text()).slice(0, 1_000_000))
      }
      const text = chunks.join("")
      return text.trim() ? text : undefined
    } catch {
      return undefined
    } finally {
      clearTimeout(timer)
    }
  }

  const automaticResearchObservation = (target, text, purpose) => {
    const excerpt = focusedWebPassage(text)
    if (!excerpt) return undefined
    const retrievedAt = new Date().toISOString()
    const publishedAt = sourceDate(excerpt)
    const receipt = evidenceReceipt(
      "webfetch",
      { url: target.sourceUrl },
      excerpt,
      {
        purpose,
        retrieved_at: retrievedAt,
        source_url: target.sourceUrl,
        ...(publishedAt ? { published_at: publishedAt } : {}),
      },
    )
    return {
      content:
        `${receipt}\nSource URL: ${target.sourceUrl}\n` +
        `Retrieved: ${retrievedAt}\n${excerpt}`,
      kind: "web",
      source: target.sourceUrl,
      status: "observed",
      url: target.sourceUrl,
      retrieved_at: retrievedAt,
      published_at: publishedAt,
      purpose,
    }
  }

  const acquireAutomaticResearch = async (hostSessionID, state) => {
    const purpose = state?.request?.parameters?.purpose
    if (
      !state?.automatic ||
      !state.active ||
      state.deliverable !== "research_answer" ||
      !state.requestID ||
      !["contradiction_check", "primary_fetch"].includes(purpose || "") ||
      state.automaticResearchAttempted
    ) {
      return state
    }
    state.automaticResearchAttempted = true
    investigations.set(hostSessionID, state)
    const targets = automaticResearchTargets(state.goal)
    if (targets.length < 2) return state
    const fetched = (
      await Promise.all(
        targets.map(async (target) => ({
          target,
          text: await fetchBoundedPublicSource(target),
        })),
      )
    ).filter((item) => typeof item.text === "string")
    if (
      new Set(fetched.map((item) => item.target.hostname)).size < 2
    ) {
      return state
    }
    const contradiction = fetched
      .map((item) =>
        automaticResearchObservation(
          item.target,
          item.text,
          "contradiction_check",
        ),
      )
      .filter(Boolean)
    if (contradiction.length < 2) return state
    const primarySource =
      fetched.find((item) => item.target.hostname === "github.com") ||
      fetched[0]
    const primary = automaticResearchObservation(
      primarySource.target,
      primarySource.text,
      "primary_fetch",
    )
    let next = state
    const submitBound = async (observations) => {
      next.hostEvidence = undefined
      next.hostEvidenceBatch = observations
      investigations.set(hostSessionID, next)
      next = await submitAutomaticObservation(hostSessionID, next)
    }
    if (purpose === "primary_fetch") {
      // The runtime asked for the primary source first; serve that request,
      // then land the corroboration set.
      if (!primary) return state
      await submitBound([primary])
      if (!next.evidenceIDs?.length) return next
      if (
        next.requestID &&
        next.request?.parameters?.purpose === "contradiction_check"
      ) {
        await submitBound(contradiction)
      } else {
        next = await submitPassiveObservations(
          hostSessionID,
          next,
          contradiction,
        )
      }
    } else {
      await submitBound(contradiction)
      if (!next.evidenceIDs?.length) return next
      if (!primary) return next
      if (
        next.requestID &&
        next.request?.parameters?.purpose === "primary_fetch"
      ) {
        await submitBound([primary])
      } else if (next.requestID) {
        return next
      } else {
        next = await submitPassiveObservations(hostSessionID, next, [primary])
      }
    }
    if (next.requestID) return next
    next.automaticResearchAcquired = true
    investigations.set(hostSessionID, next)
    await debug(
      `automatic research acquired origins=${fetched.length} ` +
        `release=${next.releaseVersion?.value || "unresolved"}`,
    )
    return next
  }

  return { acquireAutomaticResearch }
}

export { createAutoResearch }
