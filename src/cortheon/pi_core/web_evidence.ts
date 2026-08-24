import { boundedEvidence, contentText } from "./host_evidence.ts";
import { type EvidenceRequest, objectValue, stringValue } from "./protocol.ts";

const WEB_TOOLS = new Set(["websearch", "webfetch"]);
const WEB_CAPABILITIES = new Set(["search", "fetch", "search_or_fetch"]);
const MAX_WEB_RESULTS = 8;
const MAX_PURPOSE_CHARACTERS = 500;
const MAX_RESULT_TEXT_CHARACTERS = 10_000;

export function isWebTool(tool: string): boolean {
	return WEB_TOOLS.has(tool.toLowerCase());
}

export function isWebRequest(request: EvidenceRequest | undefined): boolean {
	return Boolean(
		request?.capability && WEB_CAPABILITIES.has(request.capability),
	);
}

function normalizedUrl(value: unknown): string | undefined {
	if (typeof value !== "string" || value.length > 2_000) return;
	try {
		const parsed = new URL(value);
		if (!["http:", "https:"].includes(parsed.protocol)) return;
		if (parsed.username || parsed.password || !parsed.hostname) return;
		parsed.hash = "";
		return parsed.toString();
	} catch {
		return;
	}
}

function normalizedPublished(value: unknown): string | undefined {
	if (typeof value !== "string" || value.length > 100) return;
	if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
		return Number.isNaN(Date.parse(`${value}T00:00:00Z`)) ? undefined : value;
	}
	const parsed = new Date(value);
	return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

function onePublished(
	value: Record<string, unknown>,
): string | null | undefined {
	const raw = [value.published_at, value.publishedAt, value.date].filter(
		(item): item is string => typeof item === "string" && Boolean(item),
	);
	if (raw.length === 0) return;
	const normalized = raw.map(normalizedPublished);
	if (normalized.some((item) => !item) || new Set(normalized).size !== 1)
		return null;
	return normalized[0];
}

function boundedMetadata(value: unknown): string | number | undefined {
	if (typeof value === "string" && value.length > 0 && value.length <= 200) {
		return value;
	}
	if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
		return value;
	}
	return;
}

function receiptMetadata(
	value: Record<string, unknown>,
	url: string,
): Record<string, unknown> {
	const source = objectValue(value.source);
	const provider = boundedMetadata(value.provider);
	const sourceType = boundedMetadata(
		value.source_type ?? value.sourceType ?? source?.type,
	);
	const authority = boundedMetadata(
		value.authority ?? value.authority_score ?? value.authorityScore,
	);
	return {
		lineage: {
			origin: new URL(url).origin,
			...(provider !== undefined ? { provider } : {}),
			...(sourceType !== undefined ? { source_type: sourceType } : {}),
		},
		...(authority !== undefined ? { authority } : {}),
	};
}

function purpose(request: EvidenceRequest): string | undefined {
	const value = stringValue(request.parameters?.purpose);
	return value && value.length <= MAX_PURPOSE_CHARACTERS ? value : undefined;
}

function webObservation(
	tool: string,
	request: EvidenceRequest,
	url: string,
	text: string,
	publishedAt: string | undefined,
	metadata: Record<string, unknown>,
	retrievedAt: string,
): Record<string, unknown> {
	const requestPurpose = purpose(request)!;
	const receipt = {
		tool,
		outcome: "result",
		args: {
			url,
			...(tool === "websearch"
				? { query: String(request.query || "").slice(0, 500) }
				: {}),
			purpose: requestPurpose,
		},
		retrieved_at_source: "pi_tool_result",
		...receiptMetadata(metadata, url),
	};
	return {
		kind: "web",
		content:
			`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(receipt)}\n` +
			boundedEvidence(text),
		source: url,
		url,
		retrieved_at: retrievedAt,
		...(publishedAt ? { published_at: publishedAt } : {}),
		purpose: requestPurpose,
		status: "observed",
	};
}

export function failedWebObservation(
	tool: string,
	request: EvidenceRequest,
	reason: string,
): Record<string, unknown> {
	const requestPurpose = purpose(request);
	const receipt = {
		tool,
		outcome: "error",
		args: {
			...(tool === "websearch"
				? { query: String(request.query || "").slice(0, 500) }
				: {}),
			...(requestPurpose ? { purpose: requestPurpose } : {}),
		},
	};
	return {
		kind: "web",
		content:
			`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(receipt)}\n` +
			boundedEvidence(reason),
		source: `pi:${tool}`,
		status: "failed",
		...(requestPurpose ? { purpose: requestPurpose } : {}),
	};
}

function fetchObservations(
	input: Record<string, unknown>,
	details: Record<string, unknown>,
	content: unknown,
	request: EvidenceRequest,
	retrievedAt: string,
): Array<Record<string, unknown>> {
	const candidates = [
		input.url,
		details.url,
		details.finalUrl,
		details.final_url,
	].filter((item): item is string => typeof item === "string" && Boolean(item));
	const urls = candidates.map(normalizedUrl);
	if (urls.length === 0 || urls.some((item) => !item)) {
		return [
			failedWebObservation(
				"webfetch",
				request,
				"webfetch result had no valid structured URL",
			),
		];
	}
	const origins = new Set(urls.map((item) => new URL(item!).origin));
	if (origins.size !== 1) {
		return [
			failedWebObservation(
				"webfetch",
				request,
				"webfetch result mixed URL origins",
			),
		];
	}
	const text = contentText(content);
	const publishedAt = onePublished(details);
	if (!text || publishedAt === null) {
		return [
			failedWebObservation(
				"webfetch",
				request,
				"webfetch result had invalid structured evidence",
			),
		];
	}
	return [
		webObservation(
			"webfetch",
			request,
			urls[urls.length - 1]!,
			text,
			publishedAt,
			details,
			retrievedAt,
		),
	];
}

function searchObservations(
	details: Record<string, unknown>,
	request: EvidenceRequest,
	retrievedAt: string,
): Array<Record<string, unknown>> {
	const raw = Array.isArray(details.results) ? details.results : undefined;
	if (!raw || raw.length === 0 || raw.length > MAX_WEB_RESULTS) {
		return [
			failedWebObservation(
				"websearch",
				request,
				"websearch result lacked a bounded structured result list",
			),
		];
	}
	const observations: Array<Record<string, unknown>> = [];
	const seen = new Set<string>();
	for (const candidate of raw) {
		const item = objectValue(candidate);
		const url = normalizedUrl(item?.url);
		const snippet = stringValue(item?.snippet ?? item?.content ?? item?.text);
		const title = stringValue(item?.title);
		const publishedAt = item ? onePublished(item) : null;
		if (
			!item ||
			!url ||
			!snippet ||
			snippet.length > MAX_RESULT_TEXT_CHARACTERS ||
			(title?.length || 0) > MAX_RESULT_TEXT_CHARACTERS ||
			publishedAt === null
		) {
			return [
				failedWebObservation(
					"websearch",
					request,
					"websearch result had invalid attributable fields",
				),
			];
		}
		const text = title ? `${title}\n${snippet}` : snippet;
		const key = JSON.stringify([url, text, publishedAt || ""]);
		if (seen.has(key)) continue;
		seen.add(key);
		observations.push(
			webObservation(
				"websearch",
				request,
				url,
				text,
				publishedAt,
				item,
				retrievedAt,
			),
		);
	}
	return observations.length
		? observations
		: [
				failedWebObservation(
					"websearch",
					request,
					"websearch returned only duplicate results",
				),
			];
}

export function webObservations(
	tool: string,
	input: Record<string, unknown>,
	content: unknown,
	detailsValue: unknown,
	isError: boolean,
	request: EvidenceRequest | undefined,
): Array<Record<string, unknown>> | undefined {
	const normalizedTool = tool.toLowerCase();
	if (!isWebTool(normalizedTool) || !request || !isWebRequest(request)) return;
	if (!purpose(request)) {
		return [
			failedWebObservation(
				normalizedTool,
				request,
				"web request had no bounded purpose",
			),
		];
	}
	if (isError) {
		return [
			failedWebObservation(normalizedTool, request, "host web tool failed"),
		];
	}
	const details = objectValue(detailsValue);
	if (!details) {
		return [
			failedWebObservation(
				normalizedTool,
				request,
				"host web result had no structured details",
			),
		];
	}
	const retrievedAt = new Date().toISOString();
	return normalizedTool === "webfetch"
		? fetchObservations(input, details, content, request, retrievedAt)
		: searchObservations(details, request, retrievedAt);
}
