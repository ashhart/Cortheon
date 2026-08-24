import type {
	ExtensionAPI,
	ExtensionContext,
	ToolExecutionStartEvent,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import { nextActionInstruction } from "./actions.ts";
import { OBSERVE_POLICY_REFUSAL_DISPOSITION } from "./budget.ts";
import { clearBenchmarkCandidate } from "./candidate_capture.ts";
import {
	contentText,
	hostObservation,
	observationPath,
	requestedReadPaths,
} from "./host_evidence.ts";
import { mergePayload } from "./merge.ts";
import { claimObservation, observationStillCurrent } from "./observe_claim.ts";
import { type EvidenceRequest } from "./protocol.ts";
import { debug, isRuntimePolicyRefusal, runtimeCall } from "./runtime.ts";
import {
	abandonActive,
	getActive,
	markAnswerOnly,
	setTerminalDisposition,
} from "./state.ts";
import {
	failedWebObservation,
	isWebRequest,
	isWebTool,
	webObservations,
} from "./web_evidence.ts";

function requestedWebToolAvailable(
	pi: ExtensionAPI,
	request: EvidenceRequest,
): boolean {
	const active = new Set(pi.getActiveTools().map((item) => item.toLowerCase()));
	if (request.capability === "search") return active.has("websearch");
	if (request.capability === "fetch") return active.has("webfetch");
	return active.has("websearch") || active.has("webfetch");
}

async function terminalWebUnavailable(
	context: ExtensionContext,
	reason: string,
): Promise<void> {
	clearBenchmarkCandidate();
	markAnswerOnly();
	setTerminalDisposition({ reason, causal: false, submitted: false });
	await abandonActive();
	context.abort();
}

/** Intercept an attempted absent web tool before Pi's generic not-found
 * result. One failed observation gives the runtime one chance to re-plan;
 * another unsatisfiable web request terminates the operation. */
export async function reportUnavailableWebTool(
	pi: ExtensionAPI,
	event: ToolExecutionStartEvent,
	context: ExtensionContext,
): Promise<boolean> {
	const active = getActive();
	const request = active?.request;
	if (
		!active ||
		active.completed ||
		!request ||
		!isWebRequest(request) ||
		!isWebTool(event.toolName) ||
		requestedWebToolAvailable(pi, request)
	) {
		return false;
	}
	const reason =
		"Pi has no active web tool capable of satisfying " +
		`runtime request ${request.request_id}`;
	if (active.webUnavailableReported) {
		await terminalWebUnavailable(context, `${reason}; the request repeated`);
		return true;
	}
	active.webUnavailableReported = true;
	const sessionId = active.sessionId;
	const requestId = request.request_id;
	if (!claimObservation(sessionId, requestId)) {
		await terminalWebUnavailable(
			context,
			`${reason}; the request was already reported`,
		);
		return true;
	}
	try {
		const payload = await runtimeCall(
			"/v1/observe",
			{
				session_id: sessionId,
				request_id: requestId,
				observations: [
					failedWebObservation(event.toolName.toLowerCase(), request, reason),
				],
			},
			context.signal,
		);
		if (!observationStillCurrent(getActive(), active, sessionId, requestId)) {
			context.abort();
			return true;
		}
		mergePayload(payload);
	} catch {
		clearBenchmarkCandidate();
		await abandonActive();
		context.abort();
		return true;
	}
	const replanned = getActive();
	if (
		!replanned ||
		(replanned.request &&
			isWebRequest(replanned.request) &&
			!requestedWebToolAvailable(pi, replanned.request))
	) {
		await terminalWebUnavailable(
			context,
			`${reason}; no bounded re-plan followed`,
		);
		return true;
	}
	// The current model operation still contains the unavailable call. Abort
	// it and let agent_end schedule at most the existing one continuation.
	if (replanned.request) replanned.needsContinuation = true;
	context.abort();
	return true;
}

function localObservations(
	event: ToolResultEvent,
	request: EvidenceRequest,
): Array<Record<string, unknown>> | undefined {
	if (!["read", "grep", "find", "ls", "bash"].includes(event.toolName)) return;
	const active = getActive();
	if (!active) return;
	const observation = hostObservation(
		event.toolName,
		event.input,
		contentText(event.content),
		event.isError,
		request,
	);
	const paths = requestedReadPaths(request);
	if (paths.length > 0 && event.toolName !== "read") return;
	if (paths.length <= 1 || event.toolName !== "read") return [observation];
	active.pendingReadObservations.push(observation);
	const covered = new Set(
		active.pendingReadObservations
			.map(observationPath)
			.filter((item): item is string => Boolean(item)),
	);
	return paths.every((item) => covered.has(item))
		? active.pendingReadObservations
		: undefined;
}

/** Submit one attributable Pi tool result for the pending runtime request. */
export async function observeHostToolResult(
	event: ToolResultEvent,
): Promise<{ content: ToolResultEvent["content"] } | undefined> {
	const active = getActive();
	const request = active?.request;
	if (
		!active ||
		active.completed ||
		!request ||
		event.toolName.startsWith("cortheon_")
	) {
		return;
	}
	const observations = isWebTool(event.toolName)
		? webObservations(
				event.toolName,
				event.input,
				event.content,
				event.details,
				event.isError,
				request,
			)
		: localObservations(event, request);
	if (!observations) return;
	const sessionId = active.sessionId;
	const requestId = request.request_id;
	if (!claimObservation(sessionId, requestId)) {
		debug(`observe already claimed for ${requestId}; sibling result skipped`);
		return;
	}
	try {
		const payload = await runtimeCall("/v1/observe", {
			session_id: sessionId,
			request_id: requestId,
			observations,
		});
		if (!observationStillCurrent(getActive(), active, sessionId, requestId)) {
			debug(`discarding stale observe response for ${requestId}`);
			return;
		}
		mergePayload(payload);
		return {
			content: [
				...event.content,
				{
					type: "text",
					text:
						"\n[Cortheon accepted host evidence; next action: " +
						`${nextActionInstruction(getActive())}]`,
				},
			],
		};
	} catch (error) {
		if (isRuntimePolicyRefusal(error)) {
			clearBenchmarkCandidate();
			markAnswerOnly();
			setTerminalDisposition({
				reason: OBSERVE_POLICY_REFUSAL_DISPOSITION,
				causal: false,
				submitted: false,
			});
			await abandonActive();
			return;
		}
		clearBenchmarkCandidate();
		await abandonActive();
		return;
	}
}

/** Harvest a model-owned host receipt without originating a runtime request. */
export async function observePassiveHostToolResult(event: ToolResultEvent): Promise<void> {
	const active = getActive();
	if (!active || active.completed || event.toolName.startsWith("cortheon_")) return;
	const passiveRequest: EvidenceRequest = {
		request_id: "passive_host_receipt",
		capability: isWebTool(event.toolName) ? "search_or_fetch" : "inspect",
		query: "Model-owned host receipt",
		parameters: { purpose: "fact_check" },
	};
	const observations = isWebTool(event.toolName)
		? webObservations(
				event.toolName,
				event.input,
				event.content,
				event.details,
				event.isError,
				passiveRequest,
			)
		: [
				hostObservation(
					event.toolName,
					event.input,
					contentText(event.content),
					event.isError,
					undefined,
				),
			];
	if (!observations?.length) return;
	try {
		mergePayload(
			await runtimeCall("/v1/observe", {
				session_id: active.sessionId,
				observations,
			}),
		);
	} catch {
		await abandonActive();
	}
}
