import {
	createFindTool,
	createGrepTool,
	createReadTool,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
	certifyAutomaticDiagnostic,
	certifyAutomaticGrep,
	certifyAutomaticNumericJoin,
	certifyAutomaticPlan,
	certifyAutomaticReasoning,
	certifyAutomaticSemantic,
} from "./certify.ts";
import { deriveSimpleRepairPlan } from "./derive.ts";
import {
	contentHash,
	contentText,
	hostObservation,
	projectPathAllowed,
	requestedReadPaths,
} from "./host_evidence.ts";
import { mergePayload } from "./merge.ts";
import {
	claimObservation,
	observationStillCurrent,
} from "./observe_claim.ts";
import { type EvidenceRequest, stringValue } from "./protocol.ts";
import { runtimeCall } from "./runtime.ts";
import { getActive } from "./state.ts";

export async function satisfyDeterministicRequest(context: ExtensionContext): Promise<void> {
	let active = getActive();
	if (!active || active.completed || !active.request) return;
	if (!context.isProjectTrusted()) return;
	const request = active.request;
	const capability = request.capability;
	const parameters = request.parameters || {};
	let observations: Array<Record<string, unknown>> = [];
	const readSnapshots: Array<{ path: string; source: string }> = [];

	if (capability === "grep") {
		const pattern = stringValue(parameters.pattern);
		const requestedPath = stringValue(parameters.path);
		if (
			!pattern ||
			!requestedPath ||
			!(await projectPathAllowed(context.cwd, requestedPath))
		) {
			return;
		}
		const escapedPattern = pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		const executionPattern = /\bimports?\b/i.test(request.query || "")
			? `(?:^|\\s)(?:from\\s+${escapedPattern}(?:\\.|\\s)|` +
				`import\\s+[^#\\n]*\\b${escapedPattern}\\b)`
			: `(^|[^A-Za-z0-9_])${escapedPattern}([^A-Za-z0-9_]|$)`;
		try {
			const result = await createGrepTool(context.cwd).execute(
				`cortheon-${request.request_id}`,
				{
					pattern: executionPattern,
					path: requestedPath,
					literal: false,
					limit: 200,
				},
				context.signal,
			);
			observations = [
				hostObservation(
					"grep",
					{ pattern, path: requestedPath },
					contentText(result.content),
					false,
					request,
				),
			];
		} catch {
			return;
		}
	} else if (
		capability === "search" &&
		parameters.operation === "document_discovery"
	) {
		const maximum = Number(parameters.max_candidates);
		const limit =
			Number.isInteger(maximum) && maximum >= 2 && maximum <= 12
				? maximum
				: 6;
		try {
			const result = await createFindTool(context.cwd).execute(
				`cortheon-${request.request_id}`,
				{
					pattern: "**/*.{md,markdown,rst,txt}",
					path: ".",
					limit,
				},
				context.signal,
			);
			observations = [
				hostObservation(
					"find",
					{
						pattern: "**/*.{md,markdown,rst,txt}",
						path: ".",
					},
					contentText(result.content),
					false,
					request,
				),
			];
		} catch {
			return;
		}
	} else if (capability === "read_many") {
		const paths = requestedReadPaths(request);
		if (
			paths.length === 0 ||
			!(await Promise.all(
				paths.map((item) => projectPathAllowed(context.cwd, item)),
			)).every(Boolean)
		) {
			return;
		}
		const readTool = createReadTool(context.cwd);
		try {
			for (const requestedPath of paths) {
				const result = await readTool.execute(
					`cortheon-${request.request_id}-${observations.length + 1}`,
					{ path: requestedPath },
					context.signal,
				);
				const source = contentText(result.content);
				readSnapshots.push({ path: requestedPath, source });
				observations.push(
					hostObservation(
						"read",
						{ path: requestedPath },
						source,
						false,
						request,
					),
				);
			}
		} catch {
			return;
		}
	} else {
		return;
	}

	if (!claimObservation(active.sessionId, request.request_id)) return;
	// The exact object that submits; a replacement reusing both ID strings
	// must never receive this response. An ambiguous transport failure keeps
	// the claim: the request is never resubmitted (at-most-once).
	const submittedActive = active;
	const payload = await runtimeCall(
		"/v1/observe",
		{
			session_id: active.sessionId,
			request_id: request.request_id,
			observations,
		},
		context.signal,
	);
	if (
		!observationStillCurrent(
			getActive(),
			submittedActive,
			active.sessionId,
			request.request_id,
		)
	) {
		return;
	}
	mergePayload(payload);
	active = getActive();
	if (active && capability === "read_many") {
		active.initialFileHashes = Object.fromEntries(
			readSnapshots.map((item) => [item.path, contentHash(item.source)]),
		);
		active.repairPlan = deriveSimpleRepairPlan(readSnapshots);
	}
	if (capability === "grep" && observations.length === 1) {
		await certifyAutomaticGrep(request, observations[0]);
	} else if (capability === "read_many") {
		await certifyAutomaticPlan();
		if (getActive() && !getActive()!.completed) {
			await certifyAutomaticDiagnostic();
		}
		if (getActive() && !getActive()!.completed) {
			await certifyAutomaticNumericJoin(request, readSnapshots);
		}
		if (getActive() && !getActive()!.completed) {
			await certifyAutomaticSemantic(request);
		}
		if (getActive() && !getActive()!.completed) {
			await certifyAutomaticReasoning(readSnapshots);
		}
	}
}

/** Report a deterministic request the host could not satisfy (invalid
 * parameters, outside the project, or the host tool failed) as one bounded
 * failed observation so the live runtime can re-plan. True only when the
 * runtime actually moved on (a new request, action, or completion); false
 * when the same request stays pending or the re-plan itself failed — then
 * the caller must end with a truthful explicit disposition, never fall to
 * an ungated bare-model path. */
export async function reportUnsatisfiedDeterministicRequest(
	request: EvidenceRequest,
): Promise<boolean> {
	const active = getActive();
	if (!active || active.completed || active.request?.request_id !== request.request_id) {
		return true;
	}
	const observation = hostObservation(
		request.capability === "grep" ? "grep" : "find",
		{
			pattern: stringValue(request.parameters?.pattern) || "",
			path: stringValue(request.parameters?.path) || "",
		},
		"",
		true,
		request,
	);
	if (!claimObservation(active.sessionId, request.request_id)) return true;
	// Same identity discipline as the deterministic path above.
	const submittedActive = active;
	let payload: Record<string, unknown>;
	try {
		payload = await runtimeCall("/v1/observe", {
			session_id: active.sessionId,
			request_id: request.request_id,
			observations: [observation],
		});
	} catch {
		return false;
	}
	if (
		!observationStillCurrent(
			getActive(),
			submittedActive,
			active.sessionId,
			request.request_id,
		)
	) {
		return true;
	}
	mergePayload(payload);
	const after = getActive();
	return Boolean(
		!after ||
			after.completed ||
			after.request?.request_id !== request.request_id,
	);
}
