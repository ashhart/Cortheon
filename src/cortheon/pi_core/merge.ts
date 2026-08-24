import { boundedEvidence } from "./host_evidence.ts";
import {
	type ActiveInvestigation,
	type CognitiveAction,
	type EvidenceRequest,
	objectValue,
	stringValue,
} from "./protocol.ts";
import {
	ensureHeartbeat,
	getActive,
	setActive,
	stopHeartbeat,
} from "./state.ts";

export function mergePayload(
	payload: Record<string, unknown>,
): ActiveInvestigation | undefined {
	const active = getActive();
	const session = objectValue(payload.session);
	const sessionId =
		stringValue(payload.session_id) ||
		stringValue(session?.session_id) ||
		active?.sessionId;
	if (!sessionId) return active;
	const action = objectValue(payload.next_action);
	const rawRequest = objectValue(action?.request);
	const requestId = stringValue(rawRequest?.request_id);
	const request = requestId
		? {
				request_id: requestId,
				capability: stringValue(rawRequest?.capability),
				query: stringValue(rawRequest?.query),
				parameters: objectValue(rawRequest?.parameters),
				hypothesis_id: stringValue(rawRequest?.hypothesis_id),
			}
		: undefined;
	const nextAction: CognitiveAction | undefined = action
		? {
				type: stringValue(action.type),
				instruction: stringValue(action.instruction),
				submit_via: stringValue(action.submit_via),
				required_fields: Array.isArray(action.required_fields)
					? action.required_fields.filter(
							(item): item is string => typeof item === "string",
						)
					: undefined,
				request,
			}
		: undefined;
	const accepted = Array.isArray(payload.accepted_evidence_ids)
		? payload.accepted_evidence_ids.filter(
				(item): item is string => typeof item === "string",
			)
		: [];
	const completed = payload.status === "complete";
	// Accepted evidence or a new request resets the discovery allowance.
	const redundantDiscoveryCalls =
		accepted.length > 0 || requestId ? 0 : active?.redundantDiscoveryCalls || 0;
	const context = objectValue(payload.context);
	const evidenceList: unknown[] = Array.isArray(context?.evidence)
		? context.evidence
		: [];
	// Only a payload with no evidence field may fall back to previous
	// records; an explicitly present list — even empty/all-failed/retracted —
	// is authoritative. Stale records and retracted ids never remain usable
	// grounding.
	const evidenceExplicit = Array.isArray(context?.evidence);
	const rawEvidence: unknown[] = evidenceExplicit ? evidenceList : [];
	const hypotheses = Array.isArray(context?.hypotheses)
		? context.hypotheses
				.map((item) => objectValue(item))
				.filter((item): item is Record<string, unknown> => Boolean(item))
				.map((item) => ({
					hypothesis_id: stringValue(item.hypothesis_id) || "",
					statement: stringValue(item.statement) || "",
					falsification_test: stringValue(item.falsification_test) || "",
					status: stringValue(item.status) || "open",
					supporting_evidence: Array.isArray(item.supporting_evidence)
						? item.supporting_evidence.filter(
								(value): value is string => typeof value === "string",
							)
						: [],
					contradicting_evidence: Array.isArray(item.contradicting_evidence)
						? item.contradicting_evidence.filter(
								(value): value is string => typeof value === "string",
							)
						: [],
					bearing_evidence: Array.isArray(item.bearing_evidence)
						? item.bearing_evidence.filter(
								(value): value is string => typeof value === "string",
							)
						: [],
				}))
				.filter(
					(item) =>
						Boolean(item.hypothesis_id) &&
						Boolean(item.statement) &&
						Boolean(item.falsification_test),
				)
		: active?.hypotheses || [];
	const rawDerivations = Array.isArray(context?.deterministic_derivations)
		? context.deterministic_derivations
		: [];
	const rawDiagnostic = rawDerivations
		.map((item) => objectValue(item))
		.find((item) => item?.operation === "diagnostic_chain");
	const diagnosticNodes = Array.isArray(rawDiagnostic?.nodes)
		? rawDiagnostic.nodes.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const diagnosticSources = Array.isArray(rawDiagnostic?.sources)
		? rawDiagnostic.sources.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const diagnosticAnswer = stringValue(rawDiagnostic?.answer);
	const diagnosticDerivation =
		diagnosticAnswer &&
		diagnosticNodes.length >= 2 &&
		diagnosticSources.length >= 2
			? {
					answer: diagnosticAnswer,
					nodes: diagnosticNodes,
					sources: diagnosticSources,
				}
			: undefined;
	const rawPlan = rawDerivations
		.map((item) => objectValue(item))
		.find((item) => item?.operation === "ordered_plan");
	const planNodes = Array.isArray(rawPlan?.nodes)
		? rawPlan.nodes.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const planSources = Array.isArray(rawPlan?.sources)
		? rawPlan.sources.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const rawOwners = objectValue(rawPlan?.owners);
	const planOwners = Object.fromEntries(
		Object.entries(rawOwners || {}).filter(
			(entry): entry is [string, string] => typeof entry[1] === "string",
		),
	);
	const planDerivation =
		planNodes.length >= 2 && planSources.length >= 2
			? {
					nodes: planNodes,
					owners: planOwners,
					sources: planSources,
				}
			: undefined;
	const semantic = objectValue(rawDerivations[0]);
	const semanticNodes = Array.isArray(semantic?.nodes)
		? semantic.nodes.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const semanticSources = Array.isArray(semantic?.sources)
		? semantic.sources.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const semanticDerivation =
		(semantic?.operation === "semantic_chain" ||
			semantic?.operation === "semantic_rule") &&
		semanticNodes.length >= 3 &&
		semanticSources.length >= 2
			? { nodes: semanticNodes, sources: semanticSources }
			: undefined;
	const rawCognition = objectValue(payload.cognition);
	const reasoningMoves = Array.isArray(rawCognition?.reasoning_moves)
		? rawCognition.reasoning_moves.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	const derivedInsights = Array.isArray(rawCognition?.derived_insights)
		? rawCognition.derived_insights
		: [];
	const firstInsight = objectValue(derivedInsights[0]);
	const taskFrame = objectValue(rawCognition?.task_frame);
	const unresolvedRequirements = Array.isArray(taskFrame?.requirements)
		? taskFrame.requirements
				.map((item) => objectValue(item))
				.filter(
					(item) =>
						item &&
						stringValue(item.status) !== "covered" &&
						Boolean(stringValue(item.statement)),
				)
				.slice(0, 4)
				.map(
					(item) =>
						`${stringValue(item?.requirement_id) || "requirement"}: ` +
						`${stringValue(item?.statement)} ` +
						`[${stringValue(item?.status) || "unresolved"}]`,
				)
		: [];
	const cognitionStage = stringValue(rawCognition?.stage);
	const cognition = cognitionStage
		? {
				stage: cognitionStage,
				move: reasoningMoves[0],
				derivedInsight: stringValue(firstInsight?.statement),
				unresolvedRequirements,
				decisionRule: stringValue(rawCognition?.decision_rule),
			}
		: undefined;
	// Poisoned observations must never reach deliberation.
	const cleanEvidence = rawEvidence.filter((item) => {
		const evidence = objectValue(item);
		const quarantine = evidence?.quarantine_flags;
		return (
			stringValue(evidence?.status) !== "failed" &&
			!(Array.isArray(quarantine) && quarantine.length > 0)
		);
	});
	const hasReadEvidence = cleanEvidence.some((item) =>
		(stringValue(objectValue(item)?.source) || "")
			.trim()
			.startsWith("pi:read:"),
	);
	const evidenceRecords = cleanEvidence
		.map((item): { id?: string; source: string; fact: string } | undefined => {
			const evidence = objectValue(item);
			const content = stringValue(evidence?.content);
			// Id and source are normalized here: whitespace spellings must
			// never fabricate a second identity or independent source.
			const source = (stringValue(evidence?.source) || "").trim();
			if (!content || (hasReadEvidence && source.startsWith("pi:find:")))
				return;
			const fact = content.replace(/^\[CORTHEON_HOST_EVIDENCE\] [^\n]*\n?/, "");
			// Explicit evidence without a non-empty runtime id has no provable
			// identity: it must never enter records, summary, or deliberation.
			const id = (stringValue(evidence?.evidence_id) || "").trim();
			if (!id) return;
			return {
				id,
				source: source || "host",
				fact: fact.slice(0, 700),
			};
		})
		.filter((item): item is { id?: string; source: string; fact: string } =>
			Boolean(item?.fact),
		);
	// Conflicting records behind one id exclude all records with that id;
	// exact duplicates keep one.
	const byId = new Map<
		string,
		{ id?: string; source: string; fact: string }[]
	>();
	for (const record of evidenceRecords) {
		const list = byId.get(record.id || "") || [];
		list.push(record);
		byId.set(record.id || "", list);
	}
	const usableRecords: Array<{ id: string; source: string; fact: string }> = [];
	for (const [id, list] of byId) {
		if (!id) continue;
		const signatures = new Set(list.map((r) => `${r.source}\u0000${r.fact}`));
		if (signatures.size === 1)
			usableRecords.push(
				list[0] as { id: string; source: string; fact: string },
			);
	}
	const finalRecords = usableRecords.slice(0, 6);
	const evidenceRecordList = finalRecords as Array<{
		id?: string;
		source: string;
		fact: string;
	}>;
	const evidenceSummary = [
		...finalRecords.map((item) => `[${item.source}] ${item.fact}`),
		semanticDerivation
			? `Cortheon deterministic chain: ${semanticDerivation.nodes.join(" -> ")}`
			: "",
	]
		.filter(Boolean)
		.join("\n\n");
	// A present context.evidence is the authoritative snapshot: active ids
	// come only from its final usable records (never a union with earlier
	// ids, never id-less or collided entries); an explicit empty list clears
	// ids, records, and summary.
	const snapshotIds = finalRecords.map((item) => item.id);
	setActive({
		sessionId,
		goal: stringValue(context?.goal) || active?.goal,
		deliverable: stringValue(session?.deliverable) || active?.deliverable,
		request,
		nextAction,
		evidenceIds: evidenceExplicit
			? [...new Set(snapshotIds)]
			: [...new Set([...(active?.evidenceIds || []), ...accepted])],
		hypotheses,
		completed,
		answer:
			completed && typeof payload.answer === "string"
				? payload.answer
				: active?.answer,
		automaticContinuations: active?.automaticContinuations || 0,
		admittedToolCalls: active?.admittedToolCalls || 0,
		redundantDiscoveryCalls,
		needsContinuation: false,
		pendingReadObservations: requestId
			? []
			: active?.pendingReadObservations || [],
		webUnavailableReported: active?.webUnavailableReported || false,
		evidenceRecords: evidenceExplicit
			? evidenceRecordList
			: evidenceRecordList.length > 0
				? evidenceRecordList
				: active?.evidenceRecords || [],
		evidenceSummary: evidenceExplicit
			? boundedEvidence(evidenceSummary)
			: evidenceSummary
				? boundedEvidence(evidenceSummary)
				: active?.evidenceSummary,
		semanticDerivation: semanticDerivation || active?.semanticDerivation,
		diagnosticDerivation: diagnosticDerivation || active?.diagnosticDerivation,
		planDerivation: planDerivation || active?.planDerivation,
		cognition: cognition || active?.cognition,
		repairPlan: active?.repairPlan,
		testInvocation: active?.testInvocation,
		protectedTestPaths: active?.protectedTestPaths || [],
		mutationTargets: active?.mutationTargets || [],
		mutationEvidence: active?.mutationEvidence || {},
		initialFileHashes: active?.initialFileHashes || {},
		mutationInFlight: active?.mutationInFlight || false,
	});
	if (completed) stopHeartbeat();
	else ensureHeartbeat();
	return getActive();
}
