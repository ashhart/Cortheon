import { completionHypotheses } from "./actions.ts";
import { compactReadFacts, deriveNumericJoin } from "./derive.ts";
import { observationReceipt } from "./host_evidence.ts";
import { mergePayload } from "./merge.ts";
import { objectValue, stringValue, type EvidenceRequest } from "./protocol.ts";
import { debug, runtimeCall } from "./runtime.ts";
import { getActive } from "./state.ts";
import { isAmbiguityGoal, isCausalSynthesisGoal } from "./task_analysis.ts";

export async function certifyAutomaticReasoning(
	reads: Array<{ path: string; source: string }>,
): Promise<void> {
	debug(
		`automatic reasoning candidate deliverable=${getActive()?.deliverable || "none"} ` +
			`request=${getActive()?.request?.request_id || "none"} reads=${reads.length} ` +
			`ambiguity=${isAmbiguityGoal(getActive()?.goal)} causal=${isCausalSynthesisGoal(getActive()?.goal)}`,
	);
	const active = getActive();
	if (
		!active ||
		active.completed ||
		active.request ||
		active.deliverable === "code_change" ||
		reads.length < 2
	) {
		return;
	}
	const ambiguity = isAmbiguityGoal(active.goal);
	const causal = isCausalSynthesisGoal(active.goal);
	if (!ambiguity && !causal) return;
	const facts = compactReadFacts(reads);
	if (!facts) return;
	if (causal && !ambiguity) {
		debug("automatic causal evidence ready for bounded model deliberation");
		return;
	}
	const evidenceIds = [...active.evidenceIds];
	const primary = active.hypotheses[0];
	const competing = active.hypotheses[1];
	const answer =
		`Verified competing records:\n${facts}\n\nThe request is ambiguous and the ` +
		"evidence supports multiple viable interpretations; do not choose an " +
		"alternative yet. Smallest clarification: which named component, " +
		"definition, owner, metric, or scope is intended?";
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer,
		claims: [{ claim: facts, evidence_ids: evidenceIds }],
		hypotheses: [
			{
				statement: facts,
				falsification_test:
					primary?.falsification_test ||
					"Compare the answer with every accepted record.",
				status: "supported",
				evidence_ids: evidenceIds,
			},
			{
				statement:
					competing?.statement ||
					"Another explanation may fit only a subset of the accepted clues.",
				falsification_test:
					competing?.falsification_test ||
					"Compare the alternatives across the named scope boundary.",
				status: "uncertain",
				evidence_ids: evidenceIds,
			},
		],
		completion_evidence_ids: evidenceIds,
	});
	const next = objectValue(payload.next_action);
	const nextRequest = objectValue(next?.request);
	debug(
		`automatic reasoning status=${String(payload.status || "unknown")} ` +
			`next=${stringValue(next?.type) || "none"} ` +
			`capability=${stringValue(nextRequest?.capability) || "none"}`,
	);
	mergePayload(payload);
}

export async function certifyAutomaticGrep(
	request: EvidenceRequest,
	observation: Record<string, unknown>,
): Promise<void> {
	const active = getActive();
	if (!active || active.completed || active.request) return;
	const receipt = observationReceipt(observation);
	const argumentsValue = objectValue(receipt?.args);
	const outcome = stringValue(receipt?.outcome);
	const pattern = stringValue(argumentsValue?.pattern);
	const filePath = stringValue(argumentsValue?.path);
	if (
		!pattern ||
		!filePath ||
		(outcome !== "match" && outcome !== "no_match")
	) {
		return;
	}
	const positive = outcome === "match";
	const importLookup = /\bimports?\b/i.test(request.query || "");
	const mappingLookup = /\bmaps?\s+to\b/i.test(request.query || "");
	const predicate = importLookup ? "imports" : "contains";
	const matchingLine = String(observation.content || "")
		.split("\n")
		.map((line) => line.trim())
		.find(
			(line) =>
				line &&
				!line.startsWith("[CORTHEON_HOST_EVIDENCE]") &&
				!/\bno matches?\b/i.test(line),
		);
	const answer = positive
		? mappingLookup && matchingLine
			? `Verified mapping — ${matchingLine}`
			: `Yes — ${filePath} ${predicate} ${pattern}.`
		: `No — a scoped grep found no '${pattern}' match in ${filePath}.`;
	const claim = positive
		? `${filePath} ${predicate} ${pattern}.`
		: importLookup
			? `${filePath} does not import ${pattern}.`
			: `${filePath} does not contain ${pattern}.`;
	const evidenceIds = [...active.evidenceIds];
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer,
		claims: [{ claim, evidence_ids: evidenceIds }],
		hypotheses: [
			{
				statement: claim,
				falsification_test:
					request.query || `Search ${filePath} for ${pattern}.`,
				status: "supported",
				evidence_ids: evidenceIds,
			},
		],
		completion_evidence_ids: evidenceIds,
	});
	mergePayload(payload);
}

export async function certifyAutomaticDiagnostic(): Promise<void> {
	const active = getActive();
	if (!active || active.completed || active.request) return;
	const derivation = active.diagnosticDerivation;
	if (!derivation) return;
	const evidenceIds = [...active.evidenceIds];
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer: derivation.answer,
		claims: [
			{
				claim:
					"The cited evidence establishes the stated diagnostic chain and " +
					"rules out the named alternative.",
				evidence_ids: evidenceIds,
			},
		],
		hypotheses: completionHypotheses(active, derivation.answer),
		completion_evidence_ids: evidenceIds,
	});
	mergePayload(payload);
}

export async function certifyAutomaticPlan(): Promise<void> {
	const active = getActive();
	if (!active || active.completed || active.request) return;
	const derivation = active.planDerivation;
	if (!derivation) return;
	const evidenceIds = [...active.evidenceIds];
	const steps = derivation.nodes.map((node, index) => {
		const owner = derivation.owners[node];
		return `${index + 1}. ${node}${owner ? ` — owner: ${owner}` : ""}.`;
	});
	const answer =
		`Safe dependency order:\n${steps.join("\n")}\n` +
		"Reason: each step follows its evidence-bound prerequisite; reversing the " +
		"order would violate the recorded dependency or policy gate.";
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer,
		claims: [
			{
				claim:
					"The cited evidence establishes the stated dependency order and " +
					"owner assignments.",
				evidence_ids: evidenceIds,
			},
		],
		hypotheses: completionHypotheses(active, answer),
		completion_evidence_ids: evidenceIds,
	});
	mergePayload(payload);
}

export async function certifyAutomaticSemantic(request: EvidenceRequest): Promise<void> {
	const active = getActive();
	if (!active || active.completed || active.request) return;
	const derivation = active.semanticDerivation;
	if (
		!derivation ||
		derivation.nodes.length < 3 ||
		new Set(derivation.sources).size < 2
	) {
		return;
	}
	const chain = derivation.nodes.join(" -> ");
	const terminal = derivation.nodes[derivation.nodes.length - 1];
	const answer =
		`Evidence chain: ${chain}. Therefore ${terminal} is the required ` +
		"result of the linked evidence.";
	const evidenceIds = [...active.evidenceIds];
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer,
		claims: [
			{
				claim: `The live documents establish the chain ${chain}.`,
				evidence_ids: evidenceIds,
			},
		],
		hypotheses: [
			{
				statement: `${terminal} is the terminal result of the evidence chain.`,
				falsification_test:
					request.query || "Trace every link through the named documents.",
				status: "supported",
				evidence_ids: evidenceIds,
			},
		],
		completion_evidence_ids: evidenceIds,
	});
	mergePayload(payload);
}

export async function certifyAutomaticNumericJoin(
	request: EvidenceRequest,
	reads: Array<{ path: string; source: string }>,
): Promise<void> {
	const active = getActive();
	if (!active || active.completed || active.request) return;
	const derivation = deriveNumericJoin(request, reads);
	if (!derivation) return;
	const evidenceIds = [...active.evidenceIds];
	const bindings = derivation.operands
		.map((operand) => `${operand.symbol} (${operand.path}) = ${operand.value}`)
		.join("; ");
	const arithmetic =
		derivation.operands.map((operand) => String(operand.value)).join(" + ") +
		` = ${derivation.total}`;
	const answer = `${bindings}. Arithmetic: ${arithmetic}.`;
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer,
		claims: [
			{
				claim:
					"The cited evidence establishes the arithmetic shown in the answer.",
				evidence_ids: evidenceIds,
			},
		],
		hypotheses: completionHypotheses(active, answer),
		completion_evidence_ids: evidenceIds,
	});
	mergePayload(payload);
}
