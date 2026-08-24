import type { CausalStageReason } from "./candidate_capture.ts";
import { mergePayload } from "./merge.ts";
import { mapHypothesisEvidence } from "./evidence_mapping.ts";
import type { SynthesisResult } from "./repair.ts";
import { getActive } from "./state.ts";
import { runtimeCall } from "./runtime.ts";

/** Submit the validated synthesis through /v1/complete with the grounded
 * Cause as the claim, plus Cause/Rival hypotheses sharing the Test. Cause
 * is "supported" only when clean evidence reflects it; else uncertain with
 * no ids (the runtime withholds for a grounded reason). Rival is "refuted"
 * only on a direct counterexample; else uncertain with bearing records. A
 * valid synthesis is always submitted when any usable evidence id exists
 * (never silently skip /v1/complete). Certified answer only on complete
 * status; never certifies locally. ``submitted`` separates mapping failure,
 * withheld submission, and transport failure (throws); ``reason`` is a
 * stage code, never candidate text. */
export interface CausalSubmission {
	answer?: string;
	submitted: boolean;
	reason?: CausalStageReason;
}

export async function submitCausalSynthesis(
	deliberated: SynthesisResult,
): Promise<CausalSubmission> {
	if (!deliberated.text || !deliberated.sections) {
		return { submitted: false, reason: deliberated.reason || "deliberation_empty" };
	}
	// A lost session and id-less records are one failure: nothing to bind.
	const active = getActive();
	if (!active) return { submitted: false, reason: "mapping_failed" };
	const sections = deliberated.sections;
	const mapping = mapHypothesisEvidence(sections, active.evidenceRecords);
	if (!mapping) return { submitted: false, reason: "mapping_failed" };
	const payload = await runtimeCall("/v1/complete", {
		session_id: active.sessionId,
		answer: deliberated.text,
		claims: [
			{
				claim: sections.cause,
				evidence_ids: mapping.cleanIds,
			},
		],
		hypotheses: [
			{
				statement: sections.cause,
				falsification_test: sections.test,
				status: mapping.causeStatus,
				evidence_ids: mapping.causeIds,
			},
			{
				statement: sections.rival,
				falsification_test: sections.test,
				status: mapping.rivalStatus,
				evidence_ids: mapping.rivalIds,
			},
		],
		completion_evidence_ids: mapping.cleanIds,
	});
	mergePayload(payload);
	const after = getActive();
	if (after?.completed && after.answer) return { answer: after.answer, submitted: true };
	return { submitted: true, reason: "runtime_withheld" };
}
