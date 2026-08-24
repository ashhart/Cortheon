import {
	type ActiveInvestigation,
	type CognitiveAction,
} from "./protocol.ts";

const CORTHEON_LIFECYCLE_TOOL =
	/\bcortheon_(?:observe|step|challenge|verify|complete|finish)\b/i;

export function modelFacingActionInstruction(action: CognitiveAction): string {
	if (
		action.instruction &&
		!CORTHEON_LIFECYCLE_TOOL.test(action.instruction)
	) {
		return action.instruction;
	}
	switch (action.type) {
		case "harness_tool":
			return (
				"Use Pi's native tools to satisfy the evidence request within its stated " +
				"budget. The adapter records the focused result automatically."
			);
		case "challenge":
			return (
				"Challenge the current answer against the accepted evidence, then return " +
				"the strongest supported revision. The adapter advances Cortheon automatically."
			);
		case "verify":
			return (
				"Check the answer and its claims against the accepted evidence, then return " +
				"the verified result. The adapter advances Cortheon automatically."
			);
		case "finish":
			return "Return the narrowest evidence-supported answer and stop.";
		default:
			return (
				"Perform the requested bounded reasoning step using Pi's native tools and " +
				"the accepted evidence. Return the result normally; the adapter advances " +
				"Cortheon automatically."
			);
	}
}

export function modelFacingAction(
	action: CognitiveAction | undefined,
): CognitiveAction | undefined {
	if (!action) return undefined;
	return {
		type: action.type,
		instruction: modelFacingActionInstruction(action),
		required_fields: action.required_fields,
		request: action.request,
	};
}

export function nextActionInstruction(investigation: ActiveInvestigation | undefined): string {
	const action = modelFacingAction(investigation?.nextAction);
	if (!action) return "Synthesize the narrowest evidence-supported answer.";
	return [
		action.instruction,
		action.request ? JSON.stringify(action.request) : undefined,
		action.required_fields?.length
			? `Required fields: ${action.required_fields.join(", ")}.`
			: undefined,
	]
		.filter(Boolean)
		.join(" ");
}

export function completionHypotheses(
	investigation: ActiveInvestigation,
	answer: string,
): Array<Record<string, unknown>> {
	const evidenceIds = investigation.evidenceIds;
	const falsificationTest =
		investigation.request?.query ||
		"Compare the proposed answer with the accepted live evidence.";
	return [
		{
			statement: answer,
			falsification_test: falsificationTest,
			status: "supported",
			evidence_ids: evidenceIds,
		},
		{
			statement:
				"The proposed answer conflicts with one or more accepted live observations.",
			falsification_test:
				"Check the proposed answer against every cited host receipt.",
			status: "refuted",
			evidence_ids: evidenceIds,
		},
	];
}
