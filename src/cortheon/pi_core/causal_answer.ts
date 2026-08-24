import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Usage } from "@earendil-works/pi-ai";
import { submitCausalSynthesis } from "./causal_complete.ts";
import {
	type CausalStageReason,
	clearBenchmarkCandidate,
	emitBenchmarkCandidate,
	emitBenchmarkStageReason,
} from "./candidate_capture.ts";
import { combineUsage, deliberateCausalSynthesis } from "./repair.ts";
import { WITHHELD_PREFIX } from "./protocol.ts";
import { isRuntimePolicyRefusal } from "./runtime.ts";
import {
	abandonActive,
	markAnswerDelivered,
	markAnswerOnly,
	setActive,
	setTerminalDisposition,
	stopHeartbeat,
} from "./state.ts";

/** Bounded causal answer path: at most two deliberation calls, validation,
 * one /v1/complete submission. Only the runtime's certification labels the
 * answer certified; otherwise one withheld result and an abandoned session. */
export async function causalAnswerResult<
	T extends { content: unknown[]; usage: Usage },
>(
	pi: ExtensionAPI,
	context: Parameters<typeof deliberateCausalSynthesis>[0],
	message: T,
	proposedAnswer: string,
): Promise<{ message: T & { usage: Usage } } | { message: T } | undefined> {
	const deliberated = await deliberateCausalSynthesis(context, proposedAnswer);
	if (deliberated && "unavailable" in deliberated) {
		// Substrate unavailability before any candidate/usage existed: fail
		// open to the original host-model answer after abandoning the
		// ephemeral session — no withheld result, no sticky disposition.
		await abandonActive();
		return undefined;
	}
	const deliberationUsage = deliberated?.usage;
	const withheld = () => ({
		message: {
			...message,
			content: [
				{
					type: "text" as const,
					text: `${WITHHELD_PREFIX}\nCausal synthesis could not be validated ` +
						"and certified from the accepted evidence.",
				},
			],
			...(deliberationUsage
				? { usage: combineUsage(message.usage, deliberationUsage) }
				: {}),
		},
	});
	const markFinalWithhold = (reason: string) => {
		markAnswerOnly();
		setTerminalDisposition({ reason, causal: true });
	};
	if (!deliberated?.text || !deliberated.sections) {
		emitBenchmarkStageReason(pi, deliberated?.reason || "deliberation_empty");
		markFinalWithhold(
			"causal deliberation produced no validated candidate before " +
				"certification",
		);
		await abandonActive();
		return withheld();
	}
	let certifiedAnswer: string | undefined;
	let submittedForCertification = false;
	let stageReason: CausalStageReason | undefined;
	try {
		const submission = await submitCausalSynthesis(deliberated);
		certifiedAnswer = submission.answer;
		submittedForCertification = submission.submitted;
		stageReason = submission.reason;
	} catch (error) {
		if (!isRuntimePolicyRefusal(error)) {
			// Transport/protocol/unavailability during /v1/complete: abandon
			// the ephemeral session and leave the original host-model answer
			// verbatim, with no sticky disposition that later rewrites it.
			// The path failed open, so any pending benchmark candidate is
			// cleared and none is emitted.
			emitBenchmarkStageReason(pi, "transport_failed");
			clearBenchmarkCandidate();
			await abandonActive();
			return undefined;
		}
		// An explicit live policy refusal fails closed: never fall back to
		// evidence-close, never label the draft certified. The submission DID
		// reach /v1/complete and was refused, so the withheld validated
		// candidate is captured below. The truthful fixed stage is
		// runtime_withheld — never transport_failed.
		certifiedAnswer = undefined;
		submittedForCertification = true;
		stageReason = "runtime_withheld";
	}
	if (certifiedAnswer) {
		stopHeartbeat();
		markAnswerOnly();
		markAnswerDelivered();
		setActive(undefined);
		return {
			message: {
				...message,
				content: [{ type: "text", text: certifiedAnswer }],
				usage: combineUsage(message.usage, deliberated.usage),
			},
		};
	}
	if (submittedForCertification) {
		// Benchmark-only capture of the withheld validated candidate.
		emitBenchmarkCandidate(pi, "causal_synthesis", deliberated.text);
	}
	if (stageReason) emitBenchmarkStageReason(pi, stageReason);
	markFinalWithhold(
		"the causal synthesis completion was withheld or failed " +
			"certification after validation",
	);
	await abandonActive();
	return withheld();
}
