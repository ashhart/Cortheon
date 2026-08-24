import path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { completionHypotheses, nextActionInstruction } from "./actions.ts";
import {
	certifyFinalTestNotice,
	certifyPatchNotice,
} from "./certify_mutation.ts";
import { contentText, patchFromSuccessfulEdit } from "./host_evidence.ts";
import {
	ANSWER_ONLY_TOOL_REASON,
	BUDGET_TOOL_REASON,
	CERTIFIED_TOOL_REASON,
	CONTINUATION_EXHAUSTED_TOOL_REASON,
	EVIDENCE_SUFFICIENT_TOOL_REASON,
	HOST_BOUND_DISPOSITION,
	SUBMITTED_NOT_CERTIFIED_DISPOSITION,
	causalEvidenceSufficient,
	continuationBudgetExhausted,
	discoveryExhausted,
	finishPhase,
	markTerminationState,
	maxHostToolCalls,
	toolBatchMustTerminate,
	toolBudgetExhausted,
} from "./budget.ts";
import { mergePayload } from "./merge.ts";
import { causalAnswerResult } from "./causal_answer.ts";
import {
	observeHostToolResult,
	observePassiveHostToolResult,
	reportUnavailableWebTool,
} from "./observe_tools.ts";
import {
	MAX_AUTOMATIC_CONTINUATIONS,
	WITHHELD_PREFIX,
	objectValue,
	stringValue,
} from "./protocol.ts";
import {
	clearBenchmarkCandidate,
	emitBenchmarkCandidate,
	retainBenchmarkCandidate,
} from "./candidate_capture.ts";
import {
	emitTerminalStatusOnce,
	terminalDispositionResult,
	terminalStatusText,
} from "./terminal.ts";
import { debug, isRuntimePolicyRefusal, runtimeCall } from "./runtime.ts";
import {
	abandonActive,
	answerAlreadyDelivered,
	getActive,
	isAnswerOnly,
	isMultiMutation,
	markAnswerDelivered,
	markAnswerOnly,
	peekTerminalDisposition,
	recordBoundStep,
	setTerminalDisposition,
} from "./state.ts";
import {
	commandRunsRequiredTest,
	isAmbiguityGoal,
	isCausalSynthesisGoal,
} from "./task_analysis.ts";
import { evaluationProfile, operatorEnabled } from "./protocol.ts";

export function registerToolEvents(pi: ExtensionAPI): void {
	// Unavailable tools never reach tool_call (Pi answers "not found" without
	// terminate), so a mixed batch survives and reopens the doom loop after
	// finalization; ending it requires aborting the agent operation here.
	pi.on("tool_execution_start", async (event, context) => {
		if (await reportUnavailableWebTool(pi, event, context)) return;
		if (!toolBatchMustTerminate(getActive())) return;
		if (pi.getActiveTools().includes(event.toolName)) {
			return;
		}
		debug(
			`aborting agent operation: unavailable tool ${event.toolName} ` +
				"requested after Cortheon finalization",
		);
		// Run the finalization transition before the abort, or the
		// continuation is never scheduled.
		await markTerminationState();
		context.abort();
	});

	pi.on("tool_call", async (event, context) => {
		const active = getActive();
		if (evaluationProfile() && !operatorEnabled("retrieval")) return;
		if (toolBatchMustTerminate(active)) {
			await markTerminationState();
			if (active && active.completed) {
				return {
					block: true,
					terminate: true,
					reason: CERTIFIED_TOOL_REASON,
				};
			}
			if (finishPhase(active)) {
				return {
					block: true,
					terminate: true,
					reason: ANSWER_ONLY_TOOL_REASON,
				};
			}
			if (active && toolBudgetExhausted(active)) {
				const cap = maxHostToolCalls(active.deliverable);
				return {
					block: true,
					terminate: true,
					reason: BUDGET_TOOL_REASON(active, cap),
				};
			}
			if (discoveryExhausted(active)) {
				return {
					block: true,
					terminate: true,
					reason: EVIDENCE_SUFFICIENT_TOOL_REASON,
				};
			}
			if (active && continuationBudgetExhausted(active)) {
				return {
					block: true,
					terminate: true,
					reason: CONTINUATION_EXHAUSTED_TOOL_REASON,
				};
			}
			return {
				block: true,
				terminate: true,
				reason: ANSWER_ONLY_TOOL_REASON,
			};
		}
		// No active investigation and not finalizing: normal Pi behavior.
		if (!active) return;
		if (event.toolName === "cortheon_start") {
			return {
				block: true,
				reason: "Cortheon already has an active automatic investigation.",
			};
		}
		const input = event.input as Record<string, unknown>;
		const target = stringValue(input.path);
		if (
			["edit", "write"].includes(event.toolName) &&
			target &&
			active.protectedTestPaths.some(
				(item) => path.normalize(item) === path.normalize(target),
			)
		) {
			return {
				block: true,
				reason: `Cortheon protects the requested test file from mutation: ${target}`,
			};
		}
		const plan = active.repairPlan;
		if (
			active.deliverable === "code_change" &&
			plan &&
			!isMultiMutation() &&
			!active.request &&
			active.testInvocation
		) {
			if (!context.isProjectTrusted()) {
				return {
					block: true,
					reason: "Cortheon will not mutate or test an untrusted project.",
				};
			}
			if (event.toolName !== "edit") {
				return {
					block: true,
					reason:
						`Cortheon derived a bounded repair for ${plan.path}. Use Pi's edit ` +
						"tool; the adapter will run and bind the required test.",
				};
			}
			input.path = plan.path;
			input.edits = [{ oldText: plan.oldText, newText: plan.newText }];
		}
		// Only calls that reach this point execute, so only they count
		// against the budget; discovery past sufficiency against a small allowance.
		if (operatorEnabled("adaptive_stopping") && causalEvidenceSufficient(active)) {
			active.redundantDiscoveryCalls += 1;
		}
		active.admittedToolCalls += 1;
	});

	pi.on("tool_result", async (event, context) => {
		debug(
			`tool_result name=${event.toolName} error=${event.isError} ` +
				`command=${stringValue(event.input.command) || ""} ` +
				`multi=${isMultiMutation()} pending=${getActive()?.request?.request_id || "none"}`,
		);
		if (evaluationProfile() && !operatorEnabled("retrieval")) {
			if (operatorEnabled("verification")) {
				await observePassiveHostToolResult(event);
			}
			return;
		}
		if (
			getActive() &&
			!getActive()!.completed &&
			getActive()!.deliverable === "code_change" &&
			!isMultiMutation() &&
			getActive()!.testInvocation &&
			event.toolName === "edit"
		) {
			const details = objectValue(event.details);
			const filePath =
				stringValue(event.input.path) || getActive()?.repairPlan?.path;
			const patch =
				stringValue(details?.patch) ||
				(filePath ? patchFromSuccessfulEdit(filePath, event.input) : "");
			if (!event.isError && filePath && patch) {
				return {
					content: [
						...event.content,
						{
							type: "text" as const,
							text: await certifyPatchNotice(pi, context, filePath, patch),
						},
					],
				};
			}
		}
		const mutationState = getActive();
		if (
			mutationState &&
			!mutationState.completed &&
			mutationState.deliverable === "code_change" &&
			isMultiMutation() &&
			mutationState.testInvocation &&
			event.toolName === "bash" &&
			!event.isError &&
			commandRunsRequiredTest(
				stringValue(event.input.command),
				mutationState.testInvocation,
			)
		) {
			return {
				content: [
					...event.content,
					{
						type: "text" as const,
						text: await certifyFinalTestNotice(
							context,
							contentText(event.content),
						),
					},
				],
			};
		}
	return await observeHostToolResult(event);
	});

	pi.on("message_end", async (event, context) => {
		if (event.message.role === "assistant") {
			recordBoundStep();
		}
		const session = getActive();
		if (
			event.message.role === "assistant" &&
			evaluationProfile()?.config.cleanup_before_answer
		) {
			await abandonActive();
			return;
		}
		if (!session && event.message.role === "assistant") {
			// An abandoned session must not deliver continuation text
			// unvalidated: stamp the held disposition on every answerable text.
			return terminalDispositionResult(pi, event.message);
		}
		if (!session || event.message.role !== "assistant") return;
		const message = event.message;
		if (message.content.some((item) => item.type === "toolCall")) {
			return;
		}
		const proposedAnswer = contentText(message.content);
		if (!proposedAnswer) return;
		if (session.completed && session.answer) {
			markAnswerDelivered();
			return {
				message: {
					...event.message,
					content: [{ type: "text", text: session.answer }],
				},
			};
		}
		if (
			operatorEnabled("hypothesis_framing") &&
			operatorEnabled("discriminating_evidence") &&
			operatorEnabled("contradiction_revision") &&
			operatorEnabled("cross_source_derivation") &&
			session.deliverable === "document_synthesis" &&
			isCausalSynthesisGoal(session.goal) &&
			!isAmbiguityGoal(session.goal)
		) {
			// Bounded causal path: at most two deliberation calls, then
			return await causalAnswerResult(pi, context, message, proposedAnswer);
		}
		const submitState = getActive();
		if (!submitState) {
			return;
		}
		try {
			const evidenceIds = [...submitState.evidenceIds];
			const completionClaim =
				submitState.deliverable === "document_synthesis" &&
				submitState.evidenceSummary
					? submitState.evidenceSummary
					: proposedAnswer;
			const payload = await runtimeCall("/v1/complete", {
				session_id: submitState.sessionId,
				answer: proposedAnswer,
				claims: [{ claim: completionClaim, evidence_ids: evidenceIds }],
				hypotheses: operatorEnabled("hypothesis_framing")
					? completionHypotheses(submitState, proposedAnswer)
					: [],
				completion_evidence_ids: evidenceIds,
			});
			mergePayload(payload);
			debug(
				`model answer completion status=${String(payload.status || "unknown")} ` +
					`next=${stringValue(objectValue(payload.next_action)?.type) || "none"}`,
			);
		} catch (error) {
			if (isRuntimePolicyRefusal(error)) {
				// Policy refusal fails closed: one withheld result, one
				// the draft WAS submitted and refused, so capture records one
				// candidate with that exact text.
				emitBenchmarkCandidate(pi, "completion", proposedAnswer);
				markAnswerOnly();
				setTerminalDisposition({
					reason:
						"the live runtime explicitly refused the completion for " +
						"cognitive policy reasons",
					causal: false,
				});
				await abandonActive();
				emitTerminalStatusOnce(pi);
				return {
					message: {
						...event.message,
						content: [
							{
								type: "text" as const,
								text: terminalStatusText(peekTerminalDisposition()!),
							},
						],
					},
				};
			}
			// Runtime/transport failure: abandon the ephemeral session, let the
			// host answer stand verbatim with no sticky disposition; never
			// certify or continue. The path failed open rather than
			// terminating, so any pending benchmark candidate must not be
			// graded later: clear it.
			clearBenchmarkCandidate();
			await abandonActive();
			return;
		}
		const active = getActive();
		if (active?.completed && active.answer) {
			markAnswerDelivered();
			return {
				message: {
					...event.message,
					content: [{ type: "text", text: active.answer }],
				},
			};
		}
		if (active) {
			active.needsContinuation = Boolean(
				active.request ||
				["cortheon_step", "cortheon_challenge"].includes(
					active.nextAction?.submit_via || "",
				),
			);
		}
		// PROVISIONAL withhold — never certified, never marks the answer
		// Benchmark-only, terminal-scoped: appended once at a real terminal.
		retainBenchmarkCandidate("completion", proposedAnswer);
		if (isAnswerOnly() && !peekTerminalDisposition()) {
			// A withhold inside an answer-only window did submit real text:
			// hold the truthful submitted-but-not-certified disposition
			// before agent_end, never the false "no answerable text" one.
			setTerminalDisposition({
				reason: SUBMITTED_NOT_CERTIFIED_DISPOSITION,
				causal: Boolean(
					active &&
					active.deliverable === "document_synthesis" &&
					isCausalSynthesisGoal(active.goal) &&
					!isAmbiguityGoal(active.goal),
				),
			});
		}
		if (
			isAnswerOnly() &&
			active &&
			active.automaticContinuations >= MAX_AUTOMATIC_CONTINUATIONS
		) {
			await abandonActive();
			emitTerminalStatusOnce(pi);
			return {
				message: {
					...event.message,
					content: [
						{
							type: "text" as const,
							text: terminalStatusText(peekTerminalDisposition()!),
						},
					],
				},
			};
		}
		return {
			message: {
				...event.message,
				content: [
					{
						type: "text",
						text: `${WITHHELD_PREFIX}\nNext action: ${nextActionInstruction(active)}`,
					},
				],
			},
		};
	});
}
