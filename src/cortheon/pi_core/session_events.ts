import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { modelFacingAction, nextActionInstruction } from "./actions.ts";
import {
	CONTINUATION_PREFIX,
	LEASE_SECONDS,
	MAX_AUTOMATIC_CONTINUATIONS,
	modelContext,
	protocol,
} from "./protocol.ts";
import { mergePayload } from "./merge.ts";
import { autoEnable, debug, ensureRuntime, runtimeBase, runtimeCall } from "./runtime.ts";
import {
	CONTINUATION_NO_PROGRESS_DISPOSITION,
	CONTINUATION_SPENT_DISPOSITION,
	HOST_BOUND_DISPOSITION,
	answerOnlyExhaustedDisposition,
	continuationFingerprint,
	markTerminationState,
} from "./budget.ts";
import {
	reportUnsatisfiedDeterministicRequest,
	satisfyDeterministicRequest,
} from "./satisfy.ts";
import { emitTerminalStatusOnce } from "./terminal.ts";
import {
	adapterEvaluationProfile,
	evaluationProfile,
	operatorEnabled,
} from "./protocol.ts";
import {
	abandonActive,
	answerAlreadyDelivered,
	evaluatorBoundReached,
	getActive,
	getContinuationFingerprint,
	isAnswerOnly,
	isEnabled,
	markAnswerOnly,
	peekTerminalDisposition,
	resetFinalization,
	scheduleContinuation,
	setContinuationFingerprint,
	setEnabled,
	setTerminalDisposition,
	stopHeartbeat,
	takeScheduledContinuation,
} from "./state.ts";
import {
	effortForPrompt,
	isAmbiguityGoal,
	isCausalSynthesisGoal,
	protectedTestPaths,
	requestedMutationPaths,
	requestedTestInvocation,
	shouldStartAutomatically,
} from "./task_analysis.ts";

export function registerSessionEvents(pi: ExtensionAPI): void {
	const closeEvaluatorBound = async (): Promise<boolean> => {
		if (!evaluatorBoundReached() || answerAlreadyDelivered()) return false;
		const current = getActive();
		markAnswerOnly();
		if (!peekTerminalDisposition()) {
			setTerminalDisposition({
				reason: HOST_BOUND_DISPOSITION,
				causal: Boolean(
					current?.deliverable === "document_synthesis" &&
						isCausalSynthesisGoal(current.goal) &&
						!isAmbiguityGoal(current.goal),
				),
				submitted: false,
			});
		}
		if (current && !current.completed) await abandonActive();
		emitTerminalStatusOnce(pi);
		return true;
	};

	pi.on("session_start", async () => {
		setEnabled(false);
		// resetFinalization clears all finalization/continuation state so
		// nothing from a previous task survives session start.
		resetFinalization();
		await abandonActive();
		if (autoEnable()) {
			try {
				await ensureRuntime();
				setEnabled(true);
			} catch (error) {
				setEnabled(false);
				debug(
					`automatic enable failed: ${
						error instanceof Error ? error.message : String(error)
					}`,
				);
			}
		}
	});

/** Send the one follow-up this extension schedules: the exact text is held
 * in memory first so only this prompt is ever recognized as a continuation;
 * a user-authored prompt that merely starts with the prefix is an ordinary
 * new turn. */
function sendScheduledContinuation(pi: ExtensionAPI, text: string): void {
	scheduleContinuation(text);
	pi.sendUserMessage(text, { deliverAs: "followUp" });
}

	pi.on("before_agent_start", async (event, context) => {
		if (!isEnabled()) return;
		const continuation = takeScheduledContinuation(event.prompt);
		const forgedPrefix =
			!continuation && event.prompt.startsWith(CONTINUATION_PREFIX);
		const automatic =
			!continuation && !forgedPrefix && shouldStartAutomatically(event.prompt);
		debug(
			`before_agent_start automatic=${automatic} continuation=${continuation} ` +
				`runtime=${runtimeBase()}`,
		);
		if (automatic) {
			resetFinalization();
			await abandonActive();
			try {
				const profile = adapterEvaluationProfile();
				const payload = await runtimeCall("/v1/start", {
					goal: event.prompt,
					constraints: [],
					effort: effortForPrompt(event.prompt),
					task_kind: "auto",
					lease_seconds: LEASE_SECONDS,
					...(profile ? { evaluation_profile: profile } : {}),
				});
				mergePayload(payload);
				if (getActive()) {
					const active = getActive()!;
					active.testInvocation = requestedTestInvocation(event.prompt);
					active.protectedTestPaths = protectedTestPaths(event.prompt);
					active.mutationTargets = requestedMutationPaths(
						event.prompt,
						active.protectedTestPaths,
					);
				}
				const deterministicRequest = getActive()?.request?.request_id;
				for (
					let attempt = 0;
					operatorEnabled("retrieval") && attempt < 3 && getActive()?.request;
					attempt += 1
				) {
					const requestId = getActive()!.request!.request_id;
					await satisfyDeterministicRequest(context);
					if (getActive()?.request?.request_id === requestId) break;
				}
				const unsatisfied = getActive()?.request;
				if (
					unsatisfied &&
					unsatisfied.capability === "grep" &&
					unsatisfied.request_id === deterministicRequest
				) {
					// A live runtime issued a request the host could not satisfy
					// (invalid parameters, outside the project, or the host tool
					// failed). Re-plan through the runtime via one bounded failed
					// observation; if it cannot move on, end gated with a truthful
					// explicit disposition — never a silent drop to an unbounded
					// bare-model path.
					const replanned = await reportUnsatisfiedDeterministicRequest(
						unsatisfied,
					);
					if (!replanned) {
						debug(
							`unsatisfied grep ${deterministicRequest} had no bounded re-plan`,
						);
						markAnswerOnly();
						setTerminalDisposition({
							reason:
								"the runtime's deterministic evidence request could not " +
								"be satisfied on the host (invalid parameters, outside " +
								"the project, or the host tool failed) and no bounded " +
								"re-plan followed",
							causal: false,
						});
						await abandonActive();
					}
				}
				debug(
					`automatic session ready=${Boolean(getActive())} ` +
						`completed=${Boolean(getActive()?.completed)} ` +
						`pending=${getActive()?.request?.request_id || "none"}`,
				);
			} catch (error) {
				debug(
					`automatic start failed: ${
						error instanceof Error ? error.message : String(error)
					}`,
				);
				// Clean up like every other abandonment — heartbeat, claims,
				// best-effort /v1/abandon — never a bare state drop that leaves
				// runtime-side session state alive.
				await abandonActive();
			}
		} else if (!continuation) {
			// New user turn: abandon stale finalization state so it cannot
			// block this prompt's tools.
			if (getActive() || isAnswerOnly()) {
				debug("abandoning stale investigation state before a new user turn");
				resetFinalization();
				await abandonActive();
			}
		}
		const active = getActive();
		const profile = evaluationProfile();
		if (profile && !operatorEnabled("retrieval")) {
			return { systemPrompt: `${event.systemPrompt}\n\n${modelContext}` };
		}
		if (profile?.config.cleanup_before_answer) {
			const evidenceSummary = active?.evidenceSummary;
			await abandonActive();
			const evidence = evidenceSummary
				? `\n\nCORTHEON RETRIEVAL CONDITION: Use this bounded host evidence. ` +
					`Cortheon will not reason, verify, block, or rewrite the answer.\n${evidenceSummary}`
				: "";
			return {
				systemPrompt: `${event.systemPrompt}\n\n${modelContext}${evidence}`,
			};
		}
		const causalSynthesis = Boolean(
			active &&
				!active.completed &&
				isCausalSynthesisGoal(active.goal) &&
				!isAmbiguityGoal(active.goal),
		);
		const activeInstruction = active
			? `\n\nCORTHEON_ACTIVE ${JSON.stringify({
					session_id: active.sessionId,
					completed: active.completed,
					next_action: causalSynthesis
						? undefined
						: modelFacingAction(active.nextAction),
					evidence: active.evidenceSummary,
					hypotheses:
						causalSynthesis || !operatorEnabled("hypothesis_framing")
							? undefined
							: active.hypotheses,
					cognition:
						causalSynthesis || !operatorEnabled("cross_source_derivation")
							? undefined
							: active.cognition,
					repair_plan: operatorEnabled("retrieval")
						? active.repairPlan
						: undefined,
					required_test: operatorEnabled("retrieval")
						? active.testInvocation?.commandLine
						: undefined,
					certified_answer: active.completed ? active.answer : undefined,
				})}`
			: "";
		const evidenceReadyInstruction =
			causalSynthesis && active?.evidenceSummary
				? "\n\nCORTHEON_EVIDENCE_READY: Evidence preloaded. Answer when sufficient; " +
					"use Pi tools only if needed."
				: "";
		const certifiedInstruction =
			active?.completed && active.answer
				? "\n\nCORTHEON_CERTIFIED: Return certified_answer exactly and stop. " +
					"Do not call any tool or perform another reasoning pass."
				: "";
		return {
			systemPrompt:
				`${event.systemPrompt}\n\n${protocol}${activeInstruction}` +
				evidenceReadyInstruction +
				certifiedInstruction,
		};
	});

	pi.on("turn_end", async (_event, context) => {
		if (await closeEvaluatorBound()) context.abort();
	});

	pi.on("agent_end", async () => {
		if (evaluationProfile()?.config.cleanup_before_answer) {
			await abandonActive();
			return;
		}
		if (await closeEvaluatorBound()) return;
		// Apply any terminal boundary before continuing, so a bounded run can
		// never end silently: the single follow-up or a sticky disposition.
		if (!isAnswerOnly()) {
			await markTerminationState();
		}
		if (isAnswerOnly()) {
			const current = getActive();
			// The ONE unified budget: an answer-only follow-up may run only
			// while a session is retained at the initial finish/evidence/
			// certified boundary and no repair continuation spent the budget.
			const mayAnswerFollowUp = Boolean(
				current &&
					!current.completed &&
					!answerAlreadyDelivered() &&
					current.automaticContinuations < MAX_AUTOMATIC_CONTINUATIONS,
			);
			if (mayAnswerFollowUp) {
				current!.automaticContinuations += 1;
				sendScheduledContinuation(
					pi,
					`${CONTINUATION_PREFIX} Cortheon stopped tool use for this ` +
						"investigation: the accepted evidence is sufficient. Answer now " +
						"from the accepted evidence without calling any tool.",
				);
				return;
			}
			// Budget spent, session abandoned, or answer already delivered:
			// never schedule another model turn — including one whose only
			// purpose would be to intercept its output.
			if (!answerAlreadyDelivered() && !peekTerminalDisposition()) {
				setTerminalDisposition(answerOnlyExhaustedDisposition(current));
			}
			const finalSession = getActive();
			if (finalSession && !finalSession.completed) {
				await abandonActive();
			}
			emitTerminalStatusOnce(pi);
			return;
		}
		const active = getActive();
		if (!active?.needsContinuation) return;
		const fingerprint = continuationFingerprint(active);
		if (
			active.automaticContinuations >= MAX_AUTOMATIC_CONTINUATIONS ||
			fingerprint === getContinuationFingerprint()
		) {
			// Repeated withholds terminate here, never abandon-and-wait: the
			// sticky disposition replaces later answers with one withheld
			// result, and the host sees exactly one terminal immediately. The
			// fingerprint arm catches a withhold repeating the exact granted
			// request/action.
			markAnswerOnly();
			setTerminalDisposition({
				reason:
					active.automaticContinuations >= MAX_AUTOMATIC_CONTINUATIONS
						? CONTINUATION_SPENT_DISPOSITION
						: CONTINUATION_NO_PROGRESS_DISPOSITION,
				causal: Boolean(
					active.deliverable === "document_synthesis" &&
						isCausalSynthesisGoal(active.goal) &&
						!isAmbiguityGoal(active.goal),
				),
			});
			await abandonActive();
			emitTerminalStatusOnce(pi);
			return;
		}
		setContinuationFingerprint(fingerprint);
		active.needsContinuation = false;
		active.automaticContinuations += 1;
		sendScheduledContinuation(
			pi,
			`${CONTINUATION_PREFIX} Completion was withheld. Perform this one bounded ` +
				`Cortheon reasoning or evidence action, then continue the task: ` +
				nextActionInstruction(active),
		);
	});

	pi.on("session_shutdown", async () => {
		await abandonActive();
		setEnabled(false);
		stopHeartbeat();
		resetFinalization();
	});
}
