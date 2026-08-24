import {
	type ActiveInvestigation,
	HOST_TOOL_CALLS_OVERRIDE_CEILING,
	MAX_AUTOMATIC_CONTINUATIONS,
	MAX_HOST_TOOL_CALLS_ANSWER,
	MAX_HOST_TOOL_CALLS_CODE_CHANGE,
	MAX_REDUNDANT_DISCOVERY_CALLS,
	MIN_HOST_TOOL_CALLS,
	configuredMaxHostToolCalls,
	operatorEnabled,
} from "./protocol.ts";
import {
	abandonActive,
	getActive,
	isAnswerOnly,
	markAnswerOnly,
	setTerminalDisposition,
} from "./state.ts";
import type { TerminalDisposition } from "./state.ts";
import {
	isAmbiguityGoal,
	isCausalSynthesisGoal,
} from "./task_analysis.ts";
import { canonicalEvidenceSource } from "./host_evidence.ts";

export const ANSWER_ONLY_TOOL_REASON =
	"Cortheon has all the evidence it needs for this investigation. Do not " +
	"call any tool. Answer now using only the accepted evidence above.";

export const CERTIFIED_TOOL_REASON =
	"Cortheon already certified this result. Stop using tools and report " +
	"the certified answer.";

export const BUDGET_TOOL_REASON =
	(active: ActiveInvestigation, cap: number): string =>
		`Cortheon reached its host tool budget (${active.admittedToolCalls} of ` +
		`${cap} admitted calls). ${ANSWER_ONLY_TOOL_REASON}`;

export const EVIDENCE_SUFFICIENT_TOOL_REASON =
	"Cortheon has accepted sufficient independent evidence for this causal " +
	"synthesis investigation and the runtime has no pending evidence " +
	"request. Do not persist further discovery tools. Answer now from the " +
	"accepted evidence so the result can be validated and certified.";

export const CONTINUATION_EXHAUSTED_TOOL_REASON =
	"Cortheon's bounded completion continuation budget is exhausted after " +
	"repeatedly withheld completions and no evidence request is pending. " +
	"Do not call any tool. Answer now.";

/** Truthful reason for the tool-boundary guard and agent_end backstop. */
export const CONTINUATION_EXHAUSTED_DISPOSITION =
	"completion was repeatedly withheld and the bounded " +
	"continuation budget was exhausted";

/** Truthful disposition when a withheld completion repeated the exact
 * request/action already granted a continuation. */
export const CONTINUATION_NO_PROGRESS_DISPOSITION =
	"the completion was withheld again with the same evidence request " +
	"and action, so no further continuation could make progress";

/** Disposition when the forced answer continuation ended with no answerable text. */
export const ANSWER_ONLY_EXHAUSTED_DISPOSITION =
	"the forced answer continuation ended without any answerable text";

/** Truthful disposition when the window ended after a completion was really
 * submitted but the runtime did not certify it. */
export const SUBMITTED_NOT_CERTIFIED_DISPOSITION =
	"the final completion was submitted for certification but the runtime " +
	"did not certify it";

/** Truthful disposition when a live runtime explicitly refused an
 * observation for cognitive policy reasons: fail closed once, no terminal
 * coaching, and no candidate capture (nothing was ever submitted). */
export const OBSERVE_POLICY_REFUSAL_DISPOSITION =
	"the live runtime explicitly refused the observation for " +
	"cognitive policy reasons";

/** Truthful disposition when a boundary is reached after the one unified
 * follow-up was already spent: no further model turn may be scheduled. */
export const CONTINUATION_SPENT_DISPOSITION =
	"the one allowed automatic follow-up was already spent and the " +
	"investigation ended without a certified answer";

export const HOST_BOUND_DISPOSITION =
	"the host ended its bounded execution before Cortheon certified an answer";

/** Identity of what a continuation was granted for; a repeat is not progress. */
export function continuationFingerprint(active: ActiveInvestigation): string {
	return JSON.stringify({
		request: active.request && {
			request_id: active.request.request_id,
			capability: active.request.capability,
			query: active.request.query,
			hypothesis_id: active.request.hypothesis_id,
		},
		action: active.nextAction && {
			type: active.nextAction.type,
			instruction: active.nextAction.instruction,
			submit_via: active.nextAction.submit_via,
		},
	});
}

/** Sticky terminal disposition when an answer-only continuation is spent
 * without a delivered answer. */
export function answerOnlyExhaustedDisposition(
	active: ActiveInvestigation | undefined,
): TerminalDisposition {
	return {
		reason: ANSWER_ONLY_EXHAUSTED_DISPOSITION,
		causal: causalDisposition(active),
	};
}

/** Causal synthesis whose clean records carry >=2 unique identities from
 * >=2 distinct attributable sources with no pending request. Batches/raw
 * turns never count as independence; a new request reopens discovery. */
export function causalEvidenceSufficient(
	active: ActiveInvestigation | undefined,
): boolean {
	if (
		!active ||
		active.completed ||
		active.request ||
		active.deliverable !== "document_synthesis" ||
		!isCausalSynthesisGoal(active.goal) ||
		isAmbiguityGoal(active.goal)
	) {
		return false;
	}
	// Two-source gate re-check (merge already normalizes; this re-checks so a
	// merge regression cannot silently reopen premature forcing): non-empty
	// normalized identities only, no id bound to two records, and sources are
	// canonicalized — pi:read:a.txt and pi:grep:a.txt are one source, a bare
	// pi:bash label has no attributable origin, and "host" is not a source.
	const records = active.evidenceRecords
		.map((record) => ({
			id: (typeof record.id === "string" ? record.id : "").trim(),
			source: canonicalEvidenceSource(
				typeof record.source === "string" ? record.source : "",
			),
		}))
		.filter(
			(record) => record.id && record.source && record.source !== "host",
		);
	const identities = new Set(records.map((record) => record.id));
	if (identities.size !== records.length || identities.size < 2) return false;
	const sources = new Set(records.map((record) => record.source));
	return sources.size >= 2;
}

/** Redundant-discovery allowance spent (a pending request resets it). */
export function discoveryExhausted(
	active: ActiveInvestigation | undefined,
): boolean {
	return Boolean(
		operatorEnabled("adaptive_stopping") &&
			causalEvidenceSufficient(active) &&
			active!.redundantDiscoveryCalls >= MAX_REDUNDANT_DISCOVERY_CALLS,
	);
}

/** Continuation budget spent: repeated withholds must terminate. */
export function continuationBudgetExhausted(
	active: ActiveInvestigation | undefined,
): boolean {
	return Boolean(
		active &&
			!active.completed &&
			!active.request &&
			active.deliverable !== "code_change" &&
			active.automaticContinuations >= MAX_AUTOMATIC_CONTINUATIONS,
	);
}

/** Task-aware ceiling: code-change up to 48 host tool calls, others 16;
 * override wins but clamps to [MIN, 64]. */
export function maxHostToolCalls(deliverable?: string): number {
	const override = Number(configuredMaxHostToolCalls());
	if (Number.isFinite(override)) {
		return Math.min(
			Math.max(Math.floor(override), MIN_HOST_TOOL_CALLS),
			HOST_TOOL_CALLS_OVERRIDE_CEILING,
		);
	}
	return deliverable === "code_change"
		? MAX_HOST_TOOL_CALLS_CODE_CHANGE
		: MAX_HOST_TOOL_CALLS_ANSWER;
}

/** Finish phase: no pending request, next action is finish; code-change excluded. */
export function finishPhase(active: ActiveInvestigation | undefined): boolean {
	return Boolean(
		active &&
			!active.completed &&
			!active.request &&
			active.deliverable !== "code_change" &&
			active.nextAction?.type === "finish",
	);
}

export function toolBudgetExhausted(active: ActiveInvestigation): boolean {
	return active.admittedToolCalls >= maxHostToolCalls(active.deliverable);
}

/** Host adapter boundary invariant: once true, the tool call is blocked
 * AND the batch terminated — the loop can never pass finalization or the
 * admitted-call ceiling. */
export function toolBatchMustTerminate(
	active: ActiveInvestigation | undefined,
): boolean {
	return (
		isAnswerOnly() ||
		Boolean(active && active.completed) ||
		finishPhase(active) ||
		discoveryExhausted(active) ||
		continuationBudgetExhausted(active) ||
		Boolean(active && toolBudgetExhausted(active))
	);
}

/** The one transition that must run before the tool loop is terminated
 * (per-tool terminate or whole-operation abort). Budget exhaustion also
 * abandons the session so no lease or heartbeat survives. A retained
 * session (finish/evidence-sufficiency boundary) is only kept when the
 * unified follow-up budget is still unspent; once spent the boundary is a
 * terminal, never a reason to schedule another model turn. */
export async function markTerminationState(): Promise<void> {
	const active = getActive();
	if (active && active.completed) {
		markAnswerOnly();
		return;
	}
	const budgetSpent = Boolean(
		active && active.automaticContinuations >= MAX_AUTOMATIC_CONTINUATIONS,
	);
	// Inside an already-granted answer-only window the session must survive
	// blocked tool batches: the window's final agent_end (not a per-tool
	// block) decides certification, terminal, and abandonment.
	const spentOutsideWindow = budgetSpent && !isAnswerOnly();
	if (finishPhase(active)) {
		if (spentOutsideWindow) {
			markAnswerOnly();
			setTerminalDisposition({
				reason: CONTINUATION_SPENT_DISPOSITION,
				causal: causalDisposition(active),
			});
			await abandonActive();
			return;
		}
		markAnswerOnly();
		return;
	}
	if (active && toolBudgetExhausted(active)) {
		markAnswerOnly();
		// The abandon erases the session; without a held disposition the
		// continuation's text would be delivered unvalidated (a false allow).
		// Sticky disposition: message_end replaces answerable text until reset.
		setTerminalDisposition({
			reason:
				`the host tool budget was exhausted at ${active.admittedToolCalls} ` +
				"admitted calls before any answer could be certified",
			causal: causalDisposition(active),
		});
		await abandonActive();
		return;
	}
	if (active && discoveryExhausted(active)) {
		if (spentOutsideWindow) {
			markAnswerOnly();
			setTerminalDisposition({
				reason: CONTINUATION_SPENT_DISPOSITION,
				causal: causalDisposition(active),
			});
			await abandonActive();
			return;
		}
		// Keep the session so the forced answer still flows through
		// deliberation and certification.
		markAnswerOnly();
		return;
	}
	if (active && continuationBudgetExhausted(active)) {
		// Repeated withholds end with a sticky terminal disposition, not a bare
		// abandon: the last answer is replaced with one withheld result.
		markAnswerOnly();
		setTerminalDisposition({
			reason: CONTINUATION_EXHAUSTED_DISPOSITION,
			causal: causalDisposition(active),
		});
		await abandonActive();
	}
}

function causalDisposition(active: ActiveInvestigation | undefined): boolean {
	return Boolean(
		active &&
			active.deliverable === "document_synthesis" &&
			isCausalSynthesisGoal(active.goal) &&
			!isAmbiguityGoal(active.goal),
	);
}
