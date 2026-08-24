import type { ActiveInvestigation } from "./protocol.ts";
import {
	HEARTBEAT_INTERVAL_MS,
	configuredEvaluatorMaxSteps,
} from "./protocol.ts";
import { resetBenchmarkCapture } from "./candidate_capture.ts";
import { resetObservationClaims } from "./observe_claim.ts";
import { runtimeCall } from "./runtime.ts";

let active: ActiveInvestigation | undefined;
let heartbeatTimer: ReturnType<typeof setInterval> | undefined;
let enabled = false;
let answerOnly = false;
let answerDelivered = false;
let terminalDisposition: TerminalDisposition | undefined;
let boundSteps = 0;
let scheduledContinuation: string | undefined;
let terminalStatusEmitted = false;
let lastContinuationFingerprint: string | undefined;

export interface TerminalDisposition {
	reason: string;
	causal: boolean;
	/** False when no answer was ever submitted to the runtime on this path:
	 * the terminal replacement must not capture the model's later ordinary
	 * text as a benchmark candidate. Absent means submitted. */
	submitted?: boolean;
}

export function getActive(): ActiveInvestigation | undefined {
	return active;
}

export function setActive(next: ActiveInvestigation | undefined): void {
	active = next;
}

export function isEnabled(): boolean {
	return enabled;
}

export function setEnabled(next: boolean): void {
	enabled = next;
}

export function isMultiMutation(): boolean {
	return Boolean(active && active.mutationTargets.length > 1);
}

export function isAnswerOnly(): boolean {
	return answerOnly;
}

export function markAnswerOnly(): void {
	answerOnly = true;
}

export function markAnswerDelivered(): void {
	answerDelivered = true;
}

export function answerAlreadyDelivered(): boolean {
	return answerDelivered;
}

export function setTerminalDisposition(next: TerminalDisposition): void {
	terminalDisposition = next;
}

export function peekTerminalDisposition(): TerminalDisposition | undefined {
	return terminalDisposition;
}

export function getContinuationFingerprint(): string | undefined {
	return lastContinuationFingerprint;
}

export function setContinuationFingerprint(next: string): void {
	lastContinuationFingerprint = next;
}

export function terminalStatusAlreadyEmitted(): boolean {
	return terminalStatusEmitted;
}

export function markTerminalStatusEmitted(): void {
	terminalStatusEmitted = true;
}

function boundMaxSteps(): number {
	const raw = configuredEvaluatorMaxSteps();
	return raw && /^[1-9]\d{0,3}$/.test(raw) && Number(raw) <= 1_024
		? Number(raw)
		: 0;
}

export function recordBoundStep(): void {
	boundSteps += 1;
}

export function evaluatorBoundReached(): boolean {
	const cap = boundMaxSteps();
	return cap > 0 && boundSteps >= cap;
}

export function scheduleContinuation(text: string): void {
	scheduledContinuation = text;
}

export function takeScheduledContinuation(prompt: string): boolean {
	if (scheduledContinuation && prompt === scheduledContinuation) {
		scheduledContinuation = undefined;
		return true;
	}
	return false;
}

export function resetFinalization(): void {
	answerOnly = false;
	answerDelivered = false;
	terminalDisposition = undefined;
	terminalStatusEmitted = false;
	lastContinuationFingerprint = undefined;
	scheduledContinuation = undefined;
	boundSteps = 0;
	resetBenchmarkCapture();
	resetObservationClaims();
}

export function stopHeartbeat(): void {
	if (heartbeatTimer) clearInterval(heartbeatTimer);
	heartbeatTimer = undefined;
}

export function ensureHeartbeat(): void {
	if (heartbeatTimer) return;
	heartbeatTimer = setInterval(() => {
		const sessionId = active?.sessionId;
		if (!sessionId || active?.completed) {
			stopHeartbeat();
			return;
		}
		void runtimeCall("/v1/heartbeat", { session_id: sessionId }).catch(() => {
		});
	}, HEARTBEAT_INTERVAL_MS);
	heartbeatTimer.unref?.();
}

export async function abandonActive(): Promise<void> {
	const sessionId = active?.sessionId;
	active = undefined;
	stopHeartbeat();
	resetObservationClaims();
	if (!sessionId) return;
	try {
		await runtimeCall("/v1/abandon", { session_id: sessionId });
	} catch {
	}
}
