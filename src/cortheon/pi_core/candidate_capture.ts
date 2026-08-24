import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { configuredBenchmarkCapture } from "./protocol.ts";

/** Benchmark-only capture of the exact pre-block candidate answer.
 * pi.appendEntry keeps it out of model context/stderr/project files
 * (--no-session, never persisted). Inert unless
 * CORTHEON_BENCHMARK_CAPTURE_CANDIDATE=1. The runner sees the exact
 * uncertified candidate — and the blocked text of a pre-deliberation
 * termination — so the block grades safe or false.
 *
 * Capture is terminal-scoped: a provisional withhold only retains (or
 * replaces) a pending candidate in memory; the entry is appended exactly
 * once, with the latest retained candidate, when that path actually becomes
 * terminal. A path that fails open rather than terminating clears the
 * pending candidate and emits none. All capture state is memory-only and
 * reset by resetFinalization on every new task/session/disable/shutdown. */
export const CANDIDATE_ENTRY_TYPE = "cortheon-benchmark-candidate-v1";
export const CANDIDATE_ENTRY_VERSION = 1;
export const CANDIDATE_STAGES = ["completion", "causal_synthesis"] as const;
export type CandidateStage = (typeof CANDIDATE_STAGES)[number];

export function benchmarkCaptureEnabled(): boolean {
	return configuredBenchmarkCapture() === "1";
}

interface PendingCandidate {
	stage: CandidateStage;
	candidate: string;
}

let pendingCandidate: PendingCandidate | undefined;
let candidateEmitted = false;

/** One bounded, opt-in append: the only place an entry ever reaches Pi. */
function appendBenchmarkEntry(
	pi: ExtensionAPI,
	customType: string,
	data: Record<string, unknown>,
): void {
	if (!benchmarkCaptureEnabled()) return;
	try {
		pi.appendEntry(customType, { version: CANDIDATE_ENTRY_VERSION, ...data });
	} catch {
		// Capture is measurement-only; it must never change delivery.
	}
}

/** Append one candidate entry unless this window already emitted its one
 * terminal candidate. */
function appendCandidateOnce(
	pi: ExtensionAPI,
	stage: CandidateStage,
	candidate: string,
): void {
	if (candidateEmitted) return;
	candidateEmitted = true;
	appendBenchmarkEntry(pi, CANDIDATE_ENTRY_TYPE, { stage, candidate });
}

/** Retain (or replace) the pending candidate during a provisional withhold.
 * Memory-only; nothing is appended until the path actually terminates. */
export function retainBenchmarkCandidate(
	stage: CandidateStage,
	candidate: string,
): void {
	if (!benchmarkCaptureEnabled() || !candidate) return;
	pendingCandidate = { stage, candidate };
}

/** A later fail-open (transport) never terminated: drop the pending
 * candidate so nothing from this window is graded later. */
export function clearBenchmarkCandidate(): void {
	pendingCandidate = undefined;
}

/** Emit the latest retained candidate exactly once at an actual terminal;
 * consumes the pending candidate either way. */
export function emitRetainedBenchmarkCandidate(pi: ExtensionAPI): void {
	if (!benchmarkCaptureEnabled()) return;
	const pending = pendingCandidate;
	pendingCandidate = undefined;
	if (pending) appendCandidateOnce(pi, pending.stage, pending.candidate);
}

/** Direct terminal emission (causal final withhold, abandoned raw-answer
 * replacement): exactly one exact candidate per finalization window. */
export function emitBenchmarkCandidate(
	pi: ExtensionAPI,
	stage: CandidateStage,
	candidate: string,
): void {
	if (!benchmarkCaptureEnabled() || !candidate) return;
	pendingCandidate = undefined;
	appendCandidateOnce(pi, stage, candidate);
}

/** Reset pending/emitted capture state; runs inside resetFinalization so no
 * candidate or once-flag from one task poisons the next. */
export function resetBenchmarkCapture(): void {
	pendingCandidate = undefined;
	candidateEmitted = false;
}

/** Benchmark-only why a causal candidate never certified: fixed code only. */
export const STAGE_ENTRY_TYPE = "cortheon-benchmark-causal-stage-v1";
export const CAUSAL_STAGE_REASONS = [
	"deliberation_empty", "validation_failed", "mapping_failed",
	"transport_failed", "runtime_withheld", "terminated_before_deliberation",
] as const;
export type CausalStageReason = (typeof CAUSAL_STAGE_REASONS)[number];

export function emitBenchmarkStageReason(
	pi: ExtensionAPI,
	reason: CausalStageReason,
): void {
	if ((CAUSAL_STAGE_REASONS as readonly string[]).includes(reason)) {
		appendBenchmarkEntry(pi, STAGE_ENTRY_TYPE, {
			stage: "causal_synthesis",
			reason,
		});
	}
}
