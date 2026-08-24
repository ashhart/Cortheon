import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { contentText } from "./host_evidence.ts";
import { objectValue, WITHHELD_PREFIX } from "./protocol.ts";
import {
	emitBenchmarkCandidate,
	emitBenchmarkStageReason,
	emitRetainedBenchmarkCandidate,
} from "./candidate_capture.ts";
import {
	answerAlreadyDelivered,
	isAnswerOnly,
	isEnabled,
	markAnswerDelivered,
	markTerminalStatusEmitted,
	peekTerminalDisposition,
	terminalStatusAlreadyEmitted,
} from "./state.ts";
import type { TerminalDisposition } from "./state.ts";

/** Closed channel for the host-visible terminal status: withheld status and
 * bounded host-derived reason only — never candidate/evidence text. */
export const TERMINAL_STATUS_TYPE = "cortheon-terminal-status-v1";
export const TERMINAL_STATUS_VERSION = 1;

export function terminalStatusText(disposition: TerminalDisposition): string {
	return (
		`${WITHHELD_PREFIX}\nThe Cortheon investigation ended ` +
		`without a certified answer because ${disposition.reason}.`
	);
}

/** Emit one bounded host-visible terminal status without another model turn
 * when the window closed with no answerable replacement: sticky state alone
 * is not a visible terminal. Best-effort; the sticky message_end
 * replacement still guards later raw text. */
export function emitTerminalStatusOnce(pi: ExtensionAPI): void {
	const disposition = peekTerminalDisposition();
	if (
		!disposition ||
		!isEnabled() ||
		!isAnswerOnly() ||
		answerAlreadyDelivered() ||
		terminalStatusAlreadyEmitted()
	) {
		return;
	}
	markTerminalStatusEmitted();
	// The window just became terminal: this is the one moment the retained
	// benchmark candidate (if any) may be appended — exactly once, latest
	// retained text. Benchmark-only; inert when capture is disabled.
	emitRetainedBenchmarkCandidate(pi);
	try {
		pi.sendMessage(
			{
				customType: TERMINAL_STATUS_TYPE,
				content: terminalStatusText(disposition),
				display: true,
				details: {
					version: TERMINAL_STATUS_VERSION,
					status: "withheld",
					reason: disposition.reason,
					causal: disposition.causal,
				},
			},
			{ triggerTurn: false },
		);
	} catch {
		// Host visibility is best-effort; never change delivery semantics.
	}
}

/** Stamp the disposition on an answer from an abandoned treatment: without
 * it the no-session return in message_end delivers uncertified text (a
 * false allow). Assistant text is replaced with the terminal status while
 * every non-text block (a toolCall) is retained so Pi can preserve
 * tool-result pairing; the tool boundary blocks it. Sticky until
 * resetFinalization. */
export function terminalDispositionResult<T extends { content: unknown[] }>(
	pi: ExtensionAPI,
	message: T,
):
	| { message: Omit<T, "content"> & { content: Array<{ type: "text"; text: string } | unknown> } }
	| undefined
{
	const disposition = peekTerminalDisposition();
	const candidate = contentText(message.content);
	if (
		!disposition ||
		!isEnabled() ||
		!isAnswerOnly() ||
		!candidate
	) {
		return undefined;
	}
	if (!answerAlreadyDelivered() && disposition.submitted !== false) {
		// Once: the blocked candidate, benchmark-only — but never for a
		// terminal on which nothing was submitted (an observe-path refusal).
		emitBenchmarkCandidate(
			pi,
			disposition.causal ? "causal_synthesis" : "completion",
			candidate,
		);
		if (disposition.causal) {
			emitBenchmarkStageReason(pi, "terminated_before_deliberation");
		}
	}
	markAnswerDelivered();
	return {
		message: {
			...message,
			content: message.content.map((item) =>
				objectValue(item)?.type === "text"
					? { type: "text" as const, text: terminalStatusText(disposition) }
					: item,
			),
		},
	};
}
