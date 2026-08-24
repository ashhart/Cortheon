/** Single-flight claiming for request-bound /v1/observe submissions.

 * One runtime evidence request must reach /v1/observe at most once, even
 * when several tool_result handlers for one batch race: the first ready
 * handler synchronously claims the request and submits; a sibling result
 * stays host/model context and is never resubmitted as the same runtime
 * request. A transport failure is ambiguous — the runtime may have
 * committed the observation before the response was lost — so a claim is
 * never released: the same (session, request) is never resubmitted
 * (at-most-once). Only resetObservationClaims drops claims, and it is
 * wired into every normal session/task reset path, so a claim cannot
 * poison a later request or task: each new window starts with a fresh
 * claim table. Deliberately free of local imports: state.ts depends on
 * this module for the reset wiring. */

const claims = new Set<string>();

function claimKey(sessionId: string, requestId: string): string {
	return `${sessionId}\u0000${requestId}`;
}

/** True exactly once per (session, request): only the claimant may submit. */
export function claimObservation(sessionId: string, requestId: string): boolean {
	const claim = claimKey(sessionId, requestId);
	if (claims.has(claim)) return false;
	claims.add(claim);
	return true;
}

/** Drop every claim; wired into the normal task/session reset paths. */
export function resetObservationClaims(): void {
	claims.clear();
}

/** True only while a response still belongs to the exact investigation
 * object that submitted it. Session and request IDs alone are not
 * identity: a later task may legally reuse both strings, which would let
 * a stale old response merge into (and overwrite) the new active object.
 * The exact ActiveInvestigation captured at submit time must be the same
 * object as the current one, AND both IDs must still match, before a
 * response may merge. */
export function observationStillCurrent(
	active: { sessionId: string; request?: { request_id: string } } | undefined,
	submitted: object | undefined,
	sessionId: string,
	requestId: string,
): boolean {
	return Boolean(
		active &&
			active === submitted &&
			active.sessionId === sessionId &&
			active.request?.request_id === requestId,
	);
}
