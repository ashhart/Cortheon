import { createReadTool, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { completionHypotheses } from "./actions.ts";
import { boundedEvidence, contentHash, contentText } from "./host_evidence.ts";
import { mergePayload } from "./merge.ts";
import { runtimeCall } from "./runtime.ts";
import { getActive, isMultiMutation } from "./state.ts";

export async function certifyAutomaticPatch(
	pi: ExtensionAPI,
	context: ExtensionContext,
	filePath: string,
	patch: string,
): Promise<{ passed: boolean; summary: string }> {
	const active = getActive();
	if (
		!active ||
		active.completed ||
		active.mutationInFlight ||
		active.deliverable !== "code_change" ||
		!active.testInvocation ||
		!patch.trim()
	) {
		return { passed: false, summary: "Patch certification was not applicable." };
	}
	active.mutationInFlight = true;
	const invocation = active.testInvocation;
	try {
		const result = await pi.exec(invocation.executable, invocation.args, {
			cwd: context.cwd,
			timeout: 60_000,
			signal: context.signal,
		});
		const output = boundedEvidence(
			[result.stdout, result.stderr].filter(Boolean).join("\n") ||
				`Process exited ${result.code}.`,
		);
		if (result.code !== 0 || result.killed) {
			return {
				passed: false,
				summary: output || `Required test exited ${result.code}.`,
			};
		}
		const diffReceipt = {
			tool: "diff",
			executor: "edit",
			outcome: "changed",
			args: { path: filePath },
		};
		const testReceipt = {
			tool: "test",
			executor: "pi.exec",
			outcome: "passed",
			args: { command: invocation.commandLine },
		};
		const observed = await runtimeCall(
			"/v1/observe",
			{
				session_id: active.sessionId,
				observations: [
					{
						kind: "diff",
						content:
							`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(diffReceipt)}\n` +
							boundedEvidence(patch),
						source: `pi:edit:${filePath}`,
						status: "verified",
					},
					{
						kind: "test",
						content:
							`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(testReceipt)}\n` +
							`Command: ${invocation.commandLine}\nExit: 0\n${output}`,
						source: "pi:exec:test",
						status: "verified",
					},
				],
			},
			context.signal,
		);
		mergePayload(observed);
		if (!getActive()) {
			return { passed: false, summary: "Cortheon session disappeared." };
		}
		const evidenceIds = [...getActive()!.evidenceIds];
		const answer =
			`Updated ${filePath}. Host verification passed: ` +
			`${invocation.commandLine}.`;
		const completed = await runtimeCall(
			"/v1/complete",
			{
				session_id: active.sessionId,
				answer,
				claims: [
					{
						claim:
							`${filePath} contains the captured patch and the required ` +
							"host test passed.",
						evidence_ids: evidenceIds,
					},
				],
				hypotheses: completionHypotheses(getActive()!, answer),
				completion_evidence_ids: evidenceIds,
			},
			context.signal,
		);
		mergePayload(completed);
		return getActive()?.completed
			? { passed: true, summary: output }
			: {
					passed: false,
					summary:
						"Host test passed, but Cortheon rejected the diff or evidence binding.",
				};
	} finally {
		const active = getActive();
		if (active) active.mutationInFlight = false;
	}
}

export async function certifyMultiMutationTest(
	context: ExtensionContext,
	output: string,
): Promise<{ completed: boolean; summary: string }> {
	const active = getActive();
	if (!active || active.completed || !active.testInvocation || !isMultiMutation()) {
		return { completed: false, summary: "Multi-mutation certification was not applicable." };
	}
	const invocation = active.testInvocation;
	const readTool = createReadTool(context.cwd);
	const paths = [...active.mutationTargets, ...active.protectedTestPaths];
	const finalHashes: Record<string, string> = {};
	for (const filePath of paths) {
		const result = await readTool.execute(
			`cortheon-final-${filePath}`,
			{ path: filePath },
			context.signal,
		);
		finalHashes[filePath] = contentHash(contentText(result.content));
	}
	const refreshed = getActive();
	if (!refreshed || refreshed.completed || !refreshed.testInvocation) {
		return {
			completed: false,
			summary: "Cortheon session changed during final-state verification.",
		};
	}
	const missingInitial = paths.filter((item) => !refreshed.initialFileHashes[item]);
	if (missingInitial.length) {
		return {
			completed: false,
			summary:
				`Final-state verification lacks initial hashes for: ` +
				`${missingInitial.join(", ")}.`,
		};
	}
	const unchangedTargets = refreshed.mutationTargets.filter(
		(item) => refreshed.initialFileHashes[item] === finalHashes[item],
	);
	if (unchangedTargets.length) {
		return {
			completed: false,
			summary:
				`Required deliverables remain unchanged: ${unchangedTargets.join(", ")}.`,
		};
	}
	const changedProtected = refreshed.protectedTestPaths.filter(
		(item) => refreshed.initialFileHashes[item] !== finalHashes[item],
	);
	if (changedProtected.length) {
		return {
			completed: false,
			summary: `Protected tests changed: ${changedProtected.join(", ")}.`,
		};
	}
	const mutationObservations = refreshed.mutationTargets.map((target) => {
		const receipt = {
			tool: "diff",
			executor: "pi:final-state-hash",
			outcome: "changed",
			args: { path: target },
		};
		return {
			kind: "diff",
			content:
				`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(receipt)}\n` +
				`Verified content change for ${target}. ` +
				`Before SHA-256: ${refreshed.initialFileHashes[target]}. ` +
				`After SHA-256: ${finalHashes[target]}.`,
			source: `pi:final-state:${target}`,
			status: "verified",
		};
	});
	const testReceipt = {
		tool: "test",
		executor: "pi:bash",
		outcome: "passed",
		args: { command: invocation.commandLine },
	};
	const observed = await runtimeCall(
		"/v1/observe",
		{
			session_id: refreshed.sessionId,
			observations: [
				...mutationObservations,
				{
					kind: "test",
					content:
						`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(testReceipt)}\n` +
						`Command: ${invocation.commandLine}\nExit: 0\n${boundedEvidence(output)}`,
					source: "pi:exec:final-test",
					status: "verified",
				},
			],
		},
		context.signal,
	);
	const accepted = Array.isArray(observed.accepted_evidence_ids)
		? observed.accepted_evidence_ids.filter(
				(item): item is string => typeof item === "string",
			)
		: [];
	const mutationEvidence = Object.fromEntries(
		refreshed.mutationTargets.map((target, index) => [
			target,
			accepted[index] ? [accepted[index]] : [],
		]),
	);
	const testEvidence = accepted.slice(refreshed.mutationTargets.length);
	mergePayload(observed);
	const certified = getActive();
	if (!certified) {
		return { completed: false, summary: "Cortheon session disappeared." };
	}
	certified.mutationEvidence = mutationEvidence;
	const missingEvidence = certified.mutationTargets.filter(
		(target) => !(mutationEvidence[target] || []).length,
	);
	if (missingEvidence.length || !testEvidence.length) {
		return {
			completed: false,
			summary:
				`Final-state evidence was not accepted for: ` +
				`${[...missingEvidence, ...(!testEvidence.length ? ["final test"] : [])].join(", ")}.`,
		};
	}
	const evidenceIds = [...certified.evidenceIds];
	const protectedSummary = certified.protectedTestPaths.length
		? ` Protected tests remained unchanged: ${certified.protectedTestPaths.join(", ")}.`
		: "";
	const answer =
		`Updated ${certified.mutationTargets.join(", ")}. ` +
		`Host verification passed: ${invocation.commandLine}.` +
		protectedSummary;
	const claims = certified.mutationTargets.map((target) => ({
		claim: `Applied the requested change to ${target}.`,
		evidence_ids: mutationEvidence[target] || [],
	}));
	claims.push({
		claim:
			`The final host test passed after all requested mutations.` +
			protectedSummary,
		evidence_ids: testEvidence,
	});
	const completed = await runtimeCall(
		"/v1/complete",
		{
			session_id: certified.sessionId,
			answer,
			claims,
				hypotheses: completionHypotheses(certified, answer),
			completion_evidence_ids: evidenceIds,
		},
		context.signal,
	);
	mergePayload(completed);
	return getActive()?.completed
		? { completed: true, summary: answer }
		: {
				completed: false,
				summary:
					"Host test passed, but Cortheon still requires one bounded completion step.",
			};
}

/** The one bounded host-visible notice for a single-file automatic-patch
 * certification (pass, fail, or certification error). */
export async function certifyPatchNotice(
	pi: ExtensionAPI,
	context: ExtensionContext,
	filePath: string,
	patch: string,
): Promise<string> {
	try {
		const result = await certifyAutomaticPatch(pi, context, filePath, patch);
		return result.passed
			? "\n[Cortheon captured the diff, ran the required host test, and certified completion.]"
			: `\n[Cortheon patch verification did not pass: ${result.summary}]`;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		return `\n[Cortheon patch certification error: ${message}]`;
	}
}

/** The one bounded host-visible notice for the multi-mutation final-state
 * certification (pass, incomplete, or certification error). */
export async function certifyFinalTestNotice(
	context: ExtensionContext,
	output: string,
): Promise<string> {
	try {
		const result = await certifyMultiMutationTest(context, output);
		return result.completed
			? "\n[Cortheon captured every mutation and certified the final test.]"
			: `\n[Cortheon final-state check: ${result.summary}]`;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		return `\n[Cortheon final-state certification error: ${message}]`;
	}
}
