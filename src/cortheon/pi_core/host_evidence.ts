import { realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import {
	MAX_HOST_EVIDENCE_CHARACTERS,
	type EvidenceRequest,
	objectValue,
	stringValue,
} from "./protocol.ts";

/** Lexical path normalization only: drop "." segments, redundant
 * separators, and resolve ".." against earlier segments — never resolving
 * or reading anything on disk. Leading ".." segments are kept as-is, and
 * a leading "/" is preserved so rooted and relative paths stay distinct
 * identities (/repo/a never collides with repo/a). */
function normalizeSourcePath(value: string): string {
	const rooted = value.startsWith("/");
	const segments: string[] = [];
	for (const segment of value.split("/")) {
		if (segment === "" || segment === ".") continue;
		if (
			segment === ".." &&
			segments.length > 0 &&
			segments[segments.length - 1] !== ".."
		) {
			segments.pop();
			continue;
		}
		segments.push(segment);
	}
	return (rooted ? "/" : "") + segments.join("/");
}

/** Canonical independent-source identity for the two-source check: host
 * evidence aliases (pi:read:a.txt, pi:grep:a.txt, PI:READ:./a.txt) collapse
 * to the same underlying origin/path, so one document observed by two tools
 * is never two independent sources. The pi:<tool>: prefix matches
 * case-insensitively but the captured origin/path keeps its exact case —
 * A.txt and a.txt stay distinct on a case-sensitive filesystem, and a
 * rooted origin stays distinct from a relative one. A bare pi:<tool> label
 * with no attributable origin proves nothing. Non-Pi sources (URLs,
 * documents) are byte-preserved exactly: no trimming, no case folding, no
 * separator changes. */
export function canonicalEvidenceSource(source: string): string {
	const match = /^pi:[a-z_]+(?::(.*))?$/i.exec(source);
	if (!match) return source;
	return normalizeSourcePath((match[1] || "").trim());
}

export function contentText(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.map((item) => {
			const value = objectValue(item);
			return value?.type === "text" && typeof value.text === "string"
				? value.text
				: "";
		})
		.filter(Boolean)
		.join("\n")
		.trim();
}

export function boundedEvidence(value: string): string {
	const clean = value.replace(/\u001b\[[0-9;]*m/g, "").trim();
	if (clean.length <= MAX_HOST_EVIDENCE_CHARACTERS) return clean;
	const head = Math.floor(MAX_HOST_EVIDENCE_CHARACTERS * 0.7);
	return (
		clean.slice(0, head).trimEnd() +
		"\n[CORTHEON BOUNDED OUTPUT: middle omitted]\n" +
		clean.slice(-(MAX_HOST_EVIDENCE_CHARACTERS - head)).trimStart()
	);
}

export function contentHash(value: string): string {
	return createHash("sha256").update(value, "utf8").digest("hex");
}

export function safeHostArguments(
	tool: string,
	input: Record<string, unknown>,
): Record<string, string> {
	if (tool === "grep") {
		return {
			pattern: String(input.pattern || "").slice(0, 300),
			path: String(input.path || "").slice(0, 500),
		};
	}
	if (tool === "read") {
		return {
			filePath: String(input.path || input.filePath || "").slice(0, 500),
		};
	}
	if (tool === "find") {
		return {
			pattern: String(input.pattern || "").slice(0, 300),
			path: String(input.path || "").slice(0, 500),
		};
	}
	if (tool === "bash") {
		return {
			command: String(input.command || input.cmd || "").slice(0, 1000),
		};
	}
	return {};
}

export function hostObservation(
	tool: string,
	input: Record<string, unknown>,
	output: string,
	isError: boolean,
	request: EvidenceRequest | undefined,
): Record<string, unknown> {
	const logicalTool =
		tool === "bash" && request?.capability === "test"
			? "test"
			: tool === "bash" && request?.capability === "diff"
				? "diff"
				: tool;
	const noMatch =
		tool === "grep" &&
		(!output ||
			/\b(?:no files found|no matches?(?: found)?|0 matches?)\b/i.test(output));
	const outcome = isError
		? "error"
		: logicalTool === "test"
			? "passed"
			: logicalTool === "diff"
				? "changed"
				: tool === "grep"
					? noMatch
						? "no_match"
						: "match"
					: "result";
	const receipt = {
		tool: logicalTool,
		...(logicalTool !== tool ? { executor: tool } : {}),
		outcome,
		args: safeHostArguments(tool, input),
	};
	const sourceScope =
		String(input.path || input.filePath || "").slice(0, 500) || undefined;
	return {
		kind:
			logicalTool === "test"
				? "test"
				: logicalTool === "diff"
					? "diff"
					: ["read", "grep", "find", "ls"].includes(tool)
						? "code"
						: "command",
		content:
			`[CORTHEON_HOST_EVIDENCE] ${JSON.stringify(receipt)}\n` +
			boundedEvidence(output || (isError ? "host tool failed" : "empty result")),
		source: sourceScope ? `pi:${tool}:${sourceScope}` : `pi:${tool}`,
		status:
			isError
				? "failed"
				: ["grep", "test"].includes(logicalTool)
					? "verified"
					: "observed",
	};
}

export function requestedReadPaths(request: EvidenceRequest | undefined): string[] {
	if (request?.capability !== "read_many") return [];
	const paths = request.parameters?.paths;
	return Array.isArray(paths)
		? paths.filter((item): item is string => typeof item === "string")
		: [];
}

export function observationPath(observation: Record<string, unknown>): string | undefined {
	const receipt = observationReceipt(observation);
	return stringValue(objectValue(receipt?.args)?.filePath);
}

export function observationReceipt(
	observation: Record<string, unknown>,
): Record<string, unknown> | undefined {
	const firstLine = String(observation.content || "").split("\n", 1)[0];
	if (!firstLine.startsWith("[CORTHEON_HOST_EVIDENCE] ")) return undefined;
	try {
		return objectValue(
			JSON.parse(firstLine.slice("[CORTHEON_HOST_EVIDENCE] ".length)),
		);
	} catch {
		return undefined;
	}
}

export function patchFromSuccessfulEdit(
	filePath: string,
	input: Record<string, unknown>,
): string {
	const edits = Array.isArray(input.edits)
		? input.edits
				.map((item) => objectValue(item))
				.filter((item): item is Record<string, unknown> => Boolean(item))
		: [];
	const hunks = edits
		.map((edit) => {
			const oldText = stringValue(edit.oldText);
			const newText = stringValue(edit.newText);
			if (oldText === undefined || newText === undefined) return "";
			const removed = oldText
				.split("\n")
				.map((line) => `-${line}`)
				.join("\n");
			const added = newText
				.split("\n")
				.map((line) => `+${line}`)
				.join("\n");
			return `@@\n${removed}\n${added}`;
		})
		.filter(Boolean);
	return hunks.length
		? `--- ${filePath}\n+++ ${filePath}\n${hunks.join("\n")}`
		: "";
}

export async function projectPathAllowed(cwd: string, candidate: string): Promise<boolean> {
	if (!candidate || path.isAbsolute(candidate)) return false;
	try {
		const [root, target] = await Promise.all([
			realpath(cwd),
			realpath(path.resolve(cwd, candidate)),
		]);
		const relative = path.relative(root, target);
		return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
	} catch {
		return false;
	}
}
