/** Shared strong-grounding logic; free of runtime imports for direct tests. */

import type { EvidenceRecord } from "./evidence_ledger.ts";
import { anchorTokens, contentTokens, numbersIn } from "./text.ts";

/** Structural stand-in for the synthesis sections grounding inspects: kept
 * local (never imported from repair) so the module graph stays acyclic for
 * the bundle; TypeScript's structural typing accepts SynthesisSections. */
interface GroundingSections {
	cause: string;
	rival: string;
	test: string;
}

/** Distinct shared semantic anchors between one record and a statement's
 * anchors/numbers. */
export function sharedAnchors(
	statementAnchors: Set<string>,
	statementNumbers: Set<string>,
	record: EvidenceRecord,
): { words: string[]; numbers: string[] } {
	const words = [...anchorTokens(record.fact)].filter((token) =>
		statementAnchors.has(token),
	);
	const numbers = numbersIn(record.fact).filter((token) =>
		statementNumbers.has(token),
	);
	return { words, numbers };
}

/** Distinctive anchors (long content words and numbers) of one record. */
export function recordAnchors(record: EvidenceRecord): {
	words: string[];
	numbers: string[];
} {
	return {
		words: [...anchorTokens(record.fact)],
		numbers: numbersIn(record.fact),
	};
}

/** One incidental shared token is not grounding: a record is reflected only
 * by two distinct shared anchors, or a content anchor plus a number. */
export function groundingFailures(
	sections: GroundingSections,
	records: EvidenceRecord[],
): string[] {
	const failures: string[] = [];
	const body = `${sections.cause}\n${sections.rival}\n${sections.test}`;
	const observed = contentTokens(body);
	const observedNumbers = new Set(numbersIn(body));
	for (const record of records) {
		const anchors = recordAnchors(record);
		const sharedWords = anchors.words.filter((token) => observed.has(token));
		const sharedNumbers = anchors.numbers.filter((token) =>
			observedNumbers.has(token),
		);
		if (sharedWords.length < 2 && !(sharedWords.length && sharedNumbers.length)) {
			failures.push(
				`The Cause is not grounded across every accepted source: fewer than ` +
					`two distinctive anchors from [${record.source}] appear in the ` +
					`synthesis.`,
			);
		}
	}
	const missingNumbers = records
		.flatMap((record) => numbersIn(record.fact))
		.filter((token) => !observedNumbers.has(token));
	if (missingNumbers.length) {
		failures.push(
			`Exact load-bearing numbers were dropped from the synthesis: ` +
				`${[...new Set(missingNumbers)].join(", ")}.`,
		);
	}
	return failures;
}
