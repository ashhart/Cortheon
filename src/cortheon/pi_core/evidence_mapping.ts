import type { EvidenceRecord } from "./evidence_ledger.ts";
import { canonicalEvidenceSource } from "./host_evidence.ts";
import { sharedAnchors } from "./grounding.ts";
import type { SynthesisSections } from "./repair.ts";
import {
	anchorTokens,
	contentTokens,
	isAnchorToken,
	numbersIn,
	STOPWORDS,
} from "./text.ts";

export interface IdentifiedRecord extends EvidenceRecord {
	id: string;
}

/** Strong grounding: >=2 distinct shared semantic anchors, or a content
 * anchor plus a shared number. A lone incidental token is never support. */
function supportsCause(
	sections: SynthesisSections,
	record: IdentifiedRecord,
): boolean {
	const { words, numbers } = sharedAnchors(
		anchorTokens(sections.cause),
		new Set(numbersIn(sections.cause)),
		record,
	);
	return words.length >= 2 || (words.length >= 1 && numbers.length >= 1);
}

/** Coverage fraction of the Cause's anchors one record reflects. */
function causeCoverage(
	sections: SynthesisSections,
	record: IdentifiedRecord,
): number {
	const causeAnchors = [...anchorTokens(sections.cause)];
	if (!causeAnchors.length) return 0;
	const shared = sharedAnchors(new Set(causeAnchors), new Set(), record).words
		.length;
	return shared / causeAnchors.length;
}

/** Duplicate records from one document never masquerade as independent
 * corroboration: host aliases (pi:read:x, pi:grep:x) canonicalize to the
 * underlying origin/path, and a bare pi:bash label without an attributable
 * origin collapses to nothing. */
function normalizedSource(record: IdentifiedRecord): string {
	return canonicalEvidenceSource(record.source);
}

/** Two distinct supporting sources (normalized — never one document twice),
 * or one record covering most Cause anchors alone. */
function causeGrounded(
	sections: SynthesisSections,
	supporting: IdentifiedRecord[],
): boolean {
	const sources = new Set(supporting.map(normalizedSource));
	if (sources.size >= 2) return true;
	return supporting.some(
		(record) => causeCoverage(sections, record) >= 0.6,
	);
}

/** Rival anchors not shared with the Cause; the longest is the competing
 * mechanism's name, needed for a contradiction. */
function rivalNovelAnchors(sections: SynthesisSections): string[] {
	const causeTokens = contentTokens(sections.cause);
	return [...contentTokens(sections.rival.replace(/^instead[,:]\s*/i, ""))]
		.filter(
			(token) =>
				isAnchorToken(token) && !causeTokens.has(token) && !STOPWORDS.has(token),
		)
		.sort((a, b) => b.length - a.length);
}

function rivalMechanismToken(sections: SynthesisSections): string | undefined {
	return rivalNovelAnchors(sections)[0];
}

function escapeRegExp(value: string): string {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The outcome the Cause explains. With no derivable outcome the rival can
 * never be refuted, only uncertain. */
function outcomeHeads(cause: string): string[] {
	const heads = new Set<string>();
	const before =
		/([\p{L}]{2,})\s+(?:occurs?|happens?|persists?|appears?|ensues?|results?)(?:\s+(?:because|since|when|whenever|if|after|while))?/giu;
	const after =
		/(?:leads?\s+to|results?\s+in|explains|causes?|produces?)\s+(?:the\s+|a\s+|an\s+)?([\p{L}]{2,})/giu;
	for (const pattern of [before, after]) {
		for (const match of cause.matchAll(pattern)) {
			const head = match[1].toLowerCase();
			if (isAnchorToken(head) && !STOPWORDS.has(head)) heads.add(head);
		}
	}
	return [...heads];
}

/** Absence predicates: mechanism observed as off/disabled/removed, not named. */
const ABSENCE_PREDICATE =
	"(?:disabled|removed|absent|deleted|turned\\s+off|switched\\s+off|off)";
/** A negation cue voids an affirmed absence. */
const NEGATION_CUE =
	/\b(?:not|never|cannot|can\s*not|no|without|rather\s+than|n't)\b/i;
/** Mechanism and predicate must share one clause; a disabled sibling never counts. */
const CLAUSE_SPLIT =
	/[,;]|\b(?:and|but|while|whereas|however|although|though|yet|so)\b/i;

/** Copular/auxiliary links; any other token breaks the binding and the
 * predicate belongs to something else. */
const ABSENCE_LINK =
	"(?:is|are|was|were|be|been|being|remains?|remained|stays?|stayed|becomes?|became|gets?|got|has|had|have)";

/** Determiners allowed in the absence-first form. */
const ABSENCE_FILLER = "(?:the|a|an|all|entire|whole|its|their|our)";

/** The clause asserts the mechanism itself is absent, directly or via a
 * copular link, with no negation cue. */
function clauseAssertsMechanismAbsent(clause: string, token: string): boolean {
	if (NEGATION_CUE.test(clause)) return false;
	const mech = escapeRegExp(token);
	const follows = new RegExp(
		`\\b${mech}\\b(?:\\s+${ABSENCE_LINK}){0,2}\\s+(?:${ABSENCE_PREDICATE})\\b`,
		"i",
	);
	const precedes = new RegExp(
		`\\b(?:${ABSENCE_PREDICATE})\\b(?:\\s+${ABSENCE_FILLER}){0,2}\\s*\\b${mech}\\b`,
		"i",
	);
	return follows.test(clause) || precedes.test(clause);
}

const OUTCOME_PERSISTENCE =
	/\b(?:persists?|continues?|remains?|still\s+occurs?|unchanged|reproducible)\b/i;

/** Direct contradiction: within one sentence, a clause affirming the rival
 * mechanism itself absent (never a different disabled mechanism, never
 * negated/hedged) while the Cause's outcome persists. Else the Rival stays
 * uncertain, never refuted; mention or a zero-match grep never counts. */
function contradictsRival(
	sections: SynthesisSections,
	record: IdentifiedRecord,
): boolean {
	const token = rivalMechanismToken(sections);
	if (!token) return false;
	const heads = outcomeHeads(sections.cause);
	if (!heads.length) return false;
	const containsOutcomeHead = (sentence: string) => {
		const lowered = sentence.toLowerCase();
		return heads.some(
			(head) =>
				lowered.includes(head) &&
				new RegExp(
					`(?<![\\p{L}\\p{N}_])${escapeRegExp(head)}(?![\\p{L}\\p{N}_])`,
					"iu",
				).test(sentence),
		);
	};
	return record.fact
		.split(/(?<=[.;!?)])\s+/)
		.some(
			(sentence) =>
				sentence
					.split(CLAUSE_SPLIT)
					.some((clause) => clauseAssertsMechanismAbsent(clause, token)) &&
				OUTCOME_PERSISTENCE.test(sentence) &&
				containsOutcomeHead(sentence),
		);
}

/** Mentions a Rival anchor without falsifying it. */
function bearsOnRival(
	sections: SynthesisSections,
	record: IdentifiedRecord,
): boolean {
	const lowered = record.fact.toLowerCase();
	return rivalNovelAnchors(sections).some((token) =>
		/[^\x00-\x7f]/.test(token)
			? lowered.includes(token)
			: new RegExp(`\\b${escapeRegExp(token)}\\b`, "i").test(record.fact),
	);
}

export interface HypothesisEvidence {
	/** "supported" only with strong clean grounding; else "uncertain", no ids. */
	causeStatus: "supported" | "uncertain";
	causeIds: string[];
	/** "refuted" only on a direct counterexample; an open rival is honest
	 * uncertainty, never a manufactured contradiction. */
	rivalStatus: "refuted" | "uncertain";
	/** Contradicting ids when refuted, bearing ids when uncertain. */
	rivalIds: string[];
	/** All clean accepted evidence ids. */
	cleanIds: string[];
}

/** Deterministic binding. Cause is "supported" only with credible aggregate
 * coverage; else "uncertain" with no ids — never a fallback to unrelated ids.
 * Rival is "refuted" only on direct falsification; else "uncertain" with
 * bearing records. Undefined only when no usable runtime id exists. */
export function mapHypothesisEvidence(
	sections: SynthesisSections,
	records: Array<EvidenceRecord & { id?: string }>,
): HypothesisEvidence | undefined {
	const identified: IdentifiedRecord[] = records.filter(
		(record): record is IdentifiedRecord => Boolean(record.id),
	);
	if (!identified.length) return undefined;
	const cleanIds = identified.map((record) => record.id);
	const supporting = identified.filter((record) =>
		supportsCause(sections, record),
	);
	const causeStatus = causeGrounded(sections, supporting)
		? "supported"
		: "uncertain";
	const causeIds = causeStatus === "supported" ? supporting.map((r) => r.id) : [];
	const contradicting = identified.filter((record) =>
		contradictsRival(sections, record),
	);
	if (contradicting.length) {
		return {
			causeStatus,
			causeIds,
			rivalStatus: "refuted",
			rivalIds: contradicting.map((record) => record.id),
			cleanIds,
		};
	}
	const bearing = identified.filter((record) => bearsOnRival(sections, record));
	return {
		causeStatus,
		causeIds,
		rivalStatus: "uncertain",
		rivalIds: bearing.map((record) => record.id),
		cleanIds,
	};
}
