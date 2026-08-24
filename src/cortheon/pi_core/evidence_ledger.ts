import { MAX_HOST_EVIDENCE_CHARACTERS } from "./protocol.ts";

export interface EvidenceRecord {
	source: string;
	fact: string;
}

/** Deterministic ordering: code-unit comparison, never locale collation. */
function canonicalOrder(records: EvidenceRecord[]): EvidenceRecord[] {
	return [...records].sort((a, b) => {
		if (a.source !== b.source) return a.source < b.source ? -1 : 1;
		return a.fact < b.fact ? -1 : a.fact > b.fact ? 1 : 0;
	});
}

/** One code point pre-escaped as in canonical JSON plus its UTF-16 cost, so
 * no excerpt can end between surrogate halves. */
interface EscapedPoint {
	escaped: string;
	units: number;
}

function escapeCodePoints(value: string): EscapedPoint[] {
	const points: EscapedPoint[] = [];
	for (const char of value) {
		const escaped = JSON.stringify(char).slice(1, -1);
		points.push({ escaped, units: escaped.length });
	}
	return points;
}

/** Build the host-owned Evidence ledger as canonical JSON records inside
 * MAX_HOST_EVIDENCE_CHARACTERS (JS string length). Fact content is
 * JSON-escaped, so evidence text can never inject a second source label.
 * Every accepted source gets its full label plus a nonempty excerpt: the
 * first code point of every fact is reserved, and if even that does not
 * fit, undefined is returned — the adapter must withhold. The rest is
 * whole code points by bounded fair round-robin in index order. */
export function buildEvidenceLedger(
	records: EvidenceRecord[],
): string | undefined {
	if (!records.length) return "";
	const ordered = canonicalOrder(records);
	const facts = ordered.map((record) => record.fact.trim());
	if (ordered.some((record) => !record.source.trim()) || facts.some((fact) => !fact)) {
		return undefined;
	}
	const sources = ordered.map((record) => JSON.stringify(record.source));
	// Per record: `{"source":` (10) + escaped label + `,"fact":"` (10) +
	// escaped excerpt + `"}` (2); plus n-1 commas and the enclosing brackets.
	const fixedCost = sources.map((label) => 22 + label.length);
	const overhead =
		2 + (ordered.length - 1) + fixedCost.reduce((total, cost) => total + cost, 0);
	const factsPoints = facts.map(escapeCodePoints);
	// First code point of every fact: the floor for keeping every accepted
	// source present with a nonempty excerpt.
	const reserved = factsPoints.reduce(
		(total, points) => total + points[0].units,
		0,
	);
	let budget = MAX_HOST_EVIDENCE_CHARACTERS - overhead;
	if (budget < reserved) return undefined;
	budget -= reserved;
	const assigned = factsPoints.map(() => 1);
	// Bounded fair round-robin: one point per still-hungry source per pass,
	// only when its exact escaped cost fits.
	let progressed = true;
	while (progressed) {
		progressed = false;
		for (let index = 0; index < factsPoints.length; index++) {
			const points = factsPoints[index];
			const next = points[assigned[index]];
			if (!next || next.units > budget) continue;
			budget -= next.units;
			assigned[index] += 1;
			progressed = true;
		}
	}
	const entries = sources.map(
		(label, index) =>
			`{"source":${label},"fact":"${factsPoints[index]
				.slice(0, assigned[index])
				.map((point) => point.escaped)
				.join("")}"}`,
	);
	return `[${entries.join(",")}]`;
}
