/** Single owner of tokenization, stopwords, and anchor thresholds. */

/** Content words that never distinguish mechanisms. */
export const STOPWORDS: ReadonlySet<string> = new Set(
	[
		"about", "after", "again", "against", "along", "among", "another", "because",
		"been", "before", "being", "below", "between", "both", "cannot", "cause",
		"caused", "causes", "consider", "could", "does", "done", "during", "each",
		"either", "else", "enough", "especially", "evidence", "explains", "fixed",
		"from", "further", "gives", "have", "hence", "here", "itself", "just",
		"least", "less", "makes", "matter", "might", "more", "most", "must",
		"neither", "never", "only", "other", "ought", "over", "predicts", "rather",
		"really", "results", "same", "shall", "should", "since", "some", "still",
		"take", "their", "theirs", "them", "there", "these", "they", "this",
		"those", "through", "thus", "under", "until", "upon", "very", "whereas",
		"which", "while", "whose", "with", "within", "without", "would", "rival",
		"test", "instead", "different", "genuinely", "actually", "alternative",
		"competing", "mechanism", "hypothesis", "explanation", "wrong",
		"first", "second", "third", "every", "sentence", "result",
	],
);

/** Han/kana: no dependency-free word boundaries, so anchors are
 * non-overlapping bigrams. */
const HAN_KANA = /[\u3005\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
const HAN_KANA_RUN = /[\u3005\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/u;
/** Hangul and spaced scripts use Segmenter words directly. */
const CJK_WORD = /[\u3005\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af\u1100-\u11ff]/;
/** Grammar-only Han/kana characters; a bigram containing one straddles a
 * word boundary and is not an anchor. */
const CJK_GRAMMARS = new Set(
	"的了是他它为和与或但在也被把让给从对着呢吧啊之其于而不且".split(""),
);

/** Deterministic word chunks; Segmenter never does CJK dictionary
 * segmentation; regex fallback keeps the module dependency-free. */
function chunkWords(value: string): string[] {
	if (typeof Intl.Segmenter === "function") {
		const segmenter = new Intl.Segmenter("en", { granularity: "word" });
		const chunks: string[] = [];
		for (const part of segmenter.segment(value)) {
			if (part.isWordLike) chunks.push(part.segment);
		}
		return chunks;
	}
	return value.match(/[\p{L}\p{N}][\p{L}\p{N}_-]*/gu) || [];
}

/** Non-overlapping bigrams, dropping grammar-straddling ones; a lone
 * non-grammar trailing character is kept. */
function cjkBigrams(run: string): string[] {
	const tokens: string[] = [];
	for (let index = 0; index + 1 < run.length; index += 2) {
		const token = run.slice(index, index + 2);
		if (![...token].some((char) => CJK_GRAMMARS.has(char))) tokens.push(token);
	}
	if (run.length % 2 === 1 && run.length > 1) {
		const last = run[run.length - 1];
		if (!CJK_GRAMMARS.has(last)) tokens.push(last);
	}
	return tokens;
}

/** Lowercased content tokens: words >= 4 chars, Han/kana bigrams,
 * stopwords and number-only chunks excluded. */
export function contentTokens(value: string): Set<string> {
	const tokens = new Set<string>();
	for (const chunk of chunkWords(value.toLowerCase())) {
		if (!/\p{L}/u.test(chunk)) continue;
		if (HAN_KANA.test(chunk)) {
			for (const run of chunk.match(HAN_KANA_RUN) || [chunk]) {
				for (const token of cjkBigrams(run)) tokens.add(token);
			}
			continue;
		}
		if (chunk.length >= 4 && !STOPWORDS.has(chunk)) tokens.add(chunk);
	}
	return tokens;
}

/** Anchor threshold by script: CJK bigrams at 2 units, words at 5 — one
 * incidental shared token is never support. */
export function isAnchorToken(token: string): boolean {
	return token.length >= (CJK_WORD.test(token) ? 2 : 5);
}

/** Content tokens that qualify as distinctive semantic anchors. */
export function anchorTokens(value: string): Set<string> {
	return new Set([...contentTokens(value)].filter(isAnchorToken));
}

/** Number-like tokens. A trailing separator is punctuation: untrimmed,
 * "minute 00." would never match "minute 00". */
export function numbersIn(value: string): string[] {
	return (value.match(/\d[\w.-]*/g) || []).map((token) =>
		token.replace(/[.\-_]+$/, ""),
	);
}

/** Mirrors the Python sanitizer's high-precision policy for imperative
 * role-override directives (the colon form); ordinary prose stays clean. */
const IMPERATIVE_DIRECTIVE =
	/\b(?:ignore|disregard|override|forget)\s+(?:the\s+|all\s+|any\s+)?(?:system|developer|assistant|safety)\s*:/i;

export function containsImperativeDirective(value: string): boolean {
	return IMPERATIVE_DIRECTIVE.test(value);
}
