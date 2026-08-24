import { requestedReadPaths } from "./host_evidence.ts";
import {
	type EvidenceRequest,
	type NumericJoin,
	type RepairPlan,
	MAX_HOST_EVIDENCE_CHARACTERS,
} from "./protocol.ts";

function simpleLiteral(value: string): number | boolean | undefined {
	const text = value.trim();
	if (text === "True") return true;
	if (text === "False") return false;
	if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)) return undefined;
	const parsed = Number(text);
	return Number.isFinite(parsed) ? parsed : undefined;
}

function evaluateSimpleExpression(
	expression: string,
	parameters: string[],
	values: Array<number | boolean>,
): unknown {
	if (
		expression.length > 300 ||
		!/^[A-Za-z0-9_()+\-*/%<>=!,. \t]+$/.test(expression) ||
		/__|=>|\.\s*[A-Za-z_]/.test(expression)
	) {
		return undefined;
	}
	const allowed = new Set([
		...parameters,
		"min",
		"max",
		"abs",
		"True",
		"False",
	]);
	const identifiers =
		expression.match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) || [];
	if (identifiers.some((identifier) => !allowed.has(identifier))) {
		return undefined;
	}
	const translated = expression
		.replace(/\bmin\s*\(/g, "Math.min(")
		.replace(/\bmax\s*\(/g, "Math.max(")
		.replace(/\babs\s*\(/g, "Math.abs(")
		.replace(/\bTrue\b/g, "true")
		.replace(/\bFalse\b/g, "false");
	try {
		const evaluator = Function(
			...parameters,
			`"use strict"; return (${translated});`,
		);
		return evaluator(...values);
	} catch {
		return undefined;
	}
}

function expressionDistance(left: string, right: string): number {
	const length = Math.max(left.length, right.length);
	let distance = Math.abs(left.length - right.length);
	for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
		if (left[index] !== right[index]) distance += 1;
	}
	return distance + length / 10_000;
}

export function deriveSimpleRepairPlan(
	reads: Array<{ path: string; source: string }>,
): RepairPlan | undefined {
	const implementation = reads.find(
		(item) => !/(?:^|\/)(?:test_|[^/]*_test\.)/.test(item.path),
	);
	if (!implementation) return undefined;
	const lines = implementation.source.split("\n");
	let functionName: string | undefined;
	let parameters: string[] | undefined;
	let returnLine: string | undefined;
	let returnExpression: string | undefined;
	for (let index = 0; index < lines.length; index += 1) {
		const definition = lines[index].match(
			/^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)/,
		);
		if (!definition) continue;
		const parsedParameters = definition[2]
			.split(",")
			.map((item) => item.split(/[=:]/, 1)[0].trim())
			.filter((item) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(item));
		for (let bodyIndex = index + 1; bodyIndex < lines.length; bodyIndex += 1) {
			if (/^\s*def\s+/.test(lines[bodyIndex])) break;
			const returned = lines[bodyIndex].match(/^(\s+)return\s+([^#]+?)\s*$/);
			if (!returned) continue;
			functionName = definition[1];
			parameters = parsedParameters;
			returnLine = lines[bodyIndex];
			returnExpression = returned[2].trim();
			break;
		}
		if (returnLine) break;
	}
	if (
		!functionName ||
		!parameters?.length ||
		!returnLine ||
		!returnExpression
	) {
		return undefined;
	}

	const testSource = reads
		.filter((item) => item !== implementation)
		.map((item) => item.source)
		.join("\n");
	const escapedName = functionName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const assertion = new RegExp(
		`\\b${escapedName}\\s*\\(([^()]*)\\)\\s*(?:==|is)\\s*` +
			"([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)|True|False)",
		"g",
	);
	const examples: Array<{
		values: Array<number | boolean>;
		expected: number | boolean;
	}> = [];
	for (const match of testSource.matchAll(assertion)) {
		const values = match[1].split(",").map(simpleLiteral);
		const expected = simpleLiteral(match[2]);
		if (
			values.length === parameters.length &&
			values.every((item) => item !== undefined) &&
			expected !== undefined
		) {
			examples.push({
				values: values as Array<number | boolean>,
				expected,
			});
		}
	}
	if (examples.length === 0) return undefined;

	const candidates = new Set<string>();
	for (const operator of [" + ", " - ", " * ", " / ", " % "]) {
		if (!returnExpression.includes(operator)) continue;
		for (const replacement of [" + ", " - ", " * ", " / ", " % "]) {
			if (replacement !== operator) {
				candidates.add(returnExpression.replace(operator, replacement));
			}
		}
	}
	if (/\b(?:min|max)\s*\(/.test(returnExpression)) {
		candidates.add(
			returnExpression
				.replace(/\bmin\s*\(/g, "__CORTHEON_MAX__(")
				.replace(/\bmax\s*\(/g, "min(")
				.replace(/__CORTHEON_MAX__\(/g, "max("),
		);
	}
	candidates.add(returnExpression.replace(/==\s*0\b/, "== 1"));
	candidates.add(returnExpression.replace(/==\s*1\b/, "== 0"));
	candidates.add(returnExpression.replace(/!=\s*0\b/, "!= 1"));
	candidates.add(returnExpression.replace(/!=\s*1\b/, "!= 0"));
	if (parameters.length === 2) {
		const [left, right] = parameters;
		candidates.add(`${left} + ${right}`);
		candidates.add(`${left} - ${right}`);
		candidates.add(`${left} * ${right}`);
		candidates.add(`${left} + ${left} * ${right}`);
		candidates.add(`${left} - ${left} * ${right}`);
		candidates.add(`${left} * (1 + ${right})`);
		candidates.add(`${left} * (1 - ${right})`);
	}
	candidates.delete(returnExpression);
	candidates.delete("");

	const passing = [...candidates].filter((candidate) =>
		examples.every((example) => {
			const observed = evaluateSimpleExpression(
				candidate,
				parameters,
				example.values,
			);
			return typeof example.expected === "boolean"
				? observed === example.expected
				: typeof observed === "number" &&
						Math.abs(observed - example.expected) <= 1e-9;
		}),
	);
	if (passing.length === 0) return undefined;
	const rootOperator = (expression: string): string | undefined =>
		expression.match(
			new RegExp(`^\\s*${parameters[0]}\\s*([+*/%]|-)`),
		)?.[1];
	const originalRootOperator = rootOperator(returnExpression);
	passing.sort(
		(left, right) =>
			Number(
				Boolean(originalRootOperator) &&
					rootOperator(left) !== originalRootOperator,
			) -
				Number(
					Boolean(originalRootOperator) &&
						rootOperator(right) !== originalRootOperator,
				) ||
			expressionDistance(left, returnExpression) -
				expressionDistance(right, returnExpression),
	);
	return {
		path: implementation.path,
		oldText: returnLine,
		newText: returnLine.replace(returnExpression, passing[0]),
		functionName,
		examples: examples.length,
	};
}

function escapeRegularExpression(value: string): string {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function deriveNumericJoin(
	request: EvidenceRequest,
	reads: Array<{ path: string; source: string }>,
): NumericJoin | undefined {
	if (request.parameters?.operation !== "sum") return undefined;
	const paths = requestedReadPaths(request);
	const symbols = Array.isArray(request.parameters?.symbols)
		? request.parameters.symbols.filter(
				(item): item is string => typeof item === "string" && Boolean(item),
			)
		: [];
	if (paths.length === 0 || paths.length !== symbols.length) return undefined;

	const operands: NumericJoin["operands"] = [];
	for (let index = 0; index < paths.length; index += 1) {
		const requestedPath = paths[index];
		const symbol = symbols[index];
		const snapshot = reads.find(
			(item) => item.path.toLowerCase() === requestedPath.toLowerCase(),
		);
		if (!snapshot) return undefined;
		const matcher = new RegExp(
			`\\b${escapeRegularExpression(symbol)}\\b(?:\\s*:[^=\\n]+)?` +
				"\\s*=\\s*([+-]?(?:\\d(?:_?\\d)*))\\b",
			"g",
		);
		const matches = [...snapshot.source.matchAll(matcher)].map((match) =>
			Number(match[1].replaceAll("_", "")),
		);
		const values = [...new Set(matches.filter(Number.isSafeInteger))];
		if (values.length !== 1) return undefined;
		operands.push({ path: requestedPath, symbol, value: values[0] });
	}
	return {
		operands,
		total: operands.reduce((total, operand) => total + operand.value, 0),
	};
}

export function compactReadFacts(
	reads: Array<{ path: string; source: string }>,
): string {
	if (reads.length === 0) return "";
	const perRead = Math.max(
		160,
		Math.floor((MAX_HOST_EVIDENCE_CHARACTERS - 80) / reads.length),
	);
	return reads
		.map((item) => `[${item.path}] ${item.source.trim().slice(0, perRead)}`)
		.join("\n")
		.slice(0, MAX_HOST_EVIDENCE_CHARACTERS);
}
