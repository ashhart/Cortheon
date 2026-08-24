import path from "node:path";
import { CONTINUATION_PREFIX, type TestInvocation } from "./protocol.ts";

export function shouldStartAutomatically(prompt: string): boolean {
	if (prompt.startsWith(CONTINUATION_PREFIX)) return false;
	const normalized = prompt.trim();
	if (normalized.length < 12 || normalized.startsWith("/")) return false;
	return !/^(?:hi|hello|hey|thanks|thank you|cheers|ok|okay|great|nice)[!. ]*$/i.test(
		normalized,
	);
}

export function effortForPrompt(prompt: string): "quick" | "standard" | "deep" {
	if (
		/\b(?:latest|current|research|compare sources?|independent sources?|web)\b/i.test(
			prompt,
		)
	) {
		return "deep";
	}
	if (
		/\b(?:fix|implement|change|edit|write|debug|diagnose|plan|design)\b/i.test(
			prompt,
		)
	) {
		return "standard";
	}
	if (
		/\b(?:read|inspect|find|does|what|which|sum|total)\b/i.test(prompt) &&
		/\b[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|md|txt)\b/i.test(
			prompt,
		)
	) {
		return "quick";
	}
	return "standard";
}

export function requestedTestInvocation(task: string): TestInvocation | undefined {
	const match = task.match(
		/\brun\s+(.+?)(?=\s+after\b|\s+and\s+(?:report|verify|then)\b|$)/i,
	);
	if (!match) return undefined;
	const commandLine = match[1].replace(/^[`'"]|[`'"]$/g, "").trim();
	if (
		!commandLine ||
		commandLine.length > 1_000 ||
		/[\r\n]/.test(commandLine) ||
		!/^[A-Za-z0-9_./:=,@+\- \t]+$/.test(commandLine)
	) {
		return undefined;
	}
	const tokens = commandLine.split(/[ \t]+/).filter(Boolean);
	if (
		tokens.length < 2 ||
		tokens.some((token) => token.includes("..") || path.isAbsolute(token))
	) {
		return undefined;
	}
	const executable = tokens[0];
	const args = tokens.slice(1);
	const name = executable.replace(/^\.\//, "").toLowerCase();
	const python =
		/^python(?:3(?:\.\d+)?)?$/.test(name) &&
		args[0] === "-m" &&
		["pytest", "unittest"].includes(args[1] || "");
	const directPytest = ["pytest", "py.test"].includes(name);
	const nodeTest =
		["npm", "pnpm", "yarn", "bun"].includes(name) &&
		(args[0] === "test" || (args[0] === "run" && args[1] === "test"));
	const compiledTest =
		(name === "cargo" && args[0] === "test") ||
		(name === "go" && args[0] === "test") ||
		(name === "dotnet" && args[0] === "test") ||
		(["mvn", "mvnw", "gradle", "gradlew"].includes(name) &&
			args.some((item) => /\btest\b/i.test(item)));
	if (!python && !directPytest && !nodeTest && !compiledTest) return undefined;
	if (
		args.some(
			(item) =>
				item === "--basetemp" ||
				item.startsWith("--basetemp=") ||
				item === "--rootdir" ||
				item.startsWith("--rootdir="),
		)
	) {
		return undefined;
	}
	return { commandLine, executable, args };
}

export function commandRunsRequiredTest(
	command: string | undefined,
	invocation: TestInvocation,
): boolean {
	if (!command) return false;
	const normalize = (value: string) =>
		value.replace(/\s+/g, " ").replace(/\s+2>&1$/, "").trim();
	const required = normalize(invocation.commandLine);
	return command
		.split(/\s*&&\s*/)
		.map(normalize)
		.some((segment) => segment === required);
}

export function protectedTestPaths(task: string): string[] {
	if (
		!/\b(?:do\s+not|don't|must\s+not|without)\s+(?:chang(?:e|ed|ing)|modif(?:y|ied|ying)|edit(?:ed|ing)?)\s+(?:the\s+)?tests?\b/i.test(
			task,
		)
	) {
		return [];
	}
	return [
		...new Set(
			task.match(
				/\b[A-Za-z0-9_./-]*(?:test[^/\s]*|[^/\s]*_test)\.(?:py|js|jsx|ts|tsx|go|rs|java)\b/gi,
			) || [],
		),
	].slice(0, 12);
}

export function requestedMutationPaths(task: string, protectedPaths: string[]): string[] {
	const protectedSet = new Set(protectedPaths.map((item) => path.normalize(item)));
	const paths =
		task.match(
			/\b[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|md|txt)\b/gi,
		) || [];
	return [
		...new Set(
			paths.filter((item) => {
				const normalized = path.normalize(item);
				const base = path.basename(normalized);
				return (
					!protectedSet.has(normalized) &&
					!/(?:^test[_-]|[_-]test\.|\.spec\.|\.test\.)/i.test(base)
				);
			}),
		),
	].slice(0, 12);
}

export function isAmbiguityGoal(value: string | undefined): boolean {
	return /\b(?:ambiguous|ambiguity|clarif(?:y|ication)|rather than guessing|do not (?:guess|invent)|tie-break|live alternatives)\b/i.test(
		value || "",
	);
}

export function isCausalSynthesisGoal(value: string | undefined): boolean {
	return /\b(?:caus(?:e|al)|diagnos(?:e|is)|explanation|hypotheses|hypothesis|falsif(?:y|ication)|disprov(?:e|ing))\b/i.test(
		value || "",
	);
}
