import { createHash } from "node:crypto";
import { closeSync, readFileSync } from "node:fs";

const evaluatorControlKeys = [
	"schema_version", "evaluation_profile", "cognitive_token",
	"evaluator_max_steps", "auto_enable", "benchmark_capture_candidate",
	"max_host_tool_calls",
];

function readEvaluatorControl(): { present: boolean; value?: Record<string, unknown> } {
	const descriptorText = process.env.CORTHEON_CONTROL_FD;
	delete process.env.CORTHEON_CONTROL_FD;
	if (descriptorText === undefined) return { present: false };
	if (!/^[1-9]\d{0,6}$/.test(descriptorText)) return { present: true };
	const descriptor = Number(descriptorText);
	let raw: string;
	try {
		raw = readFileSync(descriptor, { encoding: "utf8" });
	} catch {
		return { present: true };
	} finally {
		try { closeSync(descriptor); } catch {}
	}
	if (raw.length > 16_384) return { present: true };
	try {
		const value = JSON.parse(raw) as Record<string, unknown>;
		if (
			!value || typeof value !== "object" || Array.isArray(value) ||
			Object.keys(value).sort().join("\0") !== [...evaluatorControlKeys].sort().join("\0") ||
			value.schema_version !== 1 ||
			!(value.evaluation_profile === null ||
				(typeof value.evaluation_profile === "object" && !Array.isArray(value.evaluation_profile))) ||
			typeof value.cognitive_token !== "string" || value.cognitive_token.length > 4_096 ||
			!(value.evaluator_max_steps === null ||
				(Number.isInteger(value.evaluator_max_steps) && Number(value.evaluator_max_steps) >= 1 && Number(value.evaluator_max_steps) <= 1_024)) ||
			typeof value.auto_enable !== "boolean" ||
			typeof value.benchmark_capture_candidate !== "boolean" ||
			!(Number.isInteger(value.max_host_tool_calls) &&
				Number(value.max_host_tool_calls) >= 1 && Number(value.max_host_tool_calls) <= 64)
		) return { present: true };
		return { present: true, value };
	} catch {
		return { present: true };
	}
}

const evaluatorControl = readEvaluatorControl();
if (evaluatorControl.present && !evaluatorControl.value) {
	throw new Error("invalid evaluator control descriptor or payload");
}
const evaluatorProfileInput = evaluatorControl.present
	? evaluatorControl.value?.evaluation_profile
	: process.env.CORTHEON_EVALUATOR_PROFILE;
const cognitiveTokenInput = evaluatorControl.present
	? String(evaluatorControl.value?.cognitive_token || "")
	: process.env.CORTHEON_COGNITIVE_TOKEN || "";
const evaluatorMaxStepsInput = evaluatorControl.present
	? evaluatorControl.value?.evaluator_max_steps
	: process.env.CORTHEON_EVALUATOR_MAX_STEPS;
const autoEnableInput = evaluatorControl.present
	? evaluatorControl.value?.auto_enable === true
	: process.env.CORTHEON_AUTO_ENABLE === "1";
const benchmarkCaptureInput = evaluatorControl.present
	? evaluatorControl.value?.benchmark_capture_candidate === true
	: process.env.CORTHEON_BENCHMARK_CAPTURE_CANDIDATE === "1";
const maxHostToolCallsInput = evaluatorControl.present
	? evaluatorControl.value?.max_host_tool_calls
	: process.env.CORTHEON_MAX_HOST_TOOL_CALLS;
delete process.env.CORTHEON_EVALUATOR_PROFILE;
delete process.env.CORTHEON_COGNITIVE_TOKEN;
delete process.env.CORTHEON_EVALUATOR_MAX_STEPS;
delete process.env.CORTHEON_AUTO_ENABLE;
delete process.env.CORTHEON_BENCHMARK_CAPTURE_CANDIDATE;
delete process.env.CORTHEON_MAX_HOST_TOOL_CALLS;

export type EvaluationOperator =
	| "retrieval"
	| "verification"
	| "hypothesis_framing"
	| "discriminating_evidence"
	| "contradiction_revision"
	| "cross_source_derivation"
	| "adaptive_stopping";

export interface EvaluationProfile {
	schema_version: 1;
	config: {
		schema_version: 1;
		operators: Record<EvaluationOperator, boolean>;
		intercepts_final: boolean;
		cleanup_before_answer: boolean;
		hard_budgets_enforced: true;
		sticky_terminal_safety: true;
		transport_failure_fails_open: true;
	};
	config_sha256: string;
	implementation_sha256: string;
	nonce: string;
}

export interface AdapterEvaluationProfile extends EvaluationProfile {
	adapter_receipt: {
		schema_version: 1;
		host: "pi";
		control_transport: "fd" | "env";
		config_sha256: string;
		nonce: string;
		operators: Record<EvaluationOperator, boolean>;
	};
}

const evaluationOperatorKeys: EvaluationOperator[] = [
	"retrieval", "verification", "hypothesis_framing",
	"discriminating_evidence", "contradiction_revision",
	"cross_source_derivation", "adaptive_stopping",
];

function canonicalEvaluationConfig(value: unknown): string {
	if (Array.isArray(value)) {
		return `[${value.map(canonicalEvaluationConfig).join(",")}]`;
	}
	if (value && typeof value === "object") {
		return `{${Object.entries(value as Record<string, unknown>)
			.sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
			.map(([key, item]) => `${JSON.stringify(key)}:${canonicalEvaluationConfig(item)}`)
			.join(",")}}`;
	}
	return JSON.stringify(value);
}

export function evaluationProfile(): EvaluationProfile | undefined {
	if (!evaluatorProfileInput) return undefined;
	try {
		const value = (
			typeof evaluatorProfileInput === "string"
				? JSON.parse(evaluatorProfileInput)
				: evaluatorProfileInput
		) as EvaluationProfile;
		const config = value?.config;
		const operators = config?.operators;
		if (
			value.schema_version !== 1 || config?.schema_version !== 1 ||
			!operators ||
			evaluationOperatorKeys.some((key) => typeof operators[key] !== "boolean") ||
			!evaluationOperatorKeys.every((key) => Object.hasOwn(operators, key)) ||
			Object.keys(operators).length !== evaluationOperatorKeys.length ||
			typeof config.intercepts_final !== "boolean" ||
			typeof config.cleanup_before_answer !== "boolean" ||
			config.hard_budgets_enforced !== true ||
			config.sticky_terminal_safety !== true ||
			config.transport_failure_fails_open !== true ||
			!(/^[0-9a-f]{64}$/).test(value.config_sha256) ||
			!(/^[0-9a-f]{64}$/).test(value.implementation_sha256) ||
			!(/^[0-9a-f]{32}$/).test(value.nonce)
		) return undefined;
		const digest = createHash("sha256")
			.update(canonicalEvaluationConfig(config)).digest("hex");
		return digest === value.config_sha256 ? value : undefined;
	} catch {
		return undefined;
	}
}

export function configuredRuntimeToken(): string {
	return cognitiveTokenInput;
}

export function configuredEvaluatorMaxSteps(): string | undefined {
	return evaluatorMaxStepsInput === null || evaluatorMaxStepsInput === undefined
		? undefined
		: String(evaluatorMaxStepsInput);
}

export function configuredAutoEnable(): string {
	return autoEnableInput ? "1" : "";
}

export function configuredBenchmarkCapture(): string | undefined {
	return benchmarkCaptureInput ? "1" : undefined;
}

export function configuredMaxHostToolCalls(): string | undefined {
	if (Number.isInteger(maxHostToolCallsInput)) return String(maxHostToolCallsInput);
	return typeof maxHostToolCallsInput === "string" ? maxHostToolCallsInput : undefined;
}

export function operatorEnabled(operator: EvaluationOperator): boolean {
	return evaluationProfile()?.config.operators[operator] ?? true;
}

export function adapterEvaluationProfile(): AdapterEvaluationProfile | undefined {
	const profile = evaluationProfile();
	if (!profile) return undefined;
	return {
		...profile,
		adapter_receipt: {
			schema_version: 1,
			host: "pi",
			control_transport: evaluatorControl.present ? "fd" : "env",
			config_sha256: profile.config_sha256,
			nonce: profile.nonce,
			operators: { ...profile.config.operators },
		},
	};
}

export const MAX_RESPONSE_CHARACTERS = 1_000_000;
export const DEFAULT_TIMEOUT_MS = 5_000;
export const MAX_HOST_EVIDENCE_CHARACTERS = 1_800;
export const CONTINUATION_PREFIX = "[CORTHEON_CONTINUE]";
export const WITHHELD_PREFIX =
	"[Cortheon withheld: completion was not certified]";
/** The ONE unified automatic-follow-up budget per investigation: a repair
 * continuation and an answer-only continuation draw from the same single
 * allowance and are never additive. */
export const MAX_AUTOMATIC_CONTINUATIONS = 1;
/** Per-investigation ceilings on admitted host tool calls. */
export const MAX_HOST_TOOL_CALLS_ANSWER = 16;
/** Discovery past sufficient causal evidence is bounded; an accepted batch
 * or new request resets it. */
export const MAX_REDUNDANT_DISCOVERY_CALLS = 2;
export const MAX_HOST_TOOL_CALLS_CODE_CHANGE = 48;
/** Any CORTHEON_MAX_HOST_TOOL_CALLS override is clamped to [MIN, 64]. */
export const HOST_TOOL_CALLS_OVERRIDE_CEILING = 64;
export const MIN_HOST_TOOL_CALLS = 8;
export const LEASE_SECONDS = 30;
export const HEARTBEAT_INTERVAL_MS = 10_000;
export const RUNTIME_START_ATTEMPTS = 30;
export const RUNTIME_START_INTERVAL_MS = 100;

export interface EvidenceRequest {
	request_id: string;
	capability?: string;
	query?: string;
	parameters?: Record<string, unknown>;
	hypothesis_id?: string;
}

export interface CognitiveAction {
	type?: string;
	instruction?: string;
	submit_via?: string;
	required_fields?: string[];
	request?: EvidenceRequest;
}

export interface PublicHypothesis {
	hypothesis_id: string;
	statement: string;
	falsification_test: string;
	status: string;
	supporting_evidence: string[];
	contradicting_evidence: string[];
	/** Neutral bearing evidence for uncertain hypotheses. */
	bearing_evidence?: string[];
}

export interface RepairPlan {
	path: string;
	oldText: string;
	newText: string;
	functionName: string;
	examples: number;
}

export interface TestInvocation {
	commandLine: string;
	executable: string;
	args: string[];
}

export interface NumericJoin {
	operands: Array<{
		path: string;
		symbol: string;
		value: number;
	}>;
	total: number;
}

export interface ActiveInvestigation {
	sessionId: string;
	goal?: string;
	deliverable?: string;
	request?: EvidenceRequest;
	nextAction?: CognitiveAction;
	evidenceIds: string[];
	hypotheses: PublicHypothesis[];
	completed: boolean;
	answer?: string;
	/** Tool calls this adapter admitted for host execution (blocked calls are
	 * never counted). Pi's actual executions can never exceed this count. */
	admittedToolCalls: number;
	/** Discovery calls admitted while causal evidence was already sufficient
	 * and no runtime evidence request was pending. */
	redundantDiscoveryCalls: number;
	automaticContinuations: number;
	needsContinuation: boolean;
	pendingReadObservations: Array<Record<string, unknown>>;
	/** True after this investigation reported one unavailable Pi web
	 * capability. A repeated runtime request must terminate, not re-report. */
	webUnavailableReported: boolean;
	evidenceSummary?: string;
	evidenceRecords: Array<{ id?: string; source: string; fact: string }>;
	semanticDerivation?: {
		nodes: string[];
		sources: string[];
	};
	diagnosticDerivation?: {
		answer: string;
		nodes: string[];
		sources: string[];
	};
	planDerivation?: {
		nodes: string[];
		owners: Record<string, string>;
		sources: string[];
	};
	cognition?: {
		stage: string;
		move?: string;
		derivedInsight?: string;
		unresolvedRequirements?: string[];
		decisionRule?: string;
	};
	repairPlan?: RepairPlan;
	testInvocation?: TestInvocation;
	protectedTestPaths: string[];
	mutationTargets: string[];
	mutationEvidence: Record<string, string[]>;
	initialFileHashes: Record<string, string>;
	mutationInFlight: boolean;
}

export const modelContext = `
[CORTHEON_MODEL_CONTEXT_V1]
Cortheon is a lightweight reasoning runtime that gives this local model capabilities
beyond its weights. Use host tools to fetch current evidence, test explanations,
connect facts across sources, and verify work. The host runs tools; the model answers.
Follow Cortheon's current instruction, never invent evidence, and stop when released.
`.trim();

export const protocol = `${modelContext}

Cortheon gates evidence/completion. Never call its lifecycle tools.
Pi owns tools; no files persist.`;

export function objectValue(
	value: unknown,
): Record<string, unknown> | undefined {
	return value && typeof value === "object" && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

export function stringValue(value: unknown): string | undefined {
	return typeof value === "string" && value ? value : undefined;
}
