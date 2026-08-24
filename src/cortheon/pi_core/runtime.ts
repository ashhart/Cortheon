import { spawn } from "node:child_process";
import {
	DEFAULT_TIMEOUT_MS,
	MAX_RESPONSE_CHARACTERS,
	RUNTIME_START_ATTEMPTS,
	RUNTIME_START_INTERVAL_MS,
	configuredAutoEnable,
	configuredRuntimeToken,
} from "./protocol.ts";

const runtimeURL = process.env.CORTHEON_RUNTIME_URL || "http://127.0.0.1:8743";
const runtimeTimeoutInput = process.env.CORTHEON_RUNTIME_TIMEOUT_MS;
const runtimeCommand = process.env.CORTHEON_RUNTIME_COMMAND || "cortheon";
const pluginDebug = process.env.CORTHEON_PLUGIN_DEBUG === "1";

/** An explicit live-runtime cognitive-policy refusal: a typed marker on the
 * error object, never inferred from HTTP status or message text. Callers must
 * fail closed on it, while validation, correction, transport, protocol, and
 * unavailability errors fail open. */
export interface RuntimePolicyRefusal extends Error {
	readonly policyRefusal: true;
	readonly status: number;
}

export function runtimePolicyRefusal(
	message: string,
	status: number,
): RuntimePolicyRefusal {
	const error = new Error(message) as RuntimePolicyRefusal;
	error.name = "RuntimePolicyRefusal";
	Object.assign(error, { policyRefusal: true as const, status });
	return error;
}

export function isRuntimePolicyRefusal(
	error: unknown,
): error is RuntimePolicyRefusal {
	return (
		Boolean(error) &&
		typeof error === "object" &&
		(error as RuntimePolicyRefusal).policyRefusal === true
	);
}

export function runtimeBase(): string {
	return runtimeURL.replace(/\/+$/, "");
}

export function runtimeToken(): string {
	return configuredRuntimeToken();
}

export function runtimeTimeout(): number {
	const parsed = Number(runtimeTimeoutInput || DEFAULT_TIMEOUT_MS);
	return Number.isFinite(parsed) && parsed > 0
		? Math.min(parsed, 60_000)
		: DEFAULT_TIMEOUT_MS;
}

export function autoEnable(): boolean {
	return ["1", "true", "yes", "on"].includes(
		configuredAutoEnable().trim().toLowerCase(),
	);
}

export function runtimeHeaders(): Record<string, string> {
	const token = runtimeToken();
	return {
		Accept: "application/json",
		...(token ? { Authorization: `Bearer ${token}` } : {}),
	};
}

export async function runtimeHealthy(): Promise<boolean> {
	const controller = new AbortController();
	const timer = setTimeout(
		() => controller.abort(),
		Math.min(runtimeTimeout(), 1_000),
	);
	try {
		const response = await fetch(`${runtimeBase()}/healthz`, {
			headers: runtimeHeaders(),
			signal: controller.signal,
		});
		return response.ok;
	} catch {
		return false;
	} finally {
		clearTimeout(timer);
	}
}

function runtimeLaunchEnvironment(url: URL): NodeJS.ProcessEnv {
	const port = url.port || "8743";
	return {
		...process.env,
		CORTHEON_COGNITIVE_BIND: "127.0.0.1",
		CORTHEON_COGNITIVE_PORT: port,
		...(runtimeToken()
			? { CORTHEON_COGNITIVE_TOKEN: runtimeToken() }
			: {}),
	};
}

export async function ensureRuntime(): Promise<void> {
	if (await runtimeHealthy()) return;
	const url = new URL(runtimeBase());
	if (
		url.protocol !== "http:" ||
		!["127.0.0.1", "localhost", "::1"].includes(url.hostname) ||
		(url.pathname && url.pathname !== "/")
	) {
		throw new Error(
			`Cortheon runtime is unavailable at ${runtimeBase()}; ` +
				"automatic startup is limited to a local HTTP runtime.",
		);
	}
	const executable = runtimeCommand;
	let launchError: Error | undefined;
	const child = spawn(executable, ["serve"], {
		detached: true,
		env: runtimeLaunchEnvironment(url),
		stdio: "ignore",
	});
	child.once("error", (error) => {
		launchError = error;
	});
	child.unref();
	for (let attempt = 0; attempt < RUNTIME_START_ATTEMPTS; attempt += 1) {
		await new Promise((resolve) =>
			setTimeout(resolve, RUNTIME_START_INTERVAL_MS),
		);
		if (await runtimeHealthy()) return;
		if (launchError) break;
	}
	throw new Error(
		launchError
			? `could not start Cortheon runtime: ${launchError.message}`
			: `Cortheon runtime did not become healthy at ${runtimeBase()}`,
	);
}

export function debug(message: string): void {
	if (!pluginDebug) return;
	console.error(`[cortheon] ${message}`);
}

export async function runtimeCall(
	path: string,
	body: Record<string, unknown>,
	signal?: AbortSignal,
): Promise<Record<string, unknown>> {
	const controller = new AbortController();
	const abort = () => controller.abort();
	signal?.addEventListener("abort", abort, { once: true });
	const timer = setTimeout(abort, runtimeTimeout());
	try {
		const headers: Record<string, string> = {
			"Content-Type": "application/json",
			...runtimeHeaders(),
		};
		const response = await fetch(`${runtimeBase()}${path}`, {
			method: "POST",
			headers,
			body: JSON.stringify(body),
			signal: controller.signal,
		});
		const text = await response.text();
		if (text.length > MAX_RESPONSE_CHARACTERS) {
			throw new Error("Cortheon response exceeded the adapter limit");
		}
		let payload: unknown;
		try {
			payload = JSON.parse(text);
		} catch {
			throw new Error(`Cortheon returned invalid JSON (HTTP ${response.status})`);
		}
		if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
			throw new Error("Cortheon response was not an object");
		}
		if (!response.ok) {
			const error = (payload as Record<string, unknown>).error;
			const message =
				typeof error === "string"
					? error
					: `Cortheon rejected the request (HTTP ${response.status})`;
			const errorType = (payload as Record<string, unknown>).error_type;
			if (
				response.status === 422 &&
				errorType === "CognitivePolicyRefusal"
			) {
				throw runtimePolicyRefusal(message, response.status);
			}
			throw new Error(message);
		}
		return payload as Record<string, unknown>;
	} catch (error) {
		if (controller.signal.aborted) {
			throw new Error("Cortheon request was cancelled or timed out");
		}
		throw error;
	} finally {
		clearTimeout(timer);
		signal?.removeEventListener("abort", abort);
	}
}
