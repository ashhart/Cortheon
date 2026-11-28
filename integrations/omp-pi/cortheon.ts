// cortheon.ts — Native `/cortheon` slash command + `cortheon_verify` tool for OMP (oh-my-pi) and Pi.
//
// Drop into:
//   - OMP: ~/.omp/profiles/<profile>/agent/extensions/cortheon.ts  (or ~/.omp/agent/extensions/cortheon.ts)
//   - Pi:  ~/.pi/agent/extensions/cortheon.ts
//
// Works out of the box with both TensorFold native (`--cortheon --cortheon-exec`) and `cortheon proxy`.
// Optional env overrides: CORTHEON_URL (e.g. http://127.0.0.1:18651/v1), CORTHEON_MODEL, CORTHEON_API_KEY.

import { Type } from "@sinclair/typebox";

const WIDGET = "cortheon-v2";

interface CortheonResult {
	content: string;
	progress: string[];
	elapsedS: number;
	model: string;
	endpoint: string;
}

function resolveEndpoint(ctx: any, modeSuffix?: string): { url: string; model: string; apiKey: string } {
	const rawBase =
		(process.env.CORTHEON_URL || "").trim() ||
		String(ctx?.model?.baseUrl || ctx?.model?.providerConfig?.baseUrl || "").trim() ||
		"http://127.0.0.1:18651/v1";
	const base = rawBase.replace(/\/+$/, "");
	const url = base.endsWith("/chat/completions") ? base : `${base}/chat/completions`;

	let model =
		(process.env.CORTHEON_MODEL || "").trim() ||
		String(ctx?.model?.id || "").trim() ||
		"cortheon";

	// If the user switched to a "-raw" comparison alias (cortheon=false), route /cortheon to the real model ID
	if (model.endsWith("-raw")) {
		model = model.slice(0, -4);
	}
	if (modeSuffix && (model === "cortheon" || model.startsWith("cortheon:") || model.startsWith("flash-forge"))) {
		const root = model.split(":")[0];
		model = `${root}:${modeSuffix}`;
	}

	const apiKey = (process.env.CORTHEON_API_KEY || "local").trim();
	return { url, model, apiKey };
}

function renderWidget(phase: string, elapsedS: number, lines: string[], model: string): string[] {
	const header = `⚡ Cortheon v2 (${model}) · ${elapsedS.toFixed(1)}s · ${phase}`;
	const tail = lines.slice(-5).map(l => `  ${l}`);
	return [header, ...tail];
}

async function runCortheon(
	prompt: string,
	ctx: any,
	opts: {
		modeSuffix?: string;
		signal?: AbortSignal;
		onProgress?: (phase: string, elapsedS: number, lines: string[], model: string) => void;
	} = {},
): Promise<CortheonResult> {
	const { url, model, apiKey } = resolveEndpoint(ctx, opts.modeSuffix);
	const t0 = Date.now();
	const progress: string[] = [];
	let content = "";
	let reasoningBuf = "";
	let phase = "launching B=4 execution-verified consensus…";

	const tick = setInterval(() => {
		opts.onProgress?.(phase, (Date.now() - t0) / 1000, progress, model);
	}, 250);
	opts.onProgress?.(phase, 0, progress, model);

	try {
		const res = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				...(apiKey && apiKey !== "local" ? { Authorization: `Bearer ${apiKey}` } : {}),
			},
			// Explicitly send NO client `tools` and `"cortheon": true` so TensorFold --cortheon-exec
			// or `cortheon proxy` engages full B=4 parallel consensus (ReasonForge / CodeForge / ThinkForge).
			body: JSON.stringify({
				model,
				stream: true,
				cortheon: true,
				messages: [{ role: "user", content: prompt }],
			}),
			signal: opts.signal,
		});

		if (!res.ok) {
			const errText = await res.text().catch(() => "");
			throw new Error(`HTTP ${res.status} from ${url}: ${errText.slice(0, 240)}`);
		}
		if (!res.body) {
			throw new Error(`Empty response body from ${url}`);
		}

		const reader = res.body.getReader();
		const decoder = new TextDecoder();
		let sseBuf = "";

		while (true) {
			const { value, done } = await reader.read();
			if (done) break;
			sseBuf += decoder.decode(value, { stream: true });
			const parts = sseBuf.split("\n");
			sseBuf = parts.pop() ?? "";

			for (const raw of parts) {
				const line = raw.trim();
				if (!line.startsWith("data:")) continue;
				const payload = line.slice(5).trim();
				if (!payload || payload === "[DONE]") continue;
				let obj: any;
				try {
					obj = JSON.parse(payload);
				} catch {
					continue;
				}
				for (const ch of obj.choices ?? []) {
					const delta = ch?.delta ?? {};
					if (typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0) {
						reasoningBuf += delta.reasoning_content;
						const rLines = reasoningBuf.split("\n");
						reasoningBuf = rLines.pop() ?? "";
						for (const rl of rLines) {
							const clean = rl.trim();
							if (!clean) continue;
							progress.push(clean);
							phase = clean.replace(/^Cortheon:\s*/i, "");
						}
						opts.onProgress?.(phase, (Date.now() - t0) / 1000, progress, model);
					}
					if (typeof delta.content === "string" && delta.content.length > 0) {
						content += delta.content;
						phase = "streaming verified answer…";
					}
				}
			}
		}

		if (reasoningBuf.trim()) {
			progress.push(reasoningBuf.trim());
		}

		const elapsedS = (Date.now() - t0) / 1000;
		opts.onProgress?.("verified ✓", elapsedS, progress, model);
		return {
			content: content.trim(),
			progress,
			elapsedS,
			model,
			endpoint: url,
		};
	} finally {
		clearInterval(tick);
	}
}

export default function cortheonExtension(pi: any): void {
	if (typeof pi.registerCommand === "function") {
		pi.registerCommand("cortheon", {
			description:
				"Run Cortheon v2 B=4 execution-verified consensus: /cortheon <question> | /cortheon build <task> | /cortheon clear",
			getArgumentCompletions(prefix: string) {
				const items = [
					{ value: "build ", label: "build <task>", description: "Verify math/invariants via Cortheon B=4, then have the agent implement & test" },
					{ value: "reason ", label: "reason <problem>", description: "Force ReasonForge B=4 Python/SymPy/Z3 consensus (proxy mode)" },
					{ value: "code ", label: "code <spec>", description: "Force CodeForge v5 Behavioral Differential Consensus (proxy mode)" },
					{ value: "think ", label: "think <design>", description: "Force ThinkForge council + Python quantitative verifier (proxy mode)" },
					{ value: "clear", label: "clear", description: "Clear the Cortheon live progress widget" },
				];
				const p = (prefix || "").toLowerCase();
				return items.filter(i => i.value.toLowerCase().startsWith(p));
			},
			handler: async (args: string, ctx: any) => {
				const raw = (args || "").trim();
				// Read while ctx is active: in print/json mode (hasUI=false) the session ends right after the command,
				// and any later ctx access throws "ctx is stale".
				const hasUI = ctx?.hasUI === true;
				if (!raw) {
					ctx?.ui?.notify?.(
						"Usage: /cortheon <question> | /cortheon build <task> | /cortheon clear",
						"warning",
					);
					return;
				}
				if (raw.toLowerCase() === "clear") {
					ctx?.ui?.setWidget?.(WIDGET, undefined);
					return;
				}

				let triggerBuild = false;
				let modeSuffix: string | undefined;
				let prompt = raw;

				const firstSpace = raw.indexOf(" ");
				const head = (firstSpace === -1 ? raw : raw.slice(0, firstSpace)).toLowerCase();
				const rest = firstSpace === -1 ? "" : raw.slice(firstSpace + 1).trim();

				if (head === "build" && rest) {
					triggerBuild = true;
					prompt = rest;
				} else if ((head === "reason" || head === "code" || head === "think" || head === "deep") && rest) {
					modeSuffix = head;
					prompt = rest;
				}

				const draw = (phase: string, elapsedS: number, lines: string[], model: string) => {
					ctx?.ui?.setWidget?.(WIDGET, renderWidget(phase, elapsedS, lines, model));
				};

				try {
					const res = await runCortheon(prompt, ctx, { modeSuffix, onProgress: draw });
					const traceBlock =
						res.progress.length > 0
							? `**Cortheon Trace (${res.elapsedS.toFixed(1)}s · \`${res.model}\`):**\n${res.progress.map(l => `- ${l}`).join("\n")}\n\n`
							: `**Cortheon v2 (${res.elapsedS.toFixed(1)}s · \`${res.model}\`):**\n\n`;
					const fullMsg = `${traceBlock}${res.content || "(empty response)"}`;

					if (triggerBuild) {
						const handoff =
							`Cortheon v2 verified the specification/invariants below in ${res.elapsedS.toFixed(1)}s.\n\n` +
							`--- CORTHEON VERIFIED GROUND TRUTH ---\n${res.content}\n--- END VERIFIED GROUND TRUTH ---\n\n` +
							`Now implement and thoroughly test the task grounded in the verified result above:\n${prompt}`;
						if (typeof pi.sendUserMessage === "function") {
							pi.sendUserMessage(handoff);
						} else {
							pi.sendMessage({ customType: "cortheon", display: true, content: handoff }, { triggerTurn: true });
						}
					} else if (hasUI) {
						pi.sendMessage({
							customType: "cortheon",
							display: true,
							content: fullMsg,
						});
					} else {
						process.stdout.write(`${fullMsg}\n`);
					}
				} catch (e: unknown) {
					const msg = `Cortheon failed: ${e instanceof Error ? e.message : String(e)}`;
					if (hasUI) ctx?.ui?.notify?.(msg, "error");
					else process.stderr.write(`${msg}\n`);
				} finally {
					if (hasUI) {
						const clear = setTimeout(() => {
							try {
								ctx?.ui?.setWidget?.(WIDGET, undefined);
							} catch {
								// Session was replaced or reloaded within 15s; its widget is already gone.
							}
						}, 15_000);
						clear.unref?.();
					}
				}
			},
		});
	}

	if (typeof pi.registerTool === "function") {
		pi.registerTool({
			name: "cortheon_verify",
			label: "Cortheon Verify (B=4)",
			loadMode: "essential",
			description:
				"Invoke Cortheon v2 B=4 execution-verified parallel consensus (ReasonForge Python/SymPy/Z3, CodeForge v5, or ThinkForge systems/protocol verifier) directly on the local server. Pass a concise, self-contained math, algorithm, or architecture/invariant question and get back the verified answer.",
			parameters: Type.Object({
				prompt: Type.String({
					description: "Concise, self-contained math, algorithm, or protocol/systems-sizing question to verify via B=4 consensus.",
				}),
				mode: Type.Optional(
					Type.String({
						description: "Optional mode hint when using cortheon proxy: reason | code | think | deep.",
					}),
				),
			}),
			async execute(_toolCallId: string, params: any, signal: AbortSignal | undefined, onUpdate: any, ctx: any) {
				const prompt = String(params?.prompt ?? "").trim();
				const modeSuffix = params?.mode ? String(params.mode).trim().toLowerCase() : undefined;
				if (!prompt) {
					return { content: [{ type: "text", text: "Error: prompt must be non-empty." }], isError: true };
				}
				const res = await runCortheon(prompt, ctx, {
					modeSuffix,
					signal,
					onProgress: (phase, elapsedS, lines, model) => {
						onUpdate?.({
							content: [{ type: "text", text: renderWidget(phase, elapsedS, lines, model).join("\n") }],
						});
					},
				});
				const header = `[Cortheon v2 · ${res.elapsedS.toFixed(1)}s · ${res.model}]`;
				const trace = res.progress.length ? `${res.progress.join("\n")}\n\n` : "";
				return {
					content: [{ type: "text", text: `${header}\n${trace}${res.content}` }],
				};
			},
		});
	}
}
