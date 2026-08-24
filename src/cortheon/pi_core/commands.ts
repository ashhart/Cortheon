import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ensureRuntime, runtimeHealthy } from "./runtime.ts";
import { abandonActive, isEnabled, resetFinalization, setEnabled } from "./state.ts";

export function registerCortheonCommand(pi: ExtensionAPI): void {
	pi.registerCommand("cortheon", {
		description: "Enable, disable, or inspect Cortheon for this Pi session",
		getArgumentCompletions: (prefix) => {
			const actions = ["enable", "disable", "status"];
			const matches = actions.filter((action) => action.startsWith(prefix.trim()));
			return matches.length
				? matches.map((action) => ({ value: action, label: action }))
				: null;
		},
		handler: async (args, context) => {
			const action = args.trim().toLowerCase() || "status";
			if (action === "enable") {
				try {
					await ensureRuntime();
					setEnabled(true);
					context.ui.notify(
						"Cortheon enabled for this Pi session. Pi remains the model and tool harness.",
						"info",
					);
				} catch (error) {
					setEnabled(false);
					context.ui.notify(
						error instanceof Error ? error.message : String(error),
						"error",
					);
				}
				return;
			}
			if (action === "disable") {
				setEnabled(false);
				// Fail open immediately: a held terminal disposition must
				// not gate the next answer after an explicit disable.
				resetFinalization();
				await abandonActive();
				context.ui.notify("Cortheon disabled for this Pi session.", "warning");
				return;
			}
			if (action === "status") {
				const healthy = await runtimeHealthy();
				context.ui.notify(
					`Cortheon: ${isEnabled() ? "enabled" : "disabled"}; ` +
						`runtime: ${healthy ? "healthy" : "stopped"}; ` +
						"model/tools: owned by Pi.",
					healthy || !isEnabled() ? "info" : "warning",
				);
				return;
			}
			context.ui.notify(
				"Usage: /cortheon enable | disable | status",
				"warning",
			);
		},
	});
}
