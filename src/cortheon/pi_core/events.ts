import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerCortheonCommand } from "./commands.ts";
import { registerSessionEvents } from "./session_events.ts";
import { registerToolEvents } from "./tool_events.ts";

export function registerPiExtension(pi: ExtensionAPI): void {
	registerCortheonCommand(pi);
	registerSessionEvents(pi);
	registerToolEvents(pi);
}
