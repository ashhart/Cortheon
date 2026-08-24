import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerPiExtension } from "./pi_core/events.ts";

/** Stable Pi entry facade. Pi owns model/tools/permissions/context/mutations;
 * pi_core supplies bounded memory-only cognitive state. */
export default function cortheonPiExtension(pi: ExtensionAPI) {
	registerPiExtension(pi);
}
