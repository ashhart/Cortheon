"""Dependency-light MCP server for Cortheon."""

# Re-exports only. Deliberately no ``__all__``: adding one would narrow the
# star-import surface this module has always had, and the pre-split module
# also exposed its own imports as attributes, so they are kept here too.
from __future__ import annotations

import argparse  # noqa: F401
import contextlib  # noqa: F401
import json  # noqa: F401
import sys
from typing import Any, TextIO  # noqa: F401

from cortheon import __version__  # noqa: F401
from cortheon.cognitive_mcp_core.arguments import (  # noqa: F401
    _coerce_json_array,
    _observations_with_host_receipts,
    _optional_object_list,
    _optional_string,
    _optional_string_list,
    _required_object_list,
    _required_string,
    _required_string_list,
)
from cortheon.cognitive_mcp_core.compat import install_facade_patch_bridge
from cortheon.cognitive_mcp_core.protocol import (  # noqa: F401
    HOST_EVIDENCE_PREFIX,
    HOST_RECEIPT_OUTCOMES,
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_PARSE_ERROR,
    MAX_MESSAGE_CHARS,
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOLS,
    _error,
    _result,
    _tool_result,
)
from cortheon.cognitive_mcp_core.server import (  # noqa: F401
    CortheonMcpServer,
    _compact_payload,
)
from cortheon.cognitive_mcp_core.stdio import _write, main, serve  # noqa: F401
from cortheon.cognitive_mcp_core.tools import tool_definitions  # noqa: F401
from cortheon.cognitive_protocol import protocol_capabilities  # noqa: F401
from cortheon.cognitive_runtime import (  # noqa: F401
    CognitiveRuntime,
    CognitiveRuntimeError,
)

# The pre-split module resolved every name it used through its own globals, so
# monkeypatching this module changed what the running code saw. The bridge
# keeps that seam across the split; deleting the installer's own name
# afterwards leaves this module's attribute set exactly as it was.
install_facade_patch_bridge(sys.modules[__name__])
del install_facade_patch_bridge

if __name__ == "__main__":
    main()
