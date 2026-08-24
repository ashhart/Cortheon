#!/usr/bin/env python3
"""Codex entry point for the Cortheon lifecycle adapter."""

# Re-exported names preserve the original hook's import and monkeypatch surface.
# ruff: noqa: F401

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

if __package__:
    from .hook_adapter import (
        _host_adapter_argv,
        _host_adapter_environment,
        _host_adapter_output,
        _host_adapter_timeout,
        _run_host_adapter_step,
        _safe_relative_token,
        _safe_test_argv,
    )
    from .hook_config import (
        COMPLETE_STATUS_RE,
        CORTHEON_AUTO_CONTEXT,
        CORTHEON_COMPACT_AUTO_CONTEXT,
        CORTHEON_COMPACT_CONTEXT,
        CORTHEON_CONTEXT,
        CORTHEON_UNAVAILABLE_CONTEXT,
        DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS,
        EXPECTED_RUNTIME_PROTOCOL,
        MAX_HOST_ADAPTER_OUTPUT_CHARS,
        MAX_HOST_ADAPTER_STEPS,
        MAX_INPUT_CHARS,
        RUNTIME_HEALTH_TIMEOUT_SECONDS,
        RUNTIME_START_ATTEMPTS,
        RUNTIME_START_INTERVAL_SECONDS,
        RUNTIME_TIMEOUT_SECONDS,
        SENSITIVE_ENV_RE,
        SUBSTANTIVE_RE,
        _configured_strictness,
        _use_compact_context,
    )
    from .hook_entry import main
    from .hook_events import (
        _post_tool_use,
        _pre_tool_use,
        _session_end,
        _stop,
        _user_prompt_submit,
    )
    from .hook_transport import (
        _bind_facade,
        _contains_certified_completion,
        _ensure_runtime,
        _expected_runtime_identity,
        _facade,
        _identity,
        _is_cortheon_skill_bootstrap,
        _payload,
        _post,
        _runtime_command,
        _runtime_healthy,
        _runtime_url,
        _tool_metadata,
        _tool_output,
        _tool_succeeded,
    )
else:
    # Isolated Python omits the script directory from sys.path. Codex owns this
    # trusted plugin directory, so bind imports to that exact directory rather
    # than relying on the caller's working directory or PYTHONPATH.
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from hook_adapter import (
        _host_adapter_argv,
        _host_adapter_environment,
        _host_adapter_output,
        _host_adapter_timeout,
        _run_host_adapter_step,
        _safe_relative_token,
        _safe_test_argv,
    )
    from hook_config import (
        COMPLETE_STATUS_RE,
        CORTHEON_AUTO_CONTEXT,
        CORTHEON_COMPACT_AUTO_CONTEXT,
        CORTHEON_COMPACT_CONTEXT,
        CORTHEON_CONTEXT,
        CORTHEON_UNAVAILABLE_CONTEXT,
        DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS,
        EXPECTED_RUNTIME_PROTOCOL,
        MAX_HOST_ADAPTER_OUTPUT_CHARS,
        MAX_HOST_ADAPTER_STEPS,
        MAX_INPUT_CHARS,
        RUNTIME_HEALTH_TIMEOUT_SECONDS,
        RUNTIME_START_ATTEMPTS,
        RUNTIME_START_INTERVAL_SECONDS,
        RUNTIME_TIMEOUT_SECONDS,
        SENSITIVE_ENV_RE,
        SUBSTANTIVE_RE,
        _configured_strictness,
        _use_compact_context,
    )
    from hook_entry import main
    from hook_events import (
        _post_tool_use,
        _pre_tool_use,
        _session_end,
        _stop,
        _user_prompt_submit,
    )
    from hook_transport import (
        _bind_facade,
        _contains_certified_completion,
        _ensure_runtime,
        _expected_runtime_identity,
        _facade,
        _identity,
        _is_cortheon_skill_bootstrap,
        _payload,
        _post,
        _runtime_command,
        _runtime_healthy,
        _runtime_url,
        _tool_metadata,
        _tool_output,
        _tool_succeeded,
    )

_bind_facade(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
