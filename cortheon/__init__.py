"""Cortheon — Unified Consensus & Verification Runtime for Frozen Local Models."""

__version__ = "2.3.0"

from flash_forge.client import CompletionResult, ModelCapabilities, TensorFoldClient
from flash_forge.code_forge import CodeCandidate, CodeForgeTrace
from flash_forge.dev_agent import DevForgeResult, run_dev_forge
from flash_forge.engine import FlashForge as Cortheon
from flash_forge.engine import ForgeConfig as CortheonConfig
from flash_forge.engine import ForgeResult as CortheonResult
from flash_forge.guard import GuardReport, ProvenBug, format_guard_sarif, install_git_hook, run_guard
from flash_forge.integrations import install_commands, install_mcp
from flash_forge.reason_forge import ReasonForgeTrace
from flash_forge.replica_agent import ReplicaAgent, ReplicaConfig, ReplicaResult
from flash_forge.showdown import run_showdown
from flash_forge.spec_verify import SpecVerifyTrace, solve_spec_verify
from flash_forge.think_forge import ThinkForgeTrace

#: Names that require the optional ``proxy`` extra (FastAPI/uvicorn).
#: Resolved on first access so the CLI, the solvers, and the stdio MCP server
#: import without the web stack -- the point of shipping a drop-in MCP server.
_LAZY_WEB_EXPORTS = {
    "create_app": ("flash_forge.proxy", "create_app"),
    "CockpitBus": ("flash_forge.cockpit", "CockpitBus"),
    "create_cockpit_router": ("flash_forge.cockpit", "create_cockpit_router"),
}


def __getattr__(name: str):
    """Import a web-surface export on first use."""
    target = _LAZY_WEB_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        value = getattr(importlib.import_module(target[0]), target[1])
    except ImportError as exc:
        raise ImportError(
            f"cortheon.{name} needs the proxy extra: pip install 'cortheon[proxy]'"
        ) from exc
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "Cortheon",
    "CortheonConfig",
    "CortheonResult",
    "TensorFoldClient",
    "ModelCapabilities",
    "CompletionResult",
    "CockpitBus",
    "create_cockpit_router",
    "CodeCandidate",
    "CodeForgeTrace",
    "ReasonForgeTrace",
    "ThinkForgeTrace",
    "SpecVerifyTrace",
    "solve_spec_verify",
    "GuardReport",
    "ProvenBug",
    "run_guard",
    "format_guard_sarif",
    "install_git_hook",
    "install_commands",
    "install_mcp",
    "DevForgeResult",
    "run_dev_forge",
    "ReplicaAgent",
    "ReplicaConfig",
    "ReplicaResult",
    "create_app",
    "run_showdown",
]
