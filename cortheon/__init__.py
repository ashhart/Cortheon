"""Cortheon — Unified Consensus & Verification Runtime for Frozen Local Models."""

__version__ = "2.3.0"

from flash_forge.client import CompletionResult, ModelCapabilities, TensorFoldClient
from flash_forge.cockpit import CockpitBus, create_cockpit_router
from flash_forge.code_forge import CodeCandidate, CodeForgeTrace
from flash_forge.dev_agent import DevForgeResult, run_dev_forge
from flash_forge.engine import FlashForge as Cortheon
from flash_forge.engine import ForgeConfig as CortheonConfig
from flash_forge.engine import ForgeResult as CortheonResult
from flash_forge.guard import GuardReport, ProvenBug, format_guard_sarif, install_git_hook, run_guard
from flash_forge.integrations import install_commands, install_mcp
from flash_forge.proxy import create_app
from flash_forge.reason_forge import ReasonForgeTrace
from flash_forge.replica_agent import ReplicaAgent, ReplicaConfig, ReplicaResult
from flash_forge.showdown import run_showdown
from flash_forge.spec_verify import SpecVerifyTrace, solve_spec_verify
from flash_forge.think_forge import ThinkForgeTrace

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
