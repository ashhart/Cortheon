from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "cortheon"

FORBIDDEN_SIDECAR_PATHS = {
    SOURCE / "agent_core",
    SOURCE / "agent_runtime.py",
    SOURCE / "cortheon.py",
    SOURCE / "cortheon_core",
    SOURCE / "cortheon_llm.py",
    SOURCE / "cortheon_llm_core",
    SOURCE / "deliberation.py",
    SOURCE / "deliberation_core",
    SOURCE / "gateway.py",
    SOURCE / "llm_client.py",
    SOURCE / "proxy.py",
    SOURCE / "proxy_core",
    SOURCE / "source_planner_core" / "json_io.py",
    SOURCE / "source_planner_core" / "model.py",
    SOURCE / "synthesis_llm.py",
    SOURCE / "tool_plugins.py",
}


def test_repository_has_one_product_model_path() -> None:
    present = sorted(
        path.relative_to(ROOT).as_posix() for path in FORBIDDEN_SIDECAR_PATHS if path.exists()
    )
    assert present == []


def test_pi_deliberation_reuses_the_active_host_model() -> None:
    source = (SOURCE / "pi_core" / "repair.ts").read_text(encoding="utf-8")
    assert "const model = context.model;" in source
    assert "complete(\n\t\t\tmodel," in source
    assert "base_url" not in source.casefold()
    assert "sidecar" not in source.casefold()
