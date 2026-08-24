"""The deployable runtime must not transitively import the research stack.

The shipped surfaces are the MCP runtime and lean operator CLI. Neither should
pay for—or be coupled to—the repository's scholarly/research/knowledge-pool/
source-planner experiments. This test pins that boundary so a future eager
import cannot silently re-couple them.

If this test fails, remove the eager dependency or move it inside the command
that needs it. Do not weaken the boundary.
"""

import unittest

# Modules that belong to the optional research/knowledge stack. The core
# must not import any of them transitively.
RESEARCH_STACK = {
    "cortheon.research",
    "cortheon.research_plan",
    "cortheon.scholarly",
    "cortheon.source_planner",
    "cortheon.clinical_trials",
    "cortheon.synthesis",
    "cortheon.synthesis_llm",
    "cortheon.knowledge_pool",
    "cortheon.auto_evidence",
    "cortheon.artifacts",
    "cortheon.artifact_assessment",
}


class CoreBoundaryTests(unittest.TestCase):
    def _loaded_after(self, import_path: str) -> set[str]:
        import sys

        before = set(sys.modules)
        __import__(import_path)
        return {m for m in (set(sys.modules) - before) if m.startswith("cortheon")}

    def test_mcp_path_excludes_research_stack(self) -> None:
        loaded = self._loaded_after("cortheon.cognitive_mcp")
        leaked = loaded & RESEARCH_STACK
        self.assertFalse(
            leaked,
            "cortheon.cognitive_mcp transitively imports the research stack: "
            f"{sorted(leaked)}. "
            "Move the offending top-level import inside the function that uses it.",
        )

    def test_cli_help_path_excludes_research_stack(self) -> None:
        # Importing the CLI module (what `cortheon --help` does) must not pull in
        # the repository-only research and knowledge-pool machinery.
        loaded = self._loaded_after("cortheon.cognitive_cli")
        leaked = loaded & RESEARCH_STACK
        self.assertFalse(
            leaked,
            "cortheon.cognitive_cli transitively imports the research stack at "
            f"module scope: {sorted(leaked)}. Remove the eager dependency.",
        )


if __name__ == "__main__":
    unittest.main()
