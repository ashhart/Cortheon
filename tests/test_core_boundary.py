"""The deployable runtime must not import repository-only discovery code.

The shipped surfaces are the MCP runtime and lean operator CLI. Neither should
pay for repository-only discovery and assessment code. This test pins that
boundary so a future eager import cannot silently couple them.

If this test fails, remove the eager dependency or move it inside the command
that needs it. Do not weaken the boundary.
"""

import unittest

# Repository-only modules that the core must not import transitively.
RESEARCH_STACK = {
    "cortheon.research_plan",
    "cortheon.scholarly",
    "cortheon.clinical_trials",
    "cortheon.synthesis",
    "cortheon.synthesis_llm",
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
        # Importing the CLI module must not pull in repository-only machinery.
        loaded = self._loaded_after("cortheon.cognitive_cli")
        leaked = loaded & RESEARCH_STACK
        self.assertFalse(
            leaked,
            "cortheon.cognitive_cli transitively imports the research stack at "
            f"module scope: {sorted(leaked)}. Remove the eager dependency.",
        )


if __name__ == "__main__":
    unittest.main()
