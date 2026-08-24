import unittest
from pathlib import Path

from cortheon.slash import split_package_query, split_task_action
from cortheon.slash_commands import command_specs, render_command


class SlashCommandTests(unittest.TestCase):
    def test_task_action_separator(self) -> None:
        task, action = split_task_action("Build a REST API :: Use FastAPI")

        self.assertEqual(task, "Build a REST API")
        self.assertEqual(action, "Use FastAPI")

    def test_task_without_action(self) -> None:
        task, action = split_task_action("How should I build a REST API?")

        self.assertEqual(task, "How should I build a REST API?")
        self.assertIsNone(action)

    def test_package_query_separator(self) -> None:
        package, query = split_package_query("httpx :: AsyncClient.stream")

        self.assertEqual(package, "httpx")
        self.assertEqual(query, "AsyncClient.stream")

    def test_separator_does_not_require_spaces(self) -> None:
        task, action = split_task_action("Build API::Use FastAPI")
        package, query = split_package_query("httpx::AsyncClient.stream")

        self.assertEqual(task, "Build API")
        self.assertEqual(action, "Use FastAPI")
        self.assertEqual(package, "httpx")
        self.assertEqual(query, "AsyncClient.stream")

    def test_rendered_commands_use_shared_runtime(self) -> None:
        rendered = {
            spec.name: render_command(spec, root=Path("/tmp/cortheon")) for spec in command_specs()
        }

        self.assertIn("python3 -m cortheon.slash answer", rendered["cortheon-answer"])
        self.assertIn("python3 -m cortheon.slash decide", rendered["cortheon-decide"])
        self.assertIn("python3 -m cortheon.slash research", rendered["cortheon-research"])
        self.assertIn("python3 -m cortheon.slash api", rendered["cortheon-api"])
        self.assertIn("python3 -m cortheon.slash recommend", rendered["cortheon-recommend"])
        self.assertIn("$ARGUMENTS", rendered["cortheon-answer"])

    def test_default_rendered_command_does_not_embed_install_path(self) -> None:
        rendered = render_command(command_specs()[0])

        self.assertIn("python3 -m cortheon.slash answer", rendered)
        self.assertNotIn("PYTHONPATH=", rendered)


if __name__ == "__main__":
    unittest.main()
