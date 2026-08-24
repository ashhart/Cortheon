import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from cortheon.cognitive_cli import build_parser, main


class CognitiveCliTests(unittest.TestCase):
    def test_repository_only_demo_is_not_an_installed_command(self) -> None:
        parser = build_parser()
        capture = StringIO()
        with redirect_stderr(capture), self.assertRaisesRegex(SystemExit, "2"):
            parser.parse_args(["demo"])
        self.assertIn("invalid choice: 'demo'", capture.getvalue())
        self.assertNotIn("demo", parser.format_help())

    def test_capabilities_command_returns_zero(self) -> None:
        capture = StringIO()
        with redirect_stdout(capture):
            code = main(["capabilities"])
        self.assertEqual(code, 0)
        self.assertTrue(capture.getvalue().strip())


if __name__ == "__main__":
    unittest.main()
