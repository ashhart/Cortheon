import unittest
from contextlib import redirect_stdout
from io import StringIO

from cortheon.demo import run_demo


class DemoShowcaseTests(unittest.TestCase):
    def test_demo_no_verify_reports_uncertified(self) -> None:
        capture = StringIO()
        with redirect_stdout(capture):
            code = run_demo(verify=False)
        output = capture.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("NOT CERTIFIED", output)
        self.assertIn("orient", output)
        self.assertIn("discover", output)
        self.assertIn("connect", output)
        self.assertIn("challenge", output)
        self.assertIn("synthesize", output)
        self.assertIn("verify", output)

    def test_demo_certifies_when_distribution_gate_passes(self) -> None:
        capture = StringIO()
        with redirect_stdout(capture):
            code = run_demo(verify=True)
        output = capture.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("CERTIFIED", output)


if __name__ == "__main__":
    unittest.main()
