import sys
import tempfile
import unittest
from pathlib import Path

from cortheon.examples import doctest_to_script, extract_runnable_examples
from cortheon.verifier import run_python_examples

README = """
# examplepkg

Install:

```bash
pip install examplepkg
```

Quickstart:

```python
import examplepkg

client = examplepkg.Client()
print(client.ping())
```

Doctest style:

```python
>>> import examplepkg
>>> examplepkg.add(1, 2)
3
```

Needs a secret, must be rejected:

```python
import examplepkg
client = examplepkg.Client(api_key="YOUR_API_KEY")
```

Placeholder, must be rejected:

```python
import examplepkg
examplepkg.connect("<your host here>")
```

Interactive, must be rejected:

```python
import examplepkg
name = input("name? ")
```

Wrong package, must be rejected:

```python
import otherpkg
otherpkg.run()
```

Broken syntax, must be rejected:

```python
import examplepkg
def broken(:
```
"""


class ExampleExtractionTests(unittest.TestCase):
    def test_extracts_only_runnable_target_examples(self) -> None:
        examples = extract_runnable_examples(README, ["examplepkg"])

        self.assertEqual(len(examples), 2)
        self.assertIn("client = examplepkg.Client()", examples[0])
        # Doctest block converted to a plain script with output lines dropped.
        self.assertIn("examplepkg.add(1, 2)", examples[1])
        self.assertNotIn(">>>", examples[1])
        self.assertNotIn("3\n3", examples[1])

    def test_limit_and_empty_description(self) -> None:
        self.assertEqual(
            extract_runnable_examples(README, ["examplepkg"], limit=1),
            [extract_runnable_examples(README, ["examplepkg"])[0]],
        )
        self.assertEqual(extract_runnable_examples(None, ["examplepkg"]), [])
        self.assertEqual(extract_runnable_examples(README, ["examplepkg"], limit=0), [])

    def test_doctest_to_script_drops_expected_output(self) -> None:
        script = doctest_to_script(">>> value = 1 + 1\n>>> value\n2")

        self.assertEqual(script, "value = 1 + 1\nvalue")


class ExampleExecutionTests(unittest.TestCase):
    def test_runs_examples_with_pass_and_fail_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = run_python_examples(
                sys.executable,
                ["print('ok')", "raise SystemExit(2)"],
                Path(tmp),
                timeout_seconds=30,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].returncode, 0)
        self.assertIn("ok", results[0].stdout_tail)
        self.assertFalse(results[1].ok)
        self.assertEqual(results[1].returncode, 2)

    def test_scrubbed_env_hides_host_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = run_python_examples(
                sys.executable,
                ["import os; print(os.environ.get('HOME', ''))"],
                Path(tmp),
                timeout_seconds=30,
            )

        self.assertTrue(results[0].ok)
        self.assertIn(tmp, results[0].stdout_tail)
        self.assertNotIn(str(Path.home()), results[0].stdout_tail)


if __name__ == "__main__":
    unittest.main()
