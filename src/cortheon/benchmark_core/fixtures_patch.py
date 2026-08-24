"""Synthetic held-out patch-task fixtures with hidden test scaffolds."""

from __future__ import annotations

import hashlib
import random

from cortheon.benchmark_core.models import PatchCase


def discover_patch_cases(*, count: int, seed: int) -> list[PatchCase]:
    """Return hidden-grader patch tasks installed only in disposable workspaces."""

    definitions = [
        (
            "calculator",
            (
                (
                    "cortheon_fixture_calculator.py",
                    "def add(left: int, right: int) -> int:\n    return left - right\n",
                ),
                (
                    "test_cortheon_fixture_calculator.py",
                    "from cortheon_fixture_calculator import add\n\n\n"
                    "def test_adds_two_numbers() -> None:\n"
                    "    assert add(2, 3) == 5\n",
                ),
            ),
            "python3 -m pytest -q test_cortheon_fixture_calculator.py",
            (
                "from cortheon_fixture_calculator import add; "
                "assert add(-3, 5) == 2; assert add(0, 0) == 0"
            ),
            (
                "Fix add in cortheon_fixture_calculator.py so "
                "test_cortheon_fixture_calculator.py passes. Do not change the test. "
                "Run python3 -m pytest -q test_cortheon_fixture_calculator.py after "
                "the edit and report the verified result."
            ),
        ),
        (
            "clamp",
            (
                (
                    "cortheon_fixture_bounds.py",
                    "def clamp(value: int, lower: int, upper: int) -> int:\n"
                    "    return min(lower, max(upper, value))\n",
                ),
                (
                    "test_cortheon_fixture_bounds.py",
                    "from cortheon_fixture_bounds import clamp\n\n\n"
                    "def test_clamps_high_value() -> None:\n"
                    "    assert clamp(12, 0, 10) == 10\n\n\n"
                    "def test_clamps_low_value() -> None:\n"
                    "    assert clamp(-3, 0, 10) == 0\n",
                ),
            ),
            "python3 -m pytest -q test_cortheon_fixture_bounds.py",
            (
                "from cortheon_fixture_bounds import clamp; "
                "assert clamp(5, 0, 10) == 5; assert clamp(10, 0, 10) == 10"
            ),
            (
                "Fix clamp in cortheon_fixture_bounds.py so "
                "test_cortheon_fixture_bounds.py passes. Do not change the test. "
                "Run python3 -m pytest -q test_cortheon_fixture_bounds.py after "
                "the edit and report the verified result."
            ),
        ),
        (
            "parity",
            (
                (
                    "cortheon_fixture_parity.py",
                    "def is_even(value: int) -> bool:\n    return value % 2 == 1\n",
                ),
                (
                    "test_cortheon_fixture_parity.py",
                    "from cortheon_fixture_parity import is_even\n\n\n"
                    "def test_even_and_odd_values() -> None:\n"
                    "    assert is_even(4) is True\n"
                    "    assert is_even(3) is False\n",
                ),
            ),
            "python3 -m pytest -q test_cortheon_fixture_parity.py",
            (
                "from cortheon_fixture_parity import is_even; "
                "assert is_even(0) is True; assert is_even(-2) is True; "
                "assert is_even(-3) is False"
            ),
            (
                "Fix is_even in cortheon_fixture_parity.py so "
                "test_cortheon_fixture_parity.py passes. Do not change the test. "
                "Run python3 -m pytest -q test_cortheon_fixture_parity.py after "
                "the edit and report the verified result."
            ),
        ),
        (
            "tax",
            (
                (
                    "cortheon_fixture_tax.py",
                    "def total_with_tax(subtotal: float, rate: float) -> float:\n"
                    "    return subtotal + rate\n",
                ),
                (
                    "test_cortheon_fixture_tax.py",
                    "from cortheon_fixture_tax import total_with_tax\n\n\n"
                    "def test_applies_tax_rate() -> None:\n"
                    "    assert total_with_tax(100.0, 0.2) == 120.0\n",
                ),
            ),
            "python3 -m pytest -q test_cortheon_fixture_tax.py",
            (
                "from cortheon_fixture_tax import total_with_tax; "
                "assert total_with_tax(50.0, 0.1) == 55.0; "
                "assert total_with_tax(80.0, 0.0) == 80.0"
            ),
            (
                "Fix total_with_tax in cortheon_fixture_tax.py so "
                "test_cortheon_fixture_tax.py passes. Do not change the test. "
                "Run python3 -m pytest -q test_cortheon_fixture_tax.py after "
                "the edit and report the verified result."
            ),
        ),
        (
            "discovered_multiplier",
            (
                (
                    "arithmetic_engine.py",
                    "def multiply(left: int, right: int) -> int:\n    return left + right\n",
                ),
                (
                    "test_arithmetic_engine.py",
                    "from arithmetic_engine import multiply\n\n\n"
                    "def test_multiplies_two_numbers() -> None:\n"
                    "    assert multiply(3, 4) == 12\n",
                ),
            ),
            "python3 -m pytest -q",
            (
                "from arithmetic_engine import multiply; "
                "assert multiply(-3, 5) == -15; assert multiply(0, 9) == 0"
            ),
            (
                "An arithmetic component has a multiplication defect. Locate the "
                "implementation and focused test, make the smallest correction without "
                "changing tests. Run python3 -m pytest -q after the edit and report "
                "the verified result."
            ),
        ),
    ]
    if count > len(definitions):
        raise ValueError(f"patch suite has {len(definitions)} held-out cases; requested {count}")
    random.Random(seed ^ 0xFA7C4).shuffle(definitions)
    cases: list[PatchCase] = []
    for name, case_files, command, hidden, prompt in definitions[:count]:
        raw = f"{seed}\0{name}\0{command}".encode()
        cases.append(
            PatchCase(
                case_id="patch_" + hashlib.sha256(raw).hexdigest()[:12],
                files=case_files,
                protected_paths=tuple(
                    path for path, _content in case_files if path.startswith("test_")
                ),
                test_command=command,
                hidden_assertions=hidden,
                prompt=prompt,
            )
        )
    return cases
