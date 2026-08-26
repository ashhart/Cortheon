"""Operator-lift CLI surface: development bank versus held-out pack."""

from __future__ import annotations

import cortheon.operator_lift.cli as cli


def _run_args(*, heldout: bool):
    return cli.build_parser().parse_args(
        [
            "run",
            "--base-url",
            "http://127.0.0.1:8099/v1",
            "--provider",
            "oMLX",
            "--model-id",
            "model",
            "--output-dir",
            "/tmp/none",
            *(["--heldout"] if heldout else []),
        ]
    )


def test_run_parser_accepts_the_sealed_heldout_flag() -> None:
    args = _run_args(heldout=True)
    assert args.heldout is True


def test_run_parser_defaults_to_the_development_bank() -> None:
    args = _run_args(heldout=False)
    assert getattr(args, "heldout", False) is False
