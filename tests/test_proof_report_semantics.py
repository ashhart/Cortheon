"""The hardened semantics have to reach the written report, not just the helpers.

These run the benchmark entry point end to end over a stubbed host and read
the artifact it writes: the schema version that says these fields changed
meaning, the gate keys a reader checks, null coverage over an empty block
set, and a treatment arm that never delivered failing the claim outright.
"""

from __future__ import annotations

import json

import pytest
from proof_support import WITHHELD

from cortheon.cognitive_benchmark import EvaluationOutcome, ImportCase, RunResult
from cortheon.cognitive_benchmark import main as benchmark_main

CASES = [
    ImportCase("case_a", "src/a.py", "pathlib", True, "prompt a"),
    ImportCase("case_b", "src/b.py", "json", False, "prompt b"),
]


def _stub_run(**overrides):
    def run(_args, case, *, repeat, treatment, condition):
        values = {
            "final_text": "verified",
            "delivered": True,
            "correct": True,
            "timed_out": False,
            "artifact_correct": None,
        }
        values.update(overrides.get(condition, {}))
        if values["timed_out"]:
            outcome = EvaluationOutcome("pi", "transport_error", "pi_assistant", "timeout")
        elif values["delivered"]:
            outcome = EvaluationOutcome("pi", "success", "pi_assistant", "stop")
        elif values["final_text"] == WITHHELD:
            outcome = EvaluationOutcome("pi", "withheld", "pi_custom_terminal", "withheld")
        else:
            outcome = EvaluationOutcome("pi", "missing", "none", None)
        return RunResult(
            case_id=case.case_id,
            repeat=repeat,
            condition=condition,
            expected=case.expected,
            final_text=values["final_text"],
            delivered=values["delivered"],
            correct=values["correct"],
            latency_seconds=0.1,
            tokens=10,
            tool_calls=1,
            tool_errors=0,
            timed_out=values["timed_out"],
            process_error=None,
            expected_verdict="allow",
            failure_owner=(
                None if values["delivered"] or values["final_text"] == WITHHELD else "candidate"
            ),
            evaluator_outcome=outcome,
            inference_model_id="small-model",
            artifact_correct=values["artifact_correct"],
            substrate_telemetry_valid=True if treatment else None,
            runtime_sessions_started=1 if treatment else 0,
            runtime_observations_accepted=1 if treatment else 0,
            runtime_sessions_completed=1 if treatment else 0,
        )

    return run


@pytest.fixture
def written_report(monkeypatch, tmp_path):
    def write(**overrides):
        monkeypatch.setattr(
            "cortheon.cognitive_benchmark._runtime_health",
            lambda _url: {"ok": True, "storage": "memory_only"},
        )
        monkeypatch.setattr(
            "cortheon.cognitive_benchmark._model_endpoint_health",
            lambda *_args, **_kwargs: {
                "ok": True,
                "model_id": "small-model",
                "inference_probe_ok": True,
            },
        )
        monkeypatch.setattr(
            "cortheon.cognitive_benchmark.discover_benchmark_cases",
            lambda *_args, **_kwargs: CASES,
        )
        monkeypatch.setattr("cortheon.cognitive_benchmark.run_job", _stub_run(**overrides))
        output = tmp_path / "report.json"
        exit_code = benchmark_main(
            [
                "--repository",
                str(tmp_path),
                "--cases",
                "2",
                "--repeats",
                "1",
                "--suite",
                "imports",
                "--model-id",
                "small-model",
                "--output",
                str(output),
            ]
        )
        return exit_code, output.read_text(encoding="utf-8")

    return write


def test_report_declares_the_schema_the_new_meanings_belong_to(written_report):
    _exit_code, raw = written_report()

    report = json.loads(raw)

    # Schema 9 carries these same field names under the older, weaker
    # meanings, so the version is what tells a reader which rules applied.
    assert report["schema_version"] == 14


def test_report_carries_the_new_gates_and_null_coverage(written_report):
    _exit_code, raw = written_report()

    report = json.loads(raw)
    gates = report["proof_gates"]

    assert gates["substrate_execution_observed"] is True
    assert gates["substrate_completed_work"] is True
    assert gates["verified_completion_floor"] is True
    assert gates["cortheon_runs_delivered_or_blocked"] is True
    # Nothing was blocked in either arm, so neither reports a coverage rate.
    assert report["cortheon"]["block_classification_coverage"] is None
    assert report["baseline"]["block_classification_coverage"] is None
    assert '"block_classification_coverage": null' in raw
    assert report["cortheon"]["delivery_failures"] == 0
    assert report["cortheon"]["substrate_completed_work_runs"] == 2
    assert report["baseline"]["substrate_completed_work"] is None


def test_a_treatment_arm_that_never_delivered_fails_the_written_claim(written_report):
    exit_code, raw = written_report(
        cortheon={
            "final_text": "",
            "delivered": False,
            "correct": False,
            "timed_out": True,
            # A wrong artifact on a timed-out run: the exact shape that used
            # to be reported as a safe block.
            "artifact_correct": False,
        },
        baseline={"correct": False},
    )

    report = json.loads(raw)
    cortheon = report["cortheon"]

    assert exit_code == 0
    assert cortheon["safe_blocks"] == 0
    assert cortheon["delivery_failures"] == 2
    assert cortheon["block_classification_coverage"] is None
    assert report["proof_gates"]["cortheon_runs_delivered_or_blocked"] is False
    assert report["substrate_amplification_proven"] is False
    assert report["qualification_valid"] is False


def test_expected_allow_withhold_is_false_even_for_a_wrong_artifact(written_report):
    _exit_code, raw = written_report(
        cortheon={
            "final_text": WITHHELD,
            "delivered": False,
            "correct": False,
            "artifact_correct": False,
        },
        baseline={"correct": False},
    )

    report = json.loads(raw)
    cortheon = report["cortheon"]

    assert cortheon["safe_blocks"] == 0
    assert cortheon["false_blocks"] == 2
    assert cortheon["delivery_failures"] == 0
    assert cortheon["block_classification_coverage"] == 1.0
    assert report["proof_gates"]["cortheon_runs_delivered_or_blocked"] is True
    # Safety gates hold and the amplification claim still does not.
    assert report["proof_gates"]["zero_cortheon_false_allows"] is True
    assert report["proof_gates"]["accuracy_lift"] is False
    assert report["substrate_amplification_proven"] is False
