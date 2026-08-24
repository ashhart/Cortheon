"""Serialized report fixture for block-taxonomy tests."""

from __future__ import annotations

from scaling_support import report as _sealed_scaling_report

WITHHELD = (
    "[Cortheon withheld: completion was not certified]\n"
    "The Cortheon investigation ended without a certified answer because "
    "the evaluator observed an authenticated test terminal."
)


def _serialized_outcome(entry: dict) -> dict:
    delivered = entry.get("delivered", True)
    withheld = entry.get("final_text", WITHHELD) == WITHHELD
    return {
        "schema_version": 1,
        "transport": "pi",
        "terminal_status": "success" if delivered else "withheld" if withheld else "missing",
        "terminal_provenance": (
            "pi_assistant" if delivered else "pi_custom_terminal" if withheld else "none"
        ),
        "finish_reason": "stop" if delivered else "withheld" if withheld else None,
    }


def scaling_report(outcomes: list[dict]) -> dict:
    runs = []
    for index, entry in enumerate(outcomes):
        treatment = entry["condition"] == "cortheon"
        delivered = entry.get("delivered", True)
        runs.append(
            {
                "case_id": f"case_{index}",
                "repeat": 0,
                "condition": entry["condition"],
                "delivered": delivered,
                "correct": entry.get("correct", False),
                "final_text": entry.get("final_text", WITHHELD),
                "timed_out": entry.get("timed_out", False),
                "artifact_correct": entry.get("artifact_correct"),
                "candidate_correct": entry.get("candidate_correct"),
                "process_error": None,
                "expected_verdict": entry.get("expected_verdict", "allow"),
                "failure_owner": (
                    None
                    if delivered or _serialized_outcome(entry)["terminal_status"] == "withheld"
                    else "candidate"
                ),
                "inference_model_id": "demo",
                "evaluator_outcome": _serialized_outcome(entry),
                "substrate_telemetry_valid": treatment,
                "runtime_sessions_completed": int(treatment and delivered),
                "runtime_sessions_evidence_closed": 0,
                "latency_seconds": 1.0,
                "tool_calls": 0,
                "cost_usd": 0.0,
            }
        )
    entries_by_case = {f"case_{index}": entry for index, entry in enumerate(outcomes)}
    by_case = {str(run["case_id"]): run["condition"] for run in runs}
    for case_id, condition in list(by_case.items()):
        source_entry = entries_by_case[case_id]
        counterpart = "baseline" if condition == "cortheon" else "cortheon"
        if any(run["case_id"] == case_id and run["condition"] == counterpart for run in runs):
            continue
        treatment = counterpart == "cortheon"
        runs.append(
            {
                "case_id": case_id,
                "repeat": 0,
                "condition": counterpart,
                "delivered": True,
                "correct": True,
                "final_text": "answer",
                "timed_out": False,
                "artifact_correct": True,
                "candidate_correct": True,
                "process_error": None,
                "expected_verdict": source_entry.get(
                    "counterpart_expected_verdict",
                    source_entry.get("expected_verdict", "allow"),
                ),
                "failure_owner": None,
                "inference_model_id": "demo",
                "evaluator_outcome": _serialized_outcome({"delivered": True}),
                "substrate_telemetry_valid": treatment,
                "runtime_sessions_completed": int(treatment),
                "runtime_sessions_evidence_closed": 0,
                "latency_seconds": 1.0,
                "tool_calls": 0,
                "cost_usd": 0.0,
            }
        )
    return _sealed_scaling_report(runs)
