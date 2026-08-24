"""Direct tests of the plugin's pure derivation operators.

Every positive case uses surface forms unlike the benchmark fixtures, and
every operator has a negative case proving it does not guess without the
evidence its conclusion depends on.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PLUGIN = Path(__file__).parents[1] / "src" / "cortheon" / "opencode_plugin.js"


def _call(operator: str, *args: object) -> object:
    script = (
        'import { pathToFileURL } from "node:url";\n'
        "const mod = await import(pathToFileURL(process.argv[1]).href);\n"
        f"const out = mod.cortheonOperators.{operator}(...{json.dumps(list(args))});\n"
        "console.log(JSON.stringify(out === undefined ? null : out));"
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(PLUGIN)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip() or "null")


def _reads(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"path": path, "source": text} for path, text in pairs]


def test_unit_mismatch_requires_a_consumer_that_documents_the_unit() -> None:
    code = "delay = retry_delay_minutes * 60\nschedule(delay)\n"
    contract = "schedule(delay) takes minutes between attempts.\n"
    goal = "Diagnose why retries fire an hour apart."
    out = _call(
        "deriveDiagnosticConclusion", _reads(("sched.py", code), ("API.md", contract)), goal
    )
    assert out and "unit mismatch" in out["answer"]
    assert "retry_delay_minutes" in out["answer"]

    # Same rescaling with no documented consumer unit: a legitimate
    # conversion, so the operator must stay silent.
    silent = _call(
        "deriveDiagnosticConclusion",
        _reads(("sched.py", code), ("NOTES.md", "retries are configurable.\n")),
        goal,
    )
    assert silent is None


def test_one_based_index_initialized_to_zero() -> None:
    code = "offset = 0\nwhile True:\n    batch = fetch(offset)\n"
    docs = "Cursor offsets are 1-indexed; offset 0 is rejected.\n"
    goal = "Diagnose why pagination returns nothing."
    out = _call(
        "deriveDiagnosticConclusion", _reads(("client.py", code), ("CURSOR.md", docs)), goal
    )
    assert out and "offset = 0" in out["answer"] and "one-based" in out["answer"]

    fine = _call(
        "deriveDiagnosticConclusion",
        _reads(("client.py", "offset = 1\nbatch = fetch(offset)\n"), ("CURSOR.md", docs)),
        goal,
    )
    assert fine is None


def test_loop_bound_off_by_one_needs_observed_iterations() -> None:
    code = "for (let i = 0; i <= attempts; i++) { send(i) }\n"
    log = "retry #1 sent\nretry #2 sent\nretry #3 sent\nretry #4 sent\n"
    goal = "Diagnose the extra retry."
    out = _call("deriveDiagnosticConclusion", _reads(("send.js", code), ("run.log", log)), goal)
    assert out and "off-by-one" in out["answer"] and "4 retries" in out["answer"]
    assert "<= attempts" in out["answer"] and ";" not in out["answer"].split(" is an off-by-one")[0]

    unobserved = _call(
        "deriveDiagnosticConclusion",
        _reads(("send.js", "for i in range(n + 1): go(i)\n"), ("run.log", "started\n")),
        goal,
    )
    assert unobserved is None


def test_recorded_expected_actual_pair() -> None:
    log = "region expected: eu-west, got: us-east\n"
    goal = "Diagnose the routing failure."
    out = _call(
        "deriveDiagnosticConclusion",
        _reads(("router.py", "route(region)\n"), ("trace.log", log)),
        goal,
    )
    assert out and "expected eu-west but actual us-east" in out["answer"]


def test_shared_key_collision_on_unfamiliar_surface() -> None:
    segments = [
        {"path": "sync.md", "text": "The session cache is keyed only by device id."},
        {
            "path": "households.md",
            "text": "Household members share one device id across their phones.",
        },
        {
            "path": "incident.md",
            "text": "Failures appear only during concurrent syncs; serial syncs are clean.",
        },
    ]
    out = _call(
        "deriveKeyedCollisionInference",
        segments,
        "Explain the sync failures with a falsifiable test.",
    )
    assert out and "same cache key" in out["text"] and "collision" in out["text"]


def test_exact_match_mismatch_on_unfamiliar_surface() -> None:
    segments = [
        {"path": "routing.md", "text": "Notification rules require an exact tenant match."},
        {
            "path": "migration.md",
            "text": "Tenants were migrated to new ids; the rules still name the old tenant ids.",
        },
        {"path": "telemetry.md", "text": "Events continue emitting for the migrated tenants."},
    ]
    out = _call("deriveExactMatchMismatchInference", segments)
    assert out and "tenant mismatch" in out["text"] and "fail to fire" in out["text"]
