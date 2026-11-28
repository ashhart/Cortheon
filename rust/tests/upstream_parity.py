"""Compare the shared AIME controls, or all arms including ReasonForge, on a local HTTP fixture."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from model_fixture import model_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.run_aime_ablation import ALL_ARMS, load_aime2025, run_single_arm
from flash_forge.client import TensorFoldClient
from flash_forge.progress import format_event
from flash_forge.skill_loader import SkillPack


SHARED_ARMS = ("vanilla_b1", "maj4_prose", "maj8_prose", "pal_b1")
SAMPLE_COUNTS = {
    "vanilla_b1": 1, "maj4_prose": 4, "maj8_prose": 8, "pal_b1": 1,
    "cortheon_w1": 4, "cortheon_full": 4,
}


def ballot(body):
    # Three seeds vote for 17. This wins B=4 and loses B=8. PAL verifies 42.
    if "PROTOCOL (pal_" in body["messages"][-1]["content"]:
        return "```python\nassert 6 * 7 == 42\nprint('ANSWER:', 6 * 7)\n```"
    return "ANSWER: 17" if body["seed"] in (101, 202, 303) else "ANSWER: 42"


def request_counts(server):
    return Counter(json.dumps(body, sort_keys=True) for body in server.requests)


async def python_results(url, item, arms):
    skills = SkillPack()
    async with TensorFoldClient(base_url=url) as client:
        return {arm: await run_single_arm(arm, client, skills, item) for arm in arms}


def compare(binary, directory, arms):
    dataset = directory / "aime.json"
    dataset.write_text(json.dumps([{"problem_idx": 1, "problem": "What is six times seven?", "answer": 42}]))
    item, = load_aime2025(dataset)
    with model_fixture({"id": "fixture", "owned_by": "mock"}) as server:
        server.reply = ballot
        url = f"http://127.0.0.1:{server.server_port}/v1"
        reference = asyncio.run(python_results(url, item, arms))
        expected_calls = request_counts(server)
        server.requests.clear()
        subprocess.run(
            [binary, "run", "--bench", "aime2025", "--dataset", dataset,
             "--arms", ",".join(arms), "--n", "8", "--endpoints", url, "--python", sys.executable,
             "--no-stream", "--progress-jsonl", "progress.jsonl"],
            cwd=directory, check=True, timeout=120, stdout=subprocess.DEVNULL,
        )
        actual_calls = request_counts(server)

    rows = json.loads((directory / "rust_results.json").read_text())["rows"]
    assert [r["arm"] for r in rows] == list(arms)
    assert [r["samples"] for r in rows] == [SAMPLE_COUNTS[arm] for arm in arms]
    for row in rows:
        for key in ("id", "expected", "predicted", "correct"):
            assert row[key] == reference[row["arm"]][key], (row["arm"], key, row, reference[row["arm"]])
    assert reference["maj4_prose"]["predicted"] == "17"
    assert reference["maj8_prose"]["predicted"] == "42"
    assert actual_calls == expected_calls, (actual_calls - expected_calls, expected_calls - actual_calls)
    progress = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    for arm in ("cortheon_w1", "cortheon_full"):
        if arm not in arms:
            continue
        events = [ev for ev in progress if ev["arm"] == arm]
        assert events[0]["event"] == "start" and events[-1]["event"] == "done", events
        assert any(ev["event"] == "slot_done" and ev["ok"] for ev in events)
        assert all(format_event(ev) and ev["problem_id"] == item["id"] for ev in events)
    print(f"{len(rows)} arms matched Python predictions and grades, {actual_calls.total()} request payloads matched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--all-arms", action="store_true", help="also compare the upstream ReasonForge strategy")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        compare(args.binary.resolve(), Path(directory), ALL_ARMS if args.all_arms else SHARED_ARMS)
