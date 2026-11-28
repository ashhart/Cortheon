"""Compare both public clients and solvers on identical non-streaming model replies."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from model_fixture import model_fixture

root = Path(os.getenv("CORTHEON_SRC", Path.cwd())).resolve()
sys.path.insert(0, str(root))
from flash_forge.client import TensorFoldClient
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import solve_reasoning
from flash_forge.skill_loader import SkillPack


def reply(case, body):
    prompt = body["messages"][-1]["content"]
    seed = body.get("seed")
    if "Extract ONLY" in prompt:
        return '{"final_answer":"7","answer_type":"integer","confidence_notes":""}'
    if body["messages"][0]["content"].startswith("Compare Candidate"):
        return "WINNER: B"
    if seed is None:
        return "A"  # constant-position votes force the deep audit
    if case == "fallback":
        return {101: "xx", 202: "éé"}[seed]
    if seed in (404, 909, 1010):
        return "ANSWER: 7"
    answer = 7
    if case == "recovery" and seed < 808:
        answer = "invalid"
    if case == "audit" and seed == 202:
        answer = 8
    return f"```python\n# seed {seed}\nassert 1 + 1 == 2\nprint('ANSWER: {answer}')\n```"


def canonical_requests(requests):
    return sorted(json.dumps(b, sort_keys=True, ensure_ascii=False) for b in requests)


def stable_events(events):
    # Concurrent completions can arrive in different orders. Compare every event's
    # semantic fields and insist on the same sequence of waves and stage boundaries.
    result = []
    for ev in events:
        ev = {k: v for k, v in ev.items() if k != "elapsed_s"}
        if ev["event"] == "slot_done":
            if not result or not isinstance(result[-1], list) or ev["wave"] != result[-1][0]["wave"]:
                result.append([])
            result[-1].append(ev)
        else:
            result.append(ev)
    for batch in result:
        if isinstance(batch, list):
            batch.sort(key=lambda e: e["slot"])
    return result


async def compare(binary, server, case, n, deep):
    url = f"http://127.0.0.1:{server.server_port}/v1"
    server.reply = lambda body: reply(case, body)
    problem = f"Find the exact integer for test {case}."
    full = case == "recovery"
    events = []
    server.requests = []
    async with TensorFoldClient(url) as client:
        ref = await solve_reasoning(
            client, SkillPack(root / "flash_forge/_skill"), EpisodicMemory(enabled=False), problem,
            num_wave1_streams=n, deep=deep, enable_wave2=full, on_event=events.append,
        )
    requests = canonical_requests(server.requests)
    server.requests = []
    with TemporaryDirectory() as tmp:
        subprocess.run([
            binary, "run", "--bench", "single", "--problem", problem, "--n", str(n),
            "--arms", "cortheon_full" if full else "cortheon_w1", "--endpoints", url, "--no-stream",
            "--cortheon-src", str(root), "--python", sys.executable, "--exec-workers", "2", "--light-workers", "1",
            "--out", f"{tmp}/results.json", "--events", f"{tmp}/events.jsonl", *(["--deep"] if deep else []),
        ], check=True, stdout=subprocess.DEVNULL, timeout=60)
        row, = json.loads(Path(tmp, "results.json").read_text())["rows"]
        rust_events = [json.loads(e["note"]) for e in map(json.loads, Path(tmp, "events.jsonl").read_text().splitlines())
                       if e["ev"] == "progress"]
    actual = canonical_requests(server.requests)
    assert requests == actual, (case, [r for r in requests if r not in actual], [r for r in actual if r not in requests])
    assert stable_events(events) == stable_events(rust_events), (case, events, rust_events)
    assert row["predicted"] == ref.final_answer, (case, row, ref)
    assert row["extra"]["early_aborted"] is False, (case, row)
    assert row["extra"]["recovered_from_disagreement"] == ref.recovered_from_disagreement, case
    assert row["extra"]["final_explanation"] == ref.final_explanation, case


if __name__ == "__main__":
    model = {"id": "mock", "owned_by": "tensorfold", "capabilities": {
        "structures": True, "thinking_budget": True, "thinking_switch": True, "loop_guard": True,
    }}
    with model_fixture(model) as server:
        for case, n, deep in [("agreement", 2, False), ("recovery", 4, True),
                              ("audit", 2, False), ("fallback", 2, False)]:
            asyncio.run(compare(sys.argv[1], server, case, n, deep))
