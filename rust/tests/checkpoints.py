"""Exercise publication of results and diagnostic traces through the CLI."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from textwrap import dedent

from model_fixture import model_fixture


def check(binary, directory):
    dataset = directory / "questions.json"
    dataset.write_text(json.dumps([
        {"problem_idx": 1, "problem": "First question", "answer": 7},
        {"problem_idx": 2, "problem": "Second question", "answer": 7},
    ]))
    # Inject one infrastructure error while keeping the normal worker protocol.
    worker = directory / "injected_worker.py"
    worker.write_text(dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "verifier")!r})
        import worker
        original = worker.handle
        def handle(req):
            if req.get("op") == "aime_prose" and req.get("content") == "fail this worker job":
                raise RuntimeError("intentional worker failure")
            return original(req)
        worker.handle = handle
        worker.main()
    """))
    results = directory / "results.json"
    events = directory / "events.jsonl"
    with model_fixture({"id": "mock", "owned_by": "mock"}) as server:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        args = [binary, "run", "--dataset", dataset, "--endpoints", url, "--python", sys.executable,
                "--worker-script", worker, "--no-stream", "--n", "1", "--light-workers", "1",
                "--out", results, "--events", events]
        checkpoints = []
        def reply(body):
            if "Second question" in body["messages"][-1]["content"]:
                checkpoints.append(results.read_bytes() if results.exists() else None)
                return "fail this worker job"
            return "ANSWER: 7"
        server.reply = reply
        failed = subprocess.run([*args, "--arms", "vanilla_b1"], capture_output=True, text=True, timeout=45)
        assert failed.returncode != 0 and "intentional worker failure" in failed.stderr, failed.stderr
        assert checkpoints == [results.read_bytes()]
        partial = json.loads(results.read_text())
        assert partial["complete"] is False and partial["summary"]["vanilla_b1"]["scored"] == 1
        assert [(r["id"], r["correct"]) for r in partial["rows"]] == [("AIME25_01", True)]
        trace = [json.loads(line) for line in events.read_text().splitlines()]
        assert [(ev["problem"], ev["ev"]) for ev in trace if ev["ev"] in {"problem_end", "problem_error"}] == [
            ("AIME25_01", "problem_end"), ("AIME25_02", "problem_error"),
        ]

        server.reply = lambda _body: "ANSWER: 7"
        subprocess.run([*args, "--arms", "vanilla_b1"], capture_output=True, check=True, timeout=45)
        full = json.loads(results.read_text())
        assert full["complete"] is True and full["summary"]["vanilla_b1"]["scored"] == 2

        if Path("/dev/full").exists():
            failed_log = subprocess.run([*args, "--arms", "cortheon_w1", "--progress-jsonl", "/dev/full"],
                                        capture_output=True, text=True, timeout=45)
            warning = "warning: progress log write failed"
            assert failed_log.returncode != 0 and failed_log.stderr.count(warning) == 1, failed_log.stderr
            assert failed_log.stderr.index(warning) < failed_log.stderr.index("[cortheon_w1]")
            assert json.loads(results.read_text())["complete"] is True

        # A trace destination alias must never truncate a published checkpoint.
        saved = results.read_bytes()
        alias = directory / "trace-alias"
        alias.hardlink_to(results)
        collision = subprocess.run([*args[:-1], alias, "--arms", "vanilla_b1"], capture_output=True, text=True, timeout=45)
        assert collision.returncode != 0 and "distinct files" in collision.stderr
        assert results.read_bytes() == saved


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        check(Path(sys.argv[1]).resolve(), Path(temp))
