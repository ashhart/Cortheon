"""Run the verifier protocol using an installed wheel and the real Linux sandbox."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(expected_backend: str) -> None:
    worker = Path(__file__).resolve().parents[1] / "verifier" / "worker.py"
    code = "import resource\nassert resource.getrlimit(resource.RLIMIT_CPU)[0] != resource.RLIM_INFINITY\n"
    if expected_backend == "linux-bwrap":
        parent_net = os.readlink("/proc/self/ns/net")
        code += f"import os\nassert os.readlink('/proc/self/ns/net') != {parent_net!r}\n"
    requests = [
        {"op": "ping"},
        {"op": "aime_candidate", "content": "Answer follows\n007"},
        {"op": "grade_aime", "predicted": "7/1", "expected": "007"},
        {"op": "scrub_detail", "detail": "/home/test/model.py:42"},
        {"op": "exec", "code": code + "print('ANSWER: 7')", "timeout_s": 5},
        {"op": "ping"},  # Wait for the execution response before closing the worker's input.
    ]
    env = os.environ.copy()
    env.pop("CORTHEON_SRC", None)
    # Isolated mode excludes the script directory, working directory and PYTHONPATH.
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(
            [sys.executable, "-I", str(worker)], cwd=tmp, env=env,
            input="".join(json.dumps(req) + "\n" for req in requests),
            text=True, stdout=subprocess.PIPE, check=True, timeout=30,
        )
    ready, ping, candidate, grade, scrub, result, last = map(json.loads, run.stdout.splitlines())
    assert ready["ready"] and ping["sandbox"] == expected_backend, ping
    assert candidate["norm"] == candidate["valid_aime_int"] == "7", candidate
    assert grade["correct"] and grade["detail"] == "OK (7)", grade
    assert scrub["detail"] == "<HOME>:42", scrub
    assert result["ok"] and result["norm_answer"] == "7", result
    assert last["pong"], last
    print(f"Installed wheel passed AIME grading and execution using {expected_backend}")


if __name__ == "__main__":
    main(sys.argv[1])
