"""Persistent verifier worker for cortheon-rs.

The Rust orchestrator keeps a pool of these processes alive and talks to each over
stdin/stdout with one JSON object per line. Every check delegates to Cortheon's own
Python functions (imported from a Cortheon checkout), so answer extraction,
normalization, equivalence and the OS sandbox are bit-for-bit the same code paths
that Cortheon v2 uses:

  extract  -> flash_forge.sandbox.extract_python_code
  exec     -> flash_forge.sandbox.run_python (bwrap / rlimit sandbox, timeout)
              + reason_forge.extract_answer_from_stdout / normalize_answer /
                _has_executable_assert   (mirrors reason_forge._execute_pal_outcome)
  prose    -> reason_forge.extract_answer_from_prose + normalize_answer
  equiv    -> reason_forge._answers_equivalent
  normalize-> reason_forge.normalize_answer

Point it at a Cortheon checkout with CORTHEON_SRC=/path/to/Cortheon (or have
`flash_forge` importable). Cortheon is MIT licensed (c) 2026 Cortheon contributors.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

src = os.environ.get("CORTHEON_SRC")
if src and src not in sys.path:
    sys.path.insert(0, src)

from flash_forge import reason_forge as rf  # noqa: E402
from flash_forge.aime import (  # noqa: E402
    extract_prose_candidate,
    grade_aime,
    normalize_aime_integer,
    scrub_detail_paths,
)
from flash_forge.sandbox import (  # noqa: E402
    extract_python_code,
    kill_active_sandbox_children,
    run_python,
    sandbox_backend,
    sandbox_scope,
)

# Each worker is a separate interpreter with at most one active exec.
_EXEC_SCOPE = "rust-verifier"


def op_exec(code: str, timeout_s: float) -> dict:
    # Mirrors reason_forge._execute_pal_outcome after the extract step (same error strings).
    out = {"ok": False, "returncode": None, "stdout": "", "timed_out": False, "elapsed_s": 0.0,
           "raw_answer": "", "norm_answer": "", "verified_constraints": False, "error": ""}
    if not code:
        out["error"] = "No valid Python code block extracted"
        return out
    with sandbox_scope(_EXEC_SCOPE):
        res = run_python(code, timeout_s=timeout_s)
    out.update(ok=res.ok, returncode=res.returncode, stdout=res.stdout[:4000],
               timed_out=res.timed_out, elapsed_s=res.elapsed_s)
    if not res.ok:
        out["error"] = res.short_error(600)
        return out
    raw_ans = rf.extract_answer_from_stdout(res.stdout)
    if not raw_ans:
        out["error"] = f"Script exited 0 but printed no ANSWER line (stdout={res.stdout[:200]!r})"
        return out
    out["raw_answer"] = raw_ans
    out["norm_answer"] = rf.normalize_answer(raw_ans)
    out["verified_constraints"] = rf._has_executable_assert(code)
    return out


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"pong": True, "sandbox": sandbox_backend(), "pid": os.getpid()}
    if op == "extract":
        return {"code": extract_python_code(req.get("content", ""))}
    if op == "exec":
        return op_exec(req.get("code", ""), float(req.get("timeout_s", 20.0)))
    if op == "prose":
        raw = rf.extract_answer_from_prose(req.get("content", ""))
        return {"raw_answer": raw, "norm_answer": rf.normalize_answer(raw)}
    if op == "normalize":
        return {"norm": rf.normalize_answer(req.get("raw", ""))}
    if op == "fallback_answer":
        # Mirror CompletionResult.json_payload(), including its fallback for fenced
        # JSON. Use Python's str() for non-string answers from unstructured servers.
        text = req["content"].strip()
        try:
            obj = json.loads(text)
        except ValueError:
            start, end = text.find("{"), text.rfind("}")
            try:
                obj = json.loads(text[start:end + 1]) if 0 <= start < end else None
            except ValueError:
                obj = None
        raw = str(obj.get("final_answer", "")).strip() if isinstance(obj, dict) else ""
        return {"norm": rf.normalize_answer(raw or rf.extract_answer_from_prose(text))}
    if op in ("aime_candidate", "aime_prose"):
        norm = extract_prose_candidate(req["content"])
        if op == "aime_prose":
            return {"norm": norm}
        return {"norm": norm, "valid_aime_int": normalize_aime_integer(norm)}
    if op == "grade_aime":
        correct, detail = grade_aime(req["predicted"], req["expected"])
        return {"correct": correct, "detail": detail}
    if op == "scrub_detail":
        return {"detail": scrub_detail_paths(req["detail"])}
    if op == "equiv":
        return {"eq": bool(rf._answers_equivalent(req.get("a", ""), req.get("b", "")))}
    return {"error": f"unknown op {op!r}"}


def respond(req: dict) -> None:
    started = time.perf_counter()
    try:
        resp = handle(req)
    except Exception as exc:
        resp = {"error": f"{type(exc).__name__}: {exc}"}
    resp["worker_s"] = time.perf_counter() - started
    print(json.dumps(resp), flush=True)


def cancel_exec(job: Future | None) -> None:
    # A cancellation may arrive before run_python registers its child. Retry
    # until the execution finishes, then allow the next request on this pipe.
    while job is not None and not job.done():
        kill_active_sandbox_children(scope=_EXEC_SCOPE)
        wait([job], timeout=0.01)


def main() -> None:
    # Warm imports (sympy is already imported by reason_forge) so the first job is not penalized.
    rf.normalize_answer("1/2")
    print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = None
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                    op = req["op"]
                except (ValueError, KeyError, TypeError) as exc:
                    print(json.dumps({"error": f"Invalid worker request: {exc}"}), flush=True)
                    continue
                if op == "cancel":
                    cancel_exec(job)
                    continue  # The original exec supplies the only response.
                if job is not None:
                    job.result()
                    job = None
                if op == "exec":
                    job = executor.submit(respond, req)
                else:
                    respond(req)
        finally:
            cancel_exec(job)  # Parent disconnected, including on a job timeout.


if __name__ == "__main__":
    main()
