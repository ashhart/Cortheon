"""Offline unit tests for live progress streaming and release packaging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import re
import sys
from typing import Any

import pytest

from flash_forge.progress import (
    LiveProgressReporter,
    format_event,
    format_sse_comment,
    make_progress_callback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _FakeTTYStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class _BrokenTTYStream(io.StringIO):
    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")


class _BrokenWriteStream:
    def isatty(self) -> bool:
        return False

    def write(self, _s: str) -> int:
        raise BrokenPipeError("Broken pipe")

    def flush(self) -> None:
        raise BrokenPipeError("Broken pipe")


# ---------------------------------------------------------------------------
# LiveProgressReporter: all event types & TensorFold Native formatting
# ---------------------------------------------------------------------------


def test_tensorfold_native_slot_trace_alignment():
    buf = io.StringIO()
    reporter = LiveProgressReporter(stream=buf, use_color=False)

    reporter.emit({"type": "start", "engine": "reason", "streams": 4, "mode": "PAL"})
    reporter.emit(
        {
            "type": "slot_done",
            "slot": 1,
            "role": "pal_exact",
            "ok": True,
            "elapsed_s": 1.4,
            "tokens": 420,
            "summary": "PAL verified",
            "answer": "70",
        }
    )
    reporter.emit(
        {
            "type": "slot_done",
            "slot": 2,
            "role": "pal_alt_z3",
            "ok": True,
            "elapsed_s": 2.1,
            "tokens": 610,
            "summary": "2/4 consensus",
            "answer": "70",
        }
    )
    reporter.emit(
        {
            "type": "early_abort",
            "cancelled": [3, 4],
            "elapsed_s": 2.1,
            "reason": "early-abort",
        }
    )

    lines = buf.getvalue().splitlines()
    assert lines == [
        "[cortheon] reason: launching 4 parallel PAL streams...",
        "  slot 1 (pal_exact)    done   1.4s   420 tok  [PAL verified -> 70]",
        "  slot 2 (pal_alt_z3)   done   2.1s   610 tok  [2/4 consensus -> 70]",
        "  slot 3..4             ABORTED @ 2.1s (early-abort)",
    ]
    assert len(reporter.events) == 4
    assert reporter.events[0]["engine"] == "reason"
    assert reporter.events[1]["answer"] == "70"
    assert reporter.events[3]["cancelled"] == [3, 4]


def test_event_start_variants():
    assert (
        format_event({"type": "start", "engine": "code", "streams": 4})
        == "[cortheon] code: launching 4 parallel streams..."
    )
    assert (
        format_event({"type": "start", "engine": "reason", "streams": 1, "mode": "PAL"})
        == "[cortheon] reason: launching 1 PAL stream..."
    )
    assert (
        format_event({"type": "start", "engine": "reason", "mode": "reason", "streams": 4})
        == "[cortheon] reason: launching 4 parallel streams..."
    )
    assert (
        format_event({"type": "start", "mode": "fast"})
        == "[cortheon] fast: starting..."
    )


def test_event_slot_done_pass_fail_and_partial_fields():
    fail_line = format_event(
        {
            "type": "slot_done",
            "slot": 3,
            "role": "pal_sympy",
            "ok": False,
            "elapsed_s": 3.2,
            "tokens": 512,
            "summary": "TimeoutError\nstack line 2",
        }
    )
    assert "\n" not in fail_line
    assert "slot 3 (pal_sympy)" in fail_line
    assert "fail" in fail_line
    assert "3.2s" in fail_line
    assert "512 tok" in fail_line
    assert "[TimeoutError stack line 2]" in fail_line

    # Answer-only without summary or tokens
    ans_only = format_event({"type": "slot_done", "slot": 1, "ok": True, "answer": "42"})
    assert "slot 1" in ans_only
    assert "done" in ans_only
    assert "[-> 42]" in ans_only

    # Avoid duplicating answer if summary already ends with it
    no_dup = format_event(
        {
            "type": "slot_done",
            "slot": 2,
            "role": "z3",
            "ok": True,
            "summary": "PAL verified -> 42",
            "answer": "42",
        }
    )
    assert "[PAL verified -> 42]" in no_dup
    assert "-> 42 -> 42" not in no_dup


def test_event_early_abort_variants():
    line_consensus = format_event(
        {
            "type": "early_abort",
            "cancelled": [2, 3, 4],
            "elapsed_s": 1.8,
            "consensus_answer": "70",
        }
    )
    assert "slot 2..4" in line_consensus
    assert "ABORTED @ 1.8s" in line_consensus
    assert "consensus -> 70" in line_consensus

    line_non_contig = format_event(
        {
            "type": "early_abort",
            "cancelled": [2, 4],
            "elapsed_s": 2.5,
            "reason": "sample consensus",
            "consensus_answer": "cand_1",
        }
    )
    assert "slot 2,4" in line_non_contig
    assert "sample consensus; consensus -> cand_1" in line_non_contig

    line_int = format_event({"type": "early_abort", "cancelled": 2})
    assert "2 slots" in line_int
    assert "ABORTED (early-abort)" in line_int

    line_names = format_event(
        {"type": "early_abort", "cancelled": ["pal_sympy", "pal_num"], "elapsed_s": 3.0}
    )
    assert "slots pal_sympy, pal_num" in line_names


def test_event_wave2_start():
    full = format_event(
        {"type": "wave2_start", "reason": "Wave-1 disagreement", "streams": 4}
    )
    assert full == "[cortheon] wave2: launching 4 falsification streams (Wave-1 disagreement)"

    no_streams = format_event({"type": "wave2_start", "reason": "single passing script"})
    assert "wave2" in no_streams
    assert "single passing script" in no_streams

    minimal = format_event({"type": "wave2_start"})
    assert "[cortheon] wave2:" in minimal


def test_event_drafter_accepted():
    plural = format_event(
        {
            "type": "drafter_accepted",
            "drafter_model": "qwen2.5-3b",
            "answer": "70",
            "elapsed_s": 1.2,
            "verifier_calls_saved": 4,
        }
    )
    assert (
        plural
        == "[cortheon] spec: ACCEPTED by qwen2.5-3b in 1.2s -> 70 (saved 4 verifier calls)"
    )

    singular = format_event(
        {
            "type": "drafter_accepted",
            "drafter_model": "tiny-2b",
            "answer": "13/5",
            "verifier_calls_saved": 1,
        }
    )
    assert "ACCEPTED by tiny-2b -> 13/5 (saved 1 verifier call)" in singular


def test_event_escalated():
    full = format_event(
        {
            "type": "escalated",
            "reason": "drafter split 1 vs 1",
            "verifier_model": "deepseek-v4",
        }
    )
    assert full == "[cortheon] spec: ESCALATED -> deepseek-v4 (drafter split 1 vs 1)"

    minimal = format_event({"type": "escalated"})
    assert minimal == "[cortheon] spec: ESCALATED -> verifier"


def test_event_bug_proven():
    regression = format_event(
        {
            "type": "bug_proven",
            "role": "edge_boundary_breaker",
            "title": "off-by-one on empty list",
            "passed_on_base": True,
        }
    )
    assert (
        regression
        == "  [BUG PROVEN] (edge_boundary_breaker) REGRESSION: off-by-one on empty list"
    )

    pre_existing = format_event(
        {
            "type": "bug_proven",
            "role": "state_concurrency_breaker",
            "title": "shared cache mutation",
            "passed_on_base": False,
        }
    )
    assert (
        pre_existing
        == "  [BUG PROVEN] (state_concurrency_breaker) PRE-EXISTING: shared cache mutation"
    )

    new_code = format_event(
        {
            "type": "bug_proven",
            "role": "regression_breaker",
            "title": "division by zero in new helper",
            "passed_on_base": None,
        }
    )
    assert (
        new_code
        == "  [BUG PROVEN] (regression_breaker) division by zero in new helper"
    )


def test_event_done():
    verified = format_event(
        {
            "type": "done",
            "elapsed_s": 2.1,
            "summary": "answer=70 (2/4 PAL consensus)",
            "verified": True,
        }
    )
    assert verified == "[cortheon] done in 2.1s [VERIFIED] answer=70 (2/4 PAL consensus)"

    unverified = format_event(
        {
            "type": "done",
            "elapsed_s": 8.4,
            "summary": "fallback prose answer",
            "verified": False,
        }
    )
    assert unverified == "[cortheon] done in 8.4s [UNVERIFIED] fallback prose answer"

    neutral = format_event({"type": "done", "elapsed_s": 0.5, "summary": "complete"})
    assert neutral == "[cortheon] done in 0.5s complete"


def test_unknown_and_malformed_events_never_raise():
    buf = io.StringIO()
    reporter = LiveProgressReporter(stream=buf, use_color=False)

    class _ExplodingStr:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    weird_events: list[Any] = [
        {"type": "custom_stage", "step": 2, "note": "compiling"},
        {},
        {"type": "weird", "obj": object(), "bad": _ExplodingStr(), "empty": None},
        {"type": "slot_done", "slot": None, "elapsed_s": "not-a-float", "tokens": "NaN"},
        {"type": "slot_done", "slot": 9, "elapsed_s": float("inf"), "tokens": None},
        "plain string payload",
        12345,
        None,
    ]
    for ev in weird_events:
        reporter.emit(ev)

    assert len(reporter.events) == len(weird_events)
    lines = buf.getvalue().splitlines()
    assert len(lines) == len(weird_events)
    assert "[cortheon] custom_stage: step=2 note=compiling" in lines[0]
    assert "[cortheon] event" in lines[1]
    assert "inf" not in lines[4].lower()


def test_default_stream_is_stderr():
    rep = LiveProgressReporter()
    assert rep.stream is sys.stderr


def test_event_snapshot_isolation():
    buf = io.StringIO()
    reporter = LiveProgressReporter(stream=buf, use_color=False)
    payload = {"type": "start", "engine": "reason", "streams": 4}
    reporter.emit(payload)
    payload["streams"] = 99
    assert reporter.events[0]["streams"] == 4


# ---------------------------------------------------------------------------
# NO_COLOR, use_color, quiet, enabled, non-TTY / broken stream, helper factory
# ---------------------------------------------------------------------------


def test_no_color_env_disables_ansi_even_on_tty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    tty_color = _FakeTTYStream()
    rep_color = LiveProgressReporter(stream=tty_color)
    assert rep_color.use_color is True
    rep_color.emit({"type": "start", "engine": "reason", "streams": 4, "mode": "PAL"})
    assert ANSI_RE.search(tty_color.getvalue()) is not None

    monkeypatch.setenv("NO_COLOR", "1")
    tty_nocolor = _FakeTTYStream()
    rep_nocolor = LiveProgressReporter(stream=tty_nocolor)
    assert rep_nocolor.use_color is False
    rep_nocolor.emit({"type": "start", "engine": "reason", "streams": 4, "mode": "PAL"})
    rep_nocolor.emit(
        {
            "type": "slot_done",
            "slot": 1,
            "role": "pal_exact",
            "ok": True,
            "elapsed_s": 1.1,
            "tokens": 300,
            "answer": "70",
        }
    )
    assert ANSI_RE.search(tty_nocolor.getvalue()) is None

    # TERM=dumb also disables auto color
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    tty_dumb = _FakeTTYStream()
    rep_dumb = LiveProgressReporter(stream=tty_dumb)
    assert rep_dumb.use_color is False


def test_use_color_explicit_false_and_true():
    tty = _FakeTTYStream()
    rep_off = LiveProgressReporter(stream=tty, use_color=False)
    assert rep_off.use_color is False
    rep_off.emit({"type": "done", "elapsed_s": 1.0, "verified": True, "summary": "ok"})
    assert ANSI_RE.search(tty.getvalue()) is None

    pipe = io.StringIO()
    rep_on = LiveProgressReporter(stream=pipe, use_color=True)
    assert rep_on.use_color is True
    rep_on.emit({"type": "done", "elapsed_s": 1.0, "verified": True, "summary": "ok"})
    assert ANSI_RE.search(pipe.getvalue()) is not None
    # Stripping ANSI yields the exact plain line
    stripped = ANSI_RE.sub("", pipe.getvalue()).strip()
    assert stripped == "[cortheon] done in 1.0s [VERIFIED] ok"


def test_quiet_and_enabled_suppress_output_but_record_events():
    buf_q = io.StringIO()
    rep_q = LiveProgressReporter(stream=buf_q, quiet=True)
    rep_q.emit({"type": "start", "engine": "code", "streams": 4})
    rep_q.emit({"type": "done", "elapsed_s": 1.5, "verified": True})
    assert buf_q.getvalue() == ""
    assert len(rep_q.events) == 2

    buf_dis = io.StringIO()
    rep_dis = LiveProgressReporter(stream=buf_dis, enabled=False)
    rep_dis.emit({"type": "start", "engine": "code", "streams": 4})
    assert buf_dis.getvalue() == ""
    assert len(rep_dis.events) == 1


def test_non_tty_and_broken_pipe_never_crash():
    broken_tty = _BrokenTTYStream()
    rep1 = LiveProgressReporter(stream=broken_tty)
    assert rep1.use_color is False
    rep1.emit({"type": "start", "engine": "reason", "streams": 2})
    assert len(rep1.events) == 1

    broken_pipe = _BrokenWriteStream()
    rep2 = LiveProgressReporter(stream=broken_pipe)  # type: ignore[arg-type]
    rep2.emit({"type": "done", "elapsed_s": 0.4, "verified": True})
    assert len(rep2.events) == 1


def test_make_progress_callback_and_direct_call():
    buf = io.StringIO()
    reporter, cb = make_progress_callback(quiet=False, stream=buf)
    assert isinstance(reporter, LiveProgressReporter)
    assert callable(cb)

    cb({"type": "start", "engine": "reason", "streams": 4, "mode": "PAL"})
    reporter({"type": "done", "elapsed_s": 2.0, "verified": True, "summary": "70"})

    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert len(reporter.events) == 2

    q_buf = io.StringIO()
    q_rep, q_cb = make_progress_callback(quiet=True, stream=q_buf)
    q_cb({"type": "start", "engine": "code", "streams": 4})
    assert q_buf.getvalue() == ""
    assert len(q_rep.events) == 1


def test_thread_safety_concurrent_emits():
    buf = io.StringIO()
    reporter = LiveProgressReporter(stream=buf, use_color=False)

    num_threads = 8
    per_thread = 40
    total = num_threads * per_thread

    def _worker(tid: int) -> None:
        for i in range(per_thread):
            reporter.emit(
                {
                    "type": "slot_done",
                    "slot": tid * 100 + i,
                    "role": f"worker_{tid}",
                    "ok": True,
                    "elapsed_s": 0.5,
                    "tokens": 100 + i,
                    "summary": f"t{tid}-i{i}",
                    "answer": str(tid * 100 + i),
                }
            )

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_worker, tid) for tid in range(num_threads)]
        for fut in futures:
            fut.result()

    assert len(reporter.events) == total
    lines = buf.getvalue().splitlines()
    assert len(lines) == total
    line_re = re.compile(
        r"^  slot (\d+) \(worker_(\d+)\)\s+done\s+0\.5s\s+\d+ tok  \[t\2-i\d+ -> \1\]$"
    )
    for line in lines:
        assert line_re.match(line), f"Corrupted/interleaved progress line: {line!r}"


# ---------------------------------------------------------------------------
# format_sse_comment framing & newline injection hardening
# ---------------------------------------------------------------------------


def test_format_sse_comment_all_event_types():
    events = [
        {"type": "start", "engine": "reason", "streams": 4, "mode": "PAL"},
        {
            "type": "slot_done",
            "slot": 1,
            "role": "pal_exact",
            "ok": True,
            "elapsed_s": 1.4,
            "tokens": 420,
            "summary": "PAL verified",
            "answer": "70",
        },
        {"type": "early_abort", "cancelled": [3, 4], "elapsed_s": 2.1, "reason": "early-abort"},
        {"type": "wave2_start", "reason": "disagreement", "streams": 4},
        {
            "type": "drafter_accepted",
            "drafter_model": "qwen2.5-3b",
            "answer": "70",
            "elapsed_s": 1.2,
            "verifier_calls_saved": 4,
        },
        {"type": "escalated", "reason": "split", "verifier_model": "deepseek-v4"},
        {"type": "bug_proven", "role": "edge", "title": "off-by-one", "passed_on_base": True},
        {"type": "done", "elapsed_s": 2.1, "summary": "answer=70", "verified": True},
        {"type": "unknown_evt", "detail": "ok"},
    ]
    for ev in events:
        frame = format_sse_comment(ev)
        assert frame.startswith(": cortheon "), f"Bad prefix: {frame!r}"
        assert frame.endswith("\n\n"), f"Bad terminator: {frame!r}"
        assert "\r" not in frame
        assert "\n" not in frame[:-2]
        assert ANSI_RE.search(frame) is None
        # Should not duplicate [cortheon] prefix inside SSE comment
        assert not frame.startswith(": cortheon [cortheon]")


def test_format_sse_comment_blocks_newline_and_crlf_injection():
    malicious_events = [
        {
            "type": "slot_done",
            "slot": 1,
            "role": "pal\nexact",
            "ok": True,
            "summary": "ok\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"INJECTED\"}}]}\n\n",
            "answer": "70\r\nevent: hijacked\r\ndata: evil",
        },
        {
            "type": "bug_proven",
            "role": "edge\r\nbreaker",
            "title": "bad\x00title\ndata: forged",
            "passed_on_base": True,
        },
        {
            "type": "done",
            "summary": "\n\n\r\r\x00",
            "verified": True,
        },
        {
            "type": "custom\nevent",
            "key\n1": "val\r\ndata: forged",
        },
    ]
    for ev in malicious_events:
        frame = format_sse_comment(ev)
        assert frame.startswith(": cortheon ")
        assert frame.endswith("\n\n")
        body = frame[:-2]
        assert "\n" not in body
        assert "\r" not in body
        assert "\x00" not in body
        # Verify SSE frame parses as exactly ONE comment line and ZERO data/event fields
        raw_lines = body.split("\n")
        assert len(raw_lines) == 1
        assert raw_lines[0].startswith(": ")


# ---------------------------------------------------------------------------
# Release packaging validation: CHANGELOG.md & .github/workflows/release.yml
# ---------------------------------------------------------------------------


def test_changelog_has_required_versions_and_entries():
    changelog_path = REPO_ROOT / "CHANGELOG.md"
    assert changelog_path.is_file(), "CHANGELOG.md must exist at repo root"
    text = changelog_path.read_text(encoding="utf-8")

    assert "# Changelog" in text
    assert "Keep a Changelog" in text

    idx_220 = text.find("## [2.2.0]")
    idx_210 = text.find("## [2.1.0]")
    idx_200 = text.find("## [2.0.0]")
    assert idx_220 != -1, "CHANGELOG.md missing [2.2.0]"
    assert idx_210 != -1, "CHANGELOG.md missing [2.1.0]"
    assert idx_200 != -1, "CHANGELOG.md missing [2.0.0]"
    assert idx_220 < idx_210 < idx_200, "Versions must appear in reverse chronological order"

    sec_220 = text[idx_220:idx_210]
    sec_210 = text[idx_210:idx_200]
    sec_200 = text[idx_200:]

    for needle in (
        "CodeForge v7",
        "early-abort",
        "TLE",
        "install-mcp",
        "Claude Code",
        "Cursor",
        "OpenCode",
        "Codex",
        "Windsurf",
        "--record-flywheel",
        "flash_forge/progress.py",
        "--version",
        "release.yml",
    ):
        assert needle in sec_220, f"CHANGELOG [2.2.0] missing {needle!r}"

    for needle in (
        "SpecVerify",
        "cortheon guard",
        "cortheon mcp",
        "Flywheel",
        "cortheon doctor",
        "@SelfHostedSI",
        "Maj@4",
        "Maj@8",
        "wilson_ci",
    ):
        assert needle in sec_210, f"CHANGELOG [2.1.0] missing {needle!r}"

    for needle in (
        "ReasonForge",
        "CodeForge",
        "ThinkForge",
        "macos-seatbelt",
        "linux-bwrap",
        "/cortheon",
        "cortheon duel",
        "cortheon proxy",
        "Death-Spiral Rescue",
        "bench/fetch_datasets.py",
        "bench/stats.py",
    ):
        assert needle in sec_200, f"CHANGELOG [2.0.0] missing {needle!r}"


def _validate_yaml_syntax_zero_dep(text: str) -> None:
    """Zero-dependency structural YAML syntax sanity checker for GitHub Actions."""
    lines = text.splitlines()
    assert lines, "YAML file must not be empty"

    in_block_scalar_indent: int | None = None
    for lineno, raw in enumerate(lines, start=1):
        assert "\t" not in raw, f"YAML line {lineno} contains illegal tab character"
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))

        if in_block_scalar_indent is not None:
            if indent > in_block_scalar_indent:
                continue
            in_block_scalar_indent = None

        assert indent % 2 == 0, f"YAML line {lineno} has odd indentation ({indent})"

        # Check balanced quotes / brackets on single-line YAML nodes
        single_q = 0
        double_q = 0
        bracket_depth = 0
        brace_depth = 0
        in_sq = False
        in_dq = False
        for ch in stripped:
            if ch == "'" and not in_dq:
                in_sq = not in_sq
                single_q += 1
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
                double_q += 1
            elif not in_sq and not in_dq:
                if ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth -= 1
                elif ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
        assert not in_sq, f"YAML line {lineno} has unclosed single quote: {raw!r}"
        assert not in_dq, f"YAML line {lineno} has unclosed double quote: {raw!r}"
        assert bracket_depth == 0, f"YAML line {lineno} has unbalanced []: {raw!r}"
        assert brace_depth == 0, f"YAML line {lineno} has unbalanced {{}}: {raw!r}"

        # Every non-block-scalar line must be a list item or mapping key
        body = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if stripped.startswith("- "):
            # Plain list scalar (e.g., `- "v*"`) or mapping entry (`- uses: ...`)
            if ":" in body and body.endswith("|"):
                in_block_scalar_indent = indent
        else:
            assert ":" in body, f"YAML line {lineno} is neither list item nor key:value: {raw!r}"
            if re.search(r":\s*[|>][+-]?\s*$", body):
                in_block_scalar_indent = indent


def test_release_workflow_valid_yaml_and_required_gates():
    wf_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert wf_path.is_file(), ".github/workflows/release.yml must exist"
    text = wf_path.read_text(encoding="utf-8")

    # Always run zero-dependency structural YAML syntax check
    _validate_yaml_syntax_zero_dep(text)

    # Also validate with PyYAML if installed in the environment
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)
        assert "jobs" in parsed and isinstance(parsed["jobs"], dict)
        # YAML 1.1 parses unquoted `on:` as boolean True
        triggers = parsed.get("on", parsed.get(True))
        assert isinstance(triggers, dict)
        assert "push" in triggers and "workflow_dispatch" in triggers

    # Required triggers
    assert re.search(r"(?m)^on:\s*$", text)
    assert '"v*"' in text or "'v*'" in text or "- v*" in text
    assert "workflow_dispatch:" in text

    # Strict offline pytest gate before build
    assert "python -m pytest -q" in text
    assert "needs: [test]" in text or "needs: test" in text
    assert text.index("python -m pytest -q") < text.index("python -m build")

    # Build, twine check, and wheel CLI verification
    assert "python -m build" in text
    assert "twine check dist/*" in text
    assert "cortheon --version" in text
    assert "cortheon --help" in text

    # GitHub Release with dist/* artifacts
    assert "softprops/action-gh-release@v2" in text
    assert "files: dist/*" in text
    assert "contents: write" in text

    # Optional OIDC PyPI publish gated on vars.PUBLISH_TO_PYPI == 'true'
    assert "vars.PUBLISH_TO_PYPI == 'true'" in text
    assert "id-token: write" in text
    assert "pypa/gh-action-pypi-publish" in text
