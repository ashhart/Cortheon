"""Live Per-Slot Streaming Progress Renderer for Cortheon.

Zero-dependency, thread-safe, ``NO_COLOR``-aware terminal trace renderer mirroring
TensorFold Native's crisp per-slot output, plus non-breaking SSE comment framing
for ``cortheon proxy``.

Example terminal output::

    [cortheon] reason: launching 4 parallel PAL streams...
      slot 1 (pal_exact)    done   1.4s   420 tok  [PAL verified -> 70]
      slot 2 (pal_alt_z3)   done   2.1s   610 tok  [2/4 consensus -> 70]
      slot 3..4             ABORTED @ 2.1s (early-abort)
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
from typing import Any, Callable, Optional, TextIO

__all__ = [
    "LiveProgressReporter",
    "format_event",
    "format_sse_comment",
    "make_progress_callback",
]

_SLOT_COL_WIDTH = 22
_CONTROL_WS_RE = re.compile(r"[\r\n\x00]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"


def _stream_isatty(stream: Optional[TextIO]) -> bool:
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _detect_color(stream: Optional[TextIO], use_color: Optional[bool]) -> bool:
    if use_color is False:
        return False
    if use_color is True:
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return _stream_isatty(stream)


def _paint(text: str, enabled: bool, *codes: str) -> str:
    if not enabled or not codes or not text:
        return text
    return "".join(codes) + text + _Ansi.RESET


def _sanitize_inline(value: Any, max_len: int = 120) -> str:
    """Coerce any value to a single safe inline string with no control breaks."""
    if value is None:
        return ""
    try:
        raw = str(value)
    except Exception:
        raw = repr(type(value).__name__)
    cleaned = _CONTROL_WS_RE.sub(" ", raw)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    if max_len > 3 and len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _pad_slot_label(label: str) -> str:
    if len(label) < _SLOT_COL_WIDTH:
        return label + (" " * (_SLOT_COL_WIDTH - len(label)))
    return label + "  "


def _format_cancelled_slots(cancelled: Any) -> str:
    if cancelled is None:
        return "remaining slots"
    if isinstance(cancelled, bool):
        return "remaining slots"
    if isinstance(cancelled, int):
        return f"{cancelled} slot" if cancelled == 1 else f"{cancelled} slots"
    if isinstance(cancelled, str):
        s = _sanitize_inline(cancelled, 40)
        if not s:
            return "remaining slots"
        if s.lower().startswith("slot"):
            return s
        return f"slot {s}"
    if isinstance(cancelled, (list, tuple, set)):
        items = list(cancelled)
        if not items:
            return "remaining slots"
        int_slots: list[int] = []
        all_ints = True
        for item in items:
            if isinstance(item, int) and not isinstance(item, bool):
                int_slots.append(item)
            else:
                all_ints = False
                break
        if all_ints and int_slots:
            ordered = sorted(set(int_slots))
            if len(ordered) == 1:
                return f"slot {ordered[0]}"
            is_contiguous = all(
                ordered[i] + 1 == ordered[i + 1] for i in range(len(ordered) - 1)
            )
            if is_contiguous:
                return f"slot {ordered[0]}..{ordered[-1]}"
            return "slot " + ",".join(str(x) for x in ordered)
        names = [_sanitize_inline(x, 24) for x in items]
        names = [n for n in names if n]
        if not names:
            return "remaining slots"
        if len(names) == 1:
            return f"slot {names[0]}"
        return f"slots {', '.join(names[:3])}" + ("..." if len(names) > 3 else "")
    return _sanitize_inline(cancelled, 32) or "remaining slots"


def _normalize_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return dict(event)
    return {"type": "event", "message": _sanitize_inline(event, 160)}


def _event_type(event: dict[str, Any]) -> str:
    raw = event.get("type") or event.get("event") or event.get("kind") or ""
    return _sanitize_inline(raw, 48).lower()


def _format_start(ev: dict[str, Any], color: bool) -> str:
    engine = _sanitize_inline(ev.get("engine"), 32)
    mode = _sanitize_inline(ev.get("mode"), 32)
    streams = _coerce_int(ev.get("streams"))

    head = engine or mode or "solve"
    mode_suffix = ""
    if mode and mode.lower() != head.lower():
        mode_suffix = f"{mode} "

    if streams is not None and streams > 0:
        if streams == 1:
            action = f"launching 1 {mode_suffix}stream..."
        else:
            action = f"launching {streams} parallel {mode_suffix}streams..."
    elif mode_suffix:
        action = f"launching {mode_suffix.strip()}..."
    else:
        action = "starting..."

    prefix = _paint("[cortheon]", color, _Ansi.CYAN, _Ansi.BOLD)
    return f"{prefix} {head}: {action}"


def _format_slot_done(ev: dict[str, Any], color: bool) -> str:
    slot = ev.get("slot")
    role = _sanitize_inline(ev.get("role"), 32)
    ok_raw = ev.get("ok")
    ok = True if ok_raw is None else bool(ok_raw)
    elapsed = _coerce_float(ev.get("elapsed_s"))
    tokens = _coerce_int(ev.get("tokens"))
    summary = _sanitize_inline(ev.get("summary"), 96)
    answer = _sanitize_inline(ev.get("answer"), 64)

    slot_str = _sanitize_inline(slot, 16) if slot is not None else "?"
    label = f"slot {slot_str} ({role})" if role else f"slot {slot_str}"
    padded_label = _pad_slot_label(label)

    status_plain = "done" if ok else "fail"
    status = _paint(
        status_plain,
        color,
        _Ansi.GREEN if ok else _Ansi.RED,
        _Ansi.BOLD,
    )

    parts: list[str] = [status]
    if elapsed is not None:
        parts.append(f"{elapsed:4.1f}s")
    if tokens is not None:
        parts.append(f"{tokens:4d} tok")

    detail = ""
    if summary and answer:
        if summary == answer or summary.endswith(f"-> {answer}"):
            detail = f"[{summary}]"
        else:
            detail = f"[{summary} -> {answer}]"
    elif summary:
        detail = f"[{summary}]"
    elif answer:
        detail = f"[-> {answer}]"

    if detail:
        parts.append(_paint(detail, color, _Ansi.DIM if not ok else _Ansi.CYAN))

    return "  " + padded_label + "  ".join(parts)


def _format_early_abort(ev: dict[str, Any], color: bool) -> str:
    cancelled = _format_cancelled_slots(ev.get("cancelled"))
    padded = _pad_slot_label(cancelled)
    elapsed = _coerce_float(ev.get("elapsed_s"))
    reason = _sanitize_inline(ev.get("reason"), 80)
    consensus = _sanitize_inline(ev.get("consensus_answer"), 64)

    aborted_tag = _paint("ABORTED", color, _Ansi.YELLOW, _Ansi.BOLD)
    time_part = f" @ {elapsed:.1f}s" if elapsed is not None else ""

    if reason and consensus:
        if reason == consensus or f"-> {consensus}" in reason or f"={consensus}" in reason:
            note = f" ({reason})"
        else:
            note = f" ({reason}; consensus -> {consensus})"
    elif reason:
        note = f" ({reason})"
    elif consensus:
        note = f" (early-abort: consensus -> {consensus})"
    else:
        note = " (early-abort)"

    return f"  {padded}{aborted_tag}{time_part}{note}"


def _format_wave2_start(ev: dict[str, Any], color: bool) -> str:
    streams = _coerce_int(ev.get("streams"))
    reason = _sanitize_inline(ev.get("reason"), 100)
    prefix = _paint("[cortheon]", color, _Ansi.MAGENTA, _Ansi.BOLD)

    if streams is not None and streams > 0:
        noun = "stream" if streams == 1 else "streams"
        body = f"launching {streams} falsification {noun}"
    else:
        body = "escalating to Wave-2 falsification"

    if reason:
        return f"{prefix} wave2: {body} ({reason})"
    return f"{prefix} wave2: {body}..."


def _format_drafter_accepted(ev: dict[str, Any], color: bool) -> str:
    drafter = _sanitize_inline(ev.get("drafter_model"), 48) or "drafter"
    answer = _sanitize_inline(ev.get("answer"), 64)
    elapsed = _coerce_float(ev.get("elapsed_s"))
    saved = _coerce_int(ev.get("verifier_calls_saved"))

    prefix = _paint("[cortheon]", color, _Ansi.CYAN, _Ansi.BOLD)
    badge = _paint("ACCEPTED", color, _Ansi.GREEN, _Ansi.BOLD)

    parts = [f"{prefix} spec: {badge} by {drafter}"]
    if elapsed is not None:
        parts.append(f"in {elapsed:.1f}s")
    if answer:
        parts.append(f"-> {answer}")
    if saved is not None:
        noun = "call" if saved == 1 else "calls"
        parts.append(f"(saved {saved} verifier {noun})")
    return " ".join(parts)


def _format_escalated(ev: dict[str, Any], color: bool) -> str:
    verifier = _sanitize_inline(ev.get("verifier_model"), 48) or "verifier"
    reason = _sanitize_inline(ev.get("reason"), 100)

    prefix = _paint("[cortheon]", color, _Ansi.CYAN, _Ansi.BOLD)
    badge = _paint("ESCALATED", color, _Ansi.YELLOW, _Ansi.BOLD)

    if reason:
        return f"{prefix} spec: {badge} -> {verifier} ({reason})"
    return f"{prefix} spec: {badge} -> {verifier}"


def _format_bug_proven(ev: dict[str, Any], color: bool) -> str:
    role = _sanitize_inline(ev.get("role"), 40)
    title = _sanitize_inline(ev.get("title"), 110) or "verified reproducer failed"
    passed_on_base = ev.get("passed_on_base")

    badge = _paint("[BUG PROVEN]", color, _Ansi.RED, _Ansi.BOLD)
    role_part = f" ({role})" if role else ""

    if passed_on_base is True:
        cls = _paint("REGRESSION", color, _Ansi.RED, _Ansi.BOLD)
        return f"  {badge}{role_part} {cls}: {title}"
    if passed_on_base is False:
        cls = _paint("PRE-EXISTING", color, _Ansi.YELLOW, _Ansi.BOLD)
        return f"  {badge}{role_part} {cls}: {title}"
    return f"  {badge}{role_part} {title}"


def _format_done(ev: dict[str, Any], color: bool) -> str:
    elapsed = _coerce_float(ev.get("elapsed_s"))
    summary = _sanitize_inline(ev.get("summary"), 120)
    verified = ev.get("verified")

    prefix = _paint("[cortheon]", color, _Ansi.CYAN, _Ansi.BOLD)
    parts = [f"{prefix} done"]
    if elapsed is not None:
        parts.append(f"in {elapsed:.1f}s")
    if verified is True:
        parts.append(_paint("[VERIFIED]", color, _Ansi.GREEN, _Ansi.BOLD))
    elif verified is False:
        parts.append(_paint("[UNVERIFIED]", color, _Ansi.YELLOW, _Ansi.BOLD))
    if summary:
        parts.append(summary)
    return " ".join(parts)


def _format_generic(ev: dict[str, Any], color: bool) -> str:
    etype = _event_type(ev) or "event"
    prefix = _paint("[cortheon]", color, _Ansi.CYAN, _Ansi.BOLD)
    kv_parts: list[str] = []
    for k, v in ev.items():
        if k in ("type", "event", "kind") or v is None:
            continue
        key_s = _sanitize_inline(k, 32)
        val_s = _sanitize_inline(v, 64)
        if key_s and val_s:
            kv_parts.append(f"{key_s}={val_s}")
        if len(kv_parts) >= 6:
            break
    if kv_parts:
        return f"{prefix} {etype}: {' '.join(kv_parts)}"
    return f"{prefix} {etype}"


def format_event(event: dict[str, Any], *, use_color: bool = False) -> str:
    """Format any progress event dict into a single crisp terminal line.

    Never raises, even for unknown or malformed event payloads.
    """
    try:
        ev = _normalize_event(event)
        etype = _event_type(ev)
        if etype == "start":
            return _format_start(ev, use_color)
        if etype == "slot_done":
            return _format_slot_done(ev, use_color)
        if etype == "early_abort":
            return _format_early_abort(ev, use_color)
        if etype == "wave2_start":
            return _format_wave2_start(ev, use_color)
        if etype == "drafter_accepted":
            return _format_drafter_accepted(ev, use_color)
        if etype == "escalated":
            return _format_escalated(ev, use_color)
        if etype == "bug_proven":
            return _format_bug_proven(ev, use_color)
        if etype == "done":
            return _format_done(ev, use_color)
        return _format_generic(ev, use_color)
    except Exception:
        fallback = _sanitize_inline(event, 120) or "event"
        return f"[cortheon] {fallback}"


def format_sse_comment(event: dict[str, Any]) -> str:
    """Format a progress event as a spec-compliant SSE comment heartbeat.

    Always returns a single comment line terminated by ``\\n\\n`` (``: cortheon <summary>\\n\\n``).
    All CR/LF/NUL characters inside event values are sanitized so untrusted payloads
    can never inject SSE fields or break stream framing.
    """
    plain = format_event(event, use_color=False)
    body = plain.strip()
    if body.startswith("[cortheon]"):
        body = body[len("[cortheon]") :].strip()
    body = _CONTROL_WS_RE.sub(" ", body)
    body = _MULTI_SPACE_RE.sub(" ", body).strip()
    if not body:
        body = "event"
    return f": cortheon {body}\n\n"


class LiveProgressReporter:
    """Thread-safe, zero-dependency live terminal renderer for Cortheon events."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        *,
        enabled: bool = True,
        use_color: Optional[bool] = None,
        quiet: bool = False,
    ) -> None:
        self.stream: TextIO = stream if stream is not None else sys.stderr
        self.enabled: bool = bool(enabled)
        self.quiet: bool = bool(quiet)
        self.use_color: bool = _detect_color(self.stream, use_color)
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def format(self, event: dict[str, Any]) -> str:
        """Render a single event using this reporter's color setting."""
        return format_event(event, use_color=self.use_color)

    def emit(self, event: dict[str, Any]) -> None:
        """Record an event and write a formatted line to ``self.stream``.

        Thread-safe and fault-tolerant: never raises on unknown events,
        non-TTY pipes, or broken/closed streams.
        """
        snapshot = _normalize_event(event)
        line = format_event(snapshot, use_color=self.use_color)
        with self._lock:
            self.events.append(snapshot)
            if not self.enabled or self.quiet:
                return
            try:
                self.stream.write(line + "\n")
                flush = getattr(self.stream, "flush", None)
                if callable(flush):
                    flush()
            except Exception:
                # Never crash an active solve if stderr is closed or a pipe breaks.
                pass

    def __call__(self, event: dict[str, Any]) -> None:
        self.emit(event)


def make_progress_callback(
    *,
    quiet: bool = False,
    stream: Optional[TextIO] = None,
    enabled: bool = True,
    use_color: Optional[bool] = None,
) -> tuple[LiveProgressReporter, Callable[[dict[str, Any]], None]]:
    """Construct a ``(LiveProgressReporter, emit_callback)`` pair ready for ``on_event=``."""
    reporter = LiveProgressReporter(
        stream=stream,
        enabled=enabled,
        use_color=use_color,
        quiet=quiet,
    )
    return reporter, reporter.emit
