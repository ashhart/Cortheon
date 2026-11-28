"""Shared answer extraction, scoring, and reporting helpers for AIME benchmarks."""

from __future__ import annotations

import re

from .reason_forge import _answers_equivalent, extract_answer_from_prose, normalize_answer


_TMPDIR_FULL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/private)?/var/folders/[^'\"\s/]+/[^'\"\s/]+/T/(?:ff_exec_[^'\"\s/]+|tmp[^'\"\s/]+)|"
    r"(?<![A-Za-z0-9_.-])(?:/private)?/tmp/(?:ff_exec_[^'\"\s/]+|tmp[A-Za-z0-9_]+)"
)
_TMPDIR_TRUNC_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/private)?/var/folders(?:/[^'\"\s),;:]*)?|"
    r"(?:[A-Za-z0-9_]+/)?T/(?:ff_exec_[^'\"\s/]+|tmp[A-Za-z0-9_]+)"
)
_HOME_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:/Users|/home)/[^'\"\s),;:]+")


def scrub_detail_paths(text: str) -> str:
    """Scrub ephemeral sandbox temp paths and host user home paths from `detail` strings."""
    if not text:
        return text
    cleaned = _TMPDIR_FULL_RE.sub("<TMPDIR>", text)
    cleaned = _TMPDIR_TRUNC_RE.sub("<TMPDIR>", cleaned)
    cleaned = _HOME_PATH_RE.sub("<HOME>", cleaned)
    return cleaned


def extract_prose_candidate(text: str) -> str:
    """Extract normalized answer from prose using the exact `run_official_eval.py` heuristic."""
    ext = extract_answer_from_prose(text)
    if not ext and text:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        ext = lines[-1] if lines else ""
    return normalize_answer(ext)


def normalize_aime_integer(raw: str) -> str:
    """Return canonical integer string '0'..'999' if `raw` represents a valid AIME integer, else ''."""
    norm = normalize_answer(raw)
    if not norm:
        return ""
    if re.fullmatch(r"\d+", norm):
        val = int(norm)
        if 0 <= val <= 999:
            return str(val)
    return ""


def grade_aime(predicted: str, expected: str) -> tuple[bool, str]:
    """Grade an AIME 2025 prediction against the exact 0..999 integer answer."""
    norm_got = normalize_answer(predicted)
    norm_exp = str(int(normalize_answer(expected)))
    if not norm_got:
        return False, f"empty (expected {norm_exp})"
    if re.fullmatch(r"\d+", norm_got):
        val = int(norm_got)
        if 0 <= val <= 999 and str(val) == norm_exp:
            return True, f"OK ({val})"
        return False, f"WRONG: got={norm_got!r} expected={norm_exp!r}"
    if _answers_equivalent(norm_got, norm_exp):
        return True, f"OK ({norm_got})"
    return False, f"WRONG: got={norm_got!r} expected={norm_exp!r}"
