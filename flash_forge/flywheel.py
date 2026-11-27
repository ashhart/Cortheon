"""Self-Play Verification Flywheel (`export-dpo` / `export-sft`).

Turns Cortheon's mechanically verified test-time rescues into clean, PII-scrubbed
training datasets (DPO preference pairs and OpenAI chat-format SFT traces) ready
for `mlx-lm`, `TRL`, `Unsloth`, or `Axolotl`.

Core workflow:
  1. `record_live_trace(...)` — atomically appends PII-scrubbed verified rescue
     traces to a local JSONL store (default `~/.cortheon/flywheel.jsonl`).
  2. `harvest_from_results_dir(...)` — harvests every verified hard-negative flip
     (baseline or Wave-1 FAILED -> Cortheon PASSED) from paired
     `*_vanilla.json` + `*_cortheon.json` benchmarks, multi-arm ablation JSONs
     (e.g. `official_aime2025_ablation_glm53.json`), and live `*.jsonl` traces,
     optionally joining full prompts from `datasets_dir`.
  3. `export_dpo_jsonl(...)` / `export_sft_jsonl(...)` — writes deduplicated,
     PII-scrubbed training files atomically and returns structured `ExportStats`.
  4. `format_export_summary(stats)` — prints a crisp terminal summary with
     harvest/dedup counts, domain breakdown, and ready-to-run fine-tuning hints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


DEFAULT_FLYWHEEL_LOG = Path.home() / ".cortheon" / "flywheel.jsonl"

# ---------------------------------------------------------------------------
# PII Scrubbing
# ---------------------------------------------------------------------------

_MAC_VAR_FOLDERS_RE = re.compile(
    r"(?:/private)?/var/folders/[^\s\"'<>|,;)\]]+"
)
_USER_HOME_POSIX_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:Users|home)/[A-Za-z0-9._-]+(?=/|[\s\"'<>|,;)\]:]|$)"
)
_USER_HOME_WIN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:\\Users\\[A-Za-z0-9._-]+(?=\\|[\s\"'<>|,;)\]:]|$)"
)
_PRIVATE_IP_RE = re.compile(
    r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
_SECRET_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,})\b"
)


def scrub_pii(text: str) -> str:
    """Scrub local user paths, macOS ephemeral temp dirs, private LAN IPs, and API keys."""
    if not text:
        return ""
    out = _MAC_VAR_FOLDERS_RE.sub("<TMPDIR>", text)
    out = _USER_HOME_POSIX_RE.sub("<HOME>", out)
    out = _USER_HOME_WIN_RE.sub("<HOME>", out)
    out = _PRIVATE_IP_RE.sub("<PRIVATE_IP>", out)
    out = _SECRET_TOKEN_RE.sub("<REDACTED_SECRET>", out)
    return out


def _scrub_obj(obj: Any) -> Any:
    """Recursively scrub PII from strings inside nested JSON-serializable objects."""
    if isinstance(obj, str):
        return scrub_pii(obj)
    if isinstance(obj, list):
        return [_scrub_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return [_scrub_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {scrub_pii(str(k)): _scrub_obj(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FlywheelPair:
    """A single verified preference / SFT training record harvested by Cortheon."""

    prompt: str
    chosen: str
    rejected: str = ""
    domain: str = "general"
    source: str = ""
    model: str = ""
    chosen_verified: bool = True
    rejected_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def scrubbed(self) -> FlywheelPair:
        return FlywheelPair(
            prompt=scrub_pii(self.prompt).strip(),
            chosen=scrub_pii(self.chosen).strip(),
            rejected=scrub_pii(self.rejected).strip(),
            domain=scrub_pii(self.domain or "general").strip() or "general",
            source=scrub_pii(self.source).strip(),
            model=scrub_pii(self.model).strip(),
            chosen_verified=bool(self.chosen_verified),
            rejected_reason=scrub_pii(self.rejected_reason).strip(),
            metadata=_scrub_obj(self.metadata) if isinstance(self.metadata, dict) else {},
        )

    def to_dpo_dict(self) -> dict[str, Any]:
        s = self.scrubbed()
        return {
            "prompt": s.prompt,
            "chosen": s.chosen,
            "rejected": s.rejected,
            "domain": s.domain,
            "source": s.source,
        }

    def to_sft_dict(self, system_prompt: Optional[str] = None) -> dict[str, Any]:
        s = self.scrubbed()
        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": scrub_pii(system_prompt).strip()})
        messages.append({"role": "user", "content": s.prompt})
        messages.append({"role": "assistant", "content": s.chosen})
        return {"messages": messages}

    def to_record_dict(self) -> dict[str, Any]:
        s = self.scrubbed()
        return {
            "prompt": s.prompt,
            "chosen": s.chosen,
            "rejected": s.rejected,
            "domain": s.domain,
            "source": s.source,
            "model": s.model,
            "chosen_verified": s.chosen_verified,
            "rejected_reason": s.rejected_reason,
            "metadata": s.metadata,
        }


@dataclass
class ExportStats:
    """Summary counts and breakdowns returned by `export_dpo_jsonl` / `export_sft_jsonl`."""

    format: str
    out_path: str
    total_input: int = 0
    exported: int = 0
    deduplicated: int = 0
    skipped_unverified: int = 0
    skipped_empty: int = 0
    domain_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)

    @property
    def pairs_harvested(self) -> int:
        return self.total_input

    @property
    def written(self) -> int:
        return self.exported

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "out_path": self.out_path,
            "total_input": self.total_input,
            "exported": self.exported,
            "deduplicated": self.deduplicated,
            "skipped_unverified": self.skipped_unverified,
            "skipped_empty": self.skipped_empty,
            "domain_counts": dict(self.domain_counts),
            "source_counts": dict(self.source_counts),
            "model_counts": dict(self.model_counts),
        }


# ---------------------------------------------------------------------------
# Live Trace Recording (Atomic, Locked, PII-Scrubbed JSONL Append)
# ---------------------------------------------------------------------------


def record_live_trace(
    log_path: str | Path | None = None,
    *,
    prompt: str,
    chosen: str,
    rejected: str = "",
    domain: str = "general",
    model: str = "",
    chosen_verified: bool = True,
    rejected_reason: str = "",
    source: str = "live_trace",
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Atomically append one PII-scrubbed flywheel trace record to a local JSONL store."""
    target = Path(log_path).expanduser() if log_path is not None else DEFAULT_FLYWHEEL_LOG
    target.parent.mkdir(parents=True, exist_ok=True)

    pair = FlywheelPair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        domain=domain,
        source=source,
        model=model,
        chosen_verified=chosen_verified,
        rejected_reason=rejected_reason,
        metadata=dict(metadata) if metadata else {},
    ).scrubbed()

    record = pair.to_record_dict()
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(target), flags, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(payload)
        written = 0
        while written < len(payload):
            n = os.write(fd, view[written:])
            if n <= 0:  # pragma: no cover - defensive
                break
            written += n
        os.fsync(fd)
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - defensive
                pass
        os.close(fd)
    return target


# ---------------------------------------------------------------------------
# Prompt Dataset Indexing & Benchmark Harvesting
# ---------------------------------------------------------------------------

_OK_ANSWER_RE = re.compile(r"\bOK\s*\((.+?)\)\s*$")
_WRONG_GOT_SINGLE_RE = re.compile(r"WRONG:\s*got='([^']*)'")
_WRONG_GOT_DOUBLE_RE = re.compile(r'WRONG:\s*got="([^"]*)"')
_WRONG_EXP_RE = re.compile(r"expected='([^']*)'")

_CHOSEN_FULL_KEYS = (
    "chosen",
    "winner_code",
    "code",
    "solution",
    "final_explanation",
    "response",
    "explanation",
    "answer",
    "output",
)
_REJECTED_FULL_KEYS = (
    "rejected",
    "baseline_code",
    "wave1_code",
    "code",
    "solution",
    "response",
    "explanation",
    "answer",
    "output",
)

_TREATMENT_ARM_PRIORITY: dict[str, int] = {
    "cortheon_full": 5,
    "cortheon_world_compiler": 5,
    "cortheon": 4,
    "forge": 4,
    "cortheon_w1": 2,
}

_BASELINE_ARM_ORDER: dict[str, int] = {
    "vanilla_b1": 0,
    "vanilla": 1,
    "baseline": 2,
    "cortheon_w1": 3,
    "best4_sample": 4,
    "maj4_prose": 5,
    "maj4_self_consistency": 6,
    "maj8_prose": 7,
    "pal_b1": 8,
}


def _row_pid(row: dict[str, Any]) -> str:
    for k in ("pid", "id", "task_id", "question_id", "problem_id"):
        val = row.get(k)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _row_passed(row: dict[str, Any]) -> bool:
    if "passed" in row:
        return bool(row["passed"])
    if "correct" in row:
        return bool(row["correct"])
    if "chosen_verified" in row:
        return bool(row["chosen_verified"])
    return False


def _infer_domain(dataset: str = "", row: Optional[dict[str, Any]] = None) -> str:
    r = row or {}
    for k in ("domain", "track", "category"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            low = v.strip().lower()
            if low.startswith("logic"):
                return "logic"
            if low.startswith("math") or low in ("algebra", "geometry", "combinatorics", "number_theory"):
                return "math"
            if low.startswith(("code", "debug", "algo")) or low in ("coding", "algorithms", "programming"):
                return "code"
            if low.startswith(("sys", "arch")) or low in ("design", "open_ended", "think"):
                return "systems"
            return low
    extra = r.get("extra") if isinstance(r.get("extra"), dict) else {}
    routed = str(r.get("routed_mode") or extra.get("routed_mode") or "").strip().lower()
    ds = (dataset or "").lower()
    if "aime" in ds or "math" in ds:
        return "math"
    if any(tok in ds for tok in ("bigcodebench", "bcb", "livecodebench", "lcb", "humaneval", "code", "swe")):
        return "code"
    if any(tok in ds for tok in ("open_ended", "systems", "world_compiler")):
        return "systems"
    if routed == "code":
        return "code"
    if routed == "reason":
        return "math" if "aime" in ds else "reasoning"
    if routed == "think":
        return "systems"
    if "gauntlet" in ds:
        return "reasoning"
    return "general"


def _load_prompt_index(datasets_dir: Optional[str | Path]) -> dict[str, dict[str, str]]:
    """Index problem prompts by normalized ID from JSON/JSONL files in `datasets_dir`."""
    index: dict[str, dict[str, str]] = {}
    if not datasets_dir:
        return index
    root = Path(datasets_dir).expanduser()
    if not root.exists():
        return index

    files: list[Path]
    if root.is_file():
        files = [root]
    else:
        files = sorted(
            p
            for p in list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))
            if not any(part.startswith(".") for part in p.relative_to(root).parts)
        )

    def _register(key: str, default_prompt: str, instruct_prompt: str = "") -> None:
        if not key or not (default_prompt or instruct_prompt):
            return
        entry = {"default": default_prompt or instruct_prompt}
        if instruct_prompt:
            entry["instruct"] = instruct_prompt
        variants = {key, key.replace("/", "_"), key.replace("_", "/"), key.lower()}
        for v in variants:
            index.setdefault(v, entry)

    for fp in files:
        fname = fp.name.lower()
        rows: list[Any] = []
        try:
            if fp.suffix.lower() == ".jsonl":
                with fp.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            else:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    rows = raw
                elif isinstance(raw, dict):
                    if "problems" in raw and isinstance(raw["problems"], list):
                        rows = raw["problems"]
                    else:
                        for k, v in raw.items():
                            if isinstance(v, str):
                                _register(str(k), v)
                            elif isinstance(v, dict):
                                item = dict(v)
                                item.setdefault("id", k)
                                rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue

        for item in rows:
            if not isinstance(item, dict):
                continue
            # AIME-style dataset entries
            if "problem_idx" in item and "problem" in item:
                try:
                    idx = int(item["problem_idx"])
                except (TypeError, ValueError):
                    idx = -1
                prob = str(item["problem"]).strip()
                if idx >= 0 and prob:
                    if "aime" in fname:
                        _register(f"AIME25_{idx:02d}", prob)
                        _register(f"AIME25_{idx}", prob)
                    _register(str(idx), prob)
                continue

            # BigCodeBench-style dataset entries
            if "task_id" in item and ("complete_prompt" in item or "instruct_prompt" in item):
                tid = str(item["task_id"]).strip()
                comp = str(item.get("complete_prompt") or "").strip()
                inst = str(item.get("instruct_prompt") or "").strip()
                _register(tid, comp or inst, instruct_prompt=inst)
                continue

            # LiveCodeBench-style dataset entries
            if "question_id" in item and "question_content" in item:
                qid = str(item["question_id"]).strip()
                title = str(item.get("question_title") or "").strip()
                content = str(item.get("question_content") or "").strip()
                prompt_text = f"{title}\n\n{content}".strip() if title and title not in content[:120] else content
                _register(qid, prompt_text)
                continue

            # Generic prompt dataset entries
            pid = _row_pid(item)
            if not pid:
                continue
            prompt_val = ""
            for pk in ("prompt", "problem", "question_content", "complete_prompt", "instruct_prompt", "question", "text"):
                v = item.get(pk)
                if isinstance(v, str) and v.strip():
                    prompt_val = v.strip()
                    break
            if prompt_val:
                _register(pid, prompt_val, instruct_prompt=str(item.get("instruct_prompt") or "").strip())

    return index


def _resolve_prompt(
    pid: str,
    dataset: str,
    prompt_index: dict[str, dict[str, str]],
    *rows: Optional[dict[str, Any]],
) -> str:
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in ("prompt", "problem", "question", "task"):
            val = r.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()

    if pid and prompt_index:
        for cand_key in (pid, pid.replace("/", "_"), pid.replace("_", "/"), pid.lower()):
            hit = prompt_index.get(cand_key)
            if hit:
                if "instruct" in (dataset or "").lower() and hit.get("instruct"):
                    return hit["instruct"]
                return hit["default"]

    title = ""
    for r in rows:
        if isinstance(r, dict) and r.get("title"):
            title = str(r["title"]).strip()
            break
    ds_prefix = f"[{dataset}] " if dataset else ""
    if pid and title:
        return f"{ds_prefix}{pid}: {title}"
    if pid:
        return f"{ds_prefix}{pid}"
    return title or ds_prefix.strip()


def _format_answer_text(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    if re.match(r"^ANSWER\s*[:\s]", s, re.IGNORECASE):
        return s
    if len(s) > 80 and " " in s:
        return s
    return f"ANSWER: {s}"


def _format_failure_text(detail: str) -> str:
    s = detail.strip()
    if not s:
        return ""
    if re.match(r"^(?:FAIL|FAILED|WA|TLE|RE|ERROR|TIMEOUT|SyntaxError)\b", s):
        return s
    return f"FAILED: {s}"


def _extract_chosen_text(row: dict[str, Any]) -> tuple[str, str]:
    """Return `(chosen_text, trace_kind)` from a verified/passing result row."""
    for k in _CHOSEN_FULL_KEYS:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip(), "full_trace"

    pred = row.get("predicted")
    if pred is not None and str(pred).strip():
        return _format_answer_text(str(pred)), "answer_pair"

    detail = str(row.get("detail") or "").strip()
    m_ok = _OK_ANSWER_RE.search(detail)
    if m_ok:
        return _format_answer_text(m_ok.group(1)), "answer_pair"

    exp = row.get("expected")
    if exp is not None and str(exp).strip():
        return _format_answer_text(str(exp)), "answer_pair"

    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    winner = str(extra.get("winner") or "").strip()
    if detail:
        tag = f"VERIFIED [{winner}]" if winner else "VERIFIED"
        return f"{tag}: {detail}", "benchmark_flip"

    checks = row.get("checks")
    if isinstance(checks, dict) and checks:
        passed_checks = sorted(k for k, v in checks.items() if v)
        score = row.get("score")
        score_txt = f"score={score}, " if score is not None else ""
        return f"VERIFIED ({score_txt}checks={', '.join(passed_checks)})", "benchmark_flip"

    return "", "empty"


def _extract_rejected_text(row: dict[str, Any]) -> tuple[str, str]:
    """Return `(rejected_text, rejected_reason)` from a failing baseline/Wave-1 row."""
    detail = str(row.get("detail") or "").strip()
    err = str(row.get("error") or "").strip()
    reason = detail or err

    for k in _REJECTED_FULL_KEYS:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip(), reason or "failed_verification"

    pred = row.get("predicted")
    if pred is not None and str(pred).strip():
        pred_s = str(pred).strip()
        exp = str(row.get("expected") or "").strip()
        r_msg = reason or (f"wrong_answer: got={pred_s!r} expected={exp!r}" if exp else f"wrong_answer: {pred_s!r}")
        return _format_answer_text(pred_s), r_msg

    if detail:
        m_s = _WRONG_GOT_SINGLE_RE.search(detail)
        m_d = _WRONG_GOT_DOUBLE_RE.search(detail)
        m_got = m_s or m_d
        if m_got and m_got.group(1).strip():
            return _format_answer_text(m_got.group(1)), detail
        if detail.lower().startswith("empty"):
            return "<empty response>", detail
        return _format_failure_text(detail), detail

    checks = row.get("checks")
    missed = row.get("missed")
    if isinstance(missed, list) and missed:
        miss_txt = ", ".join(str(x) for x in missed)
        score = row.get("score")
        score_txt = f"score={score}, " if score is not None else ""
        msg = f"FAILED ({score_txt}missed={miss_txt})"
        return msg, msg
    if isinstance(checks, dict) and checks:
        failed_checks = sorted(k for k, v in checks.items() if not v)
        score = row.get("score")
        score_txt = f"score={score}, " if score is not None else ""
        msg = f"FAILED ({score_txt}missed={', '.join(failed_checks)})"
        return msg, msg

    if err:
        return f"ERROR: {err}", err
    return "<failed>", "failed_verification"


def _harvest_jsonl_file(path: Path) -> list[FlywheelPair]:
    pairs: list[FlywheelPair] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return pairs

    for idx, line in enumerate(lines, 1):
        raw = line.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue

        prompt = ""
        for k in ("prompt", "problem", "question"):
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                prompt = v.strip()
                break
        chosen = ""
        for k in _CHOSEN_FULL_KEYS:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                chosen = v.strip()
                break
        rejected = ""
        for k in ("rejected", "baseline_code", "baseline_answer", "wave1_code", "failed_attempt"):
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                rejected = v.strip()
                break

        if not prompt or not chosen:
            continue

        meta = dict(rec.get("metadata")) if isinstance(rec.get("metadata"), dict) else {}
        meta.setdefault("line", idx)
        pair = FlywheelPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            domain=str(rec.get("domain") or _infer_domain("", rec)),
            source=str(rec.get("source") or path.name),
            model=str(rec.get("model") or ""),
            chosen_verified=bool(rec.get("chosen_verified", rec.get("verified", True))),
            rejected_reason=str(rec.get("rejected_reason") or ""),
            metadata=meta,
        ).scrubbed()
        pairs.append(pair)
    return pairs


def _harvest_paired_runs(
    vanilla_path: Path,
    cortheon_path: Path,
    prompt_index: dict[str, dict[str, str]],
) -> list[FlywheelPair]:
    try:
        v_data = json.loads(vanilla_path.read_text(encoding="utf-8"))
        c_data = json.loads(cortheon_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(v_data, dict) or not isinstance(c_data, dict):
        return []
    v_rows = v_data.get("results")
    c_rows = c_data.get("results")
    if not isinstance(v_rows, list) or not isinstance(c_rows, list):
        return []

    dataset = str(c_data.get("dataset") or v_data.get("dataset") or "")
    model = str(c_data.get("model") or v_data.get("model") or "")
    v_by_pid = {_row_pid(r): r for r in v_rows if isinstance(r, dict) and _row_pid(r)}

    pairs: list[FlywheelPair] = []
    source_label = f"{cortheon_path.name}:vanilla->{c_data.get('arm') or 'cortheon'}"

    for c_row in c_rows:
        if not isinstance(c_row, dict):
            continue
        pid = _row_pid(c_row)
        if not pid:
            continue
        c_ok = _row_passed(c_row)
        if not c_ok:
            continue

        v_row = v_by_pid.get(pid)
        v_ok = _row_passed(v_row) if v_row is not None else None
        chosen_text, trace_kind = _extract_chosen_text(c_row)
        if not chosen_text:
            continue
        prompt = _resolve_prompt(pid, dataset, prompt_index, c_row, v_row)
        if not prompt:
            continue
        domain = _infer_domain(dataset, c_row)

        if v_row is not None and v_ok is False:
            rejected_text, rejected_reason = _extract_rejected_text(v_row)
            meta: dict[str, Any] = {
                "pid": pid,
                "dataset": dataset,
                "flip_type": "baseline_to_cortheon",
                "trace_kind": trace_kind,
                "baseline_file": vanilla_path.name,
                "cortheon_file": cortheon_path.name,
            }
            if isinstance(c_row.get("extra"), dict):
                meta["cortheon_extra"] = c_row["extra"]
            pairs.append(
                FlywheelPair(
                    prompt=prompt,
                    chosen=chosen_text,
                    rejected=rejected_text,
                    domain=domain,
                    source=source_label,
                    model=model,
                    chosen_verified=True,
                    rejected_reason=rejected_reason,
                    metadata=meta,
                ).scrubbed()
            )
        elif trace_kind == "full_trace":
            # Full verified code/proof trace even when baseline also passed (useful for SFT!)
            pairs.append(
                FlywheelPair(
                    prompt=prompt,
                    chosen=chosen_text,
                    rejected="",
                    domain=domain,
                    source=cortheon_path.name,
                    model=model,
                    chosen_verified=True,
                    rejected_reason="",
                    metadata={
                        "pid": pid,
                        "dataset": dataset,
                        "flip_type": "verified_trace",
                        "trace_kind": trace_kind,
                        "cortheon_file": cortheon_path.name,
                    },
                ).scrubbed()
            )
    return pairs


def _is_treatment_arm(arm: str) -> bool:
    low = arm.lower()
    return low in _TREATMENT_ARM_PRIORITY or "cortheon" in low or low == "forge"


def _harvest_ablation_file(
    path: Path,
    data: dict[str, Any],
    prompt_index: dict[str, dict[str, str]],
) -> list[FlywheelPair]:
    results = data.get("results")
    if not isinstance(results, dict):
        return []

    dataset = str(data.get("dataset") or "")
    model = str(data.get("model") or "")
    arm_names = list(results.keys())
    treatment_arms = sorted(
        [a for a in arm_names if _is_treatment_arm(a)],
        key=lambda a: (-_TREATMENT_ARM_PRIORITY.get(a.lower(), 3), a),
    )
    if not treatment_arms:
        return []

    # Index rows by arm -> pid -> row
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    all_pids: list[str] = []
    seen_pids: set[str] = set()
    for arm, rows in results.items():
        if not isinstance(rows, list):
            continue
        amap: dict[str, dict[str, Any]] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            pid = _row_pid(r)
            if pid:
                amap[pid] = r
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    all_pids.append(pid)
        by_arm[arm] = amap

    pairs: list[FlywheelPair] = []
    for pid in all_pids:
        # Find highest-priority passing Cortheon treatment arm for this problem
        winner_arm = ""
        winner_row: Optional[dict[str, Any]] = None
        for t_arm in treatment_arms:
            cand = by_arm.get(t_arm, {}).get(pid)
            if cand is not None and _row_passed(cand):
                winner_arm = t_arm
                winner_row = cand
                break
        if winner_row is None:
            continue

        chosen_text, trace_kind = _extract_chosen_text(winner_row)
        if not chosen_text:
            continue

        # Candidate baseline / earlier-wave arms that FAILED on this problem
        winner_prio = _TREATMENT_ARM_PRIORITY.get(winner_arm.lower(), 3)
        loser_arms: list[str] = []
        for arm in arm_names:
            if arm == winner_arm:
                continue
            # Allow strictly lower-priority Cortheon wave (e.g. cortheon_w1 -> cortheon_full)
            # AND all non-Cortheon baselines (vanilla_b1, maj4, maj8, pal_b1, best4_sample, etc.)
            if _is_treatment_arm(arm) and _TREATMENT_ARM_PRIORITY.get(arm.lower(), 3) >= winner_prio:
                continue
            row = by_arm.get(arm, {}).get(pid)
            if row is not None and not _row_passed(row):
                loser_arms.append(arm)

        loser_arms.sort(key=lambda a: (_BASELINE_ARM_ORDER.get(a.lower(), 50), a))
        prompt = _resolve_prompt(pid, dataset, prompt_index, winner_row)
        if not prompt:
            continue
        domain = _infer_domain(dataset, winner_row)

        seen_rejected_for_pid: set[str] = set()
        for b_arm in loser_arms:
            b_row = by_arm[b_arm][pid]
            rejected_text, rejected_reason = _extract_rejected_text(b_row)
            if not rejected_text or rejected_text == chosen_text:
                continue
            if rejected_text in seen_rejected_for_pid:
                continue
            seen_rejected_for_pid.add(rejected_text)
            flip_type = "wave1_to_wave2" if _is_treatment_arm(b_arm) else "baseline_to_cortheon"
            pairs.append(
                FlywheelPair(
                    prompt=prompt,
                    chosen=chosen_text,
                    rejected=rejected_text,
                    domain=domain,
                    source=f"{path.name}:{b_arm}->{winner_arm}",
                    model=model,
                    chosen_verified=True,
                    rejected_reason=rejected_reason,
                    metadata={
                        "pid": pid,
                        "dataset": dataset,
                        "baseline_arm": b_arm,
                        "treatment_arm": winner_arm,
                        "flip_type": flip_type,
                        "trace_kind": trace_kind,
                    },
                ).scrubbed()
            )

        if not loser_arms and trace_kind == "full_trace":
            pairs.append(
                FlywheelPair(
                    prompt=prompt,
                    chosen=chosen_text,
                    rejected="",
                    domain=domain,
                    source=f"{path.name}:{winner_arm}",
                    model=model,
                    chosen_verified=True,
                    rejected_reason="",
                    metadata={
                        "pid": pid,
                        "dataset": dataset,
                        "treatment_arm": winner_arm,
                        "flip_type": "verified_trace",
                        "trace_kind": trace_kind,
                    },
                ).scrubbed()
            )
    return pairs


def _harvest_standalone_cortheon_file(
    path: Path,
    data: dict[str, Any],
    prompt_index: dict[str, dict[str, str]],
) -> list[FlywheelPair]:
    """Harvest verified traces or embedded rejected/chosen flips from a single Cortheon JSON."""
    rows = data.get("results")
    if not isinstance(rows, list):
        return []
    arm = str(data.get("arm") or "").lower()
    if arm and not _is_treatment_arm(arm):
        return []

    dataset = str(data.get("dataset") or "")
    model = str(data.get("model") or "")
    pairs: list[FlywheelPair] = []
    for row in rows:
        if not isinstance(row, dict) or not _row_passed(row):
            continue
        pid = _row_pid(row)
        chosen_text, trace_kind = _extract_chosen_text(row)
        if not chosen_text:
            continue

        # Check for an embedded hard negative (e.g. failed Wave-1 candidate stored in row)
        embedded_rej = ""
        for rk in ("rejected", "wave1_code", "baseline_code", "baseline_answer", "failed_attempt"):
            val = row.get(rk)
            if isinstance(val, str) and val.strip():
                embedded_rej = val.strip()
                break

        if not embedded_rej and trace_kind != "full_trace":
            continue

        prompt = _resolve_prompt(pid, dataset, prompt_index, row)
        if not prompt:
            continue
        pairs.append(
            FlywheelPair(
                prompt=prompt,
                chosen=chosen_text,
                rejected=embedded_rej,
                domain=_infer_domain(dataset, row),
                source=path.name,
                model=model,
                chosen_verified=True,
                rejected_reason=str(row.get("rejected_reason") or ("wave1_failed" if embedded_rej else "")),
                metadata={
                    "pid": pid,
                    "dataset": dataset,
                    "flip_type": "embedded_flip" if embedded_rej else "verified_trace",
                    "trace_kind": trace_kind,
                },
            ).scrubbed()
        )
    return pairs


def harvest_from_results_dir(
    results_dir: str | Path,
    datasets_dir: Optional[str | Path] = None,
) -> list[FlywheelPair]:
    """Scan Cortheon benchmark results and JSONL trace logs and extract verified flywheel pairs.

    Supports:
      - Live JSONL trace stores (`flywheel.jsonl` or any `*.jsonl`)
      - Paired `*_vanilla.json` + `*_cortheon.json` (or `*_forge.json`) benchmark runs
      - Multi-arm ablation JSONs (`results: {arm_name: [...]}` such as 6-arm AIME & LCB ablations)
      - Standalone Cortheon result JSONs containing verified code/proof traces or embedded flips
    """
    root = Path(results_dir).expanduser()
    if not root.exists():
        return []

    prompt_index = _load_prompt_index(datasets_dir)
    if root.is_file():
        if root.suffix.lower() == ".jsonl":
            return _harvest_jsonl_file(root)
        try:
            data = json.loads(root.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            return _harvest_ablation_file(root, data, prompt_index)
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return _harvest_standalone_cortheon_file(root, data, prompt_index)
        return []

    jsonl_files = sorted(
        p
        for p in root.rglob("*.jsonl")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    json_files = sorted(
        p
        for p in root.rglob("*.json")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    )

    pairs: list[FlywheelPair] = []

    # 1. Harvest live JSONL trace stores
    for jp in jsonl_files:
        pairs.extend(_harvest_jsonl_file(jp))

    # 2. Match paired *_vanilla.json + *_cortheon.json (or *_forge.json) files in the same directory
    consumed: set[Path] = set()
    by_dir_and_name: dict[tuple[Path, str], Path] = {(p.parent, p.name): p for p in json_files}

    for p in json_files:
        if p.name.endswith("_vanilla.json"):
            prefix = p.name[: -len("_vanilla.json")]
            for suffix in ("_cortheon.json", "_forge.json"):
                partner = by_dir_and_name.get((p.parent, prefix + suffix))
                if partner is not None:
                    pairs.extend(_harvest_paired_runs(p, partner, prompt_index))
                    consumed.add(p)
                    consumed.add(partner)
                    break

    # 3. Harvest remaining JSONs (multi-arm ablations & standalone Cortheon trace JSONs)
    for p in json_files:
        if p in consumed:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "results" not in data:
            continue
        if isinstance(data["results"], dict):
            pairs.extend(_harvest_ablation_file(p, data, prompt_index))
        elif isinstance(data["results"], list):
            pairs.extend(_harvest_standalone_cortheon_file(p, data, prompt_index))

    return pairs


# ---------------------------------------------------------------------------
# Atomic JSONL Exporters (DPO & SFT)
# ---------------------------------------------------------------------------


def _write_jsonl_atomic(out_path: Path, lines: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".flywheel_", suffix=".tmp", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln)
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, out_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def export_dpo_jsonl(
    pairs: Iterable[FlywheelPair],
    out_path: str | Path,
    *,
    verified_only: bool = True,
) -> ExportStats:
    """Export exact-deduplicated, PII-scrubbed DPO JSONL (`prompt`, `chosen`, `rejected`, `domain`, `source`)."""
    target = Path(out_path).expanduser()
    pair_list = list(pairs)
    stats = ExportStats(format="dpo", out_path=str(target), total_input=len(pair_list))

    seen: set[tuple[str, str, str]] = set()
    lines: list[str] = []
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    models: Counter[str] = Counter()

    for raw_pair in pair_list:
        if verified_only and not raw_pair.chosen_verified:
            stats.skipped_unverified += 1
            continue
        clean = raw_pair.scrubbed()
        if not clean.prompt or not clean.chosen or not clean.rejected or clean.chosen == clean.rejected:
            stats.skipped_empty += 1
            continue
        key = (clean.prompt, clean.chosen, clean.rejected)
        if key in seen:
            stats.deduplicated += 1
            continue
        seen.add(key)

        row = clean.to_dpo_dict()
        lines.append(json.dumps(row, ensure_ascii=False))
        stats.exported += 1
        domains[clean.domain] += 1
        if clean.source:
            sources[clean.source] += 1
        if clean.model:
            models[clean.model] += 1

    _write_jsonl_atomic(target, lines)
    stats.domain_counts = dict(domains.most_common())
    stats.source_counts = dict(sources.most_common())
    stats.model_counts = dict(models.most_common())
    return stats


def export_sft_jsonl(
    pairs: Iterable[FlywheelPair],
    out_path: str | Path,
    *,
    verified_only: bool = True,
    system_prompt: Optional[str] = None,
) -> ExportStats:
    """Export exact-deduplicated, PII-scrubbed OpenAI chat-format SFT JSONL for verified solutions."""
    target = Path(out_path).expanduser()
    pair_list = list(pairs)
    stats = ExportStats(format="sft", out_path=str(target), total_input=len(pair_list))

    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    models: Counter[str] = Counter()

    for raw_pair in pair_list:
        if verified_only and not raw_pair.chosen_verified:
            stats.skipped_unverified += 1
            continue
        clean = raw_pair.scrubbed()
        if not clean.prompt or not clean.chosen:
            stats.skipped_empty += 1
            continue
        key = (clean.prompt, clean.chosen)
        if key in seen:
            stats.deduplicated += 1
            continue
        seen.add(key)

        row = clean.to_sft_dict(system_prompt=system_prompt)
        lines.append(json.dumps(row, ensure_ascii=False))
        stats.exported += 1
        domains[clean.domain] += 1
        if clean.source:
            sources[clean.source] += 1
        if clean.model:
            models[clean.model] += 1

    _write_jsonl_atomic(target, lines)
    stats.domain_counts = dict(domains.most_common())
    stats.source_counts = dict(sources.most_common())
    stats.model_counts = dict(models.most_common())
    return stats


# ---------------------------------------------------------------------------
# Terminal Summary Formatting & CLI
# ---------------------------------------------------------------------------


def format_export_summary(stats: ExportStats) -> str:
    """Render a crisp terminal summary with harvest/dedup stats, domains, and training hints."""
    fmt = (stats.format or "dpo").upper()
    out_p = Path(stats.out_path)
    data_dir = str(out_p.parent) if str(out_p.parent) not in ("", ".") else "."
    domain_str = (
        ", ".join(f"{k}={v}" for k, v in stats.domain_counts.items())
        if stats.domain_counts
        else "(none)"
    )

    skip_parts: list[str] = []
    if stats.skipped_unverified:
        skip_parts.append(f"{stats.skipped_unverified} unverified")
    if stats.skipped_empty:
        skip_parts.append(f"{stats.skipped_empty} empty/incomplete")
    skip_suffix = f" (skipped: {', '.join(skip_parts)})" if skip_parts else ""

    lines = [
        f"[Cortheon Flywheel] Exported {fmt} dataset -> {stats.out_path}",
        f"  Harvested   : {stats.total_input}",
        f"  Exported    : {stats.exported}",
        f"  Deduplicated: {stats.deduplicated}{skip_suffix}",
        f"  Domains     : {domain_str}",
    ]
    if fmt == "DPO":
        lines.extend(
            [
                "  Ready to train (TRL DPO):",
                f"    trl dpo --dataset_name {stats.out_path} --model_name_or_path <hf-model>",
                "  Ready to train (MLX LoRA / Unsloth):",
                f"    python -m mlx_lm.lora --model <mlx-model> --train --data {data_dir}",
            ]
        )
    else:
        lines.extend(
            [
                "  Ready to train (MLX LoRA SFT):",
                f"    python -m mlx_lm.lora --model <mlx-model> --train --data {data_dir}",
                "  Ready to train (TRL SFT):",
                f"    trl sft --dataset_name {stats.out_path} --model_name_or_path <hf-model>",
            ]
        )
    return "\n".join(lines)


def _cli_main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m flash_forge.flywheel",
        description="Cortheon Self-Play Verification Flywheel — export DPO / SFT datasets from verified rescues.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for cmd_name, help_text in (
        ("export-dpo", "Harvest verified hard-negative flips and export DPO JSONL"),
        ("export-sft", "Harvest verified chosen traces and export OpenAI chat SFT JSONL"),
    ):
        p = sub.add_parser(cmd_name, help=help_text)
        p.add_argument("--results", required=True, help="Results directory or JSON/JSONL trace file")
        p.add_argument("--datasets", default=None, help="Optional datasets directory for full prompt joining")
        p.add_argument("--out", required=True, help="Output JSONL path")
        p.add_argument("--include-unverified", action="store_true", help="Include unverified records")

    args = parser.parse_args(argv)
    pairs = harvest_from_results_dir(args.results, datasets_dir=args.datasets)
    if args.cmd == "export-dpo":
        stats = export_dpo_jsonl(pairs, args.out, verified_only=not args.include_unverified)
    else:
        stats = export_sft_jsonl(pairs, args.out, verified_only=not args.include_unverified)
    print(format_export_summary(stats))


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
