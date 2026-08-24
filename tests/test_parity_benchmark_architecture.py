"""Architecture guards for the cortheon.benchmark decomposition.

Round 8 moved the 3,283-line god file into the repository-only
``cortheon.parity_benchmark_core`` subpackage behind a stable facade. These
checks pin the split's structural invariants: size budgets, single-owner
definitions, the exact original compatibility surface, the facade-level
monkeypatch seams (``call_contender``, ``_post_json``, ``datetime``), the
repository-only distribution boundary, and deterministic case-bank behavior.
"""

import ast
import hashlib
import importlib
import json
from pathlib import Path
from unittest import mock

import cortheon.benchmark as facade
from cortheon.benchmark import (
    Contender,
    ModelResult,
    _builtin_cases,
    call_contender,
    run_benchmark,
    run_blind_submissions,
    select_case_bank,
)
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.parity import evaluation_schedule_hash

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "src" / "cortheon" / "parity_benchmark_core"
FACADE = ROOT / "src" / "cortheon" / "benchmark.py"

# Every top-level definition of the pre-round-8 benchmark.py god file, exactly
# once each, re-exported identity-preserving from the implementation package.
ORIGINAL_SURFACE = frozenset(
    {
        "Contender",
        "LoadedCasePack",
        "ModelResult",
        "_api_endpoint",
        "_benchmark_input_sha256",
        "_builtin_cases",
        "_call_cli_contender",
        "_candidate_identity",
        "_canonical_blind_submission",
        "_case_bank_hash",
        "_case_documents",
        "_case_has_frozen_oracle",
        "_case_pack_metadata",
        "_classification",
        "_completion_origin",
        "_contender_family",
        "_contender_messages",
        "_contenders",
        "_cortheon_outcome",
        "_extract_patch",
        "_frontier_tools",
        "_grade_patch_in_sandbox",
        "_input_symmetry",
        "_integer_token_count",
        "_load_case_pack",
        "_load_cases",
        "_load_public_case_pack",
        "_messages_with_documents",
        "_metric_float",
        "_nested_metric",
        "_normalize_cases",
        "_observed_model_id",
        "_observed_verdict",
        "_paired_candidate_comparisons",
        "_paired_promotion_statistics",
        "_paired_statistics",
        "_parse_cli_spec",
        "_parse_pricing",
        "_percentile",
        "_post_json",
        "_rate",
        "_ratio_gate",
        "_report_candidate_alias",
        "_report_candidate_summary",
        "_report_selection_hash",
        "_resolve_live_grader",
        "_responses_text",
        "_result_cost",
        "_run_blind_submission_command",
        "_run_sandbox_tests",
        "_sandbox_image",
        "_stable_integer_seed",
        "_summarize_candidate",
        "_summarize_slice",
        "_validate_patch_fixture",
        "_visible_input_sha256",
        "attest_blind_submission",
        "build_parser",
        "call_contender",
        "evaluate_promotion",
        "grade_answer",
        "main",
        "run_benchmark",
        "run_blind_submissions",
        "select_case_bank",
    }
)


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _owners() -> dict[str, str]:
    owners: dict[str, str] = {}
    for path in sorted(CORE.glob("*.py")):
        for name in _top_level_definitions(path):
            assert name not in owners, (name, owners[name], path.name)
            owners[name] = path.stem
    return owners


def test_facade_is_a_small_stable_surface() -> None:
    assert len(FACADE.read_text(encoding="utf-8").splitlines()) <= 250
    # The facade owns no implementation definitions; it only re-exports.
    assert not _top_level_definitions(FACADE) - {"__all__"}


def test_core_modules_respect_size_and_single_ownership() -> None:
    modules = sorted(CORE.glob("*.py"))
    assert len(modules) >= 10
    for path in modules:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path
    owners = _owners()
    assert owners.keys() >= ORIGINAL_SURFACE


def test_facade_reexports_the_complete_original_surface() -> None:
    # 65 is the independently measured count of top-level definitions in the
    # pre-round-8 snapshot; a drift here means a definition was lost or added.
    assert len(ORIGINAL_SURFACE) == 65
    assert set(facade.__all__) == ORIGINAL_SURFACE | {"UTC", "datetime"}
    # Runtime attributes and __all__ each cover the 65 originals exactly,
    # with only the UTC/datetime monkeypatch seams as extras.
    runtime = {name for name in vars(facade) if not name.startswith("__")}
    assert runtime == ORIGINAL_SURFACE | {"UTC", "datetime"}
    assert set(facade.__all__) == runtime
    owners = _owners()
    for name in ORIGINAL_SURFACE:
        assert getattr(facade, name) is getattr(
            importlib.import_module(f"cortheon.parity_benchmark_core.{owners[name]}"),
            name,
        ), name  # The facade binds each original name exactly once as an import and lists
    # it exactly once in the explicit compatibility surface.
    tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    imported: list[str] = []
    explicit: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.extend(alias.name for alias in node.names)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            explicit.extend(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    for name in ORIGINAL_SURFACE:
        assert imported.count(name) == 1, name
        assert explicit.count(name) == 1, name
    assert set(explicit) == set(facade.__all__)


def test_rate_behavior_and_identity_match_the_original() -> None:
    from cortheon.parity_benchmark_core.metrics import _rate as core_rate

    assert facade._rate is core_rate
    assert facade._rate(1, 4) == 0.25
    assert facade._rate(2, 3) == 0.6667
    assert facade._rate(0, 0) is None
    assert facade._rate(5, 0) is None


def test_dataclass_identity_is_preserved() -> None:
    assert Contender.__name__ == "Contender"
    assert Contender.__module__ == "cortheon.parity_benchmark_core.models"
    assert type(Contender.__slots__) is tuple and len(Contender.__slots__) == 12
    instance = ModelResult(
        answer="a",
        latency_ms=1.0,
        metadata={},
        evaluator_outcome=EvaluationOutcome("openai_chat", "success", "chat_finish_reason", "stop"),
    )
    assert (instance.answer, instance.latency_ms, instance.metadata) == ("a", 1.0, {})


def test_parity_benchmark_core_stays_repository_only() -> None:
    for config in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        assert "parity_benchmark_core" not in (ROOT / config).read_text(encoding="utf-8")


def test_facade_level_call_contender_patch_drives_both_run_paths() -> None:
    case = {
        "id": "seam_case",
        "category": "custom",
        "domain": "custom",
        "difficulty": "medium",
        "prompt": "p",
        "expected_verdict": "allow",
        "grader": {"type": "patterns", "required_patterns": [r"ok"], "forbidden_patterns": []},
        "documents": [],
    }
    contender = Contender("a", "stock", "http://x", "m", "k")
    with mock.patch(
        "cortheon.benchmark.call_contender",
        return_value=ModelResult(
            answer="ok",
            latency_ms=1.0,
            metadata={},
            evaluator_outcome=EvaluationOutcome(
                "openai_chat", "success", "chat_finish_reason", "stop"
            ),
        ),
    ) as fake:
        report = run_benchmark(
            [contender],
            [case],
            repetitions=1,
            seed=7,
            timeout=1,
            max_tokens=5,
            include_answers=False,
        )
    assert fake.call_count == 1
    assert report["rows"][0]["passed"] is True

    blind_case = {key: case[key] for key in ("id", "category", "domain", "difficulty", "prompt")}
    blind_case["documents"] = []
    bank = {
        "execution_seed": 7,
        "execution_repetitions": 1,
        "scheduled_contenders": ["a"],
        "precommitted_schedule_sha256": evaluation_schedule_hash(["seam_case"], ["a"], 1, 7),
    }
    with mock.patch(
        "cortheon.benchmark.call_contender",
        return_value=ModelResult(
            answer="ok",
            latency_ms=1.0,
            metadata={},
            evaluator_outcome=EvaluationOutcome(
                "openai_chat", "success", "chat_finish_reason", "stop"
            ),
        ),
    ) as fake:
        artifact = run_blind_submissions(
            [contender],
            [blind_case],
            repetitions=1,
            seed=7,
            timeout=1,
            max_tokens=5,
            case_bank=dict(bank),
        )
    assert fake.call_count == 1
    assert artifact["rows"][0]["answer"]["text"] == "ok"


def test_facade_level_post_json_patch_drives_call_contender() -> None:
    contender = Contender("a", "stock", "http://x", "m", "k")
    with mock.patch(
        "cortheon.benchmark._post_json",
        return_value={"choices": [{"message": {"content": "seam"}}]},
    ) as fake:
        result = call_contender(
            contender, {"prompt": "hi", "documents": []}, timeout=1, max_tokens=5
        )
    assert fake.call_count == 1
    assert result.answer == "seam"


def test_facade_level_datetime_patch_drives_generated_at() -> None:
    from datetime import datetime

    class _Frozen:
        @classmethod
        def now(cls, tz):
            return datetime(2020, 5, 1, 12, 0, tzinfo=tz)

    contender = Contender("a", "stock", "http://x", "m", "k")
    case = {
        "id": "clock_case",
        "category": "custom",
        "domain": "custom",
        "difficulty": "medium",
        "prompt": "p",
        "expected_verdict": "allow",
        "grader": {"type": "patterns", "required_patterns": [r"ok"], "forbidden_patterns": []},
        "documents": [],
    }
    with (
        mock.patch("cortheon.benchmark.call_contender", side_effect=RuntimeError("down")),
        mock.patch("cortheon.benchmark.datetime", _Frozen),
    ):
        report = run_benchmark(
            [contender],
            [case],
            repetitions=1,
            seed=7,
            timeout=1,
            max_tokens=5,
            include_answers=False,
        )
    assert report["generated_at"] == "2020-05-01T12:00:00+00:00"


def test_builtin_case_bank_is_deterministic_and_stable() -> None:
    cases = _builtin_cases()
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Pin the claim-grade built-in bank: case ids, order, and content are stable.
    assert digest == _BUILTIN_CASES_SHA256
    assert _builtin_cases() == cases
    for seed in (0, 7, 13, 42):
        assert select_case_bank(
            cases,
            split="heldout",
            seed=seed,
            holdout_fraction=0.3,
            rotation_index=0,
            rotation_size=0,
        ) == select_case_bank(
            _builtin_cases(),
            split="heldout",
            seed=seed,
            holdout_fraction=0.3,
            rotation_index=0,
            rotation_size=0,
        )


# SHA-256 of the canonical JSON of _builtin_cases(), pinned against the
# pre-round-8 snapshot: the split must never change case ids, order, or content.
_BUILTIN_CASES_SHA256 = "327746e9a079498e34ab045027746fe34f420a8f1b416ceb1b47d99ed52ef61e"


def test_all_core_modules_import() -> None:
    for path in sorted(CORE.glob("*.py")):
        importlib.import_module(f"cortheon.parity_benchmark_core.{path.stem}")
