"""Architecture contract for repository-only prompt evidence generation."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from typing import Any, get_type_hints

import pytest

import cortheon.prompt_evidence as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "prompt_evidence.py"
CORE = ROOT / "src" / "cortheon" / "prompt_evidence_core"
EXPECTED_CONSUMERS: set[str] = set()
FUNCTION_SIGNATURES = {
    "_install_targets": "(spec: 'str') -> 'list[str]'",
    "_candidate_tiers": "(text: 'str') -> 'list[tuple[list[str], bool]]'",
    "detect_packages": "(text: 'str', probe: 'Callable[[str], Any]') -> 'list[str]'",
    "_bound_names": "(text: 'str', package: 'str') -> 'list[str]'",
    "_constructor_params": "(qualname: 'str', symbols: 'list') -> 'str | None'",
    "_comparison_base_version": "(text: 'str') -> 'str | None'",
    "_public_added_names": "(symbols: 'list[Any]') -> 'list[str]'",
    "_official_recovery_facts": "(engine: 'Any', package: 'str', replacements: 'list[str]') -> 'list[str]'",
    "build_evidence": "(engine, text: 'str', packages: 'list[str]') -> 'tuple[str, dict[str, Any]]'",
    "wrap_for_prompt": "(facts: 'str') -> 'str'",
    "wrap_as_assumptions": "(facts: 'str', predicted_failures: 'str' = '') -> 'str'",
    "predict_failures": "(engine, text: 'str', packages: 'list[str]') -> 'str'",
}
PROMPT_HASHES = {
    "EVIDENCE_HEADER": "0b7feb2e6537a9c91923f50c8e36e928ccb2038c815d49848ff8332763162ac1",
    "ASSUMPTION_HEADER": "42145f8de768667ac5ba7f0308fb5c5e2fb7401c8f8d2d4e7900898c44b2b03e",
    "FAILURE_PREDICTOR_HEADER": "063df286e320dc10a14faaed35466d0cdd3d4270090439027f782cc513fadd79",
}


def _imports(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        (isinstance(node, ast.ImportFrom) and node.module == module)
        or (isinstance(node, ast.Import) and any(alias.name == module for alias in node.names))
        for node in ast.walk(tree)
    )


def test_facade_and_core_stay_below_file_limit() -> None:
    for path in [FACADE, *sorted(CORE.glob("*.py"))]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_core_does_not_reenter_facade() -> None:
    assert _imports(FACADE, "cortheon.prompt_evidence_core")
    for path in CORE.glob("*.py"):
        assert not _imports(path, "cortheon.prompt_evidence"), path


def test_direct_source_consumers_are_explicit() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and _imports(path, "cortheon.prompt_evidence")
    }
    assert consumers == EXPECTED_CONSUMERS


def test_prompt_evidence_is_not_in_lean_distribution() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert '"prompt_evidence"' not in setup_text
    assert "src/cortheon/prompt_evidence.py" not in manifest_text


def test_public_signatures_type_hints_and_prompt_bytes_are_stable() -> None:
    for name, expected in FUNCTION_SIGNATURES.items():
        function = getattr(facade, name)
        assert function.__module__ == "cortheon.prompt_evidence"
        assert str(inspect.signature(function)) == expected
        assert get_type_hints(function)
    for name, expected_hash in PROMPT_HASHES.items():
        value = getattr(facade, name)
        assert hashlib.sha256(value.encode()).hexdigest() == expected_hash


def test_detector_keeps_candidate_tier_patch_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = RuntimeError("patched tiers")

    def patched_tiers(_text: str) -> list[tuple[list[str], bool]]:
        raise marker

    def probe(_name: str) -> Any:
        raise AssertionError("probe must not run")

    monkeypatch.setattr(facade, "_candidate_tiers", patched_tiers)
    with pytest.raises(RuntimeError) as raised:
        facade.detect_packages("import httpx", probe)
    assert raised.value is marker
