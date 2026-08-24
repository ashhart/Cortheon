"""Build current source-derived package evidence for raw model prompts."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from typing import Any

from cortheon.code_check import _callable_param_index, is_legacy_path, suggest_replacements
from cortheon.docs_reader import GUIDE_KEYWORDS
from cortheon.prompt_evidence_core import building, detection, failures
from cortheon.prompt_evidence_core import symbols as symbol_helpers
from cortheon.prompt_evidence_core.constants import (
    ASSUMPTION_HEADER,
    COMMON_ALIASES,
    EVIDENCE_HEADER,
    FAILURE_PREDICTOR_HEADER,
    MAX_EVIDENCE_CHARS,
    MAX_PACKAGES,
    MAX_PROBES,
    PROSE_STOPWORDS,
    REVERSE_IMPORT_OVERRIDES,
)
from cortheon.prompt_evidence_core.constants import BACKTICK_RE as _BACKTICK_RE
from cortheon.prompt_evidence_core.constants import BARE_RE as _BARE_RE
from cortheon.prompt_evidence_core.constants import DOTTED_RE as _DOTTED_RE
from cortheon.prompt_evidence_core.constants import FROM_RE as _FROM_RE
from cortheon.prompt_evidence_core.constants import IMPORT_RE as _IMPORT_RE
from cortheon.prompt_evidence_core.constants import INSTALL_RE as _INSTALL_RE
from cortheon.prompt_evidence_core.constants import PACKAGE_LIST_RE as _PACKAGE_LIST_RE
from cortheon.prompt_evidence_core.constants import TLDS as _TLDS
from cortheon.prompt_evidence_core.constants import TOKEN_RE as _TOKEN_RE
from cortheon.prompt_evidence_core.constants import (
    VERSION_COMPARISON_RE as _VERSION_COMPARISON_RE,
)


def _install_targets(spec: str) -> list[str]:
    return detection.install_targets(spec, split=re.split)


def _candidate_tiers(text: str) -> list[tuple[list[str], bool]]:
    """Return package candidates grouped from strongest to weakest signal."""
    return detection.candidate_tiers(
        text,
        import_re=_IMPORT_RE,
        from_re=_FROM_RE,
        install_re=_INSTALL_RE,
        package_list_re=_PACKAGE_LIST_RE,
        dotted_re=_DOTTED_RE,
        tlds=_TLDS,
        backtick_re=_BACKTICK_RE,
        bare_re=_BARE_RE,
        install_targets=_install_targets,
    )


def detect_packages(text: str, probe: Callable[[str], Any]) -> list[str]:
    """Return verified PyPI packages mentioned by the prompt."""
    return detection.detect_packages(
        text,
        probe,
        candidate_tiers=_candidate_tiers,
        stdlib_names=sys.stdlib_module_names,
        prose_stopwords=PROSE_STOPWORDS,
        common_aliases=COMMON_ALIASES,
        reverse_overrides=REVERSE_IMPORT_OVERRIDES,
        fullmatch=re.fullmatch,
        max_probes=MAX_PROBES,
        max_packages=MAX_PACKAGES,
    )


def _bound_names(text: str, package: str) -> list[str]:
    """Return names explicitly bound to the package in the prompt."""
    return detection.bound_names(
        text,
        package,
        reverse_overrides=REVERSE_IMPORT_OVERRIDES,
        common_aliases=COMMON_ALIASES,
        escape=re.escape,
        compile_pattern=re.compile,
        multiline=re.MULTILINE,
        ignorecase=re.IGNORECASE,
    )


def _constructor_params(qualname: str, symbols: list) -> str | None:
    return symbol_helpers.constructor_params(qualname, symbols)


def _comparison_base_version(text: str) -> str | None:
    return symbol_helpers.comparison_base_version(text, pattern=_VERSION_COMPARISON_RE)


def _public_added_names(symbols: list[Any]) -> list[str]:
    return symbol_helpers.public_added_names(symbols)


def _official_recovery_facts(
    engine: Any,
    package: str,
    replacements: list[str],
) -> list[str]:
    return symbol_helpers.official_recovery_facts(
        engine,
        package,
        replacements,
        guide_keywords=GUIDE_KEYWORDS,
    )


def build_evidence(engine, text: str, packages: list[str]) -> tuple[str, dict[str, Any]]:
    """Return bounded current-source facts for prompt-mentioned names."""
    return building.build_evidence(
        engine,
        text,
        packages,
        token_pattern=_TOKEN_RE,
        comparison_base_version=_comparison_base_version,
        public_added_names=_public_added_names,
        is_legacy_path=is_legacy_path,
        constructor_params=_constructor_params,
        official_recovery_facts=_official_recovery_facts,
        bound_names=_bound_names,
        suggest_replacements=suggest_replacements,
        callable_param_index=_callable_param_index,
        max_evidence_chars=MAX_EVIDENCE_CHARS,
    )


def wrap_for_prompt(facts: str) -> str:
    """Return the original evidence block appended to a user message."""
    return f"\n\n[{EVIDENCE_HEADER}\n{facts}]"


def wrap_as_assumptions(facts: str, predicted_failures: str = "") -> str:
    """Inject verified facts as working assumptions and stale-prior warnings."""
    assumptions = f"\n\n[{ASSUMPTION_HEADER}\n{facts}]"
    if predicted_failures:
        assumptions += f"\n\n[{FAILURE_PREDICTOR_HEADER}\n{predicted_failures}]"
    return assumptions


def predict_failures(engine, text: str, packages: list[str]) -> str:
    """Name stale package priors before the model writes its answer."""
    return failures.predict_failures(
        engine,
        text,
        packages,
        is_legacy_path=is_legacy_path,
        token_pattern=_TOKEN_RE,
        bound_names=_bound_names,
    )
