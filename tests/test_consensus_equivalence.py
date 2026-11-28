"""Comprehensive unit tests for answer normalization, equivalence, extraction,
prompt-format gating, and consensus clustering in ``flash_forge.reason_forge``.
"""

from __future__ import annotations

import pytest

from flash_forge.reason_forge import (
    BranchOutcome,
    _answer_satisfies_prompt_format,
    _answers_equivalent,
    _cluster_answers,
    extract_answer_from_prose,
    extract_answer_from_stdout,
    normalize_answer,
)
from flash_forge.sandbox import ExecResult


def _mk_outcome(
    name: str,
    norm_answer: str,
    *,
    kind: str = "pal",
    wave: int = 1,
    exec_ok: bool = True,
    verified_constraints: bool = False,
) -> BranchOutcome:
    exec_res = (
        ExecResult(ok=exec_ok, returncode=0 if exec_ok else 1, stdout=f"ANSWER: {norm_answer}\n", stderr="", elapsed_s=0.05)
        if kind == "pal"
        else None
    )
    return BranchOutcome(
        name=name,
        wave=wave,
        kind=kind,
        raw_content=f"ANSWER: {norm_answer}",
        reasoning="",
        exec_result=exec_res,
        raw_answer=norm_answer,
        norm_answer=norm_answer,
        verified_constraints=verified_constraints,
    )


# ---------------------------------------------------------------------------
# 1. Exact fractions & rational normalization
# ---------------------------------------------------------------------------


def test_normalize_exact_fractions_and_half_variants():
    assert normalize_answer("1/2") == "1/2"
    assert normalize_answer("2/4") == "1/2"
    assert normalize_answer(r"\frac{3}{6}") == "1/2"
    assert normalize_answer("0.5") == "1/2"
    assert normalize_answer("Fraction(3, 6)") == "1/2"


def test_answers_equivalent_exact_fractions():
    assert _answers_equivalent("1/2", "2/4")
    assert _answers_equivalent("1/2", r"\frac{3}{6}")
    assert _answers_equivalent("1/2", "0.5")
    assert _answers_equivalent("2/4", "0.5")
    assert not _answers_equivalent("1/2", "3/5")


def test_normalize_nested_latex_fractions():
    assert normalize_answer(r"\frac{\frac{1}{2}}{\frac{3}{4}}") == "2/3"
    assert normalize_answer(r"\dfrac{18}{24}") == "3/4"
    assert normalize_answer(r"\tfrac{10}{5}") == "2"


# ---------------------------------------------------------------------------
# 2. Huge 25-digit rationals & arbitrary-precision integers (NO float loss!)
# ---------------------------------------------------------------------------


def test_huge_25_digit_rational_normalization_preserves_exact_digits():
    huge = "2630600114653531854763423/183488738276457059168800"
    assert normalize_answer(huge) == huge
    assert normalize_answer(r"\frac{2630600114653531854763423}{183488738276457059168800}") == huge
    # Unreduced multiple reduces back identically
    scaled = f"{2630600114653531854763423 * 7}/{183488738276457059168800 * 7}"
    assert normalize_answer(scaled) == huge


def test_huge_25_digit_rational_equivalence_never_loses_precision():
    a = "2630600114653531854763423/183488738276457059168800"
    a_unreduced = f"{2630600114653531854763423 * 3}/{183488738276457059168800 * 3}"
    b_off_by_one = "2630600114653531854763424/183488738276457059168800"
    assert _answers_equivalent(a, a_unreduced) is True
    # Float64 cannot distinguish these two, so exact rational comparison MUST reject!
    assert _answers_equivalent(a, b_off_by_one) is False


def test_huge_25_digit_integers_not_collapsed_by_float64():
    n1 = "1000000000000000000000001"
    n2 = "1000000000000000000000002"
    assert normalize_answer(n1) == n1
    assert _answers_equivalent(n1, n1) is True
    assert _answers_equivalent(n1, n2) is False


# ---------------------------------------------------------------------------
# 3. Tight 1e-12 decimal tolerance
# ---------------------------------------------------------------------------


def test_decimal_tolerance_rejects_near_misses():
    assert _answers_equivalent("1.0001", "1.0000") is False
    assert _answers_equivalent("0.5", "0.5000000001") is False
    assert _answers_equivalent("3.14159265", "3.14159266") is False
    assert _answers_equivalent("1e-15", "2e-15") is False


def test_decimal_tolerance_accepts_ieee754_exact_and_sub_1e12():
    assert _answers_equivalent("0.5833333333333334", "7/12") is True
    assert _answers_equivalent("1.0000000000001", "1.0") is True
    # 10-digit truncated decimal must NOT snap to 1/3
    assert normalize_answer("0.3333333333") != "1/3"
    assert _answers_equivalent("0.3333333333", "1/3") is False


# ---------------------------------------------------------------------------
# 4. LaTeX formatting, trailing punctuation, and markdown wrappers
# ---------------------------------------------------------------------------


def test_normalize_latex_boxed_cdot_times_and_periods():
    assert normalize_answer(r"\boxed{42}") == "42"
    assert normalize_answer(r"\boxed{\dfrac{6}{8}}.") == "3/4"
    assert normalize_answer(r"$3 \cdot 7.$") == "21"
    assert normalize_answer(r"\left(\dfrac{2}{3}\right) \times 9") == "6"
    assert normalize_answer(r"\text{42}.") == "42"
    assert normalize_answer("42.") == "42"


def test_normalize_markdown_bold_and_code_wrappers():
    assert normalize_answer("**42**") == "42"
    assert normalize_answer("`7/12`") == "7/12"
    assert normalize_answer("__100__") == "100"


def test_extract_answer_from_prose_markdown_and_boxed():
    assert extract_answer_from_prose("Step 1...\n**ANSWER: 42**") == "42"
    assert extract_answer_from_prose("Hence:\n> **FINAL ANSWER: 99**") == "99"
    assert extract_answer_from_prose("Check:\n`ANSWER: 7/12`") == "7/12"
    assert extract_answer_from_prose(r"First \boxed{10}, corrected to \boxed{\frac{3}{6}}.") == r"\frac{3}{6}"


def test_extract_answer_from_prose_prefers_explicit_answer_over_boxed():
    prose = r"Intermediate value \boxed{10}. Final check gives:" + "\nANSWER: 42"
    assert extract_answer_from_prose(prose) == "42"


def test_extract_answer_from_prose_inline_answer_and_empty():
    assert extract_answer_from_prose("After simplification, **ANSWER: 137**.") == "137"
    assert normalize_answer(extract_answer_from_prose("After simplification, **ANSWER: 137**.")) == "137"
    assert extract_answer_from_prose("No explicit final answer was stated here.") == ""
    assert extract_answer_from_prose("") == ""


# ---------------------------------------------------------------------------
# 5. Leading zeros / mod-1000 integers & trailing commentary
# ---------------------------------------------------------------------------


def test_leading_zeros_and_mod_1000_aime_integers():
    assert normalize_answer("042") == "42"
    assert normalize_answer("007") == "7"
    assert normalize_answer("000") == "0"
    assert _answers_equivalent("042", "42") is True
    assert _answers_equivalent("007", "7") is True
    assert normalize_answer("019. AIME answer 019 verified") == "19"


# ---------------------------------------------------------------------------
# 6. Negative values & scientific notation
# ---------------------------------------------------------------------------


def test_negative_integers_fractions_and_decimals():
    assert normalize_answer("-42") == "-42"
    assert normalize_answer("-3/6") == "-1/2"
    assert normalize_answer("-0.5") == "-1/2"
    assert normalize_answer("-0.0") == "0"
    assert _answers_equivalent("-3/6", "-0.5") is True
    assert _answers_equivalent("-42", "42") is False


def test_scientific_notation_normalization_and_equivalence():
    assert normalize_answer("1e6") == "1000000"
    assert normalize_answer("2.5e-4") == "1/4000"
    # Very small nonzero scientific notation must NEVER collapse to "0"
    assert normalize_answer("1.5e-20") == "1.5e-20"
    assert _answers_equivalent("1e6", "1000000") is True
    assert _answers_equivalent("2.5e-4", "1/4000") is True
    assert _answers_equivalent("1.5e-20", "0") is False


# ---------------------------------------------------------------------------
# 7. Symbolic radicals
# ---------------------------------------------------------------------------


def test_symbolic_radicals_normalization_and_equivalence():
    assert normalize_answer("sqrt(2)") == "sqrt(2)"
    assert normalize_answer(r"\sqrt{2}") == "sqrt(2)"
    assert normalize_answer(r"\sqrt{8}") == "2*sqrt(2)"
    assert normalize_answer(r"2\sqrt{2}") == "2*sqrt(2)"
    assert normalize_answer(r"2 \cdot \sqrt{2}") == "2*sqrt(2)"
    assert normalize_answer(r"\sqrt[3]{27}") == "3"
    assert _answers_equivalent("sqrt(2)", r"\sqrt{2}") is True
    assert _answers_equivalent("sqrt(8)", r"2\sqrt{2}") is True
    assert _answers_equivalent("1/sqrt(2)", "sqrt(2)/2") is True
    assert _answers_equivalent("sqrt(2)", "1.4142135623730951") is True
    assert _answers_equivalent("sqrt(2)", "1.4142") is False


# ---------------------------------------------------------------------------
# 8. Stdout extraction & rejecting prose garbage / status lines
# ---------------------------------------------------------------------------


def test_extract_answer_from_stdout_explicit_markers():
    assert extract_answer_from_stdout("debug: n=5\nANSWER: 42\n") == "42"
    assert extract_answer_from_stdout("FINAL ANSWER = 7/12\n") == "7/12"
    assert extract_answer_from_stdout("ANSWER 208\n") == "208"
    assert extract_answer_from_stdout("ANSWER: 10\nANSWER: 42\n") == "42"


@pytest.mark.parametrize(
    "noise_stdout",
    [
        "",
        "done\n",
        "working...\n",
        "PASS\n",
        "PASSED\n",
        "OK\n",
        "debug: checked 10 cases\n",
        "verified all constraints\n",
        "result:\n",
        "This is a long prose sentence with many words and no answer.\n",
    ],
)
def test_extract_answer_from_stdout_rejects_prose_and_status_noise(noise_stdout: str):
    assert extract_answer_from_stdout(noise_stdout) == ""


def test_extract_answer_from_stdout_accepts_clean_bare_final_value():
    assert extract_answer_from_stdout("computed\n42\n") == "42"
    assert extract_answer_from_stdout("7/12\n") == "7/12"


# ---------------------------------------------------------------------------
# 9. Prompt format gate (_answer_satisfies_prompt_format)
# ---------------------------------------------------------------------------


def test_prompt_format_gate_enforces_integer_when_requested():
    unbounded_prompts = [
        "Find N. End with ANSWER: <int>.",
        "Compute the exact integer value of S.",
        "Give the result. Output ANSWER: <integer>.",
    ]
    for p in unbounded_prompts:
        assert _answer_satisfies_prompt_format(p, "42") is True
        assert _answer_satisfies_prompt_format(p, "-7") is True
        assert _answer_satisfies_prompt_format(p, "0") is True
        assert _answer_satisfies_prompt_format(p, "3/4") is False
        assert _answer_satisfies_prompt_format(p, "3.14") is False
        assert _answer_satisfies_prompt_format(p, "sqrt(2)") is False
        assert _answer_satisfies_prompt_format(p, "") is False

    bounded_p = "The answer is an integer from 000 to 999."
    assert _answer_satisfies_prompt_format(bounded_p, "42") is True
    assert _answer_satisfies_prompt_format(bounded_p, "0") is True
    assert _answer_satisfies_prompt_format(bounded_p, "999") is True
    assert _answer_satisfies_prompt_format(bounded_p, "-7") is False
    assert _answer_satisfies_prompt_format(bounded_p, "1000") is False
    assert _answer_satisfies_prompt_format(bounded_p, "3/4") is False


def test_prompt_format_gate_allows_non_integers_for_open_prompts():
    p = "Compute the exact probability as a simplified fraction or expression."
    assert _answer_satisfies_prompt_format(p, "7/12") is True
    assert _answer_satisfies_prompt_format(p, "sqrt(2)/2") is True
    assert _answer_satisfies_prompt_format(p, "0.25") is True
    assert _answer_satisfies_prompt_format(p, "") is False


# ---------------------------------------------------------------------------
# 10. Clustering, majority selection, tie-breaking, and canonical rep preference
# ---------------------------------------------------------------------------


def test_cluster_answers_merges_equivalent_forms_and_prefers_rational_rep():
    outcomes = [
        _mk_outcome("pal_a", "0.5"),
        _mk_outcome("pal_b", "1/2"),
        _mk_outcome("pal_c", "2/4"),
    ]
    clusters = _cluster_answers(outcomes)
    assert len(clusters) == 1
    rep, members = clusters[0]
    assert rep == "1/2"
    assert len(members) == 3


def test_cluster_answers_majority_beats_earlier_minority():
    outcomes = [
        _mk_outcome("pal_wrong", "99"),
        _mk_outcome("pal_right_1", "42"),
        _mk_outcome("pal_right_2", "42"),
    ]
    clusters = _cluster_answers(outcomes)
    assert len(clusters) == 2
    assert clusters[0][0] == "42"
    assert len(clusters[0][1]) == 2
    assert clusters[1][0] == "99"
    assert len(clusters[1][1]) == 1


def test_cluster_answers_tiebreak_prefers_verified_pal_over_analytical():
    outcomes = [
        _mk_outcome("analytical", "99", kind="analytical"),
        _mk_outcome("pal_exact", "42", kind="pal", exec_ok=True),
    ]
    clusters = _cluster_answers(outcomes)
    assert len(clusters) == 2
    assert clusters[0][0] == "42"
    assert clusters[1][0] == "99"


def test_cluster_answers_tiebreak_prefers_assert_verified_pal():
    outcomes = [
        _mk_outcome("pal_unasserted", "99", kind="pal", exec_ok=True, verified_constraints=False),
        _mk_outcome("pal_asserted", "42", kind="pal", exec_ok=True, verified_constraints=True),
    ]
    clusters = _cluster_answers(outcomes)
    assert len(clusters) == 2
    assert clusters[0][0] == "42"


def test_cluster_answers_ignores_empty_and_keeps_huge_rationals_separate():
    a = "2630600114653531854763423/183488738276457059168800"
    b = "2630600114653531854763424/183488738276457059168800"
    outcomes = [
        _mk_outcome("empty", ""),
        _mk_outcome("pal_a1", a),
        _mk_outcome("pal_b1", b),
        _mk_outcome("pal_a2", a),
    ]
    clusters = _cluster_answers(outcomes)
    assert len(clusters) == 2
    assert clusters[0][0] == a and len(clusters[0][1]) == 2
    assert clusters[1][0] == b and len(clusters[1][1]) == 1


def test_normalize_booleans_and_thousands_commas():
    assert normalize_answer("1,234,567") == "1234567"
    assert normalize_answer("yes") == "True"
    assert normalize_answer("FALSE") == "False"
    assert normalize_answer("(1, 2, 3)") == "[1, 2, 3]"


def test_normalize_empty_whitespace_and_plain_strings():
    assert normalize_answer("") == ""
    assert normalize_answer("   ...   ") == ""
    assert normalize_answer("  Alice   Smith ") == "Alice Smith"


def test_answers_equivalent_empty_case_insensitive_and_non_numeric():
    assert _answers_equivalent("", "42") is False
    assert _answers_equivalent("42", "") is False
    assert _answers_equivalent("Alice", "alice") is True
    assert _answers_equivalent("Alice", "Bob") is False


def test_cluster_answers_prefers_exact_integer_over_decimal_rep():
    outcomes = [
        _mk_outcome("pal_float", "42.0"),
        _mk_outcome("pal_int", "42"),
    ]
    clusters = _cluster_answers(outcomes)
    assert len(clusters) == 1
    assert clusters[0][0] == "42"
    assert len(clusters[0][1]) == 2

