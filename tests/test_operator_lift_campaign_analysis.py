"""Campaign analysis: deterministic contrast drafts from a retained run."""

from __future__ import annotations

import json
from pathlib import Path

from cortheon.operator_lift.campaign_analysis import campaign_analysis
from cortheon.operator_lift.models import OPERATORS

ROOT = Path(__file__).parents[1]
RETAINED = ROOT / "benchmarks/frozen/operator_lift_qwen35_4b_full_20260826/release.json"


def test_campaign_analysis_is_deterministic_on_the_retained_run() -> None:
    first = campaign_analysis(RETAINED)
    second = campaign_analysis(RETAINED)
    assert first == second
    assert first["valid_cells"] == 531
    assert first["scheduled_cells"] == 540


def test_full_beats_bare_by_a_wide_lower_bound() -> None:
    analysis = campaign_analysis(RETAINED)
    assert analysis["full_rate"] > 0.7
    assert analysis["placebo_rate"] == 0.0
    assert (
        analysis["full_vs_bare_one_sided_lower_bound"]
        > analysis["thresholds"]["full_vs_bare_lower_bound_points"]
    )


def test_strongest_reduced_arm_is_selected_by_the_preregistered_rule() -> None:
    analysis = campaign_analysis(RETAINED)
    assert analysis["strongest_reduced_operator"] in OPERATORS
    # The selected arm is the one whose removal costs most on its own clusters.
    assert analysis["strongest_reduced_realized_loss"] < 0
    # The full-versus-selected contrast clears its point gate in this sample.
    assert (
        analysis["full_vs_selected_reduced_one_sided_lower_bound"]
        > analysis["thresholds"]["full_vs_selected_reduced_lower_bound_points"]
    )


def test_every_cell_is_classified() -> None:
    analysis = campaign_analysis(RETAINED)
    assert set(analysis["per_operator"]) == set(OPERATORS)
    for operator in OPERATORS:
        effects = analysis["per_operator"][operator]["cluster_effects"]
        assert effects  # every operator has paired cluster contrasts
        assert all(-1.000001 <= effect <= 1.000001 for effect in effects)


def test_analysis_output_is_closed_and_self_digesting() -> None:
    analysis = campaign_analysis(RETAINED)
    core = json.loads(
        json.dumps({k: v for k, v in analysis.items() if k != "digest"}, sort_keys=True)
    )
    # Round-trip stability: the digest binds the closed record.
    import hashlib

    rebuilt = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert rebuilt == analysis["digest"]


def test_strongest_reduced_uses_each_operators_paired_full_rate(tmp_path: Path) -> None:
    hits = {
        "hypothesis_framing": (3, 3),
        "discriminating_evidence": (3, 3),
        "contradiction_revision": (3, 3),
        "cross_source_derivation": (3, 1),
        "adaptive_stopping": (1, 0),
    }
    records = []
    for ordinal, operator in enumerate(OPERATORS, 1):
        full_hits, removed_hits = hits[operator]
        for repeat in range(3):
            common = {"case_ordinal": ordinal, "identity_valid": True}
            records.extend(
                (
                    {
                        **common,
                        "condition_id": "full",
                        "correct": repeat < full_hits,
                    },
                    {
                        **common,
                        "condition_id": f"ablation_{OPERATORS.index(operator)}",
                        "correct": repeat < removed_hits,
                    },
                    {
                        **common,
                        "condition_id": "equal_budget_placebo",
                        "correct": False,
                    },
                )
            )
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"records": records}), encoding="utf-8")

    analysis = campaign_analysis(release)

    assert analysis["strongest_reduced_operator"] == "cross_source_derivation"
    assert analysis["per_operator"]["cross_source_derivation"]["full_rate"] == 1.0
    assert analysis["per_operator"]["adaptive_stopping"]["full_rate"] == 1 / 3
