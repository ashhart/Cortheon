"""Stage 7 generic-host and retained-release attack coverage."""

from cortheon.threat_model.models import ResidualRisk, ThreatRisk

RELEASE_RISK_IDS = (
    "fused_reason_completion_bypass",
    "generic_usage_timing_spoof",
    "release_record_chain_forgery",
    "release_claim_scope_escalation",
    "release_trust_anchor_spoof",
    "retained_private_content_leak",
)

RELEASE_RISKS = (
    ThreatRisk(
        RELEASE_RISK_IDS[0],
        "critical",
        "receipts",
        "execution_receipts",
        "A fused completion can skip fields required by the active reasoning action.",
        (
            "tests/test_generic_mcp_hardening.py::test_fused_reason_completion_requires_every_projected_field[action2-arguments2-False]",
            "tests/test_generic_mcp_hardening.py::test_fused_reason_completion_requires_every_projected_field[action3-arguments3-False]",
        ),
    ),
    ThreatRisk(
        RELEASE_RISK_IDS[1],
        "high",
        "measurement",
        "measurement_integrity",
        "Malformed provider timing extensions can enter evaluator measurements.",
        tuple(
            "tests/test_generic_mcp_hardening.py::test_model_rejects_invalid_omlx_load_duration"
            f"[{value}]"
            for value in ("True", "1.0", "-1.0", "inf")
        ),
    ),
    ThreatRisk(
        RELEASE_RISK_IDS[2],
        "critical",
        "measurement",
        "measurement_integrity",
        "Deletion, reordering, duplication, or splicing can forge a retained event chain.",
        tuple(
            "tests/test_operator_lift_release.py::test_release_chain_rejects_structural_mutation"
            f"[{mutation}]"
            for mutation in ("delete", "reorder", "duplicate", "splice")
        ),
    ),
    ThreatRisk(
        RELEASE_RISK_IDS[3],
        "critical",
        "measurement",
        "measurement_integrity",
        "An incomplete pilot can be relabelled as claim-eligible.",
        (
            "tests/test_operator_lift_release.py::test_pilot_release_is_valid_but_never_claim_eligible",
        ),
    ),
    ThreatRisk(
        RELEASE_RISK_IDS[4],
        "critical",
        "measurement",
        "measurement_integrity",
        "A changed run descriptor or chain root can pass retained-release replay.",
        (
            "tests/test_operator_lift_release.py::test_replay_binds_descriptor_and_external_chain_root",
        ),
    ),
    ThreatRisk(
        RELEASE_RISK_IDS[5],
        "critical",
        "private_pack",
        "privacy_boundary",
        "Finalized artifacts can retain task, answer, tool, path, or transcript content.",
        (
            "tests/test_operator_lift_release.py::test_release_serialization_contains_no_sensitive_content_or_content_commitments",
            "tests/test_operator_lift_execution_storage.py::test_retained_artifact_scanner_rejects_raw_content_fields",
        ),
    ),
)

RELEASE_RESIDUALS = (
    ResidualRisk(
        "withheld_cause_redaction",
        "medium",
        "A content-free release retains a withheld status but not its specific cause.",
        "Treat the row as an incorrect delivery failure and reproduce it before diagnosis.",
    ),
    ResidualRisk(
        "single_cluster_ceiling",
        "medium",
        "One easy case can let both full and ablated conditions reach the score ceiling.",
        "Use the pilot for execution checks only and require the preregistered multi-case run for lift.",
    ),
)
