from cortheon.operator_lift.case_builders import _e, _stop
from cortheon.operator_lift.models import LiftCase


def _stopping_cases() -> tuple[LiftCase, ...]:
    return (
        _stop(
            1,
            "binary_config_identity",
            _e(
                "Deployment may run build_amber or build_cobalt.",
                "A signed binary hash uniquely identifies the deployed build.",
            ),
            (
                ("hash_binary", "Read signed deployed binary hash.", 1),
                ("read_changelog", "Read another changelog.", 2),
                ("scan_all_logs", "Scan all service logs.", 4),
            ),
            ("hash_binary",),
            "build_cobalt",
            (("hash_binary", "hash_matches_cobalt"),),
        ),
        _stop(
            2,
            "database_primary_region",
            _e(
                "Primary is either region_east or region_west.",
                "Consensus leader metadata is authoritative for the current term.",
            ),
            (
                ("read_consensus_term", "Read current signed leader term.", 1),
                ("sample_latency", "Sample client latency.", 2),
                ("read_old_incident", "Read an old failover incident.", 2),
            ),
            ("read_consensus_term",),
            "region_west",
            (("read_consensus_term", "leader_region_west"),),
        ),
        _stop(
            3,
            "invoice_duplicate_key",
            _e(
                "Duplicate may be retry_reuse or two real orders.",
                "The idempotency ledger binds a key to an originating order.",
            ),
            (
                ("lookup_idempotency", "Lookup the duplicate key in the ledger.", 1),
                ("interview_customer", "Ask customer to recall checkout.", 3),
                ("scan_month", "Scan every monthly invoice.", 5),
            ),
            ("lookup_idempotency",),
            "retry_reuse",
            (("lookup_idempotency", "same_key_one_order"),),
        ),
        _stop(
            4,
            "specimen_chain_break",
            _e(
                "Mismatch arose at accession or at sequencing.",
                "Tamper-evident handoff seals isolate the first broken custody edge.",
            ),
            (
                ("verify_handoff_seals", "Verify both custody seal IDs.", 2),
                ("rerun_sequence", "Sequence the received tube again.", 4),
                ("audit_all_staff", "Audit every staff shift.", 5),
            ),
            ("verify_handoff_seals",),
            "accession_swap",
            (("verify_handoff_seals", "accession_seal_mismatch"),),
        ),
        _stop(
            5,
            "feature_rollout_arm",
            _e(
                "User is assigned control or treatment by immutable allocation.",
                "The allocation receipt is sufficient; behavior logs are downstream.",
            ),
            (
                ("read_allocation_receipt", "Read signed experiment allocation.", 1),
                ("analyze_clicks", "Analyze a week of clicks.", 3),
                ("query_support", "Query support tickets.", 3),
            ),
            ("read_allocation_receipt",),
            "treatment_arm",
            (("read_allocation_receipt", "arm_treatment"),),
        ),
        _stop(
            6,
            "aircraft_part_serial",
            _e(
                "Installed pump is series_j or series_k.",
                "The physical serial and registry jointly identify the series.",
            ),
            (
                ("read_serial", "Read physical pump serial.", 1),
                ("lookup_registry", "Resolve serial in airworthiness registry.", 1),
                ("borescope", "Perform unrelated internal borescope.", 5),
            ),
            ("read_serial", "lookup_registry"),
            "series_k",
            (("read_serial", "serial_p9"), ("lookup_registry", "serial_p9_series_k")),
        ),
        _stop(
            7,
            "legal_contract_version",
            _e(
                "The agreement may be version_2 or version_3.",
                "A countersigned amendment controls only if its signature verifies.",
            ),
            (
                ("read_amendment", "Read latest amendment identifier.", 1),
                ("verify_signature", "Verify amendment countersignature.", 1),
                ("search_emails", "Search negotiation emails.", 4),
            ),
            ("read_amendment", "verify_signature"),
            "version_3",
            (("read_amendment", "amendment_v3"), ("verify_signature", "signature_valid")),
        ),
        _stop(
            8,
            "sensor_fault_localization",
            _e(
                "Fault is sensor_a or shared_bus.",
                "Cross-channel reference distinguishes a local sensor from the bus.",
            ),
            (
                ("read_reference_channel", "Compare calibrated reference channel.", 2),
                ("swap_sensor_a", "Swap sensor A after diagnosis.", 4),
                ("inspect_history", "Inspect a year of logs.", 5),
            ),
            ("read_reference_channel",),
            "sensor_a_fault",
            (("read_reference_channel", "reference_stable_a_diverges"),),
        ),
        _stop(
            9,
            "container_customs_hold",
            _e(
                "Hold is documentation or inspection.",
                "The customs status code is authoritative but requires codebook resolution.",
            ),
            (
                ("read_status_code", "Read live customs status code.", 1),
                ("resolve_codebook", "Resolve code under current codebook.", 1),
                ("call_carrier", "Call general carrier support.", 3),
            ),
            ("read_status_code", "resolve_codebook"),
            "inspection_hold",
            (("read_status_code", "code_x7"), ("resolve_codebook", "x7_inspection")),
        ),
        _stop(
            10,
            "certificate_revocation",
            _e(
                "Certificate is active or revoked.",
                "A stapled response is useful only after issuer signature validation.",
            ),
            (
                ("fetch_ocsp", "Fetch stapled OCSP response.", 1),
                ("verify_issuer", "Validate issuer signature and freshness.", 1),
                ("crawl_forums", "Search reports about the certificate.", 4),
            ),
            ("fetch_ocsp", "verify_issuer"),
            "revoked",
            (("fetch_ocsp", "status_revoked"), ("verify_issuer", "issuer_valid_fresh")),
        ),
        _stop(
            11,
            "patient_allergy_identity",
            _e(
                "Alert may belong to patient_r or a namesake.",
                "Two identifiers are required to bind the external allergy record.",
            ),
            (
                ("match_record_id", "Match external record identifier.", 1),
                ("match_birth_date", "Match birth date as second identifier.", 1),
                ("review_all_notes", "Review every clinical note.", 5),
            ),
            ("match_record_id", "match_birth_date"),
            "allergy_confirmed_patient_r",
            (("match_record_id", "id_matches"), ("match_birth_date", "birth_date_matches")),
        ),
        _stop(
            12,
            "repository_patch_state",
            _e(
                "Patch may be applied or absent.",
                "Commit containment plus working-tree diff identifies both committed and local state.",
            ),
            (
                ("check_commit", "Check target commit containment.", 1),
                ("check_diff", "Inspect focused working-tree diff.", 1),
                ("run_full_history", "Traverse complete repository history.", 5),
            ),
            ("check_commit", "check_diff"),
            "patch_applied_with_local_edit",
            (("check_commit", "commit_present"), ("check_diff", "local_followup_present")),
        ),
    )
