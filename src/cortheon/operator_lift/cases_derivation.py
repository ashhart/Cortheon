from cortheon.operator_lift.case_builders import _e, _join
from cortheon.operator_lift.models import LiftCase


def _derivation_cases() -> tuple[LiftCase, ...]:
    return (
        _join(
            1,
            "service_dataset_steward",
            _e(
                "Atlas Portal maps_to service_meridian.",
                "service_meridian reads dataset_ember.",
                "dataset_ember steward leila_hassan.",
            ),
            ("atlas_portal", "steward", "leila_hassan"),
            (
                ("source_a", "atlas_portal", "maps_to", "service_meridian"),
                ("source_b", "service_meridian", "reads", "dataset_ember"),
                ("source_c", "dataset_ember", "steward", "leila_hassan"),
            ),
        ),
        _join(
            2,
            "shipment_temperature_owner",
            _e(
                "shipment_k9 uses container_cold7.",
                "container_cold7 sensor sensor_nova.",
                "sensor_nova owner team_orbit.",
            ),
            ("shipment_k9", "sensor_owner", "team_orbit"),
            (
                ("source_a", "shipment_k9", "uses", "container_cold7"),
                ("source_b", "container_cold7", "sensor", "sensor_nova"),
                ("source_c", "sensor_nova", "owner", "team_orbit"),
            ),
        ),
        _join(
            3,
            "component_license_obligation",
            _e(
                "app_cedar bundles library_onyx.",
                "library_onyx license license_cobalt.",
                "license_cobalt requires notice_bundle.",
            ),
            ("app_cedar", "requires", "notice_bundle"),
            (
                ("source_a", "app_cedar", "bundles", "library_onyx"),
                ("source_b", "library_onyx", "license", "license_cobalt"),
                ("source_c", "license_cobalt", "requires", "notice_bundle"),
            ),
        ),
        _join(
            4,
            "patient_trial_investigator",
            _e(
                "patient_r17 enrolled_in trial_lumen.",
                "trial_lumen site clinic_harbor.",
                "clinic_harbor investigator dr_amina.",
            ),
            ("patient_r17", "investigator", "dr_amina"),
            (
                ("source_a", "patient_r17", "enrolled_in", "trial_lumen"),
                ("source_b", "trial_lumen", "site", "clinic_harbor"),
                ("source_c", "clinic_harbor", "investigator", "dr_amina"),
            ),
        ),
        _join(
            5,
            "invoice_cost_center",
            _e(
                "invoice_i44 purchase_order po_sable.",
                "po_sable project project_wren.",
                "project_wren cost_center cc_804.",
            ),
            ("invoice_i44", "cost_center", "cc_804"),
            (
                ("source_a", "invoice_i44", "purchase_order", "po_sable"),
                ("source_b", "po_sable", "project", "project_wren"),
                ("source_c", "project_wren", "cost_center", "cc_804"),
            ),
        ),
        _join(
            6,
            "alert_runbook_responder",
            _e(
                "alert_frost emitted_by service_boreal.",
                "service_boreal runbook runbook_echo.",
                "runbook_echo responder team_saffron.",
            ),
            ("alert_frost", "responder", "team_saffron"),
            (
                ("source_a", "alert_frost", "emitted_by", "service_boreal"),
                ("source_b", "service_boreal", "runbook", "runbook_echo"),
                ("source_c", "runbook_echo", "responder", "team_saffron"),
            ),
        ),
        _join(
            7,
            "specimen_freezer_location",
            _e(
                "specimen_q2 stored_in rack_ruby.",
                "rack_ruby inside freezer_f12.",
                "freezer_f12 room lab_west.",
            ),
            ("specimen_q2", "room", "lab_west"),
            (
                ("source_a", "specimen_q2", "stored_in", "rack_ruby"),
                ("source_b", "rack_ruby", "inside", "freezer_f12"),
                ("source_c", "freezer_f12", "room", "lab_west"),
            ),
        ),
        _join(
            8,
            "feature_flag_approver",
            _e(
                "release_rain enables flag_velvet.",
                "flag_velvet risk_class high_risk.",
                "high_risk approver security_duty_officer.",
            ),
            ("release_rain", "approver", "security_duty_officer"),
            (
                ("source_a", "release_rain", "enables", "flag_velvet"),
                ("source_b", "flag_velvet", "risk_class", "high_risk"),
                ("source_c", "high_risk", "approver", "security_duty_officer"),
            ),
        ),
        _join(
            9,
            "farm_water_authority",
            _e(
                "field_delta supplied_by canal_iris.",
                "canal_iris district district_north.",
                "district_north water_authority authority_mesa.",
            ),
            ("field_delta", "water_authority", "authority_mesa"),
            (
                ("source_a", "field_delta", "supplied_by", "canal_iris"),
                ("source_b", "canal_iris", "district", "district_north"),
                ("source_c", "district_north", "water_authority", "authority_mesa"),
            ),
        ),
        _join(
            10,
            "aircraft_part_directive",
            _e(
                "aircraft_tail_n7 fitted_with pump_p9.",
                "pump_p9 model pump_series_k.",
                "pump_series_k governed_by directive_ad22.",
            ),
            ("aircraft_tail_n7", "governed_by", "directive_ad22"),
            (
                ("source_a", "aircraft_tail_n7", "fitted_with", "pump_p9"),
                ("source_b", "pump_p9", "model", "pump_series_k"),
                ("source_c", "pump_series_k", "governed_by", "directive_ad22"),
            ),
        ),
        _join(
            11,
            "dataset_retention_rule",
            _e(
                "report_jade derives_from dataset_mint.",
                "dataset_mint contains biometric_template.",
                "biometric_template retention_rule delete_30_days.",
            ),
            ("report_jade", "retention_rule", "delete_30_days"),
            (
                ("source_a", "report_jade", "derives_from", "dataset_mint"),
                ("source_b", "dataset_mint", "contains", "biometric_template"),
                ("source_c", "biometric_template", "retention_rule", "delete_30_days"),
            ),
        ),
        _join(
            12,
            "warehouse_carrier_insurer",
            _e(
                "parcel_pine assigned_to route_r8.",
                "route_r8 carrier carrier_sol.",
                "carrier_sol insurer insurer_terra.",
            ),
            ("parcel_pine", "insurer", "insurer_terra"),
            (
                ("source_a", "parcel_pine", "assigned_to", "route_r8"),
                ("source_b", "route_r8", "carrier", "carrier_sol"),
                ("source_c", "carrier_sol", "insurer", "insurer_terra"),
            ),
        ),
    )
