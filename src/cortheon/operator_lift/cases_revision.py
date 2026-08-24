from cortheon.operator_lift.case_builders import _e, _revision
from cortheon.operator_lift.models import LiftCase

_DIRECT = {"supports": ("supported", False), "refutes": ("refuted", True)}
_ALTERNATE = {"confirms": ("retained", False), "disconfirms": ("superseded", True)}


def _reverse(prior: str, decisive: str) -> tuple[tuple[str, str], ...]:
    return (("source_a", decisive), ("source_b", prior))


def _revision_cases() -> tuple[LiftCase, ...]:
    return (
        _revision(
            1,
            "superseded_oncall_roster",
            _e(
                "[source_a] archived roster supports h_alex_oncall.",
                "[source_b] signed current rota refutes h_alex_oncall and supports h_priya_oncall.",
            ),
            ("h_alex_oncall", "refuted", "h_priya_oncall", "source_b"),
            effect_contract=_DIRECT,
        ),
        _revision(
            2,
            "calibrated_sensor_override",
            _reverse(
                "[source_b] dashboard reading supports h_pressure_surge.",
                "[source_a] calibrated gauge confirms h_pressure_surge.",
            ),
            ("h_pressure_surge", "retained", "h_pressure_surge", "source_a"),
            effect_contract=_ALTERNATE,
        ),
        _revision(
            3,
            "prospective_trial_reversal",
            _e(
                "[source_a] retrospective series confirms h_treatment_benefit.",
                "[source_b] randomized trial disconfirms h_treatment_benefit and "
                "confirms h_no_benefit after matching severity.",
            ),
            ("h_treatment_benefit", "superseded", "h_no_benefit", "source_b"),
            effect_contract=_ALTERNATE,
        ),
        _revision(
            4,
            "schema_version_scope",
            _reverse(
                "[source_b] deployed v3 schema supports h_field_required_v3.",
                "[source_a] signed validator report supports h_field_required_v3.",
            ),
            ("h_field_required_v3", "supported", "h_field_required_v3", "source_a"),
            effect_contract=_DIRECT,
        ),
        _revision(
            5,
            "regional_policy_exception",
            _reverse(
                "[source_b] global handbook supports h_export_allowed.",
                "[source_a] controlling EEA annex refutes h_export_allowed and supports "
                "h_export_prohibited_eea.",
            ),
            ("h_export_allowed", "refuted", "h_export_prohibited_eea", "source_a"),
            effect_contract=_DIRECT,
        ),
        _revision(
            6,
            "event_time_reordering",
            _e(
                "[source_a] arrival order supports h_charge_before_refund.",
                "[source_b] signed event timestamps support h_charge_before_refund.",
            ),
            (
                "h_charge_before_refund",
                "supported",
                "h_charge_before_refund",
                "source_b",
            ),
            effect_contract=_DIRECT,
        ),
        _revision(
            7,
            "branch_specific_implementation",
            _e(
                "[source_a] release candidate test confirms h_bug_fixed.",
                "[source_b] signed production test confirms h_bug_fixed.",
            ),
            ("h_bug_fixed", "retained", "h_bug_fixed", "source_b"),
            effect_contract=_ALTERNATE,
        ),
        _revision(
            8,
            "canonical_species_reclassification",
            _reverse(
                "[source_b] old catalog confirms h_species_alpha.",
                "[source_a] type-sequence registry disconfirms h_species_alpha and "
                "confirms h_species_beta.",
            ),
            ("h_species_alpha", "superseded", "h_species_beta", "source_a"),
            effect_contract=_ALTERNATE,
        ),
        _revision(
            9,
            "currency_normalization",
            _reverse(
                "[source_b] nominal totals support h_vendor_x_cheaper.",
                "[source_a] invoice-date FX calculation refutes h_vendor_x_cheaper and "
                "supports h_vendor_y_cheaper.",
            ),
            ("h_vendor_x_cheaper", "refuted", "h_vendor_y_cheaper", "source_a"),
            effect_contract=_DIRECT,
        ),
        _revision(
            10,
            "satellite_ephemeris_update",
            _e(
                "[source_a] predicted orbit confirms h_pass_visible.",
                "[source_b] post-maneuver ephemeris disconfirms h_pass_visible and "
                "confirms h_pass_below_horizon.",
            ),
            ("h_pass_visible", "superseded", "h_pass_below_horizon", "source_b"),
            effect_contract=_ALTERNATE,
        ),
        _revision(
            11,
            "legal_amendment_effective_date",
            _e(
                "[source_a] amended statute supports h_notice_60_days.",
                "[source_b] effective-date register supports h_notice_60_days.",
            ),
            ("h_notice_60_days", "supported", "h_notice_60_days", "source_b"),
            effect_contract=_DIRECT,
        ),
        _revision(
            12,
            "identity_resolution_correction",
            _reverse(
                "[source_b] verified identifiers confirm h_accounts_distinct_people.",
                "[source_a] independent registry match confirms h_accounts_distinct_people.",
            ),
            (
                "h_accounts_distinct_people",
                "retained",
                "h_accounts_distinct_people",
                "source_a",
            ),
            effect_contract=_ALTERNATE,
        ),
    )
