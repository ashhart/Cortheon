from cortheon.operator_lift.case_builders import _e, _hyp
from cortheon.operator_lift.models import LiftCase


def _hypothesis_cases() -> tuple[LiftCase, ...]:
    return (
        _hyp(
            1,
            "broker_capacity_boundary",
            _e(
                "[cause=legacy_broker_overload] Weekend accounts use a 500-request broker; bursts are 900.",
                "[outcome=activation_drop scope=weekend] Only weekend migration activation fell; measurement sampling is unchanged.",
            ),
            ("legacy_broker_overload", "activation_drop", "weekend"),
            ("cohort_selection_bias", "activation_drop", "weekend"),
            ("route_new_broker", "drop_persists", "legacy_broker_overload"),
        ),
        _hyp(
            2,
            "vat_rounding_accumulation",
            _e(
                "[cause=line_level_rounding] EEA invoices round VAT on every line; other invoices round after summing.",
                "[outcome=refund_spike scope=eea_many_lines] One-cent disputes rise only on EEA invoices with many low-value lines.",
            ),
            ("line_level_rounding", "refund_spike", "eea_many_lines"),
            ("gateway_currency_bug", "refund_spike", "eea_many_lines"),
            ("round_after_sum", "spike_remains", "line_level_rounding"),
        ),
        _hyp(
            3,
            "scheduled_power_mode",
            _e(
                "[cause=low_power_schedule] Battery gateways enter low-power mode from 02:00 to 02:15.",
                "[outcome=sample_dropout scope=battery_overnight] Only battery gateways lose high-frequency samples in that interval.",
            ),
            ("low_power_schedule", "sample_dropout", "battery_overnight"),
            ("night_radio_interference", "sample_dropout", "battery_overnight"),
            ("disable_low_power", "dropout_remains", "low_power_schedule"),
        ),
        _hyp(
            4,
            "defrost_door_interaction",
            _e(
                "[cause=defrost_door_overlap] Dock doors open during the 04:00 freezer defrost cycle.",
                "[outcome=spoilage scope=dock_freezers] Temperature excursions and spoilage occur only in dock-side freezers.",
            ),
            ("defrost_door_overlap", "spoilage", "dock_freezers"),
            ("sensor_calibration_drift", "spoilage", "dock_freezers"),
            ("stagger_defrost", "excursion_remains", "defrost_door_overlap"),
        ),
        _hyp(
            5,
            "underwriting_population_shift",
            _e(
                "[cause=thin_file_mix_shift] Partner Z doubled thin-file applicants after April; model weights did not change.",
                "[outcome=default_increase scope=partner_z_post_april] Default rose only for Partner Z post-April approvals.",
            ),
            ("thin_file_mix_shift", "default_increase", "partner_z_post_april"),
            ("score_model_drift", "default_increase", "partner_z_post_april"),
            ("reweight_applicant_mix", "default_gap_remains", "thin_file_mix_shift"),
        ),
        _hyp(
            6,
            "thermal_clock_drift",
            _e(
                "[cause=oscillator_heating] The packet clock drifts above 78C during sun-facing passes.",
                "[outcome=packet_loss scope=sun_facing_passes] Loss clusters after thermal peaks, not by ground station.",
            ),
            ("oscillator_heating", "packet_loss", "sun_facing_passes"),
            ("ground_station_congestion", "packet_loss", "sun_facing_passes"),
            ("cool_oscillator", "loss_remains", "oscillator_heating"),
        ),
        _hyp(
            7,
            "weekend_discharge_support",
            _e(
                "[cause=followup_delay] Weekend discharges wait three days longer for medication calls.",
                "[outcome=readmission_rise scope=weekend_discharge] Readmission rose only for weekend discharges with changed medication.",
            ),
            ("followup_delay", "readmission_rise", "weekend_discharge"),
            ("weekend_case_severity", "readmission_rise", "weekend_discharge"),
            ("same_day_followup", "rise_remains", "followup_delay"),
        ),
        _hyp(
            8,
            "cooling_rate_microcracks",
            _e(
                "[cause=rapid_quench] Line C cools castings twice as fast as other lines after a nozzle change.",
                "[outcome=microcracks scope=line_c_castings] Cracks rose on Line C across all alloy lots.",
            ),
            ("rapid_quench", "microcracks", "line_c_castings"),
            ("alloy_impurity", "microcracks", "line_c_castings"),
            ("restore_cooling_curve", "cracks_remain", "rapid_quench"),
        ),
        _hyp(
            9,
            "salinity_irrigation_load",
            _e(
                "[cause=saline_well_mix] Field K received 40 percent saline well water after the canal closure.",
                "[outcome=yield_loss scope=field_k] Yield fell in Field K across three seed varieties while nearby canal fields held.",
            ),
            ("saline_well_mix", "yield_loss", "field_k"),
            ("seed_batch_failure", "yield_loss", "field_k"),
            ("flush_with_canal_water", "yield_stays_low", "saline_well_mix"),
        ),
        _hyp(
            10,
            "timezone_rule_boundary",
            _e(
                "[cause=utc_cutoff_misread] The fraud rule treats local midnight as UTC for eastern merchants.",
                "[outcome=false_alerts scope=eastern_merchants] Alerts spike near local midnight only east of UTC.",
            ),
            ("utc_cutoff_misread", "false_alerts", "eastern_merchants"),
            ("coordinated_attack", "false_alerts", "eastern_merchants"),
            ("evaluate_local_timezone", "alerts_remain", "utc_cutoff_misread"),
        ),
        _hyp(
            11,
            "charge_firmware_overshoot",
            _e(
                "[cause=charge_voltage_overshoot] Firmware 4.2 briefly exceeds the cell voltage limit during cold starts.",
                "[outcome=thermal_event scope=cold_start_v42] Thermal events occur only on cold starts after 4.2 across two cell suppliers.",
            ),
            ("charge_voltage_overshoot", "thermal_event", "cold_start_v42"),
            ("cell_supplier_defect", "thermal_event", "cold_start_v42"),
            ("cap_charge_voltage", "events_remain", "charge_voltage_overshoot"),
        ),
        _hyp(
            12,
            "cache_eviction_storm",
            _e(
                "[cause=synchronized_ttl_expiry] Search shards share a top-of-hour cache TTL.",
                "[outcome=latency_spike scope=top_of_hour] Origin fetches and latency spike together at the hour; network RTT is flat.",
            ),
            ("synchronized_ttl_expiry", "latency_spike", "top_of_hour"),
            ("network_congestion", "latency_spike", "top_of_hour"),
            ("jitter_cache_ttl", "spike_remains", "synchronized_ttl_expiry"),
        ),
    )
