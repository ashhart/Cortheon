from cortheon.operator_lift.case_builders import _e, _probe
from cortheon.operator_lift.models import LiftCase


def _discrimination_cases() -> tuple[LiftCase, ...]:
    return (
        _probe(
            1,
            "cache_vs_network",
            _e(
                "h_cache predicts a warm-cache replay is fast.",
                "h_network predicts replay remains slow regardless of cache warmth.",
            ),
            ("h_cache", "h_network"),
            (
                ("warm_replay", "Replay the same query from a verified warm cache.", 1),
                ("ping_region", "Ping an unrelated region.", 2),
            ),
            ("warm_replay", "h_cache", "h_network"),
        ),
        _probe(
            2,
            "drug_vs_regression",
            _e(
                "h_drug predicts biomarker change precedes recovery.",
                "h_regression predicts recovery without assigned biomarker change.",
            ),
            ("h_drug", "h_regression"),
            (
                (
                    "randomized_biomarker",
                    "Compare assigned treatment and placebo biomarker trajectories.",
                    3,
                ),
                ("patient_testimonial", "Collect another treated-patient testimonial.", 1),
            ),
            ("randomized_biomarker", "h_drug", "h_regression"),
        ),
        _probe(
            3,
            "price_vs_outage_churn",
            _e(
                "h_price predicts churn at renewal after the price notice.",
                "h_outage predicts churn immediately after service incidents.",
            ),
            ("h_price", "h_outage"),
            (
                ("timing_cohort", "Stratify churn by renewal and outage timing.", 2),
                ("more_interviews", "Interview ten unstratified churned users.", 3),
            ),
            ("timing_cohort", "h_price", "h_outage"),
        ),
        _probe(
            4,
            "memory_leak_vs_load",
            _e(
                "h_leak predicts RSS grows at constant request rate.",
                "h_load predicts RSS tracks concurrent traffic and falls afterward.",
            ),
            ("h_leak", "h_load"),
            (
                ("constant_load_trace", "Hold load constant and trace RSS over time.", 2),
                ("heap_snapshot_once", "Take one heap snapshot at peak.", 2),
            ),
            ("constant_load_trace", "h_leak", "h_load"),
        ),
        _probe(
            5,
            "pollinator_vs_fertilizer",
            _e(
                "h_pollinator predicts fruit set changes with controlled bee access.",
                "h_fertilizer predicts leaf nutrient response independent of bee access.",
            ),
            ("h_pollinator", "h_fertilizer"),
            (
                ("caged_branch_trial", "Randomize bee access with matched fertilizer.", 3),
                ("soil_sample", "Measure one pooled soil sample.", 1),
            ),
            ("caged_branch_trial", "h_pollinator", "h_fertilizer"),
        ),
        _probe(
            6,
            "clock_skew_vs_queueing",
            _e(
                "h_clock predicts negative durations move with host clock offset.",
                "h_queue predicts positive delays move with queue depth.",
            ),
            ("h_clock", "h_queue"),
            (
                (
                    "monotonic_clock_replay",
                    "Replay with monotonic timestamps and fixed queue depth.",
                    2,
                ),
                ("increase_workers", "Increase workers without changing clocks.", 2),
            ),
            ("monotonic_clock_replay", "h_clock", "h_queue"),
        ),
        _probe(
            7,
            "contamination_vs_label_swap",
            _e(
                "h_contam predicts the organism appears in raw sample aliquots.",
                "h_swap predicts organism identity follows the relabeled tube.",
            ),
            ("h_contam", "h_swap"),
            (
                (
                    "split_blind_retest",
                    "Blind-retest retained aliquots and tube identities separately.",
                    3,
                ),
                ("repeat_same_tube", "Repeat the assay on the same labeled tube.", 1),
            ),
            ("split_blind_retest", "h_contam", "h_swap"),
        ),
        _probe(
            8,
            "antenna_vs_decoder",
            _e(
                "h_antenna predicts loss changes with polarization before decoding.",
                "h_decoder predicts raw signal is intact but decoded frames fail.",
            ),
            ("h_antenna", "h_decoder"),
            (
                (
                    "raw_iq_capture",
                    "Capture raw I/Q under both polarizations and decode offline.",
                    3,
                ),
                ("firmware_restart", "Restart the decoder once.", 1),
            ),
            ("raw_iq_capture", "h_antenna", "h_decoder"),
        ),
        _probe(
            9,
            "teacher_vs_curriculum",
            _e(
                "h_teacher predicts gains transfer with the teacher across curricula.",
                "h_curriculum predicts gains transfer with materials across teachers.",
            ),
            ("h_teacher", "h_curriculum"),
            (
                ("crossed_assignment", "Cross-randomize teachers and curriculum materials.", 4),
                ("class_average", "Compare current class averages.", 1),
            ),
            ("crossed_assignment", "h_teacher", "h_curriculum"),
        ),
        _probe(
            10,
            "sensor_vs_real_pressure",
            _e(
                "h_sensor predicts a reference gauge stays stable.",
                "h_pressure predicts independent gauges rise together.",
            ),
            ("h_sensor", "h_pressure"),
            (
                ("independent_gauge", "Attach a calibrated independent pressure gauge.", 2),
                ("replace_display", "Replace only the dashboard display.", 2),
            ),
            ("independent_gauge", "h_pressure", "h_sensor"),
        ),
        _probe(
            11,
            "ranking_vs_inventory",
            _e(
                "h_rank predicts impressions shift with ranking held against stock.",
                "h_stock predicts conversion recovers when availability is restored.",
            ),
            ("h_rank", "h_stock"),
            (
                ("factorial_holdout", "Factorially hold ranking and stock availability.", 4),
                ("traffic_total", "Inspect total traffic alone.", 1),
            ),
            ("factorial_holdout", "h_rank", "h_stock"),
        ),
        _probe(
            12,
            "corrosion_vs_fatigue",
            _e(
                "h_corrosion predicts chemistry-specific pitting at crack origins.",
                "h_fatigue predicts striations tied to load cycles without pitting.",
            ),
            ("h_corrosion", "h_fatigue"),
            (
                (
                    "fractography_chemistry",
                    "Blind fractography plus deposit chemistry at origins.",
                    3,
                ),
                ("visual_photo", "Take another exterior photograph.", 1),
            ),
            ("fractography_chemistry", "h_corrosion", "h_fatigue"),
        ),
    )
