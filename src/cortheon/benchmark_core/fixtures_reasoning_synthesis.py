"""Definition tables for the novel-synthesis reasoning fixtures."""

from __future__ import annotations

from typing import Any


def _novel_synthesis_definitions() -> list[tuple[Any, ...]]:
    """Return the held-out novel-synthesis reasoning case definitions."""

    synthesis_requirements = (
        ("hypothesis", "explanation", "alternative"),
        ("falsify", "distinguish", "counterexample", "test"),
        (
            "because",
            "therefore",
            "interaction",
            "combined",
            "causal",
            "causes",
            "accumulate",
            "drives",
            "explains",
            "accounts for",
            "leads",
            "matches",
            "produces",
            "results",
            "consistent",
            "triggers",
        ),
    )
    return [
        (
            "weekend_activation",
            (
                (
                    "cohort_notes.md",
                    "Activation fell only for accounts created during the weekend "
                    "migration window. Weekday migration cohorts remained stable.\n",
                ),
                (
                    "routing_map.md",
                    "Weekend migrations still route through the legacy token broker. "
                    "Weekday migrations use the new token broker.\n",
                ),
                (
                    "capacity_limits.md",
                    "The legacy token broker rejects bursts above 500 requests. "
                    "Migration bursts reach 900 requests; the new broker accepts 2000.\n",
                ),
            ),
            ("legacy token broker", "500", "900"),
            ("database saturation", "network outage"),
            (
                ("hypothesis", "explanation", "alternative"),
                ("falsify", "distinguish", "counterexample", "test"),
                (
                    "because",
                    "therefore",
                    "interaction",
                    "combined",
                    "causal",
                    "explains",
                    "accounts for",
                    "matches",
                    "consistent",
                ),
            ),
            (
                "Read the three documents as independent clues. Explain why activation "
                "fell, originate at least two genuinely competing hypotheses, select "
                "the best-supported explanation, and state one observation that would "
                "falsify it. Do not modify files."
            ),
        ),
        (
            "regional_refunds",
            (
                (
                    "refund_cohorts.md",
                    "The refund spike is isolated to EEA invoices created after the "
                    "tax rollout. US and pre-rollout EEA invoices are stable.\n",
                ),
                (
                    "billing_design.md",
                    "The EEA path rounds VAT on every invoice line. Other paths round "
                    "once after summing the invoice.\n",
                ),
                (
                    "support_patterns.md",
                    "Affected customers report repeated one-cent discrepancies on "
                    "invoices containing many low-value lines.\n",
                ),
            ),
            ("eea", "invoice line", "round"),
            ("payment gateway outage", "fraud"),
            (
                ("hypothesis", "explanation", "alternative"),
                ("falsify", "distinguish", "counterexample", "test"),
                (
                    "because",
                    "therefore",
                    "interaction",
                    "combined",
                    "causal",
                    "explains",
                    "accounts for",
                    "matches",
                    "consistent",
                ),
            ),
            (
                "Synthesize the independent documents to diagnose the refund spike. "
                "Present competing hypotheses, choose the best-supported causal "
                "explanation, and give a discriminating falsification test. "
                "Do not modify files."
            ),
        ),
        (
            "overnight_sensor_dropout",
            (
                (
                    "incident_timeline.md",
                    "Sensor samples disappear between 02:00 and 02:15, but only on "
                    "battery-powered gateways. Mains-powered gateways remain stable.\n",
                ),
                (
                    "firmware_schedule.md",
                    "Battery gateways enter low-power mode at 02:00 and leave it at "
                    "02:15. The schedule was enabled in the latest firmware.\n",
                ),
                (
                    "sampling_contract.md",
                    "High-frequency sampling is disabled while low-power mode is "
                    "active; buffered summaries resume afterward.\n",
                ),
            ),
            ("low-power mode", "02:00", "high-frequency sampling"),
            ("radio interference", "cloud outage"),
            (
                ("hypothesis", "explanation", "alternative"),
                ("falsify", "distinguish", "counterexample", "test"),
                (
                    "because",
                    "therefore",
                    "interaction",
                    "combined",
                    "causal",
                    "explains",
                    "accounts for",
                    "matches",
                    "consistent",
                ),
            ),
            (
                "Connect the separate clues to explain the overnight sensor dropout. "
                "Generate competing explanations, identify the strongest one, and "
                "state a test that could disprove it. Do not modify files."
            ),
        ),
        (
            "top_of_hour_queue_stall",
            (
                (
                    "tenant_metrics.md",
                    "Queue stalls affect only large tenants and begin exactly at the "
                    "top of each hour. Small tenants continue normally.\n",
                ),
                (
                    "scheduler.md",
                    "Hourly reconciliation fans out one job per account at minute 00. "
                    "Large tenants have tens of thousands of accounts.\n",
                ),
                (
                    "partitioning.md",
                    "Reconciliation jobs use tenant id as the queue partition key; a "
                    "single hot partition throttles while other partitions remain free.\n",
                ),
            ),
            ("hourly reconciliation", "tenant id", "hot partition"),
            ("global queue outage", "database corruption"),
            (
                ("hypothesis", "explanation", "alternative"),
                ("falsify", "distinguish", "counterexample", "test"),
                (
                    "because",
                    "therefore",
                    "interaction",
                    "combined",
                    "causal",
                    "explains",
                    "accounts for",
                    "matches",
                    "consistent",
                ),
            ),
            (
                "Infer the cause of the top-of-hour queue stalls from the independent "
                "documents. Compare at least two hypotheses and give a falsification "
                "test for the selected explanation. Do not modify files."
            ),
        ),
        (
            "compact_nonce_collision",
            (
                (
                    "incident_slice.md",
                    "Authentication failures affect only the Northstar cohort and "
                    "occur during parallel refreshes. Serial replays are clean.\n",
                ),
                (
                    "cohort_catalog.md",
                    "Northstar is the reporting label for Android v9 sessions using "
                    "the compact nonce experiment.\n",
                ),
                (
                    "nonce_design.md",
                    "Compact nonces retain the first 8 characters of a session nonce. "
                    "Household members share that 8-character prefix and differ in "
                    "the discarded suffix.\n",
                ),
                (
                    "refresh_cache.md",
                    "The refresh cache is keyed only by compact nonce. Concurrent "
                    "collisions return the token stored by the first member.\n",
                ),
            ),
            ("northstar", "compact nonce", "cache"),
            ("identity provider outage", "expired certificate"),
            synthesis_requirements,
            (
                "Discover the latent identity joins across these records and explain "
                "the cohort-specific authentication failures. Compare a genuine "
                "alternative, state the complete causal chain, and give a test whose "
                "result could falsify the selected explanation. Do not modify files."
            ),
        ),
        (
            "cold_closure_gap",
            (
                (
                    "route_codes.md",
                    "The Quartz route means reusable medical shipments transferred "
                    "overnight through Hub B.\n",
                ),
                (
                    "hub_conditions.md",
                    "Hub B holds overnight freight at -22 C. Other hubs remain above -10 C.\n",
                ),
                (
                    "closure_catalog.md",
                    "Reusable shipments use the P7 polymer closure. P7 begins rapid "
                    "radial contraction below -15 C; disposable closures do not.\n",
                ),
                (
                    "seal_tolerance.md",
                    "A radial contraction above 0.4 mm removes gasket compression and "
                    "opens a seal gap. P7 contracts 0.6 mm at -22 C.\n",
                ),
                (
                    "incident_summary.md",
                    "Integrity failures occur only on Quartz reusable shipments after "
                    "the overnight transfer. Pre-transfer checks pass.\n",
                ),
            ),
            ("quartz", "p7", "seal gap"),
            ("handling damage", "sensor calibration"),
            synthesis_requirements,
            (
                "Resolve the route and material aliases, then derive why integrity "
                "fails only after this transfer. Consider a competing explanation, "
                "show the cross-document mechanism, and propose a discriminating "
                "falsification test. Do not modify files."
            ),
        ),
        (
            "renderer_lease_overlap",
            (
                (
                    "incident_metrics.md",
                    "Jobs tagged Heron time out at 45 seconds only for large tenants. "
                    "Two workers often log the same job id before the timeout.\n",
                ),
                (
                    "service_registry.md",
                    "Heron is the internal tag for PDF export rendering.\n",
                ),
                (
                    "worker_policy.md",
                    "An unacknowledged job's worker lease expires after 30 seconds, "
                    "at which point another worker retries it.\n",
                ),
                (
                    "renderer_profile.md",
                    "Large-tenant PDF rendering has a p95 duration of 38 seconds and "
                    "holds an exclusive document lock until completion.\n",
                ),
                (
                    "timeout_contract.md",
                    "A request fails at 45 seconds if it is still waiting for the "
                    "document lock or renderer result.\n",
                ),
            ),
            ("heron", "30", "38", "lock"),
            ("database overload", "network packet loss"),
            synthesis_requirements,
            (
                "Infer the timing-dependent cause of the Heron failures. The answer "
                "must connect the alias, lease, render duration, duplicate execution, "
                "and terminal timeout; compare an alternative and give a falsification "
                "test. Do not modify files."
            ),
        ),
        (
            "orchid_rounding_drift",
            (
                (
                    "cohort_dictionary.md",
                    "Orchid denotes annual prepaid accounts billed in CAD.\n",
                ),
                (
                    "credit_policy.md",
                    "Annual prepaid accounts can receive hundreds of micro-credits. "
                    "Credits are allocated before currency conversion.\n",
                ),
                (
                    "conversion_engine.md",
                    "The CAD path converts and rounds each allocation independently "
                    "to the nearest cent. The USD path sums allocations before rounding.\n",
                ),
                (
                    "statement_checks.md",
                    "Orchid statements differ from the unrounded ledger by several "
                    "cents; the gap grows with the number of micro-credits.\n",
                ),
                (
                    "control_cohorts.md",
                    "Monthly accounts and annual USD accounts do not show the drift.\n",
                ),
            ),
            ("orchid", "micro-credit", "round"),
            ("exchange rate feed", "duplicate charge"),
            synthesis_requirements,
            (
                "Explain the Orchid-only statement drift by joining the cohort, credit, "
                "conversion, and control records. Reject or weaken a real alternative "
                "and state a falsification test that distinguishes the mechanisms. "
                "Do not modify files."
            ),
        ),
        (
            "vega_namespace_mismatch",
            (
                (
                    "dashboard_map.md",
                    "Vega dashboards cover services moved to the observability-v2 "
                    "namespace during the June migration.\n",
                ),
                (
                    "telemetry_audit.md",
                    "Affected services continue emitting complete error counters in "
                    "observability-v2. Raw metric queries return the expected spikes.\n",
                ),
                (
                    "alert_matcher.md",
                    "Alert rules require an exact namespace match. The migrated rules "
                    "still name the legacy observability namespace.\n",
                ),
                (
                    "incident_report.md",
                    "Vega dashboards show error spikes but no corresponding alerts. "
                    "Non-Vega services alert normally.\n",
                ),
            ),
            ("vega", "namespace", "legacy"),
            ("metrics outage", "notification provider outage"),
            synthesis_requirements,
            (
                "Determine why Vega has visible errors but no alerts. Connect the "
                "reporting alias and namespace migration, compare at least one plausible "
                "alternative, and supply a falsification test. Do not modify files."
            ),
        ),
        (
            "inherited_deny_loss",
            (
                (
                    "workspace_catalog.md",
                    "Lumen is the internal name for the restricted research workspace.\n",
                ),
                (
                    "access_model.md",
                    "Research-Contractors inherits its Lumen grant from Contractors "
                    "and its Lumen deny from the Restricted-Research parent policy. "
                    "Deny rules override inherited grants.\n",
                ),
                (
                    "migration_design.md",
                    "The new directory sync flattens inherited group membership into "
                    "direct memberships. It copies grants but omits inherited denies.\n",
                ),
                (
                    "audit_delta.md",
                    "After sync, Research-Contractors unexpectedly gained Lumen access. "
                    "Directly denied users remained blocked.\n",
                ),
            ),
            ("lumen", "inherited deny", "flatten"),
            ("administrator grant", "audit logging bug"),
            synthesis_requirements,
            (
                "Derive the access-control failure across the alias, precedence, and "
                "migration records. Contrast it with a credible alternative and give "
                "a test that would disprove the selected mechanism. Do not modify files."
            ),
        ),
        (
            "stale_alias_embeddings",
            (
                (
                    "release_notes.md",
                    "The Sparrow catalog revision added several former display names "
                    "as aliases. Indexed canonical record names did not change.\n",
                ),
                (
                    "index_pipeline.md",
                    "Incremental indexing re-embeds records only when canonical fields "
                    "change. Alias-only changes are skipped.\n",
                ),
                (
                    "retrieval_policy.md",
                    "Queries above the semantic confidence threshold do not run lexical "
                    "fallback, even when the semantic result set is empty.\n",
                ),
                (
                    "query_analysis.md",
                    "After Sparrow, searches using former names cross the confidence "
                    "threshold but return zero results. Canonical-name searches work.\n",
                ),
                (
                    "index_baseline.md",
                    "The current embedding index was built after canonical names were "
                    "finalized but before the Sparrow aliases were added.\n",
                ),
            ),
            ("sparrow", "alias", "incremental", "fallback"),
            ("document deletion", "permission filtering"),
            synthesis_requirements,
            (
                "Explain the counterintuitive zero-result searches after Sparrow. "
                "Join revision timing, indexing eligibility, and fallback policy; "
                "evaluate an alternative and propose a falsification test. "
                "Do not modify files."
            ),
        ),
        (
            "writer_lease_split",
            (
                (
                    "region_aliases.md",
                    "South clients resolve the write endpoint through zone C.\n",
                ),
                (
                    "failover_timeline.md",
                    "At 10:00 the writer moved from zone C to zone D. The former writer "
                    "kept its lease until 10:00:40.\n",
                ),
                (
                    "dns_policy.md",
                    "Write-endpoint DNS records have a 90-second client TTL. Existing "
                    "South clients retained the zone C address after 10:00.\n",
                ),
                (
                    "fencing.md",
                    "The promoted zone D writer accepts writes immediately. Storage "
                    "does not reject writes from the old writer while its lease remains.\n",
                ),
                (
                    "incident_window.md",
                    "Only South clients report stale writes, all between 10:00:00 and "
                    "10:00:40. Reads converge after that interval.\n",
                ),
            ),
            ("south", "90", "40", "old writer"),
            ("replication backlog", "clock skew"),
            synthesis_requirements,
            (
                "Reconstruct why stale writes are regional and bounded to forty seconds. "
                "The explanation must join routing, TTL, lease, and fencing behavior, "
                "compare a competing hypothesis, and include a falsification test. "
                "Do not modify files."
            ),
        ),
    ]
