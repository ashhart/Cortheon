"""Definition tables for the ambiguity-resolution reasoning fixtures."""

from __future__ import annotations

from typing import Any


def _ambiguity_definitions() -> list[tuple[Any, ...]]:
    """Return the held-out ambiguity-resolution reasoning case definitions."""

    ambiguity_requirements = (
        (
            "ambiguous",
            "cannot determine",
            "insufficient",
            "conflict",
            "underspecified",
            "not actionable",
            "no basis",
        ),
        ("clarify", "clarification", "which", "need to know"),
        ("alternative", "interpretation", "either"),
    )
    return [
        (
            "ambiguous_atlas",
            (
                (
                    "product_index.md",
                    "Atlas Portal is the customer-facing analytics application.\n",
                ),
                (
                    "platform_index.md",
                    "Atlas Pipeline is the internal event-ingestion service.\n",
                ),
                (
                    "request.md",
                    "The request says only: deploy Atlas. It names no environment, "
                    "component, or success criterion.\n",
                ),
            ),
            ("atlas portal", "atlas pipeline"),
            ("deploy atlas portal", "deploy atlas pipeline"),
            (
                (
                    "ambiguous",
                    "cannot determine",
                    "insufficient",
                    "conflict",
                    "underspecified",
                    "not actionable",
                    "no basis",
                ),
                ("clarify", "clarification", "which", "need to know"),
                ("alternative", "interpretation", "either"),
            ),
            (
                "Read the documents and decide what action is justified by 'deploy "
                "Atlas'. Preserve ambiguity rather than guessing: enumerate the viable "
                "interpretations and ask the smallest clarification that separates "
                "them. Do not modify files."
            ),
        ),
        (
            "ambiguous_conversion",
            (
                (
                    "growth_metrics.md",
                    "Growth defines conversion as visitor-to-account signup.\n",
                ),
                (
                    "commerce_metrics.md",
                    "Commerce defines conversion as checkout-to-paid-order.\n",
                ),
                (
                    "request.md",
                    "The request asks to improve conversion by 10% without naming the "
                    "funnel, baseline period, or owning team.\n",
                ),
            ),
            ("visitor-to-account", "checkout-to-paid-order"),
            ("optimize signup conversion", "optimize purchase conversion"),
            (
                (
                    "ambiguous",
                    "cannot determine",
                    "insufficient",
                    "conflict",
                    "underspecified",
                    "not actionable",
                    "no basis",
                ),
                ("clarify", "clarification", "which", "need to know"),
                ("alternative", "interpretation", "either"),
            ),
            (
                "Determine whether 'improve conversion by 10%' is actionable from "
                "these documents. State the competing interpretations and the minimum "
                "clarification required before planning. Do not modify files."
            ),
        ),
        (
            "ambiguous_owner",
            (
                ("roster_a.md", "Roster A names Priya Nair as the owner of Nimbus.\n"),
                ("roster_b.md", "Roster B names Elena Voss as the owner of Nimbus.\n"),
                (
                    "authority.md",
                    "Neither roster includes an effective date, supersession marker, "
                    "or authority level.\n",
                ),
            ),
            ("priya nair", "elena voss"),
            ("owner is priya nair", "owner is elena voss"),
            (
                (
                    "ambiguous",
                    "cannot determine",
                    "unresolved",
                    "conflict",
                    "no basis",
                    "nothing in them breaks the tie",
                ),
                ("effective date", "authority", "supersession"),
                ("alternative", "either", "both"),
            ),
            (
                "Who currently owns Nimbus? Use the documents, but do not invent a "
                "tie-break. State both live alternatives and the exact evidence needed "
                "to resolve the conflict. Do not modify files."
            ),
        ),
        (
            "ambiguous_latency",
            (
                ("slo.md", "Interactive traffic is governed by p99 latency.\n"),
                ("dashboard.md", "The executive dashboard reports median p50 latency.\n"),
                (
                    "request.md",
                    "The request says optimize latency without naming traffic class, "
                    "percentile, or acceptable regression elsewhere.\n",
                ),
            ),
            ("p99", "p50"),
            ("optimize p99", "optimize p50"),
            (
                (
                    "ambiguous",
                    "cannot determine",
                    "insufficient",
                    "conflict",
                    "underspecified",
                    "not actionable",
                    "no basis",
                ),
                ("clarify", "clarification", "which", "need to know"),
                ("alternative", "metric", "percentile"),
            ),
            (
                "Assess the request to 'optimize latency'. Enumerate the incompatible "
                "metrics and ask the minimum clarification needed before changing "
                "anything. Do not modify files."
            ),
        ),
        (
            "ambiguous_ttl_unit",
            (
                ("service_config.md", "The cache service stores TTL values in seconds.\n"),
                (
                    "admin_console.md",
                    "The admin console labels its cache TTL input in milliseconds.\n",
                ),
                (
                    "request.md",
                    "The request says set the cache TTL to 300 without naming an "
                    "interface or unit.\n",
                ),
            ),
            ("seconds", "milliseconds"),
            ("set ttl to 300 seconds", "set ttl to 300 milliseconds"),
            ambiguity_requirements,
            (
                "Assess whether 'set the cache TTL to 300' is actionable. Preserve "
                "the unit ambiguity, enumerate the live interpretations, and ask the "
                "smallest clarification needed before changing anything. Do not "
                "modify files."
            ),
        ),
        (
            "ambiguous_margin_change",
            (
                (
                    "finance_glossary.md",
                    "Finance reports margin improvement as a relative percent change.\n",
                ),
                (
                    "board_glossary.md",
                    "Board reporting expresses margin improvement in percentage points.\n",
                ),
                (
                    "request.md",
                    "The request says improve margin by 5 without naming the reporting "
                    "convention or baseline.\n",
                ),
            ),
            ("relative percent", "percentage points"),
            ("improve margin by 5 percent", "improve margin by 5 percentage points"),
            ambiguity_requirements,
            (
                "Determine whether 'improve margin by 5' has one justified meaning. "
                "Enumerate the incompatible interpretations and ask the minimum "
                "clarification before planning. Do not modify files."
            ),
        ),
        (
            "ambiguous_deadline_zone",
            (
                ("engineering_handbook.md", "Engineering release deadlines use UTC.\n"),
                (
                    "customer_contract.md",
                    "Customer delivery deadlines use America/New_York local time.\n",
                ),
                (
                    "request.md",
                    "The request says ship by 17:00 without naming whether this is a "
                    "release or customer delivery deadline.\n",
                ),
            ),
            ("utc", "america/new_york"),
            ("ship at 17:00 utc", "ship at 17:00 america/new_york"),
            ambiguity_requirements,
            (
                "Assess 'ship by 17:00' from the documents. Preserve the timezone and "
                "deadline ambiguity, state both live interpretations, and ask the "
                "smallest clarification required before acting. Do not modify files."
            ),
        ),
        (
            "ambiguous_account_delete",
            (
                (
                    "identity_terms.md",
                    "Identity uses account to mean an individual user profile.\n",
                ),
                (
                    "billing_terms.md",
                    "Billing uses account to mean a tenant subscription and its invoices.\n",
                ),
                (
                    "request.md",
                    "The request says delete the Orion account without naming the "
                    "system, subject, or retention requirement.\n",
                ),
            ),
            ("user profile", "tenant subscription"),
            ("delete the user profile", "delete the tenant subscription"),
            ambiguity_requirements,
            (
                "Decide whether 'delete the Orion account' authorizes one action. "
                "Enumerate the incompatible meanings and ask the minimum clarification "
                "needed before any deletion. Do not modify files."
            ),
        ),
        (
            "ambiguous_reliability",
            (
                (
                    "sre_objectives.md",
                    "SRE defines reliability as service availability during requests.\n",
                ),
                (
                    "storage_objectives.md",
                    "Storage defines reliability as data durability after acknowledged writes.\n",
                ),
                (
                    "request.md",
                    "The request says improve reliability without naming the workload, "
                    "failure mode, or objective.\n",
                ),
            ),
            ("availability", "durability"),
            ("optimize availability", "optimize durability"),
            ambiguity_requirements,
            (
                "Assess the request to 'improve reliability'. Preserve the competing "
                "definitions, enumerate the live objectives, and ask the smallest "
                "clarification before planning. Do not modify files."
            ),
        ),
        (
            "ambiguous_stable_release",
            (
                ("desktop_channel.md", "Desktop labels release 4.2 as stable.\n"),
                ("server_channel.md", "Server labels release 5.0 as stable.\n"),
                (
                    "request.md",
                    "The request says deploy the stable release without naming the "
                    "product channel or target fleet.\n",
                ),
            ),
            ("4.2", "5.0"),
            ("deploy release 4.2", "deploy release 5.0"),
            ambiguity_requirements,
            (
                "Determine what 'deploy the stable release' justifies. Enumerate the "
                "live release interpretations and ask the minimum clarification that "
                "separates them. Do not modify files."
            ),
        ),
        (
            "ambiguous_west_region",
            (
                ("eu_runbook.md", "The EU team calls eu-west-1 the West region.\n"),
                ("us_runbook.md", "The US team calls us-west-2 the West region.\n"),
                (
                    "request.md",
                    "The request says deploy to West without naming a geography, cloud "
                    "account, or runbook.\n",
                ),
            ),
            ("eu-west-1", "us-west-2"),
            ("deploy to eu-west-1", "deploy to us-west-2"),
            ambiguity_requirements,
            (
                "Assess 'deploy to West' without guessing. Enumerate both viable "
                "regions and ask the smallest clarification required before a deploy. "
                "Do not modify files."
            ),
        ),
        (
            "ambiguous_batch",
            (
                (
                    "data_glossary.md",
                    "The data platform uses batch to mean the nightly ETL workload.\n",
                ),
                (
                    "ml_glossary.md",
                    "The inference platform uses batch to mean requests grouped into "
                    "one model execution.\n",
                ),
                (
                    "request.md",
                    "The request says optimize batch without naming a platform, metric, "
                    "or workload.\n",
                ),
            ),
            ("nightly etl", "model execution"),
            ("optimize nightly etl", "optimize model execution"),
            ambiguity_requirements,
            (
                "Determine whether 'optimize batch' is actionable. State the competing "
                "platform interpretations and ask the minimum clarification before "
                "planning. Do not modify files."
            ),
        ),
    ]
