"""Synthetic held-out semantic-recall fixtures with forbidden answers."""

from __future__ import annotations

import hashlib
import random

from cortheon.benchmark_core.models import SemanticCase


def discover_semantic_cases(*, count: int, seed: int) -> list[SemanticCase]:
    """Return multi-hop document tasks installed only in disposable workspaces."""

    definitions = [
        (
            "checkout_approval",
            (
                (
                    "service_catalog.md",
                    "# Service catalog\n"
                    "Search is change class Indigo and owned by Discovery.\n"
                    "Checkout is change class Coral and owned by Commerce.\n"
                    "Messaging is change class Silver and owned by Engagement.\n",
                ),
                (
                    "change_policy.md",
                    "# Emergency change policy\n"
                    "Indigo changes need the Product Director.\n"
                    "Coral changes require approval from the Duty Security Officer.\n"
                    "Silver changes need the Communications Lead.\n",
                ),
                (
                    "org_directory.md",
                    "# Current duty directory\n"
                    "Communications Lead: Javier Ruiz\n"
                    "Duty Security Officer: Amara Okafor\n"
                    "Product Director: Lin Wei\n",
                ),
            ),
            ("amara okafor", "coral", "duty security officer"),
            ("javier ruiz", "lin wei"),
            (
                "Read service_catalog.md, change_policy.md, and org_directory.md as "
                "separate documents. Connect the facts: who must approve an emergency "
                "Checkout change, and why? State the complete evidence chain. "
                "Do not modify files."
            ),
        ),
        (
            "biometric_signoff",
            (
                (
                    "product_register.md",
                    "# Product data register\n"
                    "Nova stores anonymous usage counters.\n"
                    "Kepler Mobile stores biometric templates for device unlock.\n"
                    "Orion stores coarse regional preferences.\n",
                ),
                (
                    "privacy_standard.md",
                    "# Launch privacy standard\n"
                    "Anonymous counters need an engineering review.\n"
                    "Systems storing biometric templates require sign-off from the "
                    "Data Protection Officer before launch.\n"
                    "Regional preferences need a product review.\n",
                ),
                (
                    "governance_roster.md",
                    "# Governance roster\n"
                    "Engineering Reviewer: Mira Shah\n"
                    "Data Protection Officer: Noor Patel\n"
                    "Product Reviewer: Diego Costa\n",
                ),
            ),
            ("noor patel", "biometric templates", "data protection officer"),
            ("mira shah", "diego costa"),
            (
                "Read product_register.md, privacy_standard.md, and "
                "governance_roster.md as separate documents. Connect the facts: who "
                "must sign off Kepler Mobile before launch, and why? State the complete "
                "evidence chain. Do not modify files."
            ),
        ),
        (
            "certificate_owner",
            (
                (
                    "dependency_map.md",
                    "# Recovery dependencies\n"
                    "Atlas ingestion cannot resume until certificate Nimbus is renewed.\n"
                    "Borealis export is waiting on database Cedar.\n"
                    "Cirrus search is waiting on queue Quartz.\n",
                ),
                (
                    "asset_register.md",
                    "# Asset ownership\n"
                    "Database Cedar belongs to Storage Platform.\n"
                    "Certificate Nimbus belongs to Identity Platform.\n"
                    "Queue Quartz belongs to Messaging Platform.\n",
                ),
                (
                    "oncall_roster.md",
                    "# Current on-call roster\n"
                    "Storage Platform: Tomas Eriksen\n"
                    "Identity Platform: Elena Voss\n"
                    "Messaging Platform: Priya Nair\n",
                ),
            ),
            ("elena voss", "certificate nimbus", "identity platform"),
            ("tomas eriksen", "priya nair"),
            (
                "Read dependency_map.md, asset_register.md, and oncall_roster.md as "
                "separate documents. Connect the facts: which current responder owns "
                "the blocker preventing Atlas ingestion from resuming, and why? State "
                "the complete evidence chain. Do not modify files."
            ),
        ),
        (
            "release_delegate",
            (
                (
                    "release_brief.md",
                    "# Release brief\n"
                    "Project Alder is risk band Azure.\n"
                    "Project Aster requires launch authorization for risk band Saffron.\n"
                    "Project Aspen is risk band Olive.\n",
                ),
                (
                    "risk_policy.md",
                    "# Release authorization\n"
                    "Azure releases use the Engineering Sponsor.\n"
                    "Saffron releases are approved by the Regional Risk Delegate.\n"
                    "Olive releases use the Delivery Sponsor.\n",
                ),
                (
                    "regional_roles.md",
                    "# EMEA role holders\n"
                    "Engineering Sponsor: Sofia Rossi\n"
                    "Regional Risk Delegate: Kwame Mensah\n"
                    "Delivery Sponsor: Hannah Cole\n",
                ),
            ),
            ("kwame mensah", "saffron", "regional risk delegate"),
            ("sofia rossi", "hannah cole"),
            (
                "Read release_brief.md, risk_policy.md, and regional_roles.md as "
                "separate documents. Connect the facts: who authorizes Project Aster's "
                "EMEA launch, and why? State the complete evidence chain. "
                "Do not modify files."
            ),
        ),
        (
            "aliased_dependency_owner",
            (
                (
                    "service_index.md",
                    "# Public services\n"
                    "Payments API is internally called Lantern.\n"
                    "Search API is internally called Lattice.\n",
                ),
                (
                    "dependency_notes.md",
                    "# Runtime dependencies\n"
                    "Service Lantern depends on Queue Zephyr.\n"
                    "Service Lattice depends on Queue Quartz.\n",
                ),
                (
                    "duty_roster.md",
                    "# Current ownership\n"
                    "Current owner for Queue Zephyr is Rina Sol.\n"
                    "Current owner for Queue Quartz is Tomas Eriksen.\n",
                ),
            ),
            ("rina sol", "lantern", "queue zephyr"),
            ("tomas eriksen", "lattice", "queue quartz"),
            (
                "Read service_index.md, dependency_notes.md, and duty_roster.md as "
                "separate documents. Connect the facts despite aliases and type-name "
                "variation: who currently owns the Payments API dependency, and why? "
                "State the complete evidence chain. Do not modify files."
            ),
        ),
        (
            "superseded_directory",
            (
                (
                    "service_catalog.md",
                    "# Service catalog\nCheckout is change class Coral.\n",
                ),
                (
                    "change_policy.md",
                    "# Emergency policy\n"
                    "Coral changes require approval from the Duty Security Officer.\n",
                ),
                (
                    "old_directory.md",
                    "# Archived directory\nDuty Security Officer: Lin Wei\n",
                ),
                (
                    "new_directory.md",
                    "# Replacement directory\nDuty Security Officer: Amara Okafor\n",
                ),
                (
                    "current_authority.md",
                    "# Effective assignments\n"
                    "Current owner for Duty Security Officer is Amara Okafor.\n",
                ),
            ),
            ("amara okafor", "coral", "duty security officer"),
            ("lin wei",),
            (
                "Read service_catalog.md, change_policy.md, old_directory.md, "
                "new_directory.md, and current_authority.md as separate documents. "
                "The directories conflict. Reconcile them using explicit current "
                "authority: who approves Checkout, and why? State the complete "
                "evidence chain. Do not modify files."
            ),
        ),
        (
            "conjunctive_privacy_rule",
            (
                (
                    "data_profile.md",
                    "# Data profiles\n"
                    "Kepler processes biometric templates.\n"
                    "Orion processes anonymous usage counters.\n",
                ),
                (
                    "market_scope.md",
                    "# User scope\nKepler serves EEA residents.\nOrion serves US residents.\n",
                ),
                (
                    "privacy_policy.md",
                    "# Conjunctive approval policy\n"
                    "Systems that process biometric templates and serve EEA residents "
                    "require approval from the Data Protection Officer.\n",
                ),
                (
                    "governance_roster.md",
                    "# Current governance roster\n"
                    "Data Protection Officer: Noor Patel\n"
                    "Engineering Reviewer: Mira Shah\n",
                ),
            ),
            (
                "noor patel",
                "biometric templates",
                "eea residents",
                "data protection officer",
            ),
            ("mira shah", "anonymous usage counters", "us residents"),
            (
                "Read data_profile.md, market_scope.md, privacy_policy.md, and "
                "governance_roster.md as separate documents. Apply the policy only "
                "if every condition is established: who must approve Kepler, and why? "
                "State both independently sourced conditions, the rule, and the current "
                "role holder. Do not modify files."
            ),
        ),
        (
            "tabular_dataset_steward",
            (
                (
                    "product_registry.md",
                    "# Product registry\n"
                    "| Product | Runtime service |\n"
                    "| --- | --- |\n"
                    "| Order Console | Helios |\n"
                    "| Search Console | Boreal |\n",
                ),
                (
                    "integration_matrix.md",
                    "# Integration matrix\n"
                    "| Service | Critical dataset |\n"
                    "| --- | --- |\n"
                    "| Helios | Ledger Aurora |\n"
                    "| Boreal | Index Cobalt |\n",
                ),
                (
                    "data_stewards.md",
                    "# Data stewardship\n"
                    "| Dataset | Current steward |\n"
                    "| --- | --- |\n"
                    "| Ledger Aurora | Imani Brooks |\n"
                    "| Index Cobalt | Pavel Novak |\n",
                ),
            ),
            ("imani brooks", "helios", "ledger aurora"),
            ("pavel novak", "boreal", "index cobalt"),
            (
                "Read product_registry.md, integration_matrix.md, and "
                "data_stewards.md as separate documents. Connect the matching table "
                "rows: who is the current steward of the critical dataset used by "
                "Order Console, and why? State the complete cross-document chain. "
                "Do not modify files."
            ),
        ),
        (
            "discovered_dataset_steward",
            (
                (
                    "records/app_landscape.md",
                    "# Application landscape\n"
                    "| Product | Runtime service |\n"
                    "| --- | --- |\n"
                    "| Atlas Portal | Meridian |\n"
                    "| Beacon Portal | Solstice |\n",
                ),
                (
                    "records/data_routes.md",
                    "# Data routes\n"
                    "| Service | Critical dataset |\n"
                    "| --- | --- |\n"
                    "| Meridian | Dataset Ember |\n"
                    "| Solstice | Dataset Frost |\n",
                ),
                (
                    "records/stewardship_assignments.md",
                    "# Stewardship assignments\n"
                    "| Dataset | Current steward |\n"
                    "| --- | --- |\n"
                    "| Dataset Ember | Leila Hassan |\n"
                    "| Dataset Frost | Marco Silva |\n",
                ),
                (
                    "records/team_notes.md",
                    "# General notes\nThe enablement team meets on Tuesdays.\n",
                ),
            ),
            ("leila hassan", "meridian", "dataset ember"),
            ("marco silva", "solstice", "dataset frost"),
            (
                "Search across the live project documents without assuming filenames. "
                "Discover and connect the relevant sources: who is the current steward "
                "of the critical dataset used by Atlas Portal, and why? State the "
                "complete evidence chain and ignore unrelated documents. "
                "Do not modify files."
            ),
        ),
    ]
    if count > len(definitions):
        raise ValueError(f"semantic suite has {len(definitions)} held-out cases; requested {count}")
    random.Random(seed ^ 0x5E6A17).shuffle(definitions)
    cases: list[SemanticCase] = []
    for name, case_files, expected, forbidden_answers, prompt in definitions[:count]:
        raw = f"{seed}\0{name}\0{expected}".encode()
        cases.append(
            SemanticCase(
                case_id="semantic_" + hashlib.sha256(raw).hexdigest()[:12],
                files=case_files,
                expected=expected,
                forbidden_answers=forbidden_answers,
                prompt=prompt,
            )
        )
    return cases
