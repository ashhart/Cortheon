"""Fail-closed cross-report replication campaign gate for parity reports.

A campaign preregisters an immutable contract (the exact matrix of cells and
every commitment that exists before execution), then maps each cell to its
post-run artifacts in a separate results file. The claim
``replicated_broad_frontier_parity`` passes only when every cell regrades,
from original evidence, to an independently authenticated report whose own
``frontier_parity_gate`` passed, with the complete Cartesian product of at
least three local model families, all four supported hosts, and at least two
declared evaluators. This repository-only subpackage never ships in wheels.
"""

from cortheon.parity_campaign.cli import main
from cortheon.parity_campaign.errors import CampaignContractError
from cortheon.parity_campaign.evaluate import evaluate_replication_campaign
from cortheon.parity_campaign.receipt import (
    evaluation_receipt,
    evaluation_receipt_sha256,
)
from cortheon.parity_campaign.results import results_digest, validate_results
from cortheon.parity_campaign.schema import registration_digest, validate_registration

__all__ = [
    "CampaignContractError",
    "evaluate_replication_campaign",
    "evaluation_receipt",
    "evaluation_receipt_sha256",
    "main",
    "registration_digest",
    "results_digest",
    "validate_registration",
    "validate_results",
]
