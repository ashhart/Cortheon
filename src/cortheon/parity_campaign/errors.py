"""Shared failure type for replication-campaign validation and evaluation."""

from __future__ import annotations


class CampaignContractError(ValueError):
    """A replication campaign declaration, result set, or artifact is invalid."""
