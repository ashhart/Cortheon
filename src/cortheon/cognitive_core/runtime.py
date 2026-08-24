"""Composition root that assembles CognitiveRuntime from focused mixins."""

from __future__ import annotations

from cortheon.cognitive_core.runtime_brief import BriefMixin
from cortheon.cognitive_core.runtime_completion import CompletionMixin
from cortheon.cognitive_core.runtime_context import ContextMixin
from cortheon.cognitive_core.runtime_discovery import DiscoveryMixin
from cortheon.cognitive_core.runtime_failed_verification import FailedVerificationMixin
from cortheon.cognitive_core.runtime_hypotheses import HypothesisMixin
from cortheon.cognitive_core.runtime_lifecycle import LifecycleMixin
from cortheon.cognitive_core.runtime_observations import ObservationMixin
from cortheon.cognitive_core.runtime_recommendation import RecommendationMixin
from cortheon.cognitive_core.runtime_request_flow import RequestFlowMixin
from cortheon.cognitive_core.runtime_requests import RequestMixin
from cortheon.cognitive_core.runtime_verification import VerificationMixin


class CognitiveRuntime(
    LifecycleMixin,
    ObservationMixin,
    HypothesisMixin,
    RecommendationMixin,
    DiscoveryMixin,
    RequestMixin,
    RequestFlowMixin,
    VerificationMixin,
    FailedVerificationMixin,
    CompletionMixin,
    ContextMixin,
    BriefMixin,
):
    """Thread-safe, memory-only coordinator for bounded investigations."""
