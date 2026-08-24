"""The concrete hook tracker, composed from focused mixins."""

from __future__ import annotations

from cortheon.cognitive_hooks_core.automatic import AutomaticMixin
from cortheon.cognitive_hooks_core.lifecycle import LifecycleMixin
from cortheon.cognitive_hooks_core.patch_loop import PatchLoopMixin
from cortheon.cognitive_hooks_core.registration import RegistrationMixin


class CognitiveHookTracker(
    RegistrationMixin,
    LifecycleMixin,
    AutomaticMixin,
    PatchLoopMixin,
):
    """Enforce and, when possible, drive Cortheon through host lifecycle hooks."""
