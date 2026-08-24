"""Synthetic held-out planning fixtures with ordered obligations."""

from __future__ import annotations

import hashlib
import random

from cortheon.benchmark_core.models import PlanningCase


def discover_planning_cases(*, count: int, seed: int) -> list[PlanningCase]:
    """Return constrained plans with a uniquely gradeable dependency order."""

    definitions = [
        (
            "billing_rollout",
            (
                (
                    "planning/work.md",
                    "| Step | Owner |\n| --- | --- |\n"
                    "| Freeze invoice schema | Priya |\n"
                    "| Migrate billing data | Tomas |\n"
                    "| Deploy billing API | Mei |\n"
                    "| Enable invoice UI | Amina |\n",
                ),
                (
                    "planning/dependencies.md",
                    "Migrate billing data depends on Freeze invoice schema.\n"
                    "Deploy billing API depends on Migrate billing data.\n"
                    "Enable invoice UI depends on Deploy billing API.\n",
                ),
                (
                    "planning/release_constraints.md",
                    "Customer notification must follow Enable invoice UI.\n"
                    "The migration must not begin before the schema is frozen.\n",
                ),
            ),
            (
                "freeze invoice schema",
                "migrate billing data",
                "deploy billing api",
                "enable invoice ui",
                "customer notification",
            ),
            ("priya", "tomas", "mei", "amina"),
            ("deploy billing api first",),
            (
                "Read planning/work.md, planning/dependencies.md, and "
                "planning/release_constraints.md. Produce the only safe ordered rollout "
                "plan for billing, naming each owner and placing customer notification "
                "correctly. Explain the dependency chain. Do not modify files."
            ),
        ),
        (
            "certificate_rotation",
            (
                (
                    "planning/rotation_tasks.md",
                    "Generate new certificate is owned by Noor.\n"
                    "Install certificate in staging is owned by Luis.\n"
                    "Validate staging handshakes is owned by Hana.\n"
                    "Rotate production certificate is owned by Sam.\n",
                ),
                (
                    "planning/rotation_dependencies.md",
                    "Install certificate in staging depends on Generate new certificate.\n"
                    "Validate staging handshakes depends on Install certificate in staging.\n"
                    "Rotate production certificate depends on Validate staging handshakes.\n",
                ),
                (
                    "planning/rotation_policy.md",
                    "Revoke old certificate must follow Rotate production certificate.\n",
                ),
            ),
            (
                "generate new certificate",
                "install certificate in staging",
                "validate staging handshakes",
                "rotate production certificate",
                "revoke old certificate",
            ),
            ("noor", "luis", "hana", "sam"),
            ("revoke old certificate first",),
            (
                "Connect planning/rotation_tasks.md, "
                "planning/rotation_dependencies.md, and planning/rotation_policy.md. "
                "Give the safe ordered certificate-rotation plan with owners and the "
                "reason for the order. Do not modify files."
            ),
        ),
        (
            "warehouse_cutover",
            (
                (
                    "planning/cutover_owners.md",
                    "Snapshot inventory is owned by Ada.\n"
                    "Pause warehouse writes is owned by Ben.\n"
                    "Import inventory snapshot is owned by Chen.\n"
                    "Resume warehouse writes is owned by Dalia.\n",
                ),
                (
                    "planning/cutover_graph.md",
                    "Pause warehouse writes depends on Snapshot inventory.\n"
                    "Import inventory snapshot depends on Pause warehouse writes.\n"
                    "Resume warehouse writes depends on Import inventory snapshot.\n",
                ),
                (
                    "planning/cutover_checks.md",
                    "Reconcile stock totals must follow Import inventory snapshot and "
                    "must precede Resume warehouse writes.\n",
                ),
            ),
            (
                "snapshot inventory",
                "pause warehouse writes",
                "import inventory snapshot",
                "reconcile stock totals",
                "resume warehouse writes",
            ),
            ("ada", "ben", "chen", "dalia"),
            ("resume warehouse writes first",),
            (
                "Read planning/cutover_owners.md, planning/cutover_graph.md, and "
                "planning/cutover_checks.md. Produce a safe ordered warehouse cutover "
                "plan, including owners and the reconciliation gate. Do not modify files."
            ),
        ),
        (
            "mobile_launch",
            (
                (
                    "planning/launch_owners.md",
                    "Complete privacy review is owned by Imani.\n"
                    "Publish signed mobile build is owned by Pavel.\n"
                    "Enable store listing is owned by Sofia.\n"
                    "Open customer rollout is owned by Yun.\n",
                ),
                (
                    "planning/launch_dependencies.md",
                    "Publish signed mobile build depends on Complete privacy review.\n"
                    "Enable store listing depends on Publish signed mobile build.\n"
                    "Open customer rollout depends on Enable store listing.\n",
                ),
                (
                    "planning/launch_policy.md",
                    "Post-launch monitoring starts only after Open customer rollout.\n",
                ),
            ),
            (
                "complete privacy review",
                "publish signed mobile build",
                "enable store listing",
                "open customer rollout",
                "post-launch monitoring",
            ),
            ("imani", "pavel", "sofia", "yun"),
            ("open customer rollout first",),
            (
                "Join planning/launch_owners.md, planning/launch_dependencies.md, and "
                "planning/launch_policy.md into the safe ordered mobile launch plan. "
                "Name owners and justify the ordering. Do not modify files."
            ),
        ),
    ]
    if count > len(definitions):
        raise ValueError(f"planning suite has {len(definitions)} held-out cases; requested {count}")
    random.Random(seed ^ 0xB1A7).shuffle(definitions)
    return [
        PlanningCase(
            case_id="planning_"
            + hashlib.sha256(f"{seed}\0{name}\0{ordered}".encode()).hexdigest()[:12],
            files=case_files,
            ordered_steps=ordered,
            expected=expected,
            forbidden_answers=forbidden,
            prompt=prompt,
        )
        for name, case_files, ordered, expected, forbidden, prompt in definitions[:count]
    ]
