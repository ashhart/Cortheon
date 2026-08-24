"""Synthetic held-out long-horizon build fixtures with hidden scaffolds."""

from __future__ import annotations

import hashlib
import random

from cortheon.benchmark_core.models import LongHorizonCase


def discover_long_horizon_cases(*, count: int, seed: int) -> list[LongHorizonCase]:
    """Return multi-deliverable patches that require several verified edits."""

    definitions = [
        (
            "checkout_rules",
            (
                (
                    "journey/pricing.py",
                    "def discounted(subtotal: float, rate: float) -> float:\n"
                    "    return subtotal + rate\n",
                ),
                (
                    "journey/shipping.py",
                    "def shipping_fee(subtotal: float, threshold: float = 50.0) -> float:\n"
                    "    return 4.99 if subtotal >= threshold else 0.0\n",
                ),
                (
                    "journey/README.md",
                    "# Checkout rules\n\nDiscount and shipping helpers are available.\n",
                ),
                (
                    "test_journey_checkout.py",
                    "from journey.pricing import discounted\n"
                    "from journey.shipping import shipping_fee\n\n"
                    "def test_checkout_rules():\n"
                    "    assert discounted(100.0, 0.2) == 80.0\n"
                    "    assert shipping_fee(50.0, 50.0) == 0.0\n"
                    "    assert shipping_fee(49.0, 50.0) == 4.99\n",
                ),
            ),
            ("journey/pricing.py", "journey/shipping.py", "journey/README.md"),
            (("journey/README.md", "Orders of 50 or more receive free shipping."),),
            "python3 -m pytest -q test_journey_checkout.py",
            (
                "from journey.pricing import discounted; "
                "from journey.shipping import shipping_fee; "
                "assert discounted(0.0, 0.5) == 0.0; "
                "assert discounted(80.0, 0.25) == 60.0; "
                "assert shipping_fee(100.0) == 0.0"
            ),
            (
                "Fix discounted in journey/pricing.py and shipping_fee in "
                "journey/shipping.py so test_journey_checkout.py passes without changing "
                "the test. Also update journey/README.md with the exact sentence "
                "'Orders of 50 or more receive free shipping.' Run python3 -m pytest -q "
                "test_journey_checkout.py after the final edit and report every "
                "deliverable with verification."
            ),
        ),
        (
            "reward_calculation",
            (
                (
                    "journey/points.py",
                    "def awarded(base: int, multiplier: int) -> int:\n"
                    "    return base + multiplier\n",
                ),
                (
                    "journey/bonus.py",
                    "def total_with_bonus(points: float, rate: float) -> float:\n"
                    "    return points + rate\n",
                ),
                (
                    "journey/REWARDS.md",
                    "# Rewards\n\nPoints and bonuses are calculated for members.\n",
                ),
                (
                    "test_journey_rewards.py",
                    "from journey.bonus import total_with_bonus\n"
                    "from journey.points import awarded\n\n"
                    "def test_rewards():\n"
                    "    assert awarded(10, 3) == 30\n"
                    "    assert total_with_bonus(100.0, 0.2) == 120.0\n",
                ),
            ),
            ("journey/points.py", "journey/bonus.py", "journey/REWARDS.md"),
            (("journey/REWARDS.md", "Bonus rates are applied to the awarded points."),),
            "python3 -m pytest -q test_journey_rewards.py",
            (
                "from journey.points import awarded; "
                "from journey.bonus import total_with_bonus; "
                "assert awarded(0, 7) == 0; assert awarded(4, 2) == 8; "
                "assert total_with_bonus(80.0, 0.25) == 100.0"
            ),
            (
                "Correct awarded in journey/points.py and total_with_bonus in "
                "journey/bonus.py without changing test_journey_rewards.py. Update "
                "journey/REWARDS.md with the exact sentence 'Bonus rates are applied "
                "to the awarded points.' Run python3 -m pytest -q "
                "test_journey_rewards.py after the final edit and report verified "
                "completion of every requirement."
            ),
        ),
        (
            "inventory_reservation",
            (
                (
                    "journey/stock.py",
                    "def remaining(stock: int, requested: int) -> int:\n"
                    "    return stock + requested\n",
                ),
                (
                    "journey/audit.py",
                    "def needs_reorder(stock: int, threshold: int) -> bool:\n"
                    "    return stock > threshold\n",
                ),
                (
                    "journey/INVENTORY.md",
                    "# Inventory\n\nReservations are audited.\n",
                ),
                (
                    "test_journey_inventory.py",
                    "from journey.audit import needs_reorder\n"
                    "from journey.stock import remaining\n\n"
                    "def test_reservation():\n"
                    "    assert remaining(10, 3) == 7\n"
                    "    assert needs_reorder(3, 5) is True\n"
                    "    assert needs_reorder(8, 5) is False\n",
                ),
            ),
            ("journey/stock.py", "journey/audit.py", "journey/INVENTORY.md"),
            (("journey/INVENTORY.md", "Successful reservations reduce available stock."),),
            "python3 -m pytest -q test_journey_inventory.py",
            (
                "from journey.stock import remaining; "
                "from journey.audit import needs_reorder; "
                "assert remaining(3, 3) == 0; "
                "assert needs_reorder(5, 5) is True"
            ),
            (
                "Fix remaining in journey/stock.py and needs_reorder in "
                "journey/audit.py without changing test_journey_inventory.py. Update "
                "journey/INVENTORY.md with the exact sentence 'Successful reservations "
                "reduce available stock.' Run python3 -m pytest -q "
                "test_journey_inventory.py after the final edit and report verified "
                "completion of every requirement."
            ),
        ),
    ]
    if count > len(definitions):
        raise ValueError(
            f"long-horizon suite has {len(definitions)} held-out cases; requested {count}"
        )
    random.Random(seed ^ 0x10A6).shuffle(definitions)
    return [
        LongHorizonCase(
            case_id="long_"
            + hashlib.sha256(f"{seed}\0{name}\0{command}".encode()).hexdigest()[:12],
            files=case_files,
            protected_paths=tuple(
                path for path, _content in case_files if path.startswith("test_")
            ),
            required_paths=required_paths,
            required_content=required_content,
            test_command=command,
            hidden_assertions=hidden,
            prompt=prompt,
        )
        for (
            name,
            case_files,
            required_paths,
            required_content,
            command,
            hidden,
            prompt,
        ) in definitions[:count]
    ]
