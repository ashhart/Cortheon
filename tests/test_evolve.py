"""Offline unit tests for Cortheon Evolve (`flash_forge.evolve` + `cortheon evolve` CLI)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from flash_forge.cli import main as cli_main
from flash_forge.evolve import (
    GateConfig,
    build_weighted_curriculum,
    classify_pair_tier,
    evaluate_generation_gate,
    merge_adapter_weights,
    merge_npz_adapters,
    run_evolution_loop,
)
from flash_forge.flywheel import FlywheelPair


def test_contrastive_curriculum_tier_weighting_dedup_and_pii_scrub():
    p_disagree = FlywheelPair(
        prompt="Compute 7*11 on /Users/alice/work",
        chosen="ANSWER: 77",
        rejected="ANSWER: 74",
        domain="math",
        chosen_verified=True,
        rejected_reason="WRONG: got='74' expected='77'",
        metadata={"recovered_from_disagreement": True},
    )
    p_disagree_dup_lower = FlywheelPair(
        prompt="Compute 7*11 on /Users/alice/work",
        chosen="ANSWER: 77",
        rejected="",
        domain="math",
        chosen_verified=True,
    )
    p_crash = FlywheelPair(
        prompt="Solve DP problem",
        chosen="```python\nprint(42)\n```",
        rejected="Traceback (most recent call last):\nZeroDivisionError",
        domain="code",
        chosen_verified=True,
        rejected_reason="RuntimeError: ZeroDivisionError",
    )
    p_fast = FlywheelPair(
        prompt="Simple combinatorics",
        chosen="ANSWER: 16",
        rejected="",
        domain="math",
        chosen_verified=True,
    )
    p_unverified = FlywheelPair(
        prompt="Unverified guess",
        chosen="ANSWER: 999",
        rejected="ANSWER: 111",
        domain="math",
        chosen_verified=False,
    )

    assert classify_pair_tier(p_disagree) == ("disagreement_rescue", 3.0)
    assert classify_pair_tier(p_crash) == ("crash_rescue", 2.0)
    assert classify_pair_tier(p_fast) == ("fast_distill", 1.0)
    assert classify_pair_tier(p_unverified) == ("unverified", 0.0)

    examples, summary = build_weighted_curriculum(
        [p_disagree_dup_lower, p_disagree, p_crash, p_fast, p_unverified]
    )
    assert summary.total_input_pairs == 5
    assert summary.kept_examples == 3
    assert summary.dropped_unverified == 1
    assert summary.deduplicated == 1
    assert summary.by_tier == {"crash_rescue": 1, "disagreement_rescue": 1, "fast_distill": 1}
    assert summary.total_weight == 6.0

    # Highest-weight duplicate won and PII was scrubbed
    top = examples[0]
    assert top.tier == "disagreement_rescue" and top.weight == 3.0
    assert "/Users/alice" not in top.to_dpo_record()["prompt"]
    assert "<HOME>" in top.to_dpo_record()["prompt"]


def test_dual_regime_statistical_gate_accepts_clean_gain_and_blocks_regressions():
    base_s = {f"P{i}": (i < 10) for i in range(20)}
    # Candidate fixes 5 single-shot problems with 0 regressions
    cand_s = {f"P{i}": (i < 15) for i in range(20)}
    base_v = {f"P{i}": (i < 16) for i in range(20)}
    cand_v = {f"P{i}": (i < 17) for i in range(20)}

    ok = evaluate_generation_gate(
        base_single_shot=base_s,
        cand_single_shot=cand_s,
        base_verified=base_v,
        cand_verified=cand_v,
        base_anchor_pct=94.0,
        cand_anchor_pct=93.5,
    )
    assert ok.accepted is True
    assert ok.single_shot.n01_fixed == 5 and ok.single_shot.n10_broken == 0
    assert ok.verified is not None and ok.verified.n01_fixed == 1 and ok.verified.n10_broken == 0
    assert ok.composite_score > 0.0

    # Case 2: Single-shot improves, BUT verified consensus regresses -> MUST REJECT!
    bad_v = {f"P{i}": (i < 14) for i in range(20)}
    rej_v = evaluate_generation_gate(
        base_single_shot=base_s,
        cand_single_shot=cand_s,
        base_verified=base_v,
        cand_verified=bad_v,
    )
    assert rej_v.accepted is False
    assert any("verified consensus regressed" in r for r in rej_v.reasons)

    # Case 3: Anchor benchmark drops > 1.0 pp -> MUST REJECT!
    rej_anchor = evaluate_generation_gate(
        base_single_shot=base_s,
        cand_single_shot=cand_s,
        base_anchor_pct=94.0,
        cand_anchor_pct=91.5,
    )
    assert rej_anchor.accepted is False
    assert any("anchor benchmark dropped" in r for r in rej_anchor.reasons)

    # Case 4: Strict significance gate rejects noisy +2/-1 (p=1.0 > 0.10)
    noisy_s = dict(base_s)
    noisy_s["P0"] = False
    noisy_s["P10"] = True
    noisy_s["P11"] = True
    rej_sig = evaluate_generation_gate(
        base_single_shot=base_s,
        cand_single_shot=noisy_s,
        config=GateConfig(require_significant_or_zero_regressions=True, alpha=0.10),
    )
    assert rej_sig.accepted is False
    assert any("not statistically significant" in r for r in rej_sig.reasons)


def test_sign_consensus_adapter_merge_prevents_interference_and_checks_shapes(tmp_path: Path):
    # Adapter A (+4.0) and Adapter B (+2.0) agree in sign; Adapter C (-0.5) opposes.
    # Plain weighted avg = (4 + 2 - 0.5)/3 = 1.833; sign-consensus drops opposing C -> (4 + 2)/2 = 3.0!
    a1 = {"layer.w": np.array([4.0, -2.0], dtype=np.float32)}
    a2 = {"layer.w": np.array([2.0, -4.0], dtype=np.float32)}
    a3 = {"layer.w": np.array([-0.5, 0.2], dtype=np.float32)}

    merged_sc, norm_w = merge_adapter_weights([a1, a2, a3], sign_consensus=True)
    assert len(norm_w) == 3
    assert np.allclose(merged_sc["layer.w"], np.array([3.0, -3.0], dtype=np.float32), atol=1e-5)

    merged_lin, _ = merge_adapter_weights([a1, a2, a3], sign_consensus=False)
    assert np.allclose(merged_lin["layer.w"], np.array([5.5 / 3.0, -5.8 / 3.0], dtype=np.float32), atol=1e-5)

    # Shape mismatch must raise cleanly
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_adapter_weights([a1, {"layer.w": np.zeros((3,), dtype=np.float32)}])

    # Disk `.npz` round-trip + SHA-256 integrity audit
    p1 = tmp_path / "g1.npz"
    p2 = tmp_path / "g2.npz"
    out_p = tmp_path / "merged.npz"
    np.savez(p1, **a1)
    np.savez(p2, **a2)
    audit = merge_npz_adapters([p1, p2], out_p, weights=[0.75, 0.25], sign_consensus=True)
    assert audit.num_adapters == 2 and audit.num_tensors == 1
    assert len(audit.sha256) == 64
    with np.load(out_p) as z:
        assert np.allclose(z["layer.w"], np.array([3.5, -2.5], dtype=np.float32), atol=1e-5)


def test_autonomous_evolution_loop_rollback_and_top_k_merge(tmp_path: Path):
    async def _harvest(gen: int, active_adapter):
        return [
            FlywheelPair(
                prompt=f"Gen {gen} hard math problem",
                chosen="ANSWER: 42",
                rejected="ANSWER: 41",
                domain="math",
                chosen_verified=True,
                rejected_reason="WRONG: got='41' expected='42'",
            )
        ]

    async def _train(gen: int, dpo_path: Path, sft_path: Path, active_adapter) -> Path:
        assert dpo_path.is_file() and sft_path.is_file()
        ad_path = tmp_path / f"adapter_gen_{gen}.npz"
        val = float(gen) if gen != 2 else -5.0  # gen 2 is bad
        np.savez(ad_path, router=np.array([val, val * 2.0], dtype=np.float32))
        return ad_path

    # Gen 0 base: 4/10; Gen 1: 6/10 (ACCEPT); Gen 2: 5/10 (REJECT -> rollback to Gen 1!); Gen 3: 8/10 (ACCEPT)
    scores = {
        0: {f"P{i}": (i < 4) for i in range(10)},
        1: {f"P{i}": (i < 6) for i in range(10)},
        2: {f"P{i}": (i < 5) for i in range(10)},
        3: {f"P{i}": (i < 8) for i in range(10)},
    }

    async def _eval(gen: int, adapter_path):
        return {
            "single_shot": scores[gen],
            "verified": {f"P{i}": (i < 9) for i in range(10)},
            "anchor_pct": 94.0,
        }

    ledger = asyncio.run(
        run_evolution_loop(
            work_dir=tmp_path / "run",
            max_generations=3,
            patience=2,
            harvest_fn=_harvest,
            train_fn=_train,
            eval_fn=_eval,
        )
    )
    assert len(ledger.records) == 3
    assert [r.accepted for r in ledger.records] == [True, False, True]
    # After rejected Gen 2, active adapter stayed at Gen 1!
    assert "adapter_gen_1.npz" in ledger.records[1].active_adapter_after
    # Top-2 accepted adapters (Gen 1 + Gen 3) merged automatically!
    assert ledger.merged_audit is not None
    assert ledger.merged_audit["num_adapters"] == 2
    merged_file = tmp_path / "run" / "evolved_top_k_merged.npz"
    assert merged_file.is_file()
    assert (tmp_path / "run" / "evolve_ledger.json").is_file()


def test_cli_evolve_gate_and_merge_subcommands(tmp_path: Path, capsys):
    base_json = tmp_path / "base.json"
    cand_json = tmp_path / "cand.json"
    base_json.write_text(
        json.dumps({"results": [{"id": f"P{i}", "passed": i < 5} for i in range(10)]}),
        encoding="utf-8",
    )
    cand_json.write_text(
        json.dumps({"results": [{"id": f"P{i}", "passed": i < 8} for i in range(10)]}),
        encoding="utf-8",
    )
    cli_main(["evolve", "--gate-base", str(base_json), "--gate-cand", str(cand_json), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["accepted"] is True
    assert out["single_shot"]["n01_fixed"] == 3

    # Reversed direction (regression) MUST exit 2!
    with pytest.raises(SystemExit) as exc:
        cli_main(["evolve", "--gate-base", str(cand_json), "--gate-cand", str(base_json)])
    assert exc.value.code == 2
    capsys.readouterr()

    # Merge CLI
    p1 = tmp_path / "m1.npz"
    p2 = tmp_path / "m2.npz"
    out_npz = tmp_path / "m_out.npz"
    np.savez(p1, w=np.array([1.0, 2.0], dtype=np.float32))
    np.savez(p2, w=np.array([3.0, 4.0], dtype=np.float32))
    cli_main(["evolve", "--merge-adapters", str(p1), str(p2), "--out", str(out_npz), "--json"])
    audit = json.loads(capsys.readouterr().out)
    assert audit["num_adapters"] == 2 and Path(audit["out_path"]).is_file()
