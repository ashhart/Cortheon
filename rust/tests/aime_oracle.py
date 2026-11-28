"""Read one synthetic case and evaluate it with the checked-out Python implementation."""

import json
import sys
from types import SimpleNamespace

from bench import run_aime_ablation as aime
from flash_forge import reason_forge as reason


def main(case):
    if "samples" in case:
        config = aime.MAJ8_CONFIGS if len(case["samples"]) == 8 else aime.MAJ4_CONFIGS
        candidates = [(temp, text) for (_, temp), text in zip(config, case["samples"], strict=True)]
        winner, votes = aime.select_maj4_answer(candidates)
        return {
            "config": config,
            "predicted": winner,
            "vote": votes,
            "grade": aime.grade_aime(winner, case["expected"]),
        }

    outcomes = [
        SimpleNamespace(name=str(i), kind=kind, norm_answer=answer,
                        verified_constraints=verified, exec_result=SimpleNamespace(ok=ok))
        for i, (kind, answer, ok, verified) in enumerate(case["outcomes"])
    ]
    return {
        "weights": [reason._branch_weight(o) for o in outcomes],
        "clusters": [[rep, [int(o.name) for o in group]]
                     for rep, group in reason._cluster_answers(outcomes)],
        "format": [reason._answer_satisfies_prompt_format(*pair) for pair in case["format"]],
        "equiv": [reason._answers_equivalent(*pair) for pair in case["equiv"]],
    }


print(json.dumps(main(json.loads(sys.argv[1]))))
