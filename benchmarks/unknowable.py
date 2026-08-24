"""The mind-blowing test: an API that has never existed anywhere.

Every earlier demo exploited "released after the training cutoff" — a 500B model
might still scrape a version number from somewhere on the internet. This one is
information-theoretically impossible for ANY model, at ANY size, trained on the
entire internet:

  a package with a random name and a deliberately counter-intuitive function
  signature is MINTED AT TEST TIME and installed only on this machine. It has
  never been on PyPI, GitHub, or any corpus. No weights can encode it.

A stock model must guess the keyword names — and its prior actively hurts,
because the real ones contradict the obvious ones (the payload goes in
`cargo`, not `data`; encryption is `cipher_suite`, not `encryption`). The
substrate READS THE LOCAL ARTIFACT — the same AST symbol extractor used for
PyPI packages, pointed at the freshly installed module — and injects the real
signature. Correctness is graded by binding the generated call against the live
installed object: the package itself is the judge.

    CORTHEON_LLM_API_KEY=... python3 benchmarks/unknowable.py \
        --base-url http://localhost:9000/v1 \
        --small Llama-3.2-3B-Instruct-4bit --large Qwen3.6-27B-oQ8-mtp
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cortheon.api_indexer import extract_symbols_from_ast  # noqa: E402
from cortheon.code_check import extract_code_blocks  # noqa: E402
from cortheon.runtime_env import extract_call_probes, run_bind_probe  # noqa: E402
from cortheon.verifier import _venv_python  # noqa: E402

# The counter-intuitive signature: a model's prior points at data=/encryption=/
# shards=; every one of those is wrong. Only reading the artifact reveals cargo/
# cipher_suite/shard_factor/parity_scheme.
ADVERSARIAL_SOURCE = '''\
"""A package that has never existed until this moment."""


def transmit(cargo, *, cipher_suite, shard_factor=3, parity_scheme="reed-solomon"):
    """Transmit cargo over the wire with the given cipher and sharding."""
    return {"cargo": cargo, "cipher": cipher_suite, "shards": shard_factor, "parity": parity_scheme}
'''

# The NEUTRAL signature: parameter names match the intuitive prior
# (data/encryption/shards). This isolates the substrate's claim to ACCESS (it
# must read the artifact because the answer does not exist anywhere) from the
# amplification the adversarial names give (prior-punishment). Under a neutral
# signature a stock model's prior is no longer a liability — if it still fails,
# it fails on the access axis alone, which is the irreducible claim. If it
# happens to guess the intuitive names, that is the prior working, not access,
# and the demo must say so.
NEUTRAL_SOURCE = '''\
"""A package that has never existed until this moment."""


def transmit(data, *, encryption, shards=3, parity="reed-solomon"):
    """Transmit data over the wire with the given encryption and sharding."""
    return {"data": data, "encryption": encryption, "shards": shards, "parity": parity}
'''


def module_source(neutral: bool = False) -> str:
    return NEUTRAL_SOURCE if neutral else ADVERSARIAL_SOURCE


def mint_package(root: Path, neutral: bool = False) -> tuple[str, Path]:
    token = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    name = f"livewire_{token}"
    pkg_dir = root / name
    (pkg_dir).mkdir(parents=True)
    (pkg_dir / f"{name}.py").write_text(module_source(neutral), encoding="utf-8")
    (pkg_dir / "pyproject.toml").write_text(
        f'[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n'
        f'[project]\nname = "{name}"\nversion = "0.0.1"\n\n'
        f'[tool.setuptools]\npy-modules = ["{name}"]\n',
        encoding="utf-8",
    )
    return name, pkg_dir


def substrate_read_signature(name: str, pkg_dir: Path) -> tuple[str, str]:
    """The substrate reads the local artifact with the same extractor it uses
    for PyPI source distributions — no network, no corpus, just the bytes on
    disk that were written seconds ago. Returns (signature, valid call
    template): the raw def signature carries a ``*`` keyword-only marker a weak
    model will copy into the call (a SyntaxError), so the substrate also
    synthesizes a valid call skeleton from the same parse."""
    source = (pkg_dir / f"{name}.py").read_text(encoding="utf-8")
    symbols = extract_symbols_from_ast(ast.parse(source), name, f"{name}.py")
    transmit = next(s for s in symbols if s.name == "transmit")
    signature = f"{name}.{transmit.signature}"
    template = call_template("transmit", transmit.signature)
    return signature, template


def call_template(func_name: str, signature: str | None) -> str:
    """A valid keyword call skeleton derived from a stored signature."""
    if not signature:
        return f"{func_name}(...)"
    try:
        parsed = ast.parse(f"def {signature}: pass").body[0]
    except (SyntaxError, ValueError, IndexError):
        return f"{func_name}(...)"
    args = parsed.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs if a.arg != "self"]
    return f"{func_name}(" + ", ".join(f"{n}=<value>" for n in names) + ")"


def install(pkg_dir: Path, venv_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True, timeout=120
    )
    python = _venv_python(venv_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(pkg_dir)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return python


def prompt_for(name: str, evidence: tuple[str, str] | None) -> str:
    base = (
        f"The package `{name}` is installed in this environment. Write a ```python block that imports "
        "`transmit` from it and calls it to send the payload variable `data`, encrypting with the "
        "'aes-256-gcm' cipher, using 4 shards. Include the actual call line, not just the import."
    )
    if evidence:
        signature, template = evidence
        base += (
            "\n\nSUBSTRATE EVIDENCE read from the installed package — use these EXACT parameter names."
            f"\n  verified signature: {signature}"
            f"\n  fill in this exact call shape: {template}"
        )
    return base


def chat(base_url: str, model: str, api_key: str, content: str, timeout: int = 300) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 900,
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'x'}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def grade(answer: str, name: str, python: Path) -> dict:
    probes = []
    for block in extract_code_blocks(answer):
        probes.extend(extract_call_probes(block, name))
    transmit_probes = [p for p in probes if p["path"].split(".")[-1] == "transmit"] or probes
    results = run_bind_probe(python, name, transmit_probes) if transmit_probes else []
    if results is None:
        return {"passed": False, "detail": "runtime unavailable"}
    if not transmit_probes:
        return {"passed": False, "detail": "no transmit() call found"}
    bad = [
        (r["path"], r["unexpected_kwargs"], r["resolved"])
        for r in results
        if r["unexpected_kwargs"] or not r["resolved"]
    ]
    return {
        "passed": not bad,
        "detail": "binds against the real object" if not bad else f"rejected {bad}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default=os.environ.get("CORTHEON_LLM_BASE_URL", "http://localhost:9000/v1")
    )
    parser.add_argument("--small", default="Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--large", default="Qwen3.6-27B-oQ8-mtp")
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Use a NEUTRAL signature (intuitive param names) instead of the "
        "adversarial one. Isolates the access claim from prior-punishment.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of fresh packages to mint and grade (n>1 beats n=1 theater).",
    )
    args = parser.parse_args()
    api_key = os.environ.get("CORTHEON_LLM_API_KEY", "")
    runs = max(1, args.runs)

    all_runs: list[dict] = []
    totals: dict[str, dict[str, int]] = {}

    for run_index in range(runs):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name, pkg_dir = mint_package(root, neutral=args.neutral)
            evidence = substrate_read_signature(name, pkg_dir)
            python = install(pkg_dir, root / "venv")
            if run_index == 0:
                print(f"Signature family: {'NEUTRAL' if args.neutral else 'ADVERSARIAL'}")
                print(f"First minted package: {name}")
                print(f"Substrate read its real signature from disk: {evidence[0]}")
                print(f"Substrate-synthesized call template:         {evidence[1]}\n")

            contenders = [
                (f"stock small ({args.small})", args.small, None),
                (f"stock LARGE ({args.large})", args.large, None),
                (f"small+substrate ({args.small})", args.small, evidence),
            ]
            rows = []
            for label, model, ev in contenders:
                try:
                    answer = chat(args.base_url, model, api_key, prompt_for(name, ev))
                    result = grade(answer, name, python)
                except Exception as exc:
                    result, answer = {"passed": False, "detail": str(exc)}, ""
                rows.append((label, result, answer))
                totals.setdefault(label, {"pass": 0, "fail": 0})[
                    "pass" if result["passed"] else "fail"
                ] += 1
            all_runs.append(
                {
                    "package": name,
                    "signature": evidence[0],
                    "call_template": evidence[1],
                    "neutral": args.neutral,
                    "rows": [
                        {"contender": label, "result": result, "answer": answer}
                        for label, result, answer in rows
                    ],
                }
            )

    if runs > 1:
        print(
            f"\nAggregate over {runs} runs ({'neutral' if args.neutral else 'adversarial'} signatures):"
        )
        print(f"{'contender':32} {'pass/total':12}")
        print("-" * 50)
        for label, counts in totals.items():
            total = counts["pass"] + counts["fail"]
            print(f"{label:32} {counts['pass']}/{total}")

    out = REPO / ".cortheon" / "benchmarks" / "unknowable.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"runs": all_runs, "neutral": args.neutral, "n": runs},
            indent=2,
        )
    )
    print(f"\nsaved {out} ({runs} run(s))")


if __name__ == "__main__":
    main()
