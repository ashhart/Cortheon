"""Source-mutation helper for the terminal fail-closed tests.

Copies the pi_core graph with per-module string mutations plus the
facade, so tests can prove each mutation fails closed.
"""

from __future__ import annotations

from pathlib import Path

from pi_terminal_constants import SOURCE_DIR


def mutated_source(tmp_path: Path, mutations: dict[str, tuple[str, str]]) -> Path:
    """Copy pi_core with per-module string mutations and a facade; each
    mutation pattern must be present exactly once (load-bearing guard)."""
    root = tmp_path / "cortheon"
    (root / "pi_core").mkdir(parents=True)
    for path in sorted((SOURCE_DIR / "pi_core").glob("*.ts")):
        text = path.read_text(encoding="utf-8")
        if path.stem in mutations:
            old, new = mutations[path.stem]
            assert text.count(old) == 1, (path.stem, old)
            text = text.replace(old, new)
        (root / "pi_core" / path.name).write_text(text, encoding="utf-8")
    facade = root / "pi_extension.ts"
    facade.write_text((SOURCE_DIR / "pi_extension.ts").read_text(encoding="utf-8"))
    return facade
