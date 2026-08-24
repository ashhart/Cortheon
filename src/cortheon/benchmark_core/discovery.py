"""Repository-derived discovery and assembly of the balanced case suites."""

from __future__ import annotations

import ast
import hashlib
import random
import re
from pathlib import Path

from cortheon.benchmark_core.fixtures_diagnostic import discover_diagnostic_cases
from cortheon.benchmark_core.fixtures_long_horizon import discover_long_horizon_cases
from cortheon.benchmark_core.fixtures_patch import discover_patch_cases
from cortheon.benchmark_core.fixtures_planning import discover_planning_cases
from cortheon.benchmark_core.fixtures_reasoning import discover_reasoning_cases
from cortheon.benchmark_core.fixtures_research import discover_research_cases
from cortheon.benchmark_core.fixtures_semantic import discover_semantic_cases
from cortheon.benchmark_core.models import BenchmarkCase, ImportCase, JoinCase, _case_id


def _imports(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.Import) and node.names:
            module = node.names[0].name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.split(".", 1)[0]
        line_number = getattr(node, "lineno", None)
        if module and module not in found and isinstance(line_number, int):
            found[module] = lines[line_number - 1].strip()
    return found


def _integer_constants(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    found: dict[str, int] = {}
    for node in tree.body:
        name: str | None = None
        value_node: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if name is None or value_node is None or re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", name) is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if abs(value) <= 1_000_000_000:
            found[name] = value
    return found


def discover_cases(repository: Path, *, count: int, seed: int) -> list[ImportCase]:
    """Derive balanced held-out import lookups from the live repository."""

    source_root = repository / "src"
    candidates: list[tuple[str, str, bool]] = []
    file_data: list[tuple[str, str, dict[str, str]]] = []
    universe: set[str] = {
        "argparse",
        "asyncio",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "re",
        "sqlite3",
        "threading",
        "urllib",
    }
    for path in sorted(source_root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            imported = _imports(source)
        except (OSError, SyntaxError, UnicodeError):
            continue
        relative = path.relative_to(repository).as_posix()
        file_data.append((relative, source, imported))
        universe.update(imported)
        candidates.extend((relative, module, True) for module in imported)

    rng = random.Random(seed)
    positives = candidates[:]
    rng.shuffle(positives)
    negatives: list[tuple[str, str, bool]] = []
    for relative, source, imported in file_data:
        lowered = source.lower()
        options = [
            module
            for module in sorted(universe)
            if module not in imported
            and re.search(rf"\b{re.escape(module.lower())}\b", lowered) is None
        ]
        rng.shuffle(options)
        if options:
            negatives.append((relative, options[0], False))
    rng.shuffle(negatives)

    positive_count = (count + 1) // 2
    negative_count = count // 2
    selected = positives[:positive_count] + negatives[:negative_count]
    if len(selected) < count:
        raise ValueError(
            f"repository yielded only {len(selected)} balanced import cases; requested {count}"
        )
    rng.shuffle(selected)

    results: list[ImportCase] = []
    for path, module, expected in selected:
        prompt = (
            f"Inspect the actual repository before answering. Does {path} import "
            f"{module}? Answer yes or no and name the import if present. "
            "Do not modify files."
        )
        results.append(
            ImportCase(
                case_id=_case_id(path, module, expected, seed),
                path=path,
                module=module,
                expected=expected,
                prompt=prompt,
            )
        )
    return results


def discover_join_cases(
    repository: Path,
    *,
    count: int,
    seed: int,
) -> list[JoinCase]:
    """Derive held-out cross-file integer joins from the live repository."""

    constants: list[tuple[str, str, int]] = []
    for path in sorted((repository / "src").rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            found = _integer_constants(source)
        except (OSError, SyntaxError, UnicodeError):
            continue
        relative = path.relative_to(repository).as_posix()
        constants.extend((relative, symbol, value) for symbol, value in sorted(found.items()))

    pairs = [
        (left, right)
        for left_index, left in enumerate(constants)
        for right in constants[left_index + 1 :]
        if left[0] != right[0]
    ]
    random.Random(seed ^ 0xC2055).shuffle(pairs)
    if len(pairs) < count:
        raise ValueError(
            f"repository yielded only {len(pairs)} cross-file constant pairs; requested {count}"
        )

    results: list[JoinCase] = []
    for left, right in pairs[:count]:
        left_path, left_symbol, left_value = left
        right_path, right_symbol, right_value = right
        raw = (f"{seed}\0{left_path}\0{left_symbol}\0{right_path}\0{right_symbol}").encode()
        case_id = "join_" + hashlib.sha256(raw).hexdigest()[:12]
        prompt = (
            f"Inspect the actual repository before answering. Read {left_symbol} in "
            f"{left_path} and {right_symbol} in {right_path}. What is their sum? "
            "Give the numeric result and show the arithmetic. Do not modify files."
        )
        results.append(
            JoinCase(
                case_id=case_id,
                paths=(left_path, right_path),
                symbols=(left_symbol, right_symbol),
                values=(left_value, right_value),
                expected=left_value + right_value,
                prompt=prompt,
            )
        )
    return results


def discover_benchmark_cases(
    repository: Path,
    *,
    count: int,
    seed: int,
    suite: str,
) -> list[BenchmarkCase]:
    if suite == "imports":
        import_cases: list[BenchmarkCase] = list(discover_cases(repository, count=count, seed=seed))
        return import_cases
    if suite == "joins":
        join_cases: list[BenchmarkCase] = list(
            discover_join_cases(repository, count=count, seed=seed)
        )
        return join_cases
    if suite == "patches":
        patch_cases: list[BenchmarkCase] = list(discover_patch_cases(count=count, seed=seed))
        return patch_cases
    if suite == "semantic":
        semantic_cases: list[BenchmarkCase] = list(discover_semantic_cases(count=count, seed=seed))
        return semantic_cases
    if suite == "research":
        research_cases: list[BenchmarkCase] = list(discover_research_cases(count=count, seed=seed))
        return research_cases
    if suite == "debugging":
        diagnostic_cases: list[BenchmarkCase] = list(
            discover_diagnostic_cases(count=count, seed=seed)
        )
        return diagnostic_cases
    if suite == "planning":
        planning_cases: list[BenchmarkCase] = list(discover_planning_cases(count=count, seed=seed))
        return planning_cases
    if suite == "long-horizon":
        long_horizon_cases: list[BenchmarkCase] = list(
            discover_long_horizon_cases(count=count, seed=seed)
        )
        return long_horizon_cases
    if suite == "synthesis":
        synthesis_cases: list[BenchmarkCase] = list(
            discover_reasoning_cases(
                count=count,
                seed=seed,
                mode="novel_synthesis",
            )
        )
        return synthesis_cases
    if suite == "ambiguity":
        ambiguity_cases: list[BenchmarkCase] = list(
            discover_reasoning_cases(
                count=count,
                seed=seed,
                mode="ambiguity",
            )
        )
        return ambiguity_cases
    if suite == "reasoning":
        allocations = [0, 0]
        for index in range(count):
            allocations[index % len(allocations)] += 1
        synthesis_count, ambiguity_count = allocations
        reasoning_cases: list[BenchmarkCase] = [
            *discover_reasoning_cases(
                count=synthesis_count,
                seed=seed,
                mode="novel_synthesis",
            ),
            *discover_reasoning_cases(
                count=ambiguity_count,
                seed=seed,
                mode="ambiguity",
            ),
        ]
        random.Random(seed ^ 0xA8D).shuffle(reasoning_cases)
        return reasoning_cases
    if suite == "northstar":
        if count < 9:
            raise ValueError("the northstar suite requires at least nine cases")
        allocations = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for index in range(count):
            allocations[index % len(allocations)] += 1
        (
            join_count,
            patch_count,
            semantic_count,
            research_count,
            diagnostic_count,
            planning_count,
            long_horizon_count,
            synthesis_count,
            ambiguity_count,
        ) = allocations
        cases = [
            *discover_join_cases(repository, count=join_count, seed=seed),
            *discover_patch_cases(count=patch_count, seed=seed),
            *discover_semantic_cases(count=semantic_count, seed=seed),
            *discover_research_cases(count=research_count, seed=seed),
            *discover_diagnostic_cases(count=diagnostic_count, seed=seed),
            *discover_planning_cases(count=planning_count, seed=seed),
            *discover_long_horizon_cases(count=long_horizon_count, seed=seed),
            *discover_reasoning_cases(
                count=synthesis_count,
                seed=seed,
                mode="novel_synthesis",
            ),
            *discover_reasoning_cases(
                count=ambiguity_count,
                seed=seed,
                mode="ambiguity",
            ),
        ]
        random.Random(seed ^ 0xC067).shuffle(cases)
        return cases
    allocations = [0, 0, 0, 0]
    for index in range(count):
        allocations[index % len(allocations)] += 1
    import_count, join_count, patch_count, semantic_count = allocations
    cases: list[BenchmarkCase] = [
        *discover_cases(repository, count=import_count, seed=seed),
        *discover_join_cases(repository, count=join_count, seed=seed),
        *discover_patch_cases(count=patch_count, seed=seed),
        *discover_semantic_cases(count=semantic_count, seed=seed),
    ]
    random.Random(seed ^ 0x51A7E).shuffle(cases)
    return cases
