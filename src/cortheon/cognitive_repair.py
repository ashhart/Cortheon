"""Bounded repair helpers that never mutate files."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

from cortheon.cognitive_graph import (
    _candidate_expressions as _candidate_expressions,
)
from cortheon.cognitive_graph import (
    _evaluate_expression as _evaluate_expression,
)
from cortheon.cognitive_graph import (
    _matches_expected as _matches_expected,
)

_TEST_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_./-]*(?:test[^/\s]*|[^/\s]*_test)"
    r"\.(?:py|js|jsx|ts|tsx|go|rs|java)\b",
    flags=re.IGNORECASE,
)
_PROTECT_TEST_RE = re.compile(
    r"\b(?:do\s+not|don't|must\s+not|without)\s+"
    r"(?:chang(?:e|ing)|modif(?:y|ying)|edit(?:ing)?)\s+"
    r"(?:the\s+)?(?:tests?\b|[A-Za-z0-9_./-]*(?:test[^/\s]*|[^/\s]*_test)"
    r"\.(?:py|js|jsx|ts|tsx|go|rs|java)\b)",
    flags=re.IGNORECASE,
)
_SAFE_COMMAND_RE = re.compile(r"^[A-Za-z0-9_./:=,@+\- \t]+$")
_SIMPLE_LITERAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """One exact, single-line replacement supported by live examples."""

    path: str
    old_text: str
    new_text: str
    function_name: str
    examples: int

    def patch(self) -> str:
        """Return an ``apply_patch`` transaction for the host to execute."""

        return (
            "*** Begin Patch\n"
            f"*** Update File: {self.path}\n"
            "@@\n"
            f"-{self.old_text}\n"
            f"+{self.new_text}\n"
            "*** End Patch\n"
        )


@dataclass(frozen=True, slots=True)
class TestInvocation:
    """A parsed, allow-listed test command copied from the user's goal."""

    command_line: str
    executable: str
    arguments: tuple[str, ...]

    def shell_command(self) -> str:
        return shlex.join((self.executable, *self.arguments))


def requested_test_invocation(task: str) -> TestInvocation | None:
    """Extract one safe test invocation explicitly requested by the user."""

    match = re.search(
        r"\brun\s+(.+?)(?=\s+after\b|\s+and\s+(?:report|verify|then)\b|$)",
        task,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    command_line = match.group(1).strip("`'\" ").strip()
    if (
        not command_line
        or len(command_line) > 1_000
        or "\r" in command_line
        or "\n" in command_line
        or _SAFE_COMMAND_RE.fullmatch(command_line) is None
    ):
        return None
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return None
    if len(tokens) < 2 or any(not _safe_command_token(item) for item in tokens):
        return None

    executable, arguments = tokens[0], tokens[1:]
    name = executable.removeprefix("./").casefold()
    python_test = re.fullmatch(r"python(?:3(?:\.\d+)?)?", name) is not None and arguments[:2] in (
        ["-m", "pytest"],
        ["-m", "unittest"],
    )
    direct_pytest = name in {"pytest", "py.test"}
    node_test = name in {"npm", "pnpm", "yarn", "bun"} and (
        arguments[:1] == ["test"] or arguments[:2] == ["run", "test"]
    )
    compiled_test = (
        (name == "cargo" and arguments[:1] == ["test"])
        or (name == "go" and arguments[:1] == ["test"])
        or (name == "dotnet" and arguments[:1] == ["test"])
        or (
            name in {"mvn", "mvnw", "gradle", "gradlew"}
            and any(re.search(r"\btest\b", item, flags=re.IGNORECASE) for item in arguments)
        )
    )
    if not (python_test or direct_pytest or node_test or compiled_test):
        return None
    if any(
        item == "--basetemp"
        or item.startswith("--basetemp=")
        or item == "--rootdir"
        or item.startswith("--rootdir=")
        for item in arguments
    ):
        return None
    return TestInvocation(
        command_line=command_line,
        executable=executable,
        arguments=tuple(arguments),
    )


def requested_check_invocation(task: str) -> TestInvocation | None:
    """Extract the first explicitly requested, allow-listed quality check."""

    for match in re.finditer(
        r"\brun\s+(.+?)(?=\s+after\b|\s+and\s+(?:report|verify|then)\b|$)",
        task,
        flags=re.IGNORECASE,
    ):
        invocation = _check_invocation_from(match.group(1))
        if invocation is not None:
            return invocation
    return None


def _check_invocation_from(raw: str) -> TestInvocation | None:
    command_line = raw.strip("`'\" ").strip()
    if (
        not command_line
        or len(command_line) > 1_000
        or "\r" in command_line
        or "\n" in command_line
        or _SAFE_COMMAND_RE.fullmatch(command_line) is None
    ):
        return None
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return None
    if len(tokens) < 2 or any(not _safe_command_token(item) for item in tokens):
        return None

    executable, arguments = tokens[0], tokens[1:]
    name = executable.removeprefix("./").casefold()
    ruff_check = name == "ruff" and arguments[:1] == ["check"]
    python_check = re.fullmatch(r"python(?:3(?:\.\d+)?)?", name) is not None and arguments[:2] in (
        ["-m", "ruff"],
        ["-m", "mypy"],
        ["-m", "pyright"],
        ["-m", "flake8"],
    )
    direct_check = name in {"mypy", "pyright", "flake8", "eslint", "tsc", "biome"}
    compiled_check = (name == "cargo" and arguments[:1] == ["clippy"]) or (
        name == "go" and arguments[:1] == ["vet"]
    )
    if not (ruff_check or python_check or direct_check or compiled_check):
        return None
    return TestInvocation(
        command_line=command_line,
        executable=executable,
        arguments=tuple(arguments),
    )


def protects_tests(task: str) -> bool:
    """Return whether the user explicitly prohibited test mutation."""

    return _PROTECT_TEST_RE.search(task) is not None


def protected_test_paths(task: str) -> tuple[str, ...]:
    """Return named test paths covered by an explicit no-mutation constraint."""

    if not protects_tests(task):
        return ()
    return tuple(dict.fromkeys(_TEST_PATH_RE.findall(task)))[:12]


def is_test_path(path: str) -> bool:
    """Recognize conventional test filenames without consulting the filesystem."""

    name = PurePosixPath(path).name.casefold()
    return (
        name.startswith("test_")
        or "_test." in name
        or name.endswith((".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx"))
        or name.endswith((".test.js", ".test.jsx", ".test.ts", ".test.tsx"))
    )


def derive_repair_candidates(
    reads: Iterable[tuple[str, str]],
    *,
    limit: int = 3,
) -> list[RepairPlan]:
    """Rank bounded expression repairs against observed literal examples."""

    snapshots = [(path, source) for path, source in reads if path and source]
    implementation = next(
        ((path, source) for path, source in snapshots if not is_test_path(path)),
        None,
    )
    if implementation is None:
        return []
    implementation_path, implementation_source = implementation
    function = _first_simple_return(implementation_source)
    if function is None:
        return []
    function_name, parameters, return_line, return_expression = function

    test_source = "\n".join(source for path, source in snapshots if path != implementation_path)
    examples = _literal_examples(
        test_source,
        function_name=function_name,
        parameter_count=len(parameters),
    )
    if not examples:
        return []

    candidates = _candidate_expressions(return_expression, parameters)
    passing = [
        candidate
        for candidate in candidates
        if all(
            _matches_expected(
                _evaluate_expression(candidate, parameters, values),
                expected,
            )
            for values, expected in examples
        )
    ]
    if not passing:
        return []
    original_operator = _root_operator(return_expression, parameters[0])
    passing.sort(
        key=lambda candidate: (
            int(
                original_operator is not None
                and _root_operator(candidate, parameters[0]) != original_operator
            ),
            _expression_distance(candidate, return_expression),
            candidate,
        )
    )
    return [
        RepairPlan(
            path=implementation_path,
            old_text=return_line,
            new_text=return_line.replace(return_expression, replacement, 1),
            function_name=function_name,
            examples=len(examples),
        )
        for replacement in passing[: max(1, limit)]
    ]


def derive_simple_repair(
    reads: Iterable[tuple[str, str]],
) -> RepairPlan | None:
    """Return the single best-ranked repair candidate, or ``None``."""

    candidates = derive_repair_candidates(reads, limit=1)
    return candidates[0] if candidates else None


def changed_paths_from_diff(content: str) -> set[str]:
    """Return normalized paths named by a unified Git diff."""

    paths: set[str] = set()
    for line in content.splitlines():
        candidate: str | None = None
        if line.startswith("diff --git "):
            try:
                parts = shlex.split(line)
            except ValueError:
                continue
            if len(parts) >= 4:
                candidate = parts[3]
        elif line.startswith("+++ ") or line.startswith("--- "):
            candidate = line[4:].split("\t", 1)[0].strip()
        elif line.startswith("*** Update File: "):
            candidate = line[len("*** Update File: ") :].strip()
        elif line.startswith("*** Add File: "):
            candidate = line[len("*** Add File: ") :].strip()
        elif line.startswith("*** Delete File: "):
            candidate = line[len("*** Delete File: ") :].strip()
        if not candidate or candidate == "/dev/null":
            continue
        if candidate.startswith(("a/", "b/")):
            candidate = candidate[2:]
        if candidate:
            paths.add(candidate)
    return paths


def _safe_command_token(value: str) -> bool:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
    ):
        return False
    return all(part != ".." for part in PurePosixPath(value).parts)


def _first_simple_return(
    source: str,
) -> tuple[str, list[str], str, str] | None:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        definition = re.match(
            r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
            line,
        )
        if definition is None:
            continue
        parameters = [
            item.split("=", 1)[0].split(":", 1)[0].strip()
            for item in definition.group(2).split(",")
        ]
        parameters = [item for item in parameters if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item)]
        for body_line in lines[index + 1 :]:
            if re.match(r"^\s*def\s+", body_line):
                break
            returned = re.match(r"^(\s+)return\s+([^#]+?)\s*$", body_line)
            if returned is not None and parameters:
                return (
                    definition.group(1),
                    parameters,
                    body_line,
                    returned.group(2).strip(),
                )
    return None


def _simple_literal(value: str) -> int | float | bool | None:
    text = value.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    if _SIMPLE_LITERAL_RE.fullmatch(text) is None:
        return None
    return float(text) if "." in text else int(text)


def _literal_examples(
    source: str,
    *,
    function_name: str,
    parameter_count: int,
) -> list[tuple[tuple[int | float | bool, ...], int | float | bool]]:
    assertion = re.compile(
        rf"\b{re.escape(function_name)}\s*\(([^()]*)\)\s*"
        r"(?:==|is)\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)|True|False)"
    )
    examples: list[tuple[tuple[int | float | bool, ...], int | float | bool]] = []
    for match in assertion.finditer(source):
        values = tuple(_simple_literal(item) for item in match.group(1).split(","))
        expected = _simple_literal(match.group(2))
        if (
            len(values) == parameter_count
            and all(item is not None for item in values)
            and expected is not None
        ):
            examples.append(
                (
                    tuple(item for item in values if item is not None),
                    expected,
                )
            )
    return examples


def _root_operator(expression: str, first_parameter: str) -> str | None:
    match = re.match(
        rf"^\s*{re.escape(first_parameter)}\s*([+*/%]|-)",
        expression,
    )
    return match.group(1) if match is not None else None


def _expression_distance(left: str, right: str) -> float:
    distance = abs(len(left) - len(right))
    distance += sum(
        left_item != right_item for left_item, right_item in zip(left, right, strict=False)
    )
    return distance + max(len(left), len(right)) / 10_000
