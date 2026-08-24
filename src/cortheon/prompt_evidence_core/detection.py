"""Prompt package detection and explicit package-bound name extraction."""

from __future__ import annotations

from typing import Any, cast


def install_targets(spec: str, *, split: Any) -> list[str]:
    targets: list[str] = []
    for token in spec.split():
        token = token.strip("\"'")
        if not token or token.startswith("-"):
            continue
        cut = split(r"[\[=<>~!;]", token, maxsplit=1)[0].strip()
        if cut:
            targets.append(cut)
    return targets


def candidate_tiers(
    text: str,
    *,
    import_re: Any,
    from_re: Any,
    install_re: Any,
    package_list_re: Any,
    dotted_re: Any,
    tlds: set[str],
    backtick_re: Any,
    bare_re: Any,
    install_targets: Any,
) -> list[tuple[list[str], bool]]:
    imports: list[str] = []
    for match in import_re.finditer(text):
        for clause in match.group(1).split(","):
            root = clause.strip().split(" as ")[0].strip().split(".")[0]
            if root:
                imports.append(root)
    imports.extend(match.group(1).split(".")[0] for match in from_re.finditer(text))
    installs: list[str] = []
    for match in install_re.finditer(text):
        installs.extend(install_targets(match.group(1)))
    dotted = [
        match.group(1) for match in dotted_re.finditer(text) if match.group(2).lower() not in tlds
    ]
    backticks = [match.group(1) for match in backtick_re.finditer(text)]
    package_lists: list[str] = []
    for match in package_list_re.finditer(text):
        package_lists.extend(item.strip().rstrip(".") for item in match.group(1).split(","))
    return [
        (imports, False),
        (installs, False),
        (dotted, True),
        (backticks, False),
        (package_lists, False),
        (bare_re.findall(text), True),
    ]


def detect_packages(
    text: str,
    probe: Any,
    *,
    candidate_tiers: Any,
    stdlib_names: set[str] | frozenset[str],
    prose_stopwords: set[str],
    common_aliases: dict[str, str],
    reverse_overrides: dict[str, str],
    fullmatch: Any,
    max_probes: int,
    max_packages: int,
) -> list[str]:
    stdlib = {name.lower() for name in stdlib_names}
    found: list[str] = []
    seen: set[str] = set()
    probes = 0
    for names, prose_shaped in candidate_tiers(text):
        for name in names:
            name = cast(str, name)
            if prose_shaped and name.lower() in prose_stopwords:
                continue
            name = common_aliases.get(name, name)
            name = reverse_overrides.get(name, name)
            key = name.lower()
            if key in seen or key in stdlib:
                continue
            seen.add(key)
            if not fullmatch(r"[A-Za-z][A-Za-z0-9._-]+", name):
                continue
            if probes >= max_probes:
                continue
            probes += 1
            metadata = probe(name)
            if prose_shaped and not (getattr(metadata, "requires_python", "") or "").strip():
                continue
            found.append(name)
            if len(found) >= max_packages:
                return found
    return found


def bound_names(
    text: str,
    package: str,
    *,
    reverse_overrides: dict[str, str],
    common_aliases: dict[str, str],
    escape: Any,
    compile_pattern: Any,
    multiline: Any,
    ignorecase: Any,
) -> list[str]:
    roots = {package.lower()}
    roots.update(
        root for root, distribution in reverse_overrides.items() if distribution == package
    )
    roots.update(alias for alias, distribution in common_aliases.items() if distribution == package)
    names: list[str] = []
    root_pattern = "|".join(escape(root) for root in sorted(roots))
    from_re = compile_pattern(
        rf"^\s*from\s+(?:{root_pattern})(?:\.\w+)*\s+import\s+(.+)$",
        multiline | ignorecase,
    )
    for match in from_re.finditer(text):
        for clause in match.group(1).split(","):
            name = clause.strip().split(" as ")[0].strip()
            if name and name != "*" and name not in names:
                names.append(name)
    attr_re = compile_pattern(rf"\b(?:{root_pattern})((?:\.[A-Za-z_]\w*)+)")
    for match in attr_re.finditer(text):
        for name in match.group(1).lstrip(".").split("."):
            if name not in names:
                names.append(name)
    return names
