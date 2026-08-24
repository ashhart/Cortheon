from __future__ import annotations

from cortheon.connectors.github_core._compat import facade


def repository_metadata(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, str] = {}
    full_name = payload.get("full_name")
    if isinstance(full_name, str):
        metadata["repo"] = full_name
    description = payload.get("description")
    if isinstance(description, str) and description:
        metadata["description"] = description[:240]
    for key, target in (
        ("stargazers_count", "stars"),
        ("forks_count", "forks"),
        ("open_issues_count", "open_issues"),
    ):
        value = facade()._int_or_none(payload.get(key))
        if value is not None:
            metadata[target] = str(value)
    default_branch = payload.get("default_branch")
    if isinstance(default_branch, str):
        metadata["default_branch"] = default_branch
    pushed_at = (
        facade().parse_datetime(payload.get("pushed_at"))
        if isinstance(payload.get("pushed_at"), str)
        else None
    )
    if pushed_at:
        metadata["pushed_at"] = pushed_at.isoformat()
    metadata["archived"] = str(bool(payload.get("archived"))).lower()
    license_payload = payload.get("license")
    if isinstance(license_payload, dict) and isinstance(license_payload.get("spdx_id"), str):
        metadata["license_spdx"] = license_payload["spdx_id"]
    topics = payload.get("topics")
    if isinstance(topics, list):
        clean_topics = [item for item in topics if isinstance(item, str)]
        if clean_topics:
            metadata["topics"] = ",".join(clean_topics[:20])
    homepage = payload.get("homepage")
    if isinstance(homepage, str) and homepage:
        metadata["homepage"] = homepage[:240]
    return metadata


def language_metadata(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    languages = {str(key): value for key, value in payload.items() if isinstance(value, int)}
    if not languages:
        return {}
    total = sum(languages.values()) or 1
    sorted_languages = sorted(languages.items(), key=lambda item: item[1], reverse=True)
    primary, primary_bytes = sorted_languages[0]
    return {
        "primary_language": primary,
        "primary_language_share": f"{primary_bytes / total:.3f}",
        "languages": ",".join(f"{name}:{count}" for name, count in sorted_languages[:8]),
        "language_count": str(len(languages)),
    }


def root_content_metadata(payload: object) -> dict[str, str]:
    if not isinstance(payload, list):
        return {}
    names: list[str] = []
    dirs: list[str] = []
    files: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        item_type = item.get("type")
        if not isinstance(name, str):
            continue
        names.append(name)
        if item_type == "dir":
            dirs.append(name)
        elif item_type == "file":
            files.append(name)
    metadata: dict[str, str] = {}
    if names:
        metadata["root_entries"] = ",".join(names[:40])
    if dirs:
        metadata["root_dirs"] = ",".join(dirs[:25])
    if files:
        metadata["root_files"] = ",".join(files[:25])
    return metadata


def readme_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    encoding = payload.get("encoding")
    if not isinstance(content, str) or encoding != "base64":
        return ""
    try:
        decoded = facade().base64.b64decode(content, validate=False)
    except Exception:
        return ""
    return decoded.decode("utf-8", errors="replace")


def readme_metadata(readme: str) -> dict[str, str]:
    normalized = facade().normalize_readme(readme)
    metadata = {
        "readme_excerpt": normalized[:600],
        "readme_length": str(len(normalized)),
    }
    lower = normalized.lower()
    metadata["readme_has_install"] = str(
        any(
            cue in lower
            for cue in (
                "pip install",
                "npm install",
                "conda install",
                "cargo install",
                "installation",
            )
        )
    ).lower()
    metadata["readme_has_usage"] = str(
        any(cue in lower for cue in ("usage", "quickstart", "example", "getting started"))
    ).lower()
    metadata["readme_has_citation"] = str("citation" in lower or "bibtex" in lower).lower()
    metadata["readme_has_benchmark"] = str("benchmark" in lower or "leaderboard" in lower).lower()
    return metadata


def normalize_readme(readme: str) -> str:
    return facade().re.sub(r"\s+", " ", readme).strip()


def implementation_signals(metadata: dict[str, str]) -> list[str]:
    api = facade()
    root = set(api.split_metadata_csv(metadata.get("root_entries", "")))
    files = {item.lower() for item in api.split_metadata_csv(metadata.get("root_files", ""))}
    dirs = {item.lower() for item in api.split_metadata_csv(metadata.get("root_dirs", ""))}
    signals: list[str] = []
    if "pyproject.toml" in files or "setup.py" in files or "requirements.txt" in files:
        signals.append("python_package")
    if "package.json" in files:
        signals.append("javascript_package")
    if "cargo.toml" in files:
        signals.append("rust_package")
    if "go.mod" in files:
        signals.append("go_module")
    if any(name in files for name in ("dockerfile", "docker-compose.yml")):
        signals.append("containerized")
    if any(name.startswith(".github") for name in dirs) or ".github" in root:
        signals.append("ci_config")
    if "tests" in dirs or "test" in dirs or any(name.startswith("test_") for name in files):
        signals.append("tests")
    if "docs" in dirs or "doc" in dirs:
        signals.append("docs")
    if any(name.startswith("license") for name in files):
        signals.append("license_file")
    if metadata.get("readme_has_install") == "true":
        signals.append("install_docs")
    if metadata.get("readme_has_usage") == "true":
        signals.append("usage_docs")
    if metadata.get("readme_has_benchmark") == "true":
        signals.append("benchmark_docs")
    return signals


def split_metadata_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def repository_health_score(metadata: dict[str, str]) -> float:
    score = 0.35
    api = facade()
    stars = api.int_or_zero(metadata.get("stars"))
    if stars >= 10_000:
        score += 0.18
    elif stars >= 1_000:
        score += 0.14
    elif stars >= 100:
        score += 0.09
    elif stars >= 10:
        score += 0.04
    if metadata.get("archived") == "true":
        score -= 0.25
    if metadata.get("license_spdx") and metadata.get("license_spdx") != "NOASSERTION":
        score += 0.08
    signals = api.implementation_signals(metadata)
    score += min(0.24, len(signals) * 0.035)
    if metadata.get("pushed_at"):
        score += 0.07
    if metadata.get("readme_has_usage") == "true":
        score += 0.04
    return max(0.0, min(score, 0.98))


def adjusted_repository_confidence(current: float, metadata: dict[str, str]) -> float:
    health = facade().repository_health_score(metadata)
    return round(max(0.2, min(0.97, current * 0.68 + health * 0.32)), 3)


def int_or_zero(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0
