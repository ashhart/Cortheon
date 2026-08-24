from __future__ import annotations

from cortheon.models import DecisionCheck


def verdict_for(checks: list[DecisionCheck]) -> str:
    if any(check.status == "block" for check in checks):
        return "block"
    if any(check.status == "missing" for check in checks):
        return "needs_evidence"
    return "allow"


def confidence_for(checks: list[DecisionCheck], verdict: str) -> float:
    if verdict == "block":
        return 0.9
    passed = sum(1 for check in checks if check.status == "passed")
    total = max(len(checks), 1)
    if verdict == "allow":
        return round(0.72 + min(0.23, passed / total * 0.23), 3)
    return round(0.42 + min(0.25, passed / total * 0.25), 3)


def missing_evidence(checks: list[DecisionCheck]) -> list[str]:
    return [check.name for check in checks if check.status == "missing"]


def recommended_tools(checks: list[DecisionCheck]) -> list[str]:
    tools: list[str] = []
    for check in checks:
        if check.status != "missing":
            continue
        if check.name == "current_package_evidence":
            tools.extend(["cortheon_recommend", "cortheon_compare", "cortheon_verify"])
        elif check.name == "api_evidence":
            tools.append("cortheon_api_evidence")
        elif check.name == "api_diff_evidence":
            tools.append("cortheon_api_diff")
        elif check.name in {"research_report", "architecture_evidence"}:
            tools.append("cortheon_research")
    return list(dict.fromkeys(tools))


def notes_for(verdict: str, checks: list[DecisionCheck]) -> list[str]:
    if verdict == "allow":
        return ["Decision can proceed with the supplied evidence."]
    if verdict == "block":
        return [
            "Do not proceed until the user explicitly approves the risky action and a rollback plan exists."
        ]
    return ["Pause before acting; gather the missing evidence and ask the substrate again."]
