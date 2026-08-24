"""Translate concrete tasks into deterministic, domain-specific research missions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TaskDomain:
    name: str
    label: str
    source_plan: str
    query_templates: tuple[str, ...]
    keywords: tuple[str, ...]
    is_software: bool = False
    is_biomedical: bool = False
    is_frontier: bool = False


# Domain definitions with source-friendly query templates.
# Each template uses {task} as a placeholder for the original task text.
DOMAINS: tuple[TaskDomain, ...] = (
    TaskDomain(
        name="software_engineering",
        label="Software Engineering",
        source_plan="heuristic",
        is_software=True,
        keywords=(
            "api",
            "framework",
            "library",
            "package",
            "sdk",
            "database",
            "http client",
            "rest api",
            "graphql",
            "websocket",
            "cli",
            "orm",
            "migration",
            "cache",
            "queue",
            "search",
            "auth",
            "testing",
            "logging",
            "serialization",
            "templating",
            "config",
            "validation",
            "image",
            "pdf",
            "datetime",
            "parallel",
            "monitoring",
            "container",
            "cloud",
            "crypto",
            "compression",
            "build",
            "deploy",
            "docker",
            "kubernetes",
            "boto3",
        ),
        query_templates=(
            "current best {task} python package 2025",
            "{task} python implementation comparison",
            "{task} python library source code",
        ),
    ),
    TaskDomain(
        name="biomedical",
        label="Biomedical / Medical",
        source_plan="heuristic",
        is_biomedical=True,
        keywords=(
            "cancer",
            "tumor",
            "therapy",
            "treatment",
            "drug",
            "clinical trial",
            "disease",
            "cure",
            "immunotherapy",
            "senolytic",
            "biomarker",
            "protein",
            "enzyme",
            "pathway",
            "mutation",
            "genetic",
            "alzheimer",
            "diabetes",
            "hypertension",
            "infectious",
            "vaccine",
            "antibody",
            "antibody-drug",
            "oncology",
        ),
        query_templates=(
            "current {task} clinical evidence 2025",
            "{task} recent research findings",
            "{task} mechanism of action",
        ),
    ),
    TaskDomain(
        name="frontier_research",
        label="Frontier Research",
        source_plan="heuristic",
        is_frontier=True,
        keywords=(
            "alife",
            "artificial life",
            "open-ended evolution",
            "frontier",
            "novel architecture",
            "breakthrough",
            "emerging",
            "transformer",
            "attention mechanism",
            "neural architecture",
            "agi",
            "artificial general intelligence",
            "alignment",
            "quantum computing",
            "neuromorphic",
            "spiking neural",
        ),
        query_templates=(
            "current {task} research 2025",
            "{task} recent papers implementation",
            "{task} state of the art benchmarks",
        ),
    ),
    TaskDomain(
        name="general",
        label="General",
        source_plan="heuristic",
        keywords=(),
        query_templates=(
            "current best {task} 2025",
            "{task} recent developments",
        ),
    ),
)


@dataclass(slots=True)
class TaskResearchPlan:
    domain: str
    domain_label: str
    source_plan: str
    research_queries: list[str]
    is_software: bool
    is_biomedical: bool
    is_frontier: bool
    confidence: float
    domain_obj: TaskDomain | None = None
    notes: list[str] = field(default_factory=list)


def classify_task(task: str) -> TaskDomain:
    """Classify a task into a domain based on keyword matching."""
    normalized = " ".join(task.lower().split())
    best_domain = DOMAINS[-1]
    best_score = 0.0

    for domain in DOMAINS:
        if not domain.keywords:
            continue
        score = 0.0
        for keyword in domain.keywords:
            if keyword in normalized:
                score += 1.0
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def translate_task_to_queries(task: str, domain: TaskDomain) -> list[str]:
    """Translate a task into source-friendly research queries for the domain."""
    normalized_task = " ".join(task.lower().split())
    queries: list[str] = []
    seen: set[str] = set()

    for template in domain.query_templates:
        query = template.format(task=normalized_task)
        if query not in seen:
            queries.append(query)
            seen.add(query)

    return queries


def build_task_research_plan(task: str, proposed_action: str | None = None) -> TaskResearchPlan:
    """Build a research plan from a task, classifying domain and generating queries."""
    full_text = " ".join(part for part in [task, proposed_action or ""] if part).strip()
    domain = classify_task(full_text)
    queries = translate_task_to_queries(task, domain)

    normalized = " ".join(full_text.lower().split())
    keyword_hits = sum(1 for kw in domain.keywords if kw in normalized)
    confidence = min(1.0, keyword_hits / max(len(domain.keywords), 1)) if domain.keywords else 0.3

    notes = []
    if domain.name == "general":
        notes.append(
            "Task did not strongly match a specific domain; using general research queries."
        )
    if proposed_action:
        notes.append(f"Proposed action '{proposed_action}' was included in domain classification.")

    return TaskResearchPlan(
        domain=domain.name,
        domain_label=domain.label,
        source_plan=domain.source_plan,
        research_queries=queries,
        is_software=domain.is_software,
        is_biomedical=domain.is_biomedical,
        is_frontier=domain.is_frontier,
        confidence=round(confidence, 3),
        domain_obj=domain,
        notes=notes,
    )


def domain_source_plan_strategy(domain: TaskDomain) -> str:
    """Return the source planner strategy for a domain."""
    if domain.is_biomedical:
        return "heuristic"
    if domain.is_software:
        return "heuristic"
    if domain.is_frontier:
        return "heuristic"
    return "heuristic"


def domain_research_limits(domain: TaskDomain) -> dict[str, int]:
    """Return appropriate research limits for a domain."""
    if domain.is_biomedical:
        return {
            "max_search_results": 0,
            "max_scholarly_results": 8,
            "max_github_results": 0,
            "max_trial_results": 5,
            "max_pages": 0,
        }
    if domain.is_software:
        return {
            "max_search_results": 5,
            "max_scholarly_results": 0,
            "max_github_results": 3,
            "max_trial_results": 0,
            "max_pages": 10,
        }
    if domain.is_frontier:
        return {
            "max_search_results": 5,
            "max_scholarly_results": 8,
            "max_github_results": 3,
            "max_trial_results": 0,
            "max_pages": 10,
        }
    return {
        "max_search_results": 5,
        "max_scholarly_results": 5,
        "max_github_results": 2,
        "max_trial_results": 0,
        "max_pages": 5,
    }
