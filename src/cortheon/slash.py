from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from cortheon.auto_evidence import AutoEvidenceLimits, EvidenceAcquisitionLoop
from cortheon.engine import CortheonEngine
from cortheon.knowledge_pool import KnowledgePooler
from cortheon.ledger import EvidenceLedger
from cortheon.models import ApiEvidenceReport, RecommendationReport, ResearchReport
from cortheon.research import ResearchEngine

DEFAULT_LIMITS = AutoEvidenceLimits(
    max_search_results=5,
    max_scholarly_results=8,
    max_github_results=3,
    max_trial_results=5,
    max_follow_up_queries=2,
    max_adaptive_queries=1,
    max_artifact_inspections=2,
    max_pages=10,
    max_depth=1,
)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = CortheonEngine(ledger=EvidenceLedger(Path(args.ledger_dir)))
    text = " ".join(args.text).strip()
    if not text:
        raise SystemExit("Missing slash command arguments.")

    if args.slash_command == "answer":
        task, proposed_action = split_task_action(text)
        report = KnowledgePooler(engine, source_planner_strategy=args.source_planner).run(
            task,
            proposed_action=proposed_action,
            limits=DEFAULT_LIMITS,
        )
        print("# Cortheon Answer")
        print_verdict(report.verdict, report.answer_status)
        print(f"Task: {report.task}")
        if report.proposed_action:
            print(f"Proposed action: {report.proposed_action}")
        print_section("Best Supported Approach", report.best_supported_approach)
        print_list("Evidence Gaps", report.evidence_gaps[:8])
        print_sources(report.source_summaries)
        print_counts(report.discovery_counts)
        print_guardrail(report.verdict)
        return

    if args.slash_command == "decide":
        task, proposed_action = split_task_action(text)
        if not proposed_action:
            raise SystemExit("Use: /cortheon-decide <task> :: <proposed action>")
        report = EvidenceAcquisitionLoop(engine, source_planner_strategy=args.source_planner).run(
            task,
            proposed_action=proposed_action,
            limits=DEFAULT_LIMITS,
        )
        print("# Cortheon Decision")
        print_verdict(report.final_decision.verdict, None)
        print(f"Task: {report.task}")
        print(f"Proposed action: {report.proposed_action}")
        print_list("Required Evidence", report.final_decision.required_evidence)
        print_list("Evidence Tags", report.evidence_tags)
        if report.agent_runs:
            print("\nEvidence Agents:")
            for run in report.agent_runs:
                print(f"- {run.agent}: {run.status} ({run.missing_evidence})")
                print(f"  {run.summary}")
        print_guardrail(report.final_decision.verdict)
        return

    if args.slash_command == "research":
        report = ResearchEngine(
            ledger=engine.ledger,
            source_planner_strategy=args.source_planner,
        ).research(
            text,
            max_search_results=DEFAULT_LIMITS.max_search_results,
            max_scholarly_results=DEFAULT_LIMITS.max_scholarly_results,
            max_github_results=DEFAULT_LIMITS.max_github_results,
            max_trial_results=DEFAULT_LIMITS.max_trial_results,
            max_follow_up_queries=DEFAULT_LIMITS.max_follow_up_queries,
            max_adaptive_queries=DEFAULT_LIMITS.max_adaptive_queries,
            max_artifact_inspections=DEFAULT_LIMITS.max_artifact_inspections,
            max_pages=DEFAULT_LIMITS.max_pages,
            max_depth=DEFAULT_LIMITS.max_depth,
        )
        print_research(report)
        return

    if args.slash_command == "api":
        package, query = split_package_query(text)
        report = engine.retrieve_api_evidence(package, query)
        print_api(report)
        return

    if args.slash_command == "recommend":
        report = engine.recommend(text)
        print_recommendation(report)
        return

    raise SystemExit(f"Unknown slash command: {args.slash_command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cortheon.slash",
        description="Slash-command runtime for Cortheon harness integrations.",
    )
    parser.add_argument(
        "--ledger-dir",
        default=".cortheon",
        help="Directory for Cortheon audit reports.",
    )
    parser.add_argument(
        "--source-planner",
        choices=["heuristic", "llm", "auto"],
        default="auto",
        help="Source planner for research-backed slash commands.",
    )
    parser.add_argument(
        "slash_command",
        choices=["answer", "decide", "research", "api", "recommend"],
    )
    parser.add_argument("text", nargs=argparse.REMAINDER)
    return parser


def split_task_action(text: str) -> tuple[str, str | None]:
    if "::" in text:
        task, action = text.split("::", 1)
        return task.strip(), action.strip() or None
    if "\n---\n" in text:
        task, action = text.split("\n---\n", 1)
        return task.strip(), action.strip() or None
    return text.strip(), None


def split_package_query(text: str) -> tuple[str, str]:
    if "::" in text:
        package, query = text.split("::", 1)
        package = package.strip()
        query = query.strip()
    else:
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            raise SystemExit(
                "Use: /cortheon-api <package> :: <symbol> or /cortheon-api <package> <symbol>"
            )
        package, query = parts[0], parts[1]
    if not package or not query:
        raise SystemExit("Use: /cortheon-api <package> :: <symbol>")
    return package, query


def print_verdict(verdict: str, status: str | None) -> None:
    line = f"Verdict: {verdict}"
    if status:
        line += f" | answer_status: {status}"
    print(line)


def print_section(title: str, body: str | None) -> None:
    if body:
        print(f"\n{title}:")
        print(body)


def print_list(title: str, items: list[str]) -> None:
    if not items:
        return
    print(f"\n{title}:")
    for item in items:
        print(f"- {item}")


def print_sources(sources: Sequence[object]) -> None:
    if not sources:
        return
    print("\nTop Sources:")
    for source in sources[:6]:
        title = getattr(source, "title", None) or getattr(source, "url", "")
        url = getattr(source, "url", "")
        source_type = getattr(source, "source_type", "source")
        print(f"- {title} [{source_type}]")
        if url:
            print(f"  {url}")
        claims = getattr(source, "derived_claims", [])
        for claim in claims[:1]:
            print(f"  claim: {claim}")


def print_counts(counts: dict[str, int]) -> None:
    if counts:
        joined = ", ".join(f"{key}={value}" for key, value in counts.items())
        print(f"\nDiscovery Counts: {joined}")


def print_guardrail(verdict: str) -> None:
    print("\nAgent Instruction:")
    if verdict == "allow":
        print(
            "- You may proceed, but only inside the supported approach and visible source constraints."
        )
    elif verdict == "needs_evidence":
        print("- Do not write production code or make a final commitment yet.")
        print("- Gather the missing evidence above, then ask Cortheon again.")
    elif verdict == "block":
        print("- Do not perform the requested action.")
    else:
        print("- Treat this result as advisory and keep uncertainty visible.")


def print_research(report: ResearchReport) -> None:
    print("# Cortheon Research")
    print(f"Topic: {report.topic}")
    print(
        "Counts: "
        f"search_results={len(report.search_results)}, "
        f"scholarly_works={len(report.scholarly_works)}, "
        f"pages={len(report.crawled_pages)}, "
        f"artifacts={len(report.artifacts)}, "
        f"claims={len(report.claims)}"
    )
    if report.synthesis:
        print(f"Synthesis status: {report.synthesis.status}")
        print(f"Confidence: {report.synthesis.confidence}")
        print_section("Current Best Direction", report.synthesis.current_best_direction)
        print_list("Evidence Gaps", report.synthesis.evidence_gaps[:8])
    if report.source_plan:
        print("\nSource Plan:")
        for item in report.source_plan:
            status = "use" if item.selected else "skip"
            print(f"- {status}: {item.name} [{item.source_type}] budget={item.budget}")
            print(f"  {item.reason}")
    if report.source_lineage:
        print("\nTop Sources:")
        for item in report.source_lineage[:6]:
            title = item.source_title or item.source_url
            print(f"- {title} [{item.source_type}]")
            print(f"  {item.source_url}")


def print_api(report: ApiEvidenceReport) -> None:
    print("# Cortheon API Evidence")
    print(f"Package: {report.package}")
    print(f"Version: {report.version}")
    print(f"Query: {report.query}")
    print(f"Artifact: {report.artifact_filename or 'none'}")
    print(f"Matches: {len(report.matches)} / indexed_symbols={report.total_symbols}")
    for symbol in report.matches[:10]:
        signature = f" {symbol.signature}" if symbol.signature else ""
        print(f"- {symbol.qualname} [{symbol.kind}]{signature}")
        print(f"  {symbol.file_path}:{symbol.line}")
    if not report.matches:
        print("\nAgent Instruction:")
        print(
            "- Do not use this API in production code unless another source-derived match is found."
        )


def print_recommendation(report: RecommendationReport) -> None:
    print("# Cortheon Recommendation")
    print(f"Task: {report.task}")
    print(f"Profile: {report.profile or 'adhoc'}")
    print(f"Winner: {report.winner or 'none'}")
    if report.notes:
        print_list("Notes", report.notes)
    if report.candidates:
        print("\nCandidates:")
        for candidate in report.candidates[:8]:
            score = candidate.score.overall if candidate.score else None
            score_text = f"{score:.3f}" if score is not None else "unscored"
            print(f"- {candidate.package} {candidate.version or 'unknown'} score={score_text}")
            if candidate.score and candidate.score.reasons:
                print(f"  reason: {candidate.score.reasons[0]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except BrokenPipeError:
        sys.stderr.close()
