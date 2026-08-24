from __future__ import annotations

from cortheon.clinical_trials import ClinicalTrialsGovDiscovery
from cortheon.connectors.github import GitHubRepositorySearch
from cortheon.ledger import EvidenceLedger
from cortheon.models import (
    Evidence,
    ResearchArtifact,
    ResearchDiscoveryPass,
    ResearchQuery,
    ResearchReport,
    ScholarlyWork,
    SearchResult,
)
from cortheon.research_core._compat import facade
from cortheon.scholarly import CompositeScholarlyDiscovery
from cortheon.search import SearchProvider
from cortheon.source_planner import (
    SourcePlanner,
)
from cortheon.web_crawler import WebCrawler


class ResearchEngine:
    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        scholarly_discovery: CompositeScholarlyDiscovery | None = None,
        github_discovery: GitHubRepositorySearch | None = None,
        trial_discovery: ClinicalTrialsGovDiscovery | None = None,
        crawler: WebCrawler | None = None,
        ledger: EvidenceLedger | None = None,
        source_planner: SourcePlanner | None = None,
        source_planner_strategy: str | None = None,
    ) -> None:
        self.search_provider = search_provider or facade().ConfiguredSearchProvider()
        self.scholarly_discovery = scholarly_discovery or facade().CompositeScholarlyDiscovery()
        self.github_discovery = github_discovery or facade().GitHubRepositorySearch()
        self.trial_discovery = trial_discovery or facade().ClinicalTrialsGovDiscovery()
        self.crawler = crawler or facade().WebCrawler()
        self.ledger = ledger or facade().EvidenceLedger()
        self.source_planner = source_planner or facade().default_source_planner(
            source_planner_strategy
        )

    def research(
        self,
        topic: str,
        *,
        seed_urls: list[str] | None = None,
        max_search_results: int = 10,
        max_scholarly_results: int = 10,
        max_github_results: int = 5,
        max_trial_results: int = 5,
        max_follow_up_queries: int = 2,
        max_adaptive_queries: int = 1,
        max_artifact_inspections: int = 3,
        max_pages: int = 25,
        max_depth: int = 1,
        allowed_domains: list[str] | None = None,
        write_report: bool = True,
    ) -> ResearchReport:
        user_seeds = list(seed_urls or [])
        evidence: list[Evidence] = []
        errors: list[str] = []
        notes: list[str] = []

        mission_queries = facade().plan_research_queries(topic, max_follow_up_queries)
        if not mission_queries:
            mission_queries = [
                facade().ResearchQuery(
                    query=topic, purpose="primary mission query", source="user_topic"
                )
            ]
        source_plan = self.source_planner.plan(
            topic,
            facade().build_research_source_profiles(
                facade().scholarly_source_profiles(self.scholarly_discovery),
                facade().trial_registry_source_profiles(self.trial_discovery),
                search_provider_name=self.search_provider.name,
                seed_url_count=len(user_seeds),
            ),
            facade().SourcePlanningConstraints(
                max_search_results=max_search_results,
                max_scholarly_results=max_scholarly_results,
                max_github_results=max_github_results,
                max_trial_results=max_trial_results,
                seed_url_count=len(user_seeds),
                search_provider_name=self.search_provider.name,
            ),
        )
        selected_scholarly_connectors = facade().selected_source_names(source_plan, "scholarly")
        scholarly_budget = max_scholarly_results if selected_scholarly_connectors else 0
        search_budget = (
            max_search_results if facade().is_source_selected(source_plan, "web_search") else 0
        )
        github_budget = (
            max_github_results
            if facade().is_source_selected(source_plan, "github_repositories")
            else 0
        )
        trial_budget = (
            max_trial_results
            if facade().is_source_selected(source_plan, "clinicaltrials_gov")
            else 0
        )
        per_scholarly_limit = facade().per_query_limit(scholarly_budget, len(mission_queries))
        per_search_limit = facade().per_query_limit(search_budget, len(mission_queries))
        per_github_limit = facade().per_query_limit(github_budget, len(mission_queries))
        per_trial_limit = facade().per_query_limit(trial_budget, len(mission_queries))
        evidence.append(facade().source_plan_evidence(topic, source_plan))
        notes.extend(facade().source_plan_notes(source_plan))

        (
            all_scholarly_works,
            all_search_results,
            all_discovered_artifacts,
            discovery_evidence,
            discovery_errors,
            discovery_passes,
        ) = self._run_discovery_queries(
            mission_queries,
            scholarly_limit=per_scholarly_limit,
            search_limit=per_search_limit,
            github_limit=per_github_limit,
            trial_limit=per_trial_limit,
            scholarly_connectors=selected_scholarly_connectors,
        )
        evidence.extend(discovery_evidence)
        errors.extend(discovery_errors)

        preliminary_works = facade().merge_scholarly_works(
            topic, all_scholarly_works, max_scholarly_results
        )
        preliminary_claims = facade().extract_claims(topic, preliminary_works, [])
        preliminary_synthesis = facade().synthesize_research(topic, preliminary_claims)
        preliminary_source_count = len({claim.source_url for claim in preliminary_claims})
        adaptive_queries = facade().plan_gap_follow_up_queries(
            topic,
            preliminary_synthesis.evidence_gaps,
            mission_queries,
            max_adaptive_queries,
        )
        if adaptive_queries:
            adaptive_query_count = len(mission_queries) + len(adaptive_queries)
            (
                adaptive_works,
                adaptive_search_results,
                adaptive_artifacts,
                adaptive_evidence,
                adaptive_errors,
                adaptive_passes,
            ) = self._run_discovery_queries(
                adaptive_queries,
                scholarly_limit=facade().per_query_limit(scholarly_budget, adaptive_query_count),
                search_limit=facade().per_query_limit(search_budget, adaptive_query_count),
                github_limit=facade().per_query_limit(github_budget, adaptive_query_count),
                trial_limit=facade().per_query_limit(trial_budget, adaptive_query_count),
                scholarly_connectors=selected_scholarly_connectors,
            )
            mission_queries.extend(adaptive_queries)
            all_scholarly_works.extend(adaptive_works)
            all_search_results.extend(adaptive_search_results)
            all_discovered_artifacts.extend(adaptive_artifacts)
            evidence.extend(adaptive_evidence)
            errors.extend(adaptive_errors)
            discovery_passes.extend(adaptive_passes)

        scholarly_works = facade().merge_scholarly_works(
            topic, all_scholarly_works, max_scholarly_results
        )
        search_results = facade().merge_search_results(all_search_results, max_search_results)
        discovered_artifacts = facade().limit_discovered_artifacts(
            all_discovered_artifacts,
            max_github_results=max_github_results,
            max_trial_results=max_trial_results,
        )
        seeds = list(user_seeds)
        seeds.extend(
            work.url for work in scholarly_works if work.url.startswith(("http://", "https://"))
        )
        seeds.extend(result.url for result in search_results)
        seeds = facade().dedupe(seeds)
        evidence.append(facade().mission_plan_evidence(topic, mission_queries, discovery_passes))
        notes.extend(facade().mission_plan_notes(mission_queries, discovery_passes))

        if not seeds:
            notes.append(
                "No seed URLs or configured search results were available, so no crawl was performed."
            )
            claims = facade().extract_claims(topic, scholarly_works, [])
            source_lineage = facade().build_source_lineage(claims, scholarly_works, [])
            synthesis = facade().synthesize_research(topic, claims)
            gap_closures = facade().build_gap_closures(
                adaptive_queries,
                preliminary_synthesis.evidence_gaps,
                synthesis.evidence_gaps,
                before_claim_count=len(preliminary_claims),
                after_claim_count=len(claims),
                before_source_count=preliminary_source_count,
                after_source_count=len({claim.source_url for claim in claims}),
            )
            artifacts = facade().derive_research_artifacts(
                topic,
                scholarly_works,
                [],
                search_results=search_results,
                discovered_artifacts=discovered_artifacts,
            )
            artifacts, artifact_inspection_evidence, artifact_inspection_errors = (
                self.github_discovery.inspect_artifacts(artifacts, max_artifact_inspections)
            )
            artifact_assessments = facade().assess_artifacts(topic, artifacts)
            source_coverage = facade().analyze_source_coverage(
                topic,
                source_plan=source_plan,
                discovery_passes=discovery_passes,
                scholarly_works=scholarly_works,
                search_results=search_results,
                crawled_pages=[],
                artifacts=artifacts,
                claims=claims,
            )
            evidence.extend(artifact_inspection_evidence)
            errors.extend(artifact_inspection_errors)
            evidence.append(facade().artifact_assessment_evidence(topic, artifact_assessments))
            evidence.append(facade().source_coverage_evidence(topic, source_coverage))
            evidence.append(facade().grounding_evidence(topic, claims))
            evidence.append(
                facade().synthesis_evidence(topic, synthesis.status, synthesis.confidence)
            )
            evidence.append(facade().gap_closure_evidence(topic, gap_closures))
            evidence.append(facade().lineage_evidence(topic, source_lineage))
            evidence.append(facade().artifact_evidence(topic, artifacts))
            notes.extend(facade().artifact_notes(artifacts))
            notes.extend(facade().coverage_notes(source_coverage))
            report = facade().ResearchReport(
                topic=topic,
                generated_at=facade().utc_now(),
                search_provider=self.search_provider.name,
                seed_urls=[],
                search_results=search_results,
                scholarly_works=scholarly_works,
                crawled_pages=[],
                artifacts=artifacts,
                claims=claims,
                source_lineage=source_lineage,
                synthesis=synthesis,
                evidence=evidence,
                notes=notes,
                errors=facade().dedupe(errors),
                mission_queries=mission_queries,
                source_plan=source_plan,
                discovery_passes=discovery_passes,
                source_coverage=source_coverage,
                artifact_assessments=artifact_assessments,
                gap_closures=gap_closures,
            )
            if write_report:
                self.ledger.write_research_report(report)
            return report

        domains = allowed_domains or []
        if not allowed_domains and not search_results:
            domains = [
                facade().urlparse(seed).netloc for seed in seeds if facade().urlparse(seed).netloc
            ]

        pages, crawl_evidence = self.crawler.crawl(
            seeds,
            allowed_domains=domains or None,
            budget=facade().CrawlBudget(max_pages=max_pages, max_depth=max_depth),
        )
        pages = sorted(pages, key=lambda page: page.authority_score, reverse=True)
        claims = facade().extract_claims(topic, scholarly_works, pages)
        source_lineage = facade().build_source_lineage(claims, scholarly_works, pages)
        synthesis = facade().synthesize_research(topic, claims)
        gap_closures = facade().build_gap_closures(
            adaptive_queries,
            preliminary_synthesis.evidence_gaps,
            synthesis.evidence_gaps,
            before_claim_count=len(preliminary_claims),
            after_claim_count=len(claims),
            before_source_count=preliminary_source_count,
            after_source_count=len({claim.source_url for claim in claims}),
        )
        artifacts = facade().derive_research_artifacts(
            topic,
            scholarly_works,
            pages,
            search_results=search_results,
            discovered_artifacts=discovered_artifacts,
        )
        artifacts, artifact_inspection_evidence, artifact_inspection_errors = (
            self.github_discovery.inspect_artifacts(artifacts, max_artifact_inspections)
        )
        artifact_assessments = facade().assess_artifacts(topic, artifacts)
        source_coverage = facade().analyze_source_coverage(
            topic,
            source_plan=source_plan,
            discovery_passes=discovery_passes,
            scholarly_works=scholarly_works,
            search_results=search_results,
            crawled_pages=pages,
            artifacts=artifacts,
            claims=claims,
        )
        evidence.extend(crawl_evidence)
        evidence.extend(artifact_inspection_evidence)
        errors.extend(artifact_inspection_errors)
        evidence.append(facade().artifact_assessment_evidence(topic, artifact_assessments))
        evidence.append(facade().source_coverage_evidence(topic, source_coverage))
        evidence.append(
            facade().Evidence(
                claim=(
                    f"Research report collected {len(scholarly_works)} scholarly work(s), "
                    f"{len(pages)} crawled page(s), {len(artifacts)} artifact(s), "
                    f"and {len(claims)} extracted claim(s) for topic: {topic}"
                ),
                source_type="research_report",
                source_url=None,
                support=facade().SupportLevel.INFERRED,
                details={
                    "topic": topic,
                    "search_provider": self.search_provider.name,
                    "seed_count": len(seeds),
                    "mission_query_count": len(mission_queries),
                    "scholarly_work_count": len(scholarly_works),
                    "page_count": len(pages),
                    "artifact_count": len(artifacts),
                    "artifact_mix": facade().artifact_mix(artifacts),
                    "claim_count": len(claims),
                    "source_mix": facade().source_mix(pages),
                },
            )
        )
        evidence.append(facade().grounding_evidence(topic, claims))
        evidence.append(facade().synthesis_evidence(topic, synthesis.status, synthesis.confidence))
        evidence.append(facade().gap_closure_evidence(topic, gap_closures))
        evidence.append(facade().lineage_evidence(topic, source_lineage))
        evidence.append(facade().artifact_evidence(topic, artifacts))
        notes.extend(facade().research_notes(search_results, scholarly_works, pages))
        notes.extend(facade().quarantine_notes(pages))
        notes.extend(facade().artifact_notes(artifacts))
        notes.extend(facade().coverage_notes(source_coverage))
        report = facade().ResearchReport(
            topic=topic,
            generated_at=facade().utc_now(),
            search_provider=self.search_provider.name,
            seed_urls=seeds,
            search_results=search_results,
            scholarly_works=scholarly_works,
            crawled_pages=pages,
            artifacts=artifacts,
            claims=claims,
            source_lineage=source_lineage,
            synthesis=synthesis,
            evidence=evidence,
            notes=notes,
            errors=facade().dedupe(errors),
            mission_queries=mission_queries,
            source_plan=source_plan,
            discovery_passes=discovery_passes,
            source_coverage=source_coverage,
            artifact_assessments=artifact_assessments,
            gap_closures=gap_closures,
        )
        if write_report:
            self.ledger.write_research_report(report)
        return report

    def _run_discovery_queries(
        self,
        planned_queries: list[ResearchQuery],
        *,
        scholarly_limit: int,
        search_limit: int,
        github_limit: int,
        trial_limit: int,
        scholarly_connectors: list[str],
    ) -> tuple[
        list[ScholarlyWork],
        list[SearchResult],
        list[ResearchArtifact],
        list[Evidence],
        list[str],
        list[ResearchDiscoveryPass],
    ]:
        return facade().run_discovery_queries(
            self,
            planned_queries,
            scholarly_limit=scholarly_limit,
            search_limit=search_limit,
            github_limit=github_limit,
            trial_limit=trial_limit,
            scholarly_connectors=scholarly_connectors,
        )
