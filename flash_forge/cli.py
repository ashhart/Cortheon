"""FlashForge command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from .code_forge import CodeForgeTrace
from .dev_agent import run_dev_forge
from .doctor import format_doctor_report, run_doctor
from .duel import DUEL_PRESETS, run_duel
from .engine import FlashForge, ForgeConfig
from .integrations import ALL_HARNESSES, install_commands
from .memory import EpisodicMemory
from .proxy import create_app
from .reason_forge import ReasonForgeTrace
from .replica_agent import ReplicaAgent, ReplicaConfig
from .showdown import run_showdown
from .think_forge import ThinkForgeTrace

_STREAMS_HELP = (
    "Parallel verification streams (2, 4, or 8): "
    "2=fast sanity check ~2-3 calls, 4=balanced default ~4-7 calls, 8=max diversity"
)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="cortheon", description="Cortheon — Unified consensus & verification runtime for frozen local models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser(
        "doctor",
        help="Run local diagnostics (OS sandbox, solver stack, harnesses, backend capabilities & live fast-route probe)",
    )
    p_doc.add_argument("--url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible backend URL to probe")
    p_doc.add_argument("--api-key", default=None, help="Optional Bearer API key for the backend")
    p_doc.add_argument("--model", default=None, help="Optional model ID override when backend serves multiple models")
    p_doc.add_argument("--skip-live", action="store_true", help="Skip the live fast-route completion & throughput probe")
    p_doc.add_argument("--json", dest="as_json", action="store_true", help="Emit structured JSON diagnostics report")

    p_solve = sub.add_parser("solve", help="Solve any hard math, logic, coding, debugging, or open-ended architecture/design problem")
    p_solve.add_argument("prompt", help="Problem statement (or @path/to/file.txt)")
    p_solve.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p_solve.add_argument("--referee-url", default=None, help="Optional second model URL for Cross-Model Referee/Critic")
    p_solve.add_argument("--mode", choices=["auto", "code", "reason", "think", "fast"], default="auto")
    p_solve.add_argument("--streams", type=int, default=4, help=_STREAMS_HELP)
    p_solve.add_argument("--deep", action="store_true")
    p_solve.add_argument("--no-memory", action="store_true")

    p_proxy = sub.add_parser("proxy", help="Start drop-in OpenAI (/v1/chat/completions) + Anthropic (/v1/messages) proxy")
    p_proxy.add_argument("--upstream", default="http://127.0.0.1:8080/v1")
    p_proxy.add_argument("--referee-url", default=None)
    p_proxy.add_argument("--host", default="127.0.0.1")
    p_proxy.add_argument("--port", type=int, default=18652)
    p_proxy.add_argument("--streams", type=int, default=4, help=_STREAMS_HELP)
    p_proxy.add_argument("--no-auto-rescue", action="store_true", help="Disable automatic B=3 death-spiral rescue on tool-calling loops")

    p_show = sub.add_parser("showdown", help="Run a live 60-second Vanilla vs Cortheon knockout on your local model")
    p_show.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p_show.add_argument("--referee-url", default=None)
    p_show.add_argument("--streams", type=int, default=4, help=_STREAMS_HELP)

    p_duel = sub.add_parser("duel", help="Live side-by-side split-screen race: Vanilla (B=1) vs Cortheon v2 (B=4 early-abort)")
    p_duel.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p_duel.add_argument("--preset", choices=list(DUEL_PRESETS.keys()), default="ring")
    p_duel.add_argument("--prompt", default=None, help="Custom prompt instead of a built-in preset")
    p_duel.add_argument("--expected", default="", help="Optional exact expected answer for custom prompt")
    p_duel.add_argument("--streams", type=int, default=4, help=_STREAMS_HELP)

    p_repo = sub.add_parser("repo-fix", help="Patch a repository file verified in ephemeral git worktrees")
    p_repo.add_argument("--repo", required=True)
    p_repo.add_argument("--file", required=True)
    p_repo.add_argument("--verify", required=True)
    p_repo.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p_repo.add_argument("--apply", action="store_true", help="Write winning verified patch back to the repo")
    p_repo.add_argument("task")

    p_dev = sub.add_parser("dev", help="Autonomous multi-file DevForge agent with ephemeral git worktree racing")
    p_dev.add_argument("--repo", required=True)
    p_dev.add_argument("--verify", required=True)
    p_dev.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p_dev.add_argument("--streams", type=int, default=4, help=_STREAMS_HELP)
    p_dev.add_argument("--apply", action="store_true", help="Apply winning verified multi-file patch to repo")
    p_dev.add_argument("task")

    p_agent = sub.add_parser("agent", help="Run full autonomous ReplicaAgent (BrainState + tools + B=3 forks + Referee Gate)")
    p_agent.add_argument("--workspace", default=".")
    p_agent.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p_agent.add_argument("--referee-url", default=None, help="Optional second model URL (e.g. http://127.0.0.1:8081/v1) for cross-model Referee Gate")
    p_agent.add_argument("--max-turns", type=int, default=24)
    p_agent.add_argument("--read-only", action="store_true")
    p_agent.add_argument("--no-fork", action="store_true")
    p_agent.add_argument("--no-referee", action="store_true")
    p_agent.add_argument("goal")

    p_mem = sub.add_parser("memory", help="Inspect episodic reflexion memory")
    p_mem.add_argument("--limit", type=int, default=20)

    p_inst = sub.add_parser(
        "install-commands",
        help="Install /cortheon slash command & tool across OMP, Pi, OpenCode, Claude Code, Cursor, and Codex",
    )
    p_inst.add_argument(
        "--harness",
        default="all",
        help=f"Comma-separated list or 'all' ({', '.join(ALL_HARNESSES)})",
    )
    p_inst.add_argument("--project", default=None, help="Install project-locally into this directory instead of $HOME")
    p_inst.add_argument("--profile", default=None, help="Optional OMP profile name (e.g. cortheon)")
    p_inst.add_argument("--url", default="http://127.0.0.1:18651/v1", help="Default Cortheon endpoint URL for markdown commands")

    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        report = asyncio.run(
            run_doctor(
                url=args.url,
                api_key=args.api_key,
                model=args.model,
                skip_live=args.skip_live,
            )
        )
        if args.as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_doctor_report(report))
        if not report.ok:
            sys.exit(1)
        return

    if args.cmd == "install-commands":
        targets = ALL_HARNESSES if args.harness.strip().lower() == "all" else [x.strip() for x in args.harness.split(",") if x.strip()]
        proj_path = Path(args.project) if args.project else None
        written = install_commands(targets, project_dir=proj_path, omp_profile=args.profile, default_url=args.url)
        print(f"[Cortheon] Installed /cortheon across {len(written)} target(s):")
        for p in written:
            print(f"  - {p}")
        return

    if args.cmd == "proxy":
        app = create_app(
            upstream_url=args.upstream,
            referee_url=args.referee_url,
            num_streams=args.streams,
            auto_rescue=not args.no_auto_rescue,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.cmd == "showdown":
        asyncio.run(run_showdown(args.url, referee_url=args.referee_url, streams=args.streams))
        return

    if args.cmd == "duel":
        asyncio.run(
            run_duel(
                args.url,
                preset=args.preset,
                custom_prompt=args.prompt,
                expected=args.expected,
                streams=args.streams,
            )
        )
        return

    if args.cmd == "memory":
        mem = EpisodicMemory()
        lessons = mem.list_recent(args.limit)
        print(f"Total lessons stored: {mem.count()}")
        for l in lessons:
            print(f"#{l.id} (hits={l.hits}) {l.format_compact()}")
        mem.close()
        return

    if args.cmd == "solve":
        prompt = args.prompt
        if prompt.startswith("@"):
            prompt = Path(prompt[1:]).read_text(encoding="utf-8")

        async def _run() -> None:
            cfg = ForgeConfig(
                base_url=args.url,
                referee_url=args.referee_url,
                num_streams=args.streams,
                deep=args.deep,
                enable_memory=not args.no_memory,
            )
            async with FlashForge(cfg) as ff:
                force = None if args.mode == "auto" else args.mode
                res = await ff.solve(prompt, force_mode=force)
                print(f"[Cortheon] mode={res.mode} wall={res.wall_s:.2f}s stats={res.token_stats}")
                if isinstance(res.raw_trace, ReasonForgeTrace):
                    print(f"[ReasonForge] consensus={res.raw_trace.consensus_type} wave2={res.raw_trace.wave2_triggered}")
                    print(res.raw_trace.final_explanation)
                elif isinstance(res.raw_trace, CodeForgeTrace):
                    print(
                        f"[CodeForge] winner={res.raw_trace.winner_name} "
                        f"redteam_kills={len(res.raw_trace.redteam_valid_kills)} wave2={res.raw_trace.wave2_triggered}"
                    )
                    print(res.answer)
                else:
                    print(res.answer)
                if res.lesson_learned:
                    print(f"[EpisodicMemory] Learned: {res.lesson_learned}")

        asyncio.run(_run())
        return

    if args.cmd == "agent":
        async def _run_agent() -> None:
            from .client import TensorFoldClient
            async with FlashForge(ForgeConfig(base_url=args.url)) as ff:
                ref_client = None
                if args.referee_url:
                    ref_client = TensorFoldClient(base_url=args.referee_url)
                    await ref_client.probe()
                try:
                    rcfg = ReplicaConfig(
                        max_turns=args.max_turns,
                        read_only=args.read_only,
                        enable_strategic_fork=not args.no_fork,
                        enable_referee=not args.no_referee,
                        verbose=True,
                    )
                    agent = ReplicaAgent(
                        client=ff.client,
                        skills=ff.skills,
                        memory=ff.memory,
                        workspace=args.workspace,
                        config=rcfg,
                        referee_client=ref_client,
                    )
                    res = await agent.run(args.goal)
                finally:
                    if ref_client:
                        await ref_client.close()
                print(
                    f"\n[ReplicaAgent DONE] ok={res.ok} referee_confirmed={res.confirmed_by_referee} "
                    f"turns={res.turns_used} forks={res.strategic_forks} ref_rejects={res.referee_rejections} "
                    f"wall={res.wall_s:.1f}s files={res.modified_files}"
                )
                print("\n--- FINAL SUMMARY ---\n" + res.summary)
                if not res.ok:
                    sys.exit(1)

        asyncio.run(_run_agent())
        return

    if args.cmd == "dev":
        async def _run_dev() -> None:
            async with FlashForge(ForgeConfig(base_url=args.url, num_streams=args.streams)) as ff:
                res = await run_dev_forge(
                    client=ff.client,
                    skills=ff.skills,
                    memory=ff.memory,
                    repo_dir=args.repo,
                    task=args.task,
                    verify_cmd=args.verify,
                    num_streams=args.streams,
                    apply_winner=args.apply,
                )
                print(f"[DevForge] ok={res.ok} winner={res.winner_name} files={res.touched_files} wave2={res.wave2_triggered} rt_kills={len(res.redteam_kills)} wall={res.wall_s:.1f}s")
                if res.diff_text:
                    print(res.diff_text[:3000])
                if not res.ok:
                    sys.exit(1)

        asyncio.run(_run_dev())
        return

    if args.cmd == "repo-fix":
        async def _run_repo() -> None:
            async with FlashForge(ForgeConfig(base_url=args.url)) as ff:
                out = await ff.solve_repo_file(args.repo, args.file, args.task, args.verify)
                print(json.dumps({k: v for k, v in out.items() if k != "code"}, indent=2))
                if out["ok"] and args.apply and out["code"]:
                    (Path(args.repo) / args.file).write_text(out["code"], encoding="utf-8")
                    print(f"Applied verified patch from {out['winner']} to {args.file}")
                elif not out["ok"]:
                    sys.exit(1)

        asyncio.run(_run_repo())


if __name__ == "__main__":
    main()
