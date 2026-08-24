"""Command dispatch for the operator CLI."""

from __future__ import annotations

from cortheon import cognitive_cli as surface


def main(argv: list[str] | None = None) -> int:
    args = surface.build_parser().parse_args(argv)
    try:
        if args.command == "serve":
            return surface._serve(args)
        if args.command == "mcp":
            return surface._mcp(args)
        if args.command == "doctor":
            payload = surface.doctor(
                args.url,
                token=args.token,
                require_runtime=args.require_runtime,
                hosts=args.host,
                scope=args.scope,
                project_dir=args.project_dir,
            )
            print(surface.json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["ok"] else 1
        if args.command == "conformance":
            payload = surface.host_conformance(
                args.url,
                token=args.token,
                hosts=args.host,
                timeout_seconds=args.timeout_seconds,
            )
            print(surface.json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["ok"] else 1
        if args.command == "install":
            results = surface._install(args)
            print(
                surface.json.dumps(
                    {"ok": True, "results": [item.public() for item in results]},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "uninstall":
            results = surface._uninstall(args)
            print(
                surface.json.dumps(
                    {"ok": True, "results": [item.public() for item in results]},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "configure":
            results = surface._configure(args)
            print(
                surface.json.dumps(
                    {"ok": True, "results": [item.public() for item in results]},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "capabilities":
            print(surface.json.dumps(surface.protocol_capabilities(), indent=2, sort_keys=True))
            return 0
        if args.command == "paths":
            print(surface.json.dumps(surface._asset_paths(), indent=2, sort_keys=True))
            return 0
    except (OSError, ValueError) as exc:
        print(f"cortheon: {exc}", file=surface.sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")
