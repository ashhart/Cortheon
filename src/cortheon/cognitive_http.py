"""Local HTTP transport for Cortheon's memory-only cognitive runtime."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from cortheon import __version__
from cortheon.cognitive_hooks import CognitiveHookTracker
from cortheon.cognitive_protocol import (
    CORTHEON_PROTOCOL_VERSION,
    protocol_capabilities,
)
from cortheon.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeError

DEFAULT_PORT = 8743
MAX_REQUEST_BYTES = 1_000_000


def _source_fingerprint() -> str:
    """Hash of the runtime sources at process start.

    Adapters compare it to detect a stale long-running server.
    """

    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    core = (package / "cognitive_core").glob("*.py")
    hooks_core = (package / "cognitive_hooks_core").glob("*.py")
    codex_plugin = (
        path
        for path in (package / "codex_plugins" / "cortheon").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    names = (
        "cognitive_runtime.py",
        *(f"cognitive_core/{path.name}" for path in sorted(core, key=lambda p: p.name)),
        "cognitive_http.py",
        "cognitive_hooks.py",
        *(f"cognitive_hooks_core/{path.name}" for path in sorted(hooks_core, key=lambda p: p.name)),
        *(path.relative_to(package).as_posix() for path in sorted(codex_plugin)),
    )
    for name in names:
        try:
            digest.update((package / name).read_bytes())
        except OSError:
            digest.update(name.encode())
    return digest.hexdigest()[:16]


_SOURCE_FINGERPRINT = _source_fingerprint()


class CognitiveHTTPServer(ThreadingHTTPServer):
    """Bounded-process server; task data remains only in ``runtime`` memory."""

    daemon_threads = True
    request_queue_size = 128

    def __init__(
        self,
        address: tuple[str, int],
        runtime: CognitiveRuntime,
        *,
        token: str = "",
        max_concurrent_requests: int = 64,
        hook_tracker: CognitiveHookTracker | None = None,
    ) -> None:
        if not 1 <= max_concurrent_requests <= 1_024:
            raise ValueError("max_concurrent_requests must be between 1 and 1024")
        super().__init__(address, CognitiveHandler)
        self.runtime = runtime
        self.token = token
        self.max_concurrent_requests = max_concurrent_requests
        self.hook_tracker = hook_tracker or CognitiveHookTracker(runtime=runtime)
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)

    def process_request(
        self,
        request: Any,
        client_address: Any,
    ) -> None:
        if not self._request_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 31\r\n"
                    b"Cache-Control: no-store\r\n"
                    b"Connection: close\r\n\r\n"
                    b'{"error":"server is saturated"}'
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(
        self,
        request: Any,
        client_address: Any,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class CognitiveHandler(BaseHTTPRequestHandler):
    """Small JSON API used by harness adapters; never a project-file API."""

    server: Any
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "cortheon-cognitive",
                    "version": __version__,
                    "protocol_version": CORTHEON_PROTOCOL_VERSION,
                    "source_fingerprint": _SOURCE_FINGERPRINT,
                    "runtime_instance_id": os.environ.get("CORTHEON_RUNTIME_INSTANCE_ID", ""),
                    "storage": "memory_only",
                    "active_sessions": self.server.runtime.active_sessions,
                    "active_hook_turns": self.server.hook_tracker.active_turns,
                },
            )
            return
        if path == "/v1/capabilities":
            self._json(HTTPStatus.OK, {"ok": True, **protocol_capabilities()})
            return
        if path == "/metrics":
            if not self._authorized():
                self._json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": "invalid cognitive runtime bearer token"},
                )
                return
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    **self.server.runtime.metrics,
                    **self.server.hook_tracker.metrics,
                },
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "invalid cognitive runtime bearer token"},
            )
            return
        try:
            body = self._body()
            payload = self._dispatch(urlsplit(self.path).path, body)
        except KeyError as exc:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"missing required field: {exc.args[0]}"},
            )
            return
        except (ValueError, CognitiveRuntimeError) as exc:
            self._json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            return
        except (OSError, TimeoutError):
            self._json(
                HTTPStatus.REQUEST_TIMEOUT,
                {"error": "request body timed out"},
            )
            return
        self._json(HTTPStatus.OK, payload)

    def _authorized(self) -> bool:
        expected = self.server.token
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return supplied.startswith(prefix) and hmac.compare_digest(
            supplied[len(prefix) :],
            expected,
        )

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("request body must be a JSON object")
        if length > MAX_REQUEST_BYTES:
            raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("request body is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _dispatch(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        runtime = self.server.runtime
        if path == "/v1/start":
            return runtime.start(
                body["goal"],
                constraints=body.get("constraints") or (),
                effort=body.get("effort") or "quick",
                task_kind=body.get("task_kind") or "auto",
                strictness=body.get("strictness") or "standard",
                lease_seconds=body.get("lease_seconds"),
                evaluation_profile=body.get("evaluation_profile"),
            )
        if path == "/v1/evaluation-receipt":
            return runtime.consume_evaluation_receipt(body["nonce"])
        if path == "/v1/heartbeat":
            return runtime.heartbeat(body["session_id"])
        if path == "/v1/observe":
            return runtime.observe(
                body["session_id"],
                body["observations"],
                request_id=body.get("request_id"),
            )
        if path == "/v1/step":
            return runtime.step(
                body["session_id"],
                hypotheses=body.get("hypotheses") or (),
                hypothesis_updates=body.get("hypothesis_updates") or (),
                open_questions=body.get("open_questions") or (),
                draft=body.get("draft"),
            )
        if path == "/v1/challenge":
            return runtime.challenge(
                body["session_id"],
                draft=body["draft"],
                claims=body["claims"],
            )
        if path == "/v1/complete":
            return runtime.complete(
                body["session_id"],
                answer=body["answer"],
                claims=body["claims"],
                hypotheses=body["hypotheses"],
                completion_evidence_ids=body["completion_evidence_ids"],
            )
        if path == "/v1/resume":
            return runtime.describe_sessions(limit=body.get("limit") or 3)
        if path == "/v1/retract":
            return runtime.retract(
                body["session_id"],
                body["evidence_ids"],
                reason=body.get("reason") or "",
            )
        if path == "/v1/abandon":
            return runtime.finish(body["session_id"], mode="abandon")
        if path == "/v1/evidence-close":
            return runtime.close_evidence(body["session_id"])
        if path == "/v1/hooks/register":
            return self.server.hook_tracker.register(
                body["host"],
                body["host_session_id"],
                body["turn_id"],
                goal=body.get("goal"),
                effort=body.get("effort") or "quick",
                strictness=body.get("strictness") or "standard",
                task_kind=body.get("task_kind") or "auto",
            )
        if path == "/v1/hooks/pre-tool":
            tool_input = body.get("tool_input") or {}
            if not isinstance(tool_input, dict):
                raise ValueError("tool_input must be an object")
            return self.server.hook_tracker.pre_tool(
                body["host"],
                body["host_session_id"],
                body["turn_id"],
                body["tool_name"],
                tool_input=tool_input,
            )
        if path == "/v1/hooks/post-tool":
            tool_metadata = body.get("tool_metadata") or {}
            if not isinstance(tool_metadata, dict):
                raise ValueError("tool_metadata must be an object")
            return self.server.hook_tracker.post_tool(
                body["host"],
                body["host_session_id"],
                body["turn_id"],
                body["tool_name"],
                succeeded=body.get("succeeded") is True,
                certified=body.get("certified") is True,
                tool_output=body.get("tool_output") or "",
                tool_metadata=tool_metadata,
            )
        if path == "/v1/hooks/stop":
            return self.server.hook_tracker.stop(
                body["host"],
                body["host_session_id"],
                body["turn_id"],
                answer=body.get("answer"),
            )
        if path == "/v1/hooks/end":
            return self.server.hook_tracker.end_session(
                body["host"],
                body["host_session_id"],
            )
        raise ValueError(f"unknown route: {path}")

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Cortheon-Protocol", CORTHEON_PROTOCOL_VERSION)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def build_server(
    bind: str,
    port: int,
    *,
    max_sessions: int = 32,
    ttl_seconds: float = 1_800.0,
    token: str = "",
    max_concurrent_requests: int = 64,
) -> CognitiveHTTPServer:
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    runtime = CognitiveRuntime(
        max_sessions=max_sessions,
        ttl_seconds=ttl_seconds,
        require_host_receipts=True,
    )
    return CognitiveHTTPServer(
        (bind, port),
        runtime,
        token=token,
        max_concurrent_requests=max_concurrent_requests,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon cognitive-http",
        description="Serve the memory-only cognitive runtime over local HTTP.",
    )
    parser.add_argument(
        "--bind",
        default=os.environ.get("CORTHEON_COGNITIVE_BIND", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CORTHEON_COGNITIVE_PORT", str(DEFAULT_PORT))),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=int(os.environ.get("CORTHEON_COGNITIVE_MAX_SESSIONS", "32")),
    )
    parser.add_argument(
        "--ttl-seconds",
        type=float,
        default=float(os.environ.get("CORTHEON_COGNITIVE_TTL_SECONDS", "1800")),
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=int(os.environ.get("CORTHEON_COGNITIVE_MAX_CONCURRENT_REQUESTS", "64")),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CORTHEON_COGNITIVE_TOKEN", ""),
        help="Optional bearer token; prefer CORTHEON_COGNITIVE_TOKEN.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = build_server(
        args.bind,
        args.port,
        max_sessions=args.max_sessions,
        ttl_seconds=args.ttl_seconds,
        token=args.token,
        max_concurrent_requests=args.max_concurrent_requests,
    )

    def stop(signum: int, _frame: object) -> None:
        print(f"cortheon cognitive: received signal {signum}; shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        "cortheon cognitive: memory-only runtime listening on "
        f"http://{args.bind}:{server.server_port}",
        flush=True,
    )
    if not args.token and args.bind not in {"127.0.0.1", "::1", "localhost"}:
        print(
            "cortheon cognitive: WARNING unauthenticated non-loopback bind; "
            "set CORTHEON_COGNITIVE_TOKEN",
            flush=True,
        )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
