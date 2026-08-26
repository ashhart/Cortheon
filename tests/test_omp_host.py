"""Native OMP host adapter: honest receipts and the runtime roundtrip."""

from __future__ import annotations

import functools
import http.server
import tempfile
import threading
from pathlib import Path

import pytest

from cortheon.omp_host import OmpHost


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "widgets"\nrequires-python = ">=3.12"\n', encoding="utf-8"
    )
    (root / "src").mkdir()
    (root / "src" / "widgets.py").write_text(
        "def widget_size():\n    return 42\n", encoding="utf-8"
    )
    return root


_SERVED: list[http.server.ThreadingHTTPServer] = []


@functools.cache
def _served_fixture() -> str:
    root = tempfile.mkdtemp()
    Path(root, "paper.txt").write_text(
        "Widget benchmark reliability: method controlled trials; result 94%; "
        "limitations narrow domain.",
        encoding="utf-8",
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    _SERVED.append(server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/paper.txt"


@pytest.fixture(scope="session", autouse=True)
def _shutdown_served():
    yield
    for server in _SERVED:
        server.shutdown()


def test_omp_host_ops_are_receipted_read_only_and_bounded(tmp_path: Path) -> None:
    root = _project(tmp_path)
    host = OmpHost(root=str(root))

    read = host.run("read", path="pyproject.toml")
    assert read.receipt == {
        "tool": "read",
        "outcome": "result",
        "args": {"filePath": "pyproject.toml"},
    }
    assert "requires-python" in read.content

    hit = host.run("grep", pattern="widget", path=".")
    assert hit.receipt["outcome"] == "match"
    assert "widget_size" in hit.content
    miss = host.run("grep", pattern="zebra", path=".")
    assert miss.receipt == {
        "tool": "grep",
        "outcome": "no_match",
        "args": {"pattern": "zebra", "path": "."},
    }

    found = host.run("glob", pattern="*.toml")
    assert "pyproject.toml" in found.content

    listed = host.run("shell", command="ls")
    assert listed.receipt["outcome"] == "result"
    blocked = host.run("shell", command="echo owned > owned.txt")
    assert blocked.receipt["outcome"] == "error"
    assert not (root / "owned.txt").exists()

    escaped = host.run("read", path="..")
    assert escaped.receipt["outcome"] == "error"
    assert escaped.receipt["args"]["filePath"] == ".."


def test_omp_host_websearch_without_engine_is_error_never_no_match(tmp_path: Path) -> None:
    host = OmpHost(root=str(_project(tmp_path)))
    result = host.run("websearch", query="widget pricing")
    assert result.receipt["outcome"] == "error"
    assert result.receipt["tool"] == "websearch"
    assert result.receipt["outcome"] != "no_match"


def test_omp_host_drives_a_local_inspection_request_and_observes(tmp_path: Path) -> None:
    root = _project(tmp_path)
    host = OmpHost(root=str(root))
    payload = host.start(
        "Inspect the pinned dependencies and report the Python version constraints.",
        task_kind="code",
        effort="quick",
    )
    first_request_id = payload["next_action"]["request"]["request_id"]
    next_step = host.execute_next()
    assert next_step["ran"] is False
    assert next_step["request_id"] == first_request_id
    assert next_step["tool"] in {"grep", "read"}

    if next_step["tool"] == "grep":
        result = host.run("grep", pattern="dependencies|python", path=".")
    else:
        result = host.run("read", path="pyproject.toml")
    payload = host.observe(result, kind="documentation")
    assert payload["session"]["session_id"] == host.session_id
    assert payload["next_action"]["request"]["request_id"] != first_request_id


def test_omp_host_webfetch_attests_a_real_url_and_is_accepted(tmp_path: Path) -> None:
    url = _served_fixture()
    root = _project(tmp_path)
    host = OmpHost(root=str(root))
    payload = host.start(
        "Research the current widget pricing guidance from published sources.",
        task_kind="research",
        effort="quick",
    )
    first_request_id = payload["next_action"]["request"]["request_id"]

    fetched = host.run("webfetch", url=url)
    assert fetched.receipt == {"tool": "webfetch", "outcome": "result", "args": {"url": url}}
    assert fetched.url == url
    accepted = host.observe(
        fetched,
        purpose="contradiction_check",
        url=url,
        content=fetched.content,
    )
    assert "accepted_evidence_ids" in accepted
    assert accepted["next_action"]["request"]["request_id"] != first_request_id


def test_omp_host_forged_source_review_observation_is_rejected(tmp_path: Path) -> None:
    url = _served_fixture()
    root = _project(tmp_path)
    host = OmpHost(root=str(root))
    payload = host.start(
        "Find a scientific paper on widget benchmark reliability and report what it finds.",
        task_kind="research",
        effort="quick",
        strictness="strict",
    )
    paper_request = payload["next_action"]["request"]
    assert paper_request["parameters"]["purpose"] == "scholarly_validation"

    fetched = host.run("webfetch", url=url)
    with pytest.raises(ValueError, match="source_record"):
        host.observe(
            fetched, purpose="scholarly_validation", url=fetched.url, content=fetched.content
        )
    with pytest.raises(ValueError, match="URL"):
        host.observe(
            fetched,
            purpose="scholarly_validation",
            url="https://example.org/other-paper",
            content=fetched.content,
            source_record={
                "identifier": "10.0000/example",
                "method": "controlled benchmark",
                "limitations": "single domain",
            },
        )
    accepted = host.observe(
        fetched,
        purpose="scholarly_validation",
        url=fetched.url,
        content=fetched.content,
        source_record={
            "identifier": "10.0000/example",
            "method": "controlled benchmark",
            "limitations": "single domain",
        },
    )
    assert accepted["next_action"]["request"]["request_id"] != paper_request["request_id"]
    assert accepted["session"]["session_id"] == host.session_id


def test_omp_host_tools_list_is_the_compact_surface(tmp_path: Path) -> None:
    host = OmpHost(root=str(_project(tmp_path)))
    names = {tool["name"] for tool in host.tools()}
    assert {"cortheon_start", "cortheon_observe", "cortheon_complete"} <= names
    assert "cortheon_observe" in names
