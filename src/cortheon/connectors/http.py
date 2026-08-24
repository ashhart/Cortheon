from __future__ import annotations

import http.client
import ipaddress
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast


class ConnectorError(RuntimeError):
    pass


@dataclass(slots=True)
class HttpResponse:
    url: str
    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class JsonHttpClient:
    def __init__(
        self,
        timeout_seconds: float = 15.0,
        url_validator: Callable[[str], None] | None = None,
        max_response_bytes: int = 5_000_000,
    ) -> None:
        if not 1_024 <= max_response_bytes <= 100_000_000:
            raise ValueError("max_response_bytes must be between 1024 and 100000000")
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.ssl_context = _ssl_context()
        self.url_validator = url_validator
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self.ssl_context),
            _ValidatingRedirectHandler(self._validate_url),
        )

    def get(self, url: str, headers: dict[str, str] | None = None) -> HttpResponse:
        self._validate_url(url)
        request = urllib.request.Request(url, headers=self._headers(headers))
        for attempt in (0, 1):
            try:
                with self._open(request, self.timeout_seconds) as response:
                    self._validate_url(response.geturl())
                    return HttpResponse(
                        url=response.geturl(),
                        status=response.status,
                        body=self._read_bounded(response),
                        headers=dict(response.headers.items()),
                    )
            except urllib.error.HTTPError as exc:
                # Be a polite client: honor Retry-After once on rate limits
                # instead of failing the whole discovery pass immediately.
                if exc.code == 429 and attempt == 0:
                    time.sleep(
                        _retry_after_seconds(
                            exc.headers.get("Retry-After") if exc.headers else None
                        )
                    )
                    continue
                raise ConnectorError(f"GET {url} failed with HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                raise ConnectorError(f"GET {url} failed: {exc.reason}") from exc
            except (OSError, http.client.HTTPException) as exc:
                # urlopen only wraps connect-phase failures; a mid-read socket
                # timeout or dropped connection raises raw OSError/HTTPException.
                # One slow source must degrade to a connector error, never crash
                # a whole mission.
                raise ConnectorError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
        raise ConnectorError(f"GET {url} failed after retry.")

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        return self.get(url, headers=headers).json()

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request_headers = self._headers(headers)
        request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        self._validate_url(url)
        try:
            with self._open(request, self.timeout_seconds) as response:
                self._validate_url(response.geturl())
                return json.loads(self._read_bounded(response).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ConnectorError(f"POST {url} failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ConnectorError(f"POST {url} failed: {exc.reason}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise ConnectorError(f"POST {url} failed: {type(exc).__name__}: {exc}") from exc

    def head_or_get_status(self, url: str) -> int | None:
        for method in ("HEAD", "GET"):
            self._validate_url(url)
            request = urllib.request.Request(url, headers=self._headers(), method=method)
            try:
                with self._open(request, min(self.timeout_seconds, 8.0)) as response:
                    self._validate_url(response.geturl())
                    return response.status
            except urllib.error.HTTPError as exc:
                if method == "GET":
                    return exc.code
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                if method == "GET":
                    return None
        return None

    def _headers(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        merged = {
            "Accept": "application/json",
            "User-Agent": "cortheon/0.1",
        }
        if headers:
            merged.update(headers)
        return merged

    def _validate_url(self, url: str) -> None:
        validate_http_url(url)
        if self.url_validator is not None:
            self.url_validator(url)

    def _read_bounded(self, response: Any) -> bytes:
        body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise ConnectorError(f"HTTP response exceeded {self.max_response_bytes} bytes")
        return body

    def _open(self, request: urllib.request.Request, timeout: float) -> Any:
        if self.url_validator is not None:
            return self._open_pinned(request, timeout)
        return self._opener.open(request, timeout=timeout)

    def _open_pinned(
        self,
        request: urllib.request.Request,
        timeout: float,
    ) -> Any:
        """Resolve, validate, and connect to the same public IP address."""

        current_url = request.full_url
        method = request.get_method()
        body = request.data
        headers = dict(request.header_items())
        previous_origin: tuple[str, str, int] | None = None
        for _redirect in range(6):
            self._validate_url(current_url)
            parsed = urllib.parse.urlparse(current_url)
            host = parsed.hostname
            if host is None:
                raise ConnectorError("HTTP URL has no host")
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addresses = _public_addresses(host, port)
            origin = (parsed.scheme, host.casefold(), port)
            if previous_origin is not None and origin != previous_origin:
                headers = {
                    name: value
                    for name, value in headers.items()
                    if name.casefold() not in {"authorization", "cookie", "proxy-authorization"}
                }
            previous_origin = origin
            connection = _pinned_connection(
                parsed.scheme,
                host,
                port,
                addresses[0],
                timeout,
                self.ssl_context,
            )
            path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            request_headers = dict(headers)
            request_headers.setdefault(
                "Host",
                host if port == (443 if parsed.scheme == "https" else 80) else f"{host}:{port}",
            )
            try:
                connection.request(
                    method,
                    path,
                    body=body,
                    headers=request_headers,
                )
                response = connection.getresponse()
            except BaseException:
                connection.close()
                raise
            wrapped = _PinnedResponse(response, connection, current_url)
            if response.status not in {301, 302, 303, 307, 308}:
                if response.status >= 400:
                    raise urllib.error.HTTPError(
                        current_url,
                        response.status,
                        response.reason,
                        response.headers,
                        cast(IO[bytes], wrapped),
                    )
                return wrapped
            location = response.headers.get("Location")
            if not location:
                return wrapped
            if method not in {"GET", "HEAD"}:
                wrapped.close()
                raise ConnectorError("redirects are disabled for authenticated POST tools")
            wrapped.close()
            current_url = urllib.parse.urljoin(current_url, location)
        raise ConnectorError("HTTP redirect limit exceeded")


class _PinnedResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPConnection,
        url: str,
    ) -> None:
        self._response = response
        self._connection = connection
        self._url = url
        self.status = response.status
        self.headers = response.headers

    def __enter__(self) -> _PinnedResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def read(self, amount: int | None = None) -> bytes:
        return self._response.read(amount)

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        pinned_address: str,
        *,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._pinned_address = pinned_address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            getattr(self, "source_address", None),
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        pinned_address: str,
        *,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=context,
        )
        self._pinned_address = pinned_address
        self._ssl_context = context

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            getattr(self, "source_address", None),
        )
        self.sock = self._ssl_context.wrap_socket(
            raw_socket,
            server_hostname=self.host,
        )


def _public_addresses(host: str, port: int) -> list[str]:
    try:
        addresses = list(
            dict.fromkeys(
                str(item[4][0])
                for item in socket.getaddrinfo(
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                )
            )
        )
    except OSError as exc:
        raise ConnectorError(f"HTTP host could not be resolved: {exc}") from exc
    if not addresses:
        raise ConnectorError("HTTP host resolved to no addresses")
    if any(not ipaddress.ip_address(value).is_global for value in addresses):
        raise ConnectorError("HTTP host resolved to a non-public address")
    return addresses


def _pinned_connection(
    scheme: str,
    host: str,
    port: int,
    address: str,
    timeout: float,
    context: ssl.SSLContext,
) -> http.client.HTTPConnection:
    if scheme == "https":
        return _PinnedHTTPSConnection(
            host,
            address,
            port=port,
            timeout=timeout,
            context=context,
        )
    return _PinnedHTTPConnection(
        host,
        address,
        port=port,
        timeout=timeout,
    )


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], None]) -> None:
        super().__init__()
        self.validator = validator

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        self.validator(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_http_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConnectorError("URL must be absolute HTTP(S)")
    if parsed.username or parsed.password:
        raise ConnectorError("URL credentials are not allowed")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ConnectorError("URL port is invalid") from exc


def validate_public_http_url(url: str) -> None:
    validate_http_url(url)
    parsed = urllib.parse.urlparse(url)
    if parsed.port not in {None, 80, 443}:
        raise ConnectorError("public HTTP requests require port 80 or 443")


def normalize_package_name(name: str) -> str:
    return urllib.parse.quote(name.strip())


def _retry_after_seconds(value: str | None, default: float = 3.0, cap: float = 10.0) -> float:
    if not value:
        return default
    try:
        return max(0.5, min(float(value), cap))
    except ValueError:
        return default


def _ssl_context() -> ssl.SSLContext:
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
        "/etc/ssl/cert.pem",
        "/opt/homebrew/etc/ca-certificates/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()
