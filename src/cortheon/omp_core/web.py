"""Public-network validation for bounded OMP web operations."""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request
from typing import Any


def _validate_web_url(url: str, *, allow_private_network: bool) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("web access requires an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("web URLs cannot contain credentials")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("web URL is missing a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("web URL has an invalid port") from exc
    if allow_private_network:
        return
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    hostname,
                    port or 443,
                    type=socket.SOCK_STREAM,
                )
            }
        except (OSError, ValueError) as exc:
            raise ValueError("web URL hostname could not be resolved") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("web access to private or non-public addresses is blocked")


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, allow_private_network: bool) -> None:
        super().__init__()
        self._allow_private_network = allow_private_network

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_web_url(newurl, allow_private_network=self._allow_private_network)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_web_url(
    url: str,
    *,
    allow_private_network: bool,
    timeout: float,
) -> Any:
    """Open one HTTP(S) URL after validating its origin and every redirect."""
    _validate_web_url(url, allow_private_network=allow_private_network)
    opener = urllib.request.build_opener(
        _ValidatingRedirectHandler(allow_private_network=allow_private_network)
    )
    response = opener.open(url, timeout=timeout)
    try:
        _validate_web_url(response.geturl(), allow_private_network=allow_private_network)
    except Exception:
        response.close()
        raise
    return response
