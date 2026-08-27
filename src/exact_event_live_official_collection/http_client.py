from __future__ import annotations

import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin

from src.exact_event_live_official_collection.domain import SourceStatus


@dataclass(frozen=True, slots=True)
class FetchResult:
    request_url: str
    final_url: str | None
    status: int | None
    content_type: str | None
    body: bytes
    redirects: int
    redirect_chain: tuple[str, ...]
    blocker: str | None


class HttpClient(Protocol):
    def get(self, url: str) -> FetchResult: ...


class BoundedHttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        redirect_limit: int = 3,
        max_response_bytes: int = 1_000_000,
        user_agent: str = "trade-ai-live-official-exact-collector/1.0",
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._redirect_limit = redirect_limit
        self._max_response_bytes = max_response_bytes
        self._user_agent = user_agent
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def get(self, url: str) -> FetchResult:
        current = url
        redirect_chain: list[str] = []
        while True:
            request = urllib.request.Request(
                current,
                headers={"User-Agent": self._user_agent, "Accept": "*/*"},
                method="GET",
            )
            try:
                with self._opener.open(request, timeout=self._timeout_seconds) as response:
                    body = response.read(self._max_response_bytes + 1)
                    if len(body) > self._max_response_bytes:
                        return FetchResult(
                            url,
                            str(response.url),
                            int(response.status),
                            _content_type(response),
                            b"",
                            len(redirect_chain),
                            tuple(redirect_chain),
                            SourceStatus.TECHNICAL_FAILURE.value,
                        )
                    return FetchResult(
                        url,
                        str(response.url),
                        int(response.status),
                        _content_type(response),
                        body,
                        len(redirect_chain),
                        tuple(redirect_chain),
                        None,
                    )
            except urllib.error.HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                    redirect_chain.append(current)
                    if len(redirect_chain) > self._redirect_limit:
                        return _failed(
                            url,
                            current,
                            exc.code,
                            redirect_chain,
                            SourceStatus.TECHNICAL_FAILURE.value,
                        )
                    current = urljoin(current, str(exc.headers["Location"]))
                    continue
                return _failed(url, current, exc.code, redirect_chain, _http_blocker(exc.code))
            except urllib.error.URLError as exc:
                return _failed(
                    url,
                    current,
                    None,
                    redirect_chain,
                    _url_error_blocker(exc),
                )
            except TimeoutError:
                return _failed(url, current, None, redirect_chain, SourceStatus.TIMEOUT.value)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def _failed(
    request_url: str,
    final_url: str | None,
    status: int | None,
    redirect_chain: list[str],
    blocker: str,
) -> FetchResult:
    return FetchResult(
        request_url,
        final_url,
        status,
        None,
        b"",
        len(redirect_chain),
        tuple(redirect_chain),
        blocker,
    )


def _content_type(response: Any) -> str | None:
    return str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()


def _http_blocker(status: int) -> str:
    if status in {401, 407}:
        return SourceStatus.AUTH_REQUIRED.value
    if status == 403:
        return SourceStatus.POLICY_BLOCKED.value
    if status == 429:
        return SourceStatus.RATE_LIMITED.value
    return SourceStatus.HTTP_FAILURE.value


def _url_error_blocker(exc: urllib.error.URLError) -> str:
    if isinstance(exc.reason, socket.timeout):
        return SourceStatus.TIMEOUT.value
    if isinstance(exc.reason, ssl.SSLError):
        return SourceStatus.TECHNICAL_FAILURE.value
    return SourceStatus.TECHNICAL_FAILURE.value
