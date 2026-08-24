from __future__ import annotations

import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

from src.exact_event_live_source_snapshot.domain import (
    MAX_REDIRECTS,
    MAX_RESPONSE_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
    LiveBlocker,
)


@dataclass(frozen=True, slots=True)
class HttpResult:
    request_url: str
    final_url: str | None
    status: int | None
    content_type: str | None
    body: bytes
    redirects: int
    blocker: str | None


class HttpClient(Protocol):
    def get(self, url: str) -> HttpResult: ...


class BoundedHttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        user_agent: str = USER_AGENT,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_redirects = max_redirects
        self._user_agent = user_agent
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def get(self, url: str) -> HttpResult:
        current = url
        redirects = 0
        while True:
            request = urllib.request.Request(
                current,
                headers={"User-Agent": self._user_agent, "Accept": "*/*"},
                method="GET",
            )
            try:
                with self._opener.open(
                    request,
                    timeout=self._timeout_seconds,
                ) as response:
                    status = int(response.status)
                    final_url = str(response.url)
                    content_type = str(response.headers.get("Content-Type") or "").split(";")[0]
                    body = response.read(self._max_response_bytes + 1)
                    if len(body) > self._max_response_bytes:
                        return HttpResult(
                            url,
                            final_url,
                            status,
                            content_type,
                            b"",
                            redirects,
                            LiveBlocker.RESPONSE_TOO_LARGE.value,
                        )
                    return HttpResult(url, final_url, status, content_type, body, redirects, None)
            except urllib.error.HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                    redirects += 1
                    if redirects > self._max_redirects:
                        return HttpResult(
                            url,
                            current,
                            exc.code,
                            None,
                            b"",
                            redirects,
                            LiveBlocker.TECHNICAL_FETCH_FAILED.value,
                        )
                    current = urljoin(current, str(exc.headers["Location"]))
                    continue
                blocker = (
                    LiveBlocker.RATE_LIMITED.value if exc.code == 429 else _http_blocker(exc.code)
                )
                return HttpResult(url, current, exc.code, None, b"", redirects, blocker)
            except urllib.error.URLError as exc:
                return HttpResult(url, current, None, None, b"", redirects, _url_error_blocker(exc))
            except TimeoutError:
                return HttpResult(
                    url, current, None, None, b"", redirects, LiveBlocker.TIMEOUT.value
                )


class PoliteDomainClient:
    def __init__(self, client: HttpClient, *, min_delay_seconds: float) -> None:
        self._client = client
        self._min_delay_seconds = min_delay_seconds
        self._last_request_by_domain: dict[str, float] = {}

    def get(self, url: str) -> HttpResult:
        domain = urlsplit(url).netloc.lower()
        now = time.monotonic()
        last = self._last_request_by_domain.get(domain)
        if last is not None:
            wait = self._min_delay_seconds - (now - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request_by_domain[domain] = time.monotonic()
        return self._client.get(url)


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


def _http_blocker(status: int) -> str:
    if status in {401, 407}:
        return LiveBlocker.AUTH_REQUIRED.value
    if status == 402:
        return LiveBlocker.PAYMENT_REQUIRED.value
    if 400 <= status < 500:
        return LiveBlocker.HTTP_4XX.value
    if status >= 500:
        return LiveBlocker.HTTP_5XX.value
    return LiveBlocker.TECHNICAL_FETCH_FAILED.value


def _url_error_blocker(exc: urllib.error.URLError) -> str:
    reason = exc.reason
    if isinstance(reason, socket.timeout):
        return LiveBlocker.TIMEOUT.value
    if isinstance(reason, ssl.SSLError):
        return LiveBlocker.TLS_FAILED.value
    if isinstance(reason, OSError):
        return LiveBlocker.DNS_FAILED.value
    return LiveBlocker.TECHNICAL_FETCH_FAILED.value
