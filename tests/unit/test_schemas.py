from __future__ import annotations

from pydantic import ValidationError

from src.news.presentation.schemas import NewsCreateRequest


def valid_payload() -> dict[str, str]:
    return {
        "source_id": "test-source-001",
        "source_name": "Test News",
        "source_url": "https://example.com/news/1",
        "title": "Company published financial results",
        "raw_content": "Original publication text.",
        "language": "en",
        "published_at": "2026-08-06T08:00:00Z",
        "received_at": "2026-08-06T08:00:01Z",
    }


def test_request_accepts_valid_payload() -> None:
    request = NewsCreateRequest.model_validate(valid_payload())

    assert request.source_id == "test-source-001"


def test_invalid_url_is_rejected() -> None:
    payload = valid_payload()
    payload["source_url"] = "not-a-url"

    try:
        NewsCreateRequest.model_validate(payload)
    except ValidationError as exc:
        assert "source_url" in str(exc)
    else:
        raise AssertionError("invalid URL should fail validation")


def test_missing_published_at_is_rejected() -> None:
    payload = valid_payload()
    del payload["published_at"]

    try:
        NewsCreateRequest.model_validate(payload)
    except ValidationError as exc:
        assert "published_at" in str(exc)
    else:
        raise AssertionError("missing published_at should fail validation")


def test_timezone_naive_datetime_is_rejected() -> None:
    payload = valid_payload()
    payload["published_at"] = "2026-08-06T08:00:00"

    try:
        NewsCreateRequest.model_validate(payload)
    except ValidationError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("timezone-naive published_at should fail validation")
