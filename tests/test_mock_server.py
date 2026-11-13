"""The mock is a test double other tools also talk to, so its API surface is pinned here."""

from __future__ import annotations

import httpx
import pytest

from inferops.mock.server import DEFAULT_MODEL, create_app


async def test_health_and_model_listing(mock_url: str) -> None:
    root = mock_url.removesuffix("/v1")
    async with httpx.AsyncClient() as client:
        health = await client.get(f"{root}/health")
        models = await client.get(f"{mock_url}/models")

    assert health.status_code == 200
    assert [model["id"] for model in models.json()["data"]] == [DEFAULT_MODEL]


async def test_non_streaming_completion_reports_usage(mock_url: str) -> None:
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "one two three four"}],
        "max_tokens": 3,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{mock_url}/chat/completions", json=body)

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }


async def test_max_completion_tokens_is_honoured(mock_url: str) -> None:
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 7,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{mock_url}/chat/completions", json=body)

    assert response.json()["usage"]["completion_tokens"] == 7


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"model": DEFAULT_MODEL}, 400),
        ({"model": DEFAULT_MODEL, "messages": [], "max_tokens": 0}, 400),
        ({"model": DEFAULT_MODEL, "messages": [], "max_tokens": "many"}, 400),
        ({"model": "other", "messages": []}, 404),
    ],
)
async def test_bad_requests_get_an_openai_style_error(
    mock_url: str, body: dict[str, object], status: int
) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{mock_url}/chat/completions", json=body)

    assert response.status_code == status
    assert response.json()["error"]["code"] == status


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        create_app(max_concurrency=0)
