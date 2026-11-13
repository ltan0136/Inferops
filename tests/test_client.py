"""The client talks to the mock server over a real socket, not through a patched HTTP layer."""

from __future__ import annotations

import httpx

from inferops.bench.client import stream_chat
from inferops.mock.server import DEFAULT_MODEL
from tests.conftest import MOCK_TOKEN_INTERVAL, MOCK_TTFT


def _client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url + "/", timeout=httpx.Timeout(None, connect=5.0))


async def test_successful_request_is_timed_and_counted(mock_url: str) -> None:
    async with _client(mock_url) as client:
        record = await stream_chat(
            client, model=DEFAULT_MODEL, prompt="one two three", max_tokens=6
        )

    assert record.ok
    assert record.output_tokens == 6
    assert record.output_tokens_estimated is False
    assert record.prompt_tokens == 3
    # The mock cannot answer faster than its configured delays.
    assert record.ttft is not None and record.ttft >= MOCK_TTFT
    assert record.tpot is not None and record.tpot >= MOCK_TOKEN_INTERVAL * 0.9
    assert record.latency >= record.ttft


async def test_extra_body_is_sent_to_the_server(mock_url: str) -> None:
    # `stream_options` from extra_body overrides the default, so the server sends no usage
    # and the client has to fall back to counting chunks, and must say so.
    async with _client(mock_url) as client:
        record = await stream_chat(
            client,
            model=DEFAULT_MODEL,
            prompt="hello",
            max_tokens=4,
            extra_body={"stream_options": {"include_usage": False}},
        )

    assert record.ok
    assert record.output_tokens == 4
    assert record.output_tokens_estimated is True
    assert record.prompt_tokens is None


async def test_wrong_model_name_is_reported_not_raised(mock_url: str) -> None:
    async with _client(mock_url) as client:
        record = await stream_chat(client, model="no-such-model", prompt="hello", max_tokens=4)

    assert not record.ok
    assert record.error is not None
    assert "HTTP 404" in record.error
    assert "no-such-model" in record.error
    assert record.ttft is None


async def test_request_slower_than_the_timeout_is_an_error(mock_url: str) -> None:
    async with _client(mock_url) as client:
        record = await stream_chat(
            client, model=DEFAULT_MODEL, prompt="hello", max_tokens=200, timeout_seconds=0.1
        )

    assert not record.ok
    assert record.error == "timed out after 0.1s"


async def test_unreachable_server_is_an_error() -> None:
    # Port 9 (discard) is not listening on localhost in CI or on dev machines.
    async with _client("http://127.0.0.1:9/v1") as client:
        record = await stream_chat(client, model=DEFAULT_MODEL, prompt="hello", max_tokens=4)

    assert not record.ok
    assert record.error is not None
    assert "ConnectError" in record.error
