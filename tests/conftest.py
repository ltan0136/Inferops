"""Shared fixtures. Every test runs against the mock server: no GPU, no network, no secrets."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from inferops.bench.scenario import Load, RequestShape, Scenario, Target
from inferops.mock.server import DEFAULT_MODEL, running_mock_server

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mock timings used across tests. Small enough to keep the suite fast, large enough that
# the assertions below are not at the mercy of scheduler jitter.
MOCK_TTFT = 0.03
MOCK_TOKEN_INTERVAL = 0.005
MOCK_CAPACITY = 2


@pytest.fixture
async def mock_url() -> AsyncIterator[str]:
    """Base URL of a mock server that serves `MOCK_CAPACITY` requests at once."""
    async with running_mock_server(
        ttft_seconds=MOCK_TTFT,
        token_interval_seconds=MOCK_TOKEN_INTERVAL,
        max_concurrency=MOCK_CAPACITY,
    ) as base_url:
        yield base_url


def make_scenario(base_url: str, *, concurrency: list[int] | None = None) -> Scenario:
    """A short scenario aimed at the mock server."""
    return Scenario(
        name="test",
        target=Target(base_url=base_url, model=DEFAULT_MODEL),
        request=RequestShape(prompt_words=20, max_tokens=5),
        load=Load(
            concurrency=concurrency or [1],
            warmup_seconds=0.0,
            duration_seconds=0.6,
            request_timeout_seconds=10.0,
        ),
    )
