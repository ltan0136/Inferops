"""Closed-loop load generator: N workers, each sending the next request as soon as one ends.

Closed loop means the offered load adapts to the server: `concurrency` is the number of
requests in flight, which is what a serving engine's batch size reacts to. Open-loop load
(fixed arrival rate) is a different experiment and is not part of v0.1.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from inferops.bench.client import Clock, RequestRecord, stream_chat
from inferops.bench.scenario import Scenario

# A fixed vocabulary keeps prompts deterministic apart from the optional unique marker.
_FILLER = (
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
    "while",
    "a",
    "distant",
    "train",
    "crosses",
    "the",
    "quiet",
    "valley",
)

# Pause after a failed request so a dead server is not hammered in a tight loop.
_ERROR_BACKOFF_SECONDS = 0.1


class BenchmarkError(RuntimeError):
    """The run cannot produce a trustworthy result (target down, every request failing)."""


@dataclass(frozen=True, slots=True)
class LevelResult:
    """Raw outcome of one concurrency level."""

    concurrency: int
    window_start: float
    records: list[RequestRecord]


def build_prompt(words: int, marker: str | None = None) -> str:
    """Synthetic prompt of roughly `words` words, optionally starting with a unique marker."""
    body = " ".join(_FILLER[i % len(_FILLER)] for i in range(words))
    head = f"[request {marker}] " if marker else ""
    return f"{head}Continue this text in the same style.\n\n{body}"


def new_client(scenario: Scenario, api_key: str | None = None) -> httpx.AsyncClient:
    """HTTP client for the scenario's target.

    Only the connect phase gets an httpx timeout. The whole request is bounded by
    `request_timeout_seconds` in `stream_chat`, because a per-read timeout would cut off
    a healthy but slow token stream.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    connections = max(scenario.load.concurrency) + 4
    return httpx.AsyncClient(
        base_url=scenario.target.base_url.rstrip("/") + "/",
        headers=headers,
        timeout=httpx.Timeout(None, connect=10.0),
        limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
    )


async def _one_request(
    client: httpx.AsyncClient, scenario: Scenario, *, clock: Clock, max_tokens: int | None = None
) -> RequestRecord:
    request = scenario.request
    marker = uuid.uuid4().hex[:12] if request.unique_prefix else None
    return await stream_chat(
        client,
        model=scenario.target.model,
        prompt=build_prompt(request.prompt_words, marker),
        max_tokens=max_tokens or request.max_tokens,
        temperature=request.temperature,
        extra_body=request.extra_body,
        timeout_seconds=scenario.load.request_timeout_seconds,
        clock=clock,
    )


async def preflight(
    client: httpx.AsyncClient, scenario: Scenario, *, clock: Clock = time.perf_counter
) -> None:
    """Send one small request before the sweep, so a wrong URL or model name fails fast."""
    record = await _one_request(
        client, scenario, clock=clock, max_tokens=min(scenario.request.max_tokens, 8)
    )
    if not record.ok:
        raise BenchmarkError(
            f"preflight request to {scenario.target.base_url} failed: {record.error}"
        )


async def run_level(
    client: httpx.AsyncClient,
    scenario: Scenario,
    concurrency: int,
    *,
    clock: Clock = time.perf_counter,
) -> LevelResult:
    """Hold `concurrency` requests in flight through warmup plus the measurement window.

    Requests that start during warmup are sent but not recorded. Requests that start inside
    the window are recorded even if they finish after it ends.
    """
    window_start = clock() + scenario.load.warmup_seconds
    window_end = window_start + scenario.load.duration_seconds
    records: list[RequestRecord] = []

    async def worker() -> None:
        while clock() < window_end:
            record = await _one_request(client, scenario, clock=clock)
            if record.started_at >= window_start:
                records.append(record)
            if not record.ok:
                await asyncio.sleep(_ERROR_BACKOFF_SECONDS)

    async with asyncio.TaskGroup() as group:
        for _ in range(concurrency):
            group.create_task(worker())

    if not any(record.ok for record in records):
        first_error = next((record.error for record in records if record.error), "no request ran")
        raise BenchmarkError(
            f"no successful request at concurrency {concurrency}; first error: {first_error}"
        )
    return LevelResult(concurrency=concurrency, window_start=window_start, records=records)


async def run_scenario(
    scenario: Scenario,
    *,
    api_key: str | None = None,
    on_level: Callable[[LevelResult], None] | None = None,
    clock: Clock = time.perf_counter,
) -> list[LevelResult]:
    """Run every concurrency level of the scenario, in the order the scenario lists them."""
    results: list[LevelResult] = []
    async with new_client(scenario, api_key) as client:
        await preflight(client, scenario, clock=clock)
        for concurrency in scenario.load.concurrency:
            result = await run_level(client, scenario, concurrency, clock=clock)
            results.append(result)
            if on_level is not None:
                on_level(result)
    return results
