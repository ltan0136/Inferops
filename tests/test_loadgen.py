from __future__ import annotations

import pytest

from inferops.bench.loadgen import (
    BenchmarkError,
    build_prompt,
    new_client,
    preflight,
    run_level,
    run_scenario,
)
from inferops.bench.metrics import summarize
from tests.conftest import MOCK_CAPACITY, make_scenario


def test_prompt_has_the_requested_length_and_marker() -> None:
    prompt = build_prompt(30, marker="abc123")
    assert prompt.startswith("[request abc123] ")
    body = prompt.split("\n\n", 1)[1]
    assert len(body.split()) == 30


def test_prompts_without_marker_are_identical() -> None:
    assert build_prompt(10) == build_prompt(10)


async def test_level_records_only_successful_timed_requests(mock_url: str) -> None:
    scenario = make_scenario(mock_url)
    async with new_client(scenario) as client:
        result = await run_level(client, scenario, concurrency=1)

    assert result.concurrency == 1
    assert len(result.records) >= 2
    assert all(record.ok for record in result.records)
    assert all(record.started_at >= result.window_start for record in result.records)


async def test_requests_started_during_warmup_are_not_recorded(mock_url: str) -> None:
    scenario = make_scenario(mock_url)
    scenario = scenario.model_copy(
        update={"load": scenario.load.model_copy(update={"warmup_seconds": 0.3})}
    )
    async with new_client(scenario) as client:
        result = await run_level(client, scenario, concurrency=1)

    assert result.records
    assert min(record.started_at for record in result.records) >= result.window_start


async def test_queueing_past_capacity_shows_up_as_ttft(mock_url: str) -> None:
    """Past the server's capacity, extra requests wait: TTFT rises and throughput stops growing."""
    scenario = make_scenario(mock_url)
    async with new_client(scenario) as client:
        within = await run_level(client, scenario, concurrency=MOCK_CAPACITY)
        beyond = await run_level(client, scenario, concurrency=MOCK_CAPACITY * 4)

    at_capacity = summarize(within.concurrency, within.records, within.window_start)
    saturated = summarize(beyond.concurrency, beyond.records, beyond.window_start)

    assert at_capacity.ttft is not None and saturated.ttft is not None
    # Each queued request waits for about three service times before it is served.
    assert saturated.ttft.p50 > 2 * at_capacity.ttft.p50
    # Four times the load must not buy anywhere near four times the throughput.
    assert saturated.request_throughput < 2 * at_capacity.request_throughput


async def test_preflight_fails_fast_on_a_wrong_model(mock_url: str) -> None:
    scenario = make_scenario(mock_url)
    scenario = scenario.model_copy(
        update={"target": scenario.target.model_copy(update={"model": "typo"})}
    )
    async with new_client(scenario) as client:
        with pytest.raises(BenchmarkError, match="HTTP 404"):
            await preflight(client, scenario)


async def test_scenario_runs_every_level_in_order(mock_url: str) -> None:
    scenario = make_scenario(mock_url, concurrency=[1, 3])
    seen: list[int] = []
    results = await run_scenario(scenario, on_level=lambda result: seen.append(result.concurrency))

    assert [result.concurrency for result in results] == [1, 3]
    assert seen == [1, 3]


async def test_scenario_against_a_dead_target_raises() -> None:
    scenario = make_scenario("http://127.0.0.1:9/v1")
    with pytest.raises(BenchmarkError, match="preflight"):
        await run_scenario(scenario)
