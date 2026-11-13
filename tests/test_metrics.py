"""Metric definitions are the contract of the whole project: they are tested with exact numbers."""

from __future__ import annotations

import pytest

from inferops.bench.client import RequestRecord
from inferops.bench.metrics import Distribution, percentile, summarize


def test_percentile_interpolates_between_ranks() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0
    assert percentile(values, 50) == pytest.approx(25.0)
    # rank = 3 * 0.9 = 2.7 -> 30 + 0.7 * (40 - 30)
    assert percentile(values, 90) == pytest.approx(37.0)


def test_percentile_does_not_depend_on_input_order() -> None:
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([0.25], 99) == 0.25


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_distribution_of_nothing_is_absent_not_zero() -> None:
    assert Distribution.from_values([]) is None


def test_record_derives_ttft_latency_and_tpot() -> None:
    record = RequestRecord(started_at=10.0, first_token_at=10.5, finished_at=12.5, output_tokens=5)
    assert record.ttft == pytest.approx(0.5)
    assert record.latency == pytest.approx(2.5)
    # 2.0 s of decoding spread over the 4 tokens after the first one.
    assert record.tpot == pytest.approx(0.5)


def test_tpot_is_undefined_for_a_single_token() -> None:
    record = RequestRecord(started_at=0.0, first_token_at=0.2, finished_at=0.2, output_tokens=1)
    assert record.tpot is None


def test_summarize_counts_throughput_over_the_window() -> None:
    records = [
        RequestRecord(started_at=100.0, first_token_at=100.1, finished_at=101.0, output_tokens=10),
        RequestRecord(started_at=101.0, first_token_at=101.3, finished_at=104.0, output_tokens=30),
    ]
    summary = summarize(4, records, window_seconds=4.0)

    assert summary.concurrency == 4
    assert summary.requests_ok == 2
    assert summary.requests_failed == 0
    assert summary.window_seconds == pytest.approx(4.0)
    assert summary.request_throughput == pytest.approx(0.5)
    assert summary.output_token_throughput == pytest.approx(10.0)
    assert summary.mean_output_tokens == pytest.approx(20.0)
    assert summary.ttft is not None
    assert summary.ttft.p50 == pytest.approx(0.2)
    assert summary.ttft.max == pytest.approx(0.3)
    assert summary.output_tokens_estimated is False


def test_a_request_running_past_the_window_does_not_stretch_it() -> None:
    """One closed-loop worker, 1 s per request: the true rate is 1 request per second.

    The window is [0, 3). A request started during warmup ends at 0.5 and is not recorded;
    the one started at 2.5 is recorded and ends at 3.5, after the window closed. Dividing by
    the time to the last completion would report 3 / 3.5 = 0.86 req/s.
    """
    records = [
        RequestRecord(started_at=start, first_token_at=start + 0.1, finished_at=start + 1.0)
        for start in (0.5, 1.5, 2.5)
    ]
    summary = summarize(1, records, window_seconds=3.0)

    assert summary.request_throughput == pytest.approx(1.0)
    assert summary.latency is not None
    assert summary.latency.max == pytest.approx(1.0)


def test_failed_requests_are_counted_but_never_enter_latency_or_throughput() -> None:
    records = [
        RequestRecord(started_at=0.0, first_token_at=0.1, finished_at=1.0, output_tokens=10),
        RequestRecord(started_at=0.0, finished_at=2.0, error="HTTP 500: boom"),
    ]
    summary = summarize(2, records, window_seconds=2.0)

    assert summary.requests_ok == 1
    assert summary.requests_failed == 1
    assert summary.request_throughput == pytest.approx(0.5)
    assert summary.latency is not None
    assert summary.latency.max == pytest.approx(1.0)


def test_summarize_with_no_records_reports_zero_throughput_and_no_latency() -> None:
    summary = summarize(1, [], window_seconds=5.0)
    assert summary.requests_ok == 0
    assert summary.request_throughput == 0.0
    assert summary.ttft is None
    assert summary.latency is None


def test_summary_flags_estimated_token_counts() -> None:
    records = [
        RequestRecord(
            started_at=0.0,
            first_token_at=0.1,
            finished_at=1.0,
            output_tokens=7,
            output_tokens_estimated=True,
        )
    ]
    assert summarize(1, records, window_seconds=1.0).output_tokens_estimated is True


@pytest.mark.parametrize("window", [0.0, -1.0])
def test_summarize_rejects_an_empty_window(window: float) -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        summarize(1, [], window_seconds=window)
