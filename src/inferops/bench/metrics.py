"""Turn per-request records into the numbers on a benchmark curve.

Definitions (see docs/metrics.md):

- TTFT: send -> first streamed token.
- TPOT: (last token - first token) / (output tokens - 1).
- Latency: send -> end of stream.
- Throughput: successful requests (or their output tokens) per second of measured time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from inferops.bench.client import RequestRecord


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between the two nearest ranks."""
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be within [0, 100], got {q}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(frozen=True, slots=True)
class Distribution:
    """Summary of one latency metric, in seconds."""

    mean: float
    p50: float
    p90: float
    p99: float
    max: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Distribution | None:
        """None when there is nothing to summarize: an absent number, not a zero."""
        if not values:
            return None
        return cls(
            mean=sum(values) / len(values),
            p50=percentile(values, 50),
            p90=percentile(values, 90),
            p99=percentile(values, 99),
            max=max(values),
        )


@dataclass(frozen=True, slots=True)
class LevelSummary:
    """One point on the curve: everything measured at one concurrency level."""

    concurrency: int
    elapsed_seconds: float
    requests_ok: int
    requests_failed: int
    # Successful requests per second.
    request_throughput: float
    # Output tokens of successful requests per second.
    output_token_throughput: float
    mean_prompt_tokens: float | None
    mean_output_tokens: float | None
    ttft: Distribution | None
    tpot: Distribution | None
    latency: Distribution | None
    # True when at least one request had no usage from the server and chunks were counted.
    output_tokens_estimated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    concurrency: int, records: Sequence[RequestRecord], window_start: float
) -> LevelSummary:
    """Aggregate the records of one level.

    `records` are the requests that started inside the measurement window. Measured time runs
    from `window_start` to the last completion, so requests still in flight at the end of the
    window are neither dropped (which would hide the slowest ones) nor counted for free.
    """
    ok = [record for record in records if record.ok]
    failed = len(records) - len(ok)
    elapsed = max((record.finished_at for record in records), default=window_start) - window_start

    output_tokens = sum(record.output_tokens for record in ok)
    prompt_tokens = [record.prompt_tokens for record in ok if record.prompt_tokens is not None]
    measurable = elapsed > 0

    return LevelSummary(
        concurrency=concurrency,
        elapsed_seconds=elapsed,
        requests_ok=len(ok),
        requests_failed=failed,
        request_throughput=len(ok) / elapsed if measurable else 0.0,
        output_token_throughput=output_tokens / elapsed if measurable else 0.0,
        mean_prompt_tokens=sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        mean_output_tokens=output_tokens / len(ok) if ok else None,
        ttft=Distribution.from_values([r.ttft for r in ok if r.ttft is not None]),
        tpot=Distribution.from_values([r.tpot for r in ok if r.tpot is not None]),
        latency=Distribution.from_values([r.latency for r in ok]),
        output_tokens_estimated=any(record.output_tokens_estimated for record in ok),
    )
