"""Write a benchmark run to disk: raw records, machine-readable summary, readable table."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from inferops.bench.loadgen import LevelResult
from inferops.bench.metrics import Distribution, LevelSummary
from inferops.bench.scenario import Scenario

_CSV_COLUMNS = (
    "concurrency",
    "requests_ok",
    "requests_failed",
    "window_seconds",
    "request_throughput",
    "output_token_throughput",
    "ttft_p50",
    "ttft_p99",
    "tpot_p50",
    "tpot_p99",
    "latency_p50",
    "latency_p99",
)


def _ms(distribution: Distribution | None, field: str) -> str:
    """Milliseconds for the table; a dash when the metric could not be measured."""
    if distribution is None:
        return "-"
    return f"{getattr(distribution, field) * 1000:.1f}"


def render_markdown(
    summaries: Sequence[LevelSummary], *, stopped_because: str | None = None
) -> str:
    """One table row per concurrency level, latencies in milliseconds.

    `stopped_because` marks a run that ended before its last level, so a partial table is
    never mistaken for the whole sweep.
    """
    lines = [
        "| concurrency | req/s | output tok/s | TTFT p50 (ms) | TTFT p99 (ms) "
        "| TPOT p50 (ms) | TPOT p99 (ms) | latency p50 (ms) | latency p99 (ms) | failed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.concurrency} | {s.request_throughput:.2f} | {s.output_token_throughput:.1f} "
            f"| {_ms(s.ttft, 'p50')} | {_ms(s.ttft, 'p99')} "
            f"| {_ms(s.tpot, 'p50')} | {_ms(s.tpot, 'p99')} "
            f"| {_ms(s.latency, 'p50')} | {_ms(s.latency, 'p99')} | {s.requests_failed} |"
        )
    if any(s.output_tokens_estimated for s in summaries):
        lines.append("")
        lines.append(
            "Note: the server sent no token usage for some requests, so output tokens were "
            "estimated by counting streamed chunks."
        )
    if stopped_because is not None:
        lines.append("")
        lines.append(f"Note: the run stopped early; completed levels only: {stopped_because}")
    return "\n".join(lines) + "\n"


def _csv_row(summary: LevelSummary) -> dict[str, object]:
    def pick(distribution: Distribution | None, field: str) -> float | str:
        return "" if distribution is None else getattr(distribution, field)

    return {
        "concurrency": summary.concurrency,
        "requests_ok": summary.requests_ok,
        "requests_failed": summary.requests_failed,
        "window_seconds": summary.window_seconds,
        "request_throughput": summary.request_throughput,
        "output_token_throughput": summary.output_token_throughput,
        "ttft_p50": pick(summary.ttft, "p50"),
        "ttft_p99": pick(summary.ttft, "p99"),
        "tpot_p50": pick(summary.tpot, "p50"),
        "tpot_p99": pick(summary.tpot, "p99"),
        "latency_p50": pick(summary.latency, "p50"),
        "latency_p99": pick(summary.latency, "p99"),
    }


def write_run(
    out_dir: Path,
    scenario: Scenario,
    results: Sequence[LevelResult],
    summaries: Sequence[LevelSummary],
    *,
    run_info: Mapping[str, Any] | None = None,
) -> None:
    """Persist one run.

    - summary.json: run metadata, the scenario and every level's summary (seconds).
    - summary.csv:  one row per level, for plotting.
    - summary.md:   the readable table (milliseconds).
    - requests.jsonl: every recorded request, so percentiles can be recomputed later.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    info = dict(run_info or {})
    summary = {
        "run": info,
        "scenario": scenario.model_dump(mode="json"),
        "levels": [s.to_dict() for s in summaries],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for s in summaries:
            writer.writerow(_csv_row(s))

    table = render_markdown(summaries, stopped_because=info.get("stopped_because"))
    (out_dir / "summary.md").write_text(table, encoding="utf-8")

    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for record in result.records:
                row = {"concurrency": result.concurrency, **asdict(record)}
                handle.write(json.dumps(row) + "\n")
