from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from inferops.bench.client import RequestRecord
from inferops.bench.loadgen import LevelResult
from inferops.bench.metrics import summarize
from inferops.bench.report import render_markdown, write_run
from tests.conftest import make_scenario


def _level(concurrency: int, *, estimated: bool = False) -> LevelResult:
    records = [
        RequestRecord(
            started_at=0.0,
            first_token_at=0.1,
            finished_at=1.1,
            prompt_tokens=20,
            output_tokens=11,
            output_tokens_estimated=estimated,
        ),
        RequestRecord(started_at=0.0, finished_at=0.5, error="HTTP 500: boom"),
    ]
    return LevelResult(concurrency=concurrency, window_start=0.0, records=records)


def test_markdown_has_one_row_per_level_in_milliseconds() -> None:
    results = [_level(1), _level(8)]
    summaries = [summarize(r.concurrency, r.records, r.window_start) for r in results]

    table = render_markdown(summaries)
    # Data rows start with the concurrency number; the header and separator rows do not.
    rows = [line for line in table.splitlines() if re.match(r"\| \d", line)]

    assert len(rows) == 2
    # TTFT 0.1 s -> 100.0 ms; TPOT (1.1 - 0.1) / 10 = 0.1 s -> 100.0 ms; one failure.
    assert rows[0].startswith("| 1 | ")
    assert "| 100.0 | 100.0 | 100.0 | 100.0 | 1100.0 | 1100.0 | 1 |" in rows[0]
    assert "estimated" not in table


def test_markdown_says_when_token_counts_are_estimated() -> None:
    result = _level(1, estimated=True)
    table = render_markdown([summarize(1, result.records, result.window_start)])
    assert "estimated by counting streamed chunks" in table


def test_write_run_persists_everything_needed_to_recompute(tmp_path: Path) -> None:
    scenario = make_scenario("http://127.0.0.1:8001/v1", concurrency=[1, 8])
    results = [_level(1), _level(8)]
    summaries = [summarize(r.concurrency, r.records, r.window_start) for r in results]

    out_dir = tmp_path / "run"
    write_run(out_dir, scenario, results, summaries)

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scenario"]["name"] == "test"
    assert [level["concurrency"] for level in summary["levels"]] == [1, 8]
    assert summary["levels"][0]["ttft"]["p50"] == 0.1

    with (out_dir / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["concurrency"] for row in rows] == ["1", "8"]
    assert float(rows[0]["ttft_p50"]) == 0.1

    lines = (out_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[1])["error"] == "HTTP 500: boom"

    assert (out_dir / "summary.md").read_text(encoding="utf-8") == render_markdown(summaries)
