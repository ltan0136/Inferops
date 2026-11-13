from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from inferops.bench import loadgen
from inferops.bench.loadgen import BenchmarkError, LevelResult
from inferops.bench.scenario import Scenario
from inferops.cli import main
from inferops.mock.server import MOCK_VERSION

FAST_SCENARIO = """\
name: cli-test
target:
  base_url: http://127.0.0.1:9/v1
  model: mock-model
request:
  prompt_words: 10
  max_tokens: 4
load:
  concurrency: [1, 2]
  warmup_seconds: 0
  duration_seconds: 0.4
  request_timeout_seconds: 10
"""


@pytest.fixture
def scenario_path(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(FAST_SCENARIO, encoding="utf-8")
    return path


def test_bench_against_spawned_mock_writes_a_full_run(
    scenario_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "run"
    code = main(["bench", "--scenario", str(scenario_path), "--spawn-mock", "--out", str(out_dir)])

    assert code == 0
    for name in ("summary.json", "summary.csv", "summary.md", "requests.jsonl"):
        assert (out_dir / name).is_file()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert [level["concurrency"] for level in summary["levels"]] == [1, 2]
    assert all(level["requests_ok"] > 0 for level in summary["levels"])
    assert all(level["requests_failed"] == 0 for level in summary["levels"])
    # The saved scenario records the address that was really benchmarked, not the one in the file.
    assert summary["scenario"]["target"]["base_url"] != "http://127.0.0.1:9/v1"
    assert summary["run"]["complete"] is True
    assert summary["run"]["stopped_because"] is None
    assert summary["run"]["server_version"] == MOCK_VERSION
    assert summary["run"]["inferops_version"]

    captured = capsys.readouterr()
    assert "| concurrency |" in captured.out
    assert "concurrency 2:" in captured.err


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (BenchmarkError("no successful request at concurrency 2"), "concurrency 2"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_a_run_that_stops_part_way_keeps_its_completed_levels(
    scenario_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    reason: str,
) -> None:
    """A sweep that dies at its highest level must not throw away the levels before it."""
    real_run_level = loadgen.run_level

    async def failing_at_two(
        client: httpx.AsyncClient, scenario: Scenario, concurrency: int, **kwargs: Any
    ) -> LevelResult:
        if concurrency == 2:
            raise failure
        return await real_run_level(client, scenario, concurrency, **kwargs)

    monkeypatch.setattr(loadgen, "run_level", failing_at_two)
    out_dir = tmp_path / "run"
    code = main(["bench", "--scenario", str(scenario_path), "--spawn-mock", "--out", str(out_dir)])

    assert code == 1
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert [level["concurrency"] for level in summary["levels"]] == [1]
    assert summary["run"]["complete"] is False
    assert reason in summary["run"]["stopped_because"]
    assert "stopped early" in (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "completed levels were written" in capsys.readouterr().err


def test_bench_against_a_dead_target_exits_nonzero_and_writes_nothing(
    scenario_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "run"
    code = main(["bench", "--scenario", str(scenario_path), "--out", str(out_dir)])

    assert code == 1
    assert not out_dir.exists()
    assert "preflight" in capsys.readouterr().err


def test_bench_with_a_broken_scenario_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\n", encoding="utf-8")

    assert main(["bench", "--scenario", str(path)]) == 1
    assert "cannot load scenario" in capsys.readouterr().err


def test_bench_with_a_missing_api_key_variable_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INFEROPS_TEST_KEY", raising=False)
    path = tmp_path / "keyed.yaml"
    path.write_text(
        FAST_SCENARIO.replace(
            "model: mock-model", "model: mock-model\n  api_key_env: INFEROPS_TEST_KEY"
        ),
        encoding="utf-8",
    )

    assert main(["bench", "--scenario", str(path), "--spawn-mock"]) == 1
    assert "INFEROPS_TEST_KEY is not set" in capsys.readouterr().err
