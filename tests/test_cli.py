from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferops.cli import main

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

    captured = capsys.readouterr()
    assert "| concurrency |" in captured.out
    assert "concurrency 2:" in captured.err


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
