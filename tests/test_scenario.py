from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from inferops.bench.scenario import load_scenario
from tests.conftest import REPO_ROOT

SCENARIO_DIR = REPO_ROOT / "bench" / "scenarios"


@pytest.mark.parametrize("path", sorted(SCENARIO_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_scenarios_are_valid(path: Path) -> None:
    scenario = load_scenario(path)
    assert scenario.name == path.stem
    assert scenario.load.concurrency == sorted(scenario.load.concurrency)


def test_both_scenarios_ship() -> None:
    names = {path.name for path in SCENARIO_DIR.glob("*.yaml")}
    assert {"smoke.yaml", "baseline.yaml"} <= names


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "name: x\ntarget: {model: m}\nload: {concurrency: [1], duration_second: 3}\n",
    )
    with pytest.raises(ValidationError, match="duration_second"):
        load_scenario(path)


@pytest.mark.parametrize("levels", ["[0]", "[2, 2]", "[]"])
def test_bad_concurrency_levels_are_rejected(tmp_path: Path, levels: str) -> None:
    path = _write(tmp_path, f"name: x\ntarget: {{model: m}}\nload: {{concurrency: {levels}}}\n")
    with pytest.raises(ValidationError):
        load_scenario(path)


def test_a_file_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_scenario(_write(tmp_path, "- just\n- a list\n"))
