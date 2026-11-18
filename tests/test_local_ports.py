"""Check the host-side commands together without starting a cluster or contacting a model."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from inferops.bench.scenario import load_scenario
from tests.conftest import REPO_ROOT


@pytest.mark.parametrize(
    ("settings", "port", "bench_url", "guide_url"),
    [
        ([], 8002, "http://localhost:8002/v1", "http://localhost:8002"),
        (["INFEROPS_PORT=8012"], 8012, "http://localhost:8012/v1", "http://localhost:8012"),
        (
            ["BASE_URL=https://example.org/api/v1", "TARGET=https://example.org"],
            8002,
            "https://example.org/api/v1",
            "https://example.org",
        ),
    ],
)
def test_forward_and_clients(
    tmp_path: Path, settings: list[str], port: int, bench_url: str, guide_url: str
) -> None:
    for command in ("kubectl", "inferops", "guidellm"):
        stub = tmp_path / command
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
        stub.chmod(0o755)
    env = os.environ.copy()
    for option in ("INFEROPS_PORT", "TARGET", "BASE_URL", "MAKEFLAGS", "NAMESPACE", "SCENARIO"):
        env.pop(option, None)
    env.update(PATH=f"{tmp_path}:{env['PATH']}", OUT_DIR=str(tmp_path / "results"))
    run = subprocess.run(
        ["make", "-s", "port-forward", "bench", "guidellm", *settings],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    arguments = run.stdout.splitlines()
    assert arguments[arguments.index("service/vllm") + 1] == f"{port}:8000"
    assert arguments[arguments.index("--base-url") + 1] == bench_url
    assert arguments[arguments.index("--backend") + 1] == f"kind=openai_http,target={guide_url}"
    if not settings:
        scenario = load_scenario(REPO_ROOT / "bench/scenarios/baseline.yaml")
        assert scenario.target.base_url == bench_url
