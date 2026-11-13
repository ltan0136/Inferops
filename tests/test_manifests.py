"""Guard the deployment decisions that are easy to undo by accident."""

from __future__ import annotations

import re
from typing import Any

import yaml

from inferops.bench.scenario import load_scenario
from tests.conftest import REPO_ROOT

VLLM_DIR = REPO_ROOT / "deploy" / "vllm"


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((VLLM_DIR / name).read_text(encoding="utf-8"))


def _container() -> dict[str, Any]:
    return _load("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]


def test_kustomization_lists_files_that_exist() -> None:
    for resource in _load("kustomization.yaml")["resources"]:
        assert (VLLM_DIR / resource).is_file(), resource


def test_image_tag_is_pinned_to_a_release() -> None:
    assert re.fullmatch(r"vllm/vllm-openai:v\d+\.\d+\.\d+", _container()["image"])


def test_one_gpu_is_requested_and_the_rollout_releases_it_first() -> None:
    deployment = _load("deployment.yaml")
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    resources = _container()["resources"]
    assert resources["requests"]["nvidia.com/gpu"] == "1"
    assert resources["limits"]["nvidia.com/gpu"] == "1"
    assert deployment["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"


def test_all_probes_use_the_health_endpoint() -> None:
    container = _container()
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"]["path"] == "/health", probe


def test_service_selects_the_deployment_pods() -> None:
    service = _load("service.yaml")
    labels = _load("deployment.yaml")["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"].items() <= labels.items()


def test_baseline_scenario_targets_the_model_the_deployment_serves() -> None:
    served_model = _container()["args"][0]
    scenario = load_scenario(REPO_ROOT / "bench" / "scenarios" / "baseline.yaml")
    assert scenario.target.model == served_model
