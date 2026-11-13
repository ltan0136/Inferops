.PHONY: help install test lint format mock bench-smoke bench guidellm deploy-vllm undeploy-vllm port-forward

SCENARIO ?= bench/scenarios/baseline.yaml
BASE_URL ?= http://localhost:8000/v1
NAMESPACE ?= inferops

help:  ## List the available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install the package with dev tools into the active virtualenv
	pip install -e ".[dev]"

test:  ## Run the test suite (mock server only: no GPU, no network)
	pytest -q -rs

lint:  ## Lint and check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-fix lint issues and format
	ruff check --fix .
	ruff format .

mock:  ## Run the mock OpenAI-compatible server on :8001
	inferops mock-server --port 8001

bench-smoke:  ## Benchmark an in-process mock server end to end (no GPU needed)
	inferops bench --scenario bench/scenarios/smoke.yaml --spawn-mock

bench:  ## Run the concurrency sweep against a real server (override SCENARIO, BASE_URL)
	inferops bench --scenario $(SCENARIO) --base-url $(BASE_URL)

guidellm:  ## Cross-check the numbers with GuideLLM (pip install guidellm first)
	./scripts/guidellm_sweep.sh

deploy-vllm:  ## Deploy vLLM to the current kube context and wait until it is ready
	kubectl apply -k deploy/vllm
	kubectl -n $(NAMESPACE) rollout status deployment/vllm --timeout=35m

undeploy-vllm:  ## Remove vLLM but keep the namespace and the model cache
	kubectl -n $(NAMESPACE) delete deployment/vllm service/vllm

port-forward:  ## Expose the vLLM service on localhost:8000
	kubectl -n $(NAMESPACE) port-forward service/vllm 8000:8000
