# ADR 0002: vLLM as the first engine, with a pinned image

- Status: accepted
- Date: 2026-11-13

## Context

M1 needs an engine that serves the OpenAI-compatible API, batches concurrent requests, exports
Prometheus metrics, and is widely used, so that what is learned here transfers. The model has
to fit a 32 GB GPU with room left for the KV cache.

## Decision

Serve `Qwen/Qwen3-8B` (Apache-2.0, about 16 GB of weights in BF16) with the official
`vllm/vllm-openai` image, pinned to `v0.33.0`. The tag lives in one place,
`deploy/vllm/deployment.yaml`.

## Alternatives considered

| Option | Good at | Why not first |
| --- | --- | --- |
| SGLang | A comparable high-throughput engine | Kept as the comparison group for M1, under the same conditions. |
| llama.cpp | GGUF models on modest hardware | A different sweet spot from high-concurrency serving on a large GPU. |
| TensorRT-LLM | NVIDIA-specific compiled engines | A build step per model and GPU before the first number; revisit once the baseline exists. |

## Consequences

- An upgrade is deliberate: change the tag, rerun the baseline, compare the curves. vLLM
  releases often, and an unpinned image would make results impossible to reproduce.
- `tests/test_manifests.py` fails if the tag is not a release tag or if the baseline scenario
  and the deployment disagree about the model name.
- Qwen3 models can reason before they answer. The benchmark fixes the output length with
  `ignore_eos`, so thinking does not change the token count, and TTFT is timed from the first
  streamed token of either kind (`docs/metrics.md`).
