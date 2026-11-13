# Roadmap

InferOps grows in four milestones. Each one ends with something that can be shown on its own:
a curve, a comparison, or a postmortem. Order: M1 → M2 → M4 → M3.

## M1: inference baseline (in progress)

Serve one model on one GPU and measure it honestly.

- [x] Kubernetes manifests for vLLM on a one-GPU k3s node
- [x] `inferops bench`: closed-loop concurrency sweep with TTFT, TPOT, latency and throughput
- [x] Mock server, so tests and CI cover the whole benchmark path without a GPU
- [ ] First baseline on the RTX 5090, with the curves in the README
- [ ] Cross-check against GuideLLM on the same server
- [ ] One-variable experiments: batch limit (`--max-num-seqs`), quantization, prefix cache
      on and off, context length
- [ ] Same-conditions comparison: vLLM and SGLang
- [ ] Plot generation from `summary.csv`

## M2: gateway and smart routing

Put several replicas behind an inference-aware gateway and compare routing policies.

- [ ] Two or three small-model replicas on one GPU, plus simulated replicas
- [ ] Gateway API Inference Extension in front of them
- [ ] Round robin, least queue and prefix-cache-aware routing compared on TTFT and cache hits
- [ ] Token-based rate limits and priority for interactive traffic
- [ ] SwarmOps agent traffic as a workload with heavy prefix sharing
- [ ] A scoring plugin for the endpoint picker, written in Go

## M4: observability and failure drills

Make the stack observable, then break it on purpose.

- [ ] Prometheus and Grafana with vLLM and GPU (DCGM) metrics, SLO dashboards and alerts
- [ ] Fault injection: replica kills, network delay and loss, slow disk, GPU memory pressure,
      a power-capped GPU
- [ ] Self-healing: health checks, outlier ejection, retries, autoscaling on queue length
- [ ] One postmortem per drill, with detection and recovery times judged against the SLO

## M3: scheduling and GPU sharing

Share one real GPU, and test scheduling policies on a simulated cluster.

- [ ] Time-slicing, MPS and HAMi compared on one GPU
- [ ] Inference and fine-tuning on the same GPU, with preemption and checkpoint resume
- [ ] kind + KWOK cluster with hundreds of simulated GPU nodes and a public trace replayed
- [ ] A fragmentation-aware scheduler plugin compared with the default policy

## Out of scope

Prefill/decode disaggregation, multi-node training, RDMA networking and kernel or compiler
optimization need more than one GPU or a different kind of project.
