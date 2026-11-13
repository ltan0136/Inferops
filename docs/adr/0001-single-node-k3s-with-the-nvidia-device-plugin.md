# ADR 0001: Single-node k3s with the NVIDIA device plugin

- Status: accepted
- Date: 2026-11-13

## Context

The hardware is one workstation with one RTX 5090. Later milestones use standard Kubernetes
components (an inference gateway, a job queue, chaos tooling), so the stack needs a real
Kubernetes API from the start, with as little overhead as possible between the workload and
the GPU.

## Decision

Run k3s directly on the GPU host. The host owns the NVIDIA driver and the NVIDIA Container
Toolkit; k3s detects the NVIDIA runtime at startup and exposes it as the `nvidia`
RuntimeClass. The NVIDIA device plugin, installed with Helm, advertises `nvidia.com/gpu`.
GPU workloads set `runtimeClassName: nvidia` and request one GPU.

## Alternatives considered

| Option | Good at | Why not now |
| --- | --- | --- |
| kind | Disposable clusters in CI; simulated nodes with KWOK | No built-in GPU support. Planned for the simulated cluster in M3. |
| minikube | Local clusters with a GPU through its Docker driver | Adds a layer between the workload and the host for no gain on a dedicated machine. |
| Docker Compose | The shortest path to one running container | No Kubernetes API, so M2 to M4 could not use standard components. |
| NVIDIA GPU Operator from day one | Managing driver, toolkit, device plugin and DCGM exporter together | More moving parts than one node needs today. Adopt it in M4, when DCGM metrics are needed. |

## Consequences

- Driver and toolkit versions on the host are managed by hand and recorded in
  `deploy/README.md`.
- One node means no high availability, and a rollout has to release the GPU before the new pod
  starts (`strategy: Recreate`, guarded by `tests/test_manifests.py`).
- Moving to the GPU Operator later changes how the GPU is exposed, not how workloads ask for it.
