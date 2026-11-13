# Deploying vLLM on a one-GPU k3s node

> **Status:** written from the upstream k3s, NVIDIA and vLLM documentation. It has not been run
> on the RTX 5090 host yet. Whatever the first real run teaches goes back into this file.

Assumptions: a Linux host with an NVIDIA GPU and a working driver (`nvidia-smi` prints the
card). Windows with WSL2 is not covered yet.

## 1. NVIDIA Container Toolkit

Install it with NVIDIA's instructions for your distribution:
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

## 2. k3s

```bash
curl -sfL https://get.k3s.io | sh -

# Use the cluster as your own user.
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$USER" ~/.kube/config
```

k3s looks for the NVIDIA runtime when it starts. Check that it found it:

```bash
sudo grep nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

If the toolkit was installed after k3s, restart k3s (`sudo systemctl restart k3s`) and check again.

## 3. NVIDIA device plugin

The device plugin is what makes `nvidia.com/gpu` a resource pods can request.

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --create-namespace \
  --set runtimeClassName=nvidia
```

The node should now report one GPU:

```bash
kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}{"\n"}'
```

## 4. vLLM

```bash
make deploy-vllm
```

The first start downloads about 16 GB of weights into the `hf-cache` volume, so the rollout
can take a while; the startup probe allows 30 minutes. Later restarts reuse the cache.

The image tag in `deployment.yaml` follows vLLM's release names. If the pull fails, check the
tag exists: <https://hub.docker.com/r/vllm/vllm-openai/tags>

## 5. Talk to it

```bash
make port-forward            # leave running in one terminal
curl -s localhost:8000/v1/models
make bench                   # in another terminal
```

## Changing the model or an engine flag

Edit `deployment.yaml` and run `make deploy-vllm` again. If the model changes, update
`bench/scenarios/baseline.yaml` too: a test fails when the two disagree.

Change one thing per run and keep the rest fixed, so a difference in the curves has one cause.

## If something is stuck

| Symptom | Look at |
| --- | --- |
| Pod stays `Pending` | `kubectl -n inferops describe pod -l app=vllm`. "Insufficient nvidia.com/gpu" means the device plugin is not advertising the GPU (step 3), or another pod holds it. |
| Pod restarts during startup | `kubectl -n inferops logs -l app=vllm --previous`. An out-of-memory error on the GPU means the model plus KV cache does not fit: lower `--max-model-len` or pick a smaller model. |
| `make bench` fails at preflight | The port-forward is not running, or `target.model` in the scenario is not the served model name. |
