# InferOps

**一张显卡上的迷你推理云。** 用 vLLM 在 Kubernetes 上把大模型服务跑起来，再去测量它、调度它、让它保持可用，每个数字都能复现。

[English](README.md)

InferOps 是 [SwarmOps](https://github.com/ltan0136/swarmops) 的底座：SwarmOps 产生智能体流量，InferOps 负责后面的模型服务。

**当前状态：v0.1，里程碑 M1 的第一部分。** 现在能用的有三块：

- **部署：** 单显卡 k3s 节点上的 vLLM 清单。镜像版本锁定；更新时先释放显卡再起新副本；启动探针给首次下载模型留足了时间。
- **压测：** `inferops bench`，一个闭环压测器，逐档提高并发，输出 TTFT、TPOT、延迟分位数和吞吐。
- **没有显卡也能跑：** 自带一个兼容 OpenAI 接口的模拟服务，测试和 CI 用它走完整条压测链路。

RTX 5090 上的实测结果还没有发布，会随 M1 的第一份报告一起放出来。

架构图见[英文 README](README.md#architecture)。

## 快速开始（无需显卡）

```bash
python -m venv .venv && source .venv/bin/activate
make install
make bench-smoke
```

`make bench-smoke` 会在进程内启动模拟服务，把并发从 1 扫到 16，结果写到 `results/`。模拟服务同时只处理 8 个请求，所以表里能看到引擎饱和时的典型形状：吞吐不再增长，首 token 延迟（TTFT）陡增。

这些数字描述的是模拟服务，不是真实模型。

## 在显卡上运行

1. 准备节点：按 [deploy/README.md](deploy/README.md) 安装 NVIDIA Container Toolkit、k3s 和 device plugin。
2. 部署 vLLM 并等待模型加载完成：

   ```bash
   make deploy-vllm
   ```

3. 压测：

   ```bash
   make port-forward     # 终端 1
   make bench            # 终端 2
   ```

每次运行会在 `results/<场景名>-<UTC 时间>/` 下写出 `summary.md`、`summary.csv`、`summary.json` 和原始的 `requests.jsonl`。`summary.json` 还会记下 `inferops` 版本、开始时间和服务端报告的 vLLM 版本。某一档失败或者按了 Ctrl-C，已经跑完的几档照样写出来，并标为未完成。压测场景是一个 YAML 文件：[bench/scenarios/baseline.yaml](bench/scenarios/baseline.yaml)。

部署步骤是照上游文档写的，还没有在 5090 机器上实际跑过，首次实跑后会回来修正。

## 压测量的是什么

| 指标 | 定义 |
| --- | --- |
| TTFT（首 token 延迟） | 发出请求 → 收到第一个流式 token |
| TPOT（每 token 耗时） | （流结束 − 第一个 token）/（输出 token 数 − 1） |
| 延迟 | 发出请求 → 流结束 |
| 吞吐 | 测量窗口内发出且成功的请求数，以及它们的输出 token 数，除以窗口时长 |

精确定义、对应的 vLLM 服务端指标、闭环压测的已知局限，见 [docs/metrics.md](docs/metrics.md)。

为了让数字可信，做了四个选择：

- 每个 prompt 开头带一个唯一标记，前缀缓存没法让服务看起来比实际更快。
- token 数取自服务端返回的 usage。服务端不返回时，报告会注明这是估算值。
- 失败的请求记为失败，不进入延迟和吞吐统计。某一档并发没有任何成功请求时，整次运行直接终止。
- 吞吐除以测量窗口的时长，而不是一直算到最后一个请求结束，这样每档收尾时的排空阶段不会把饱和点之后的曲线压低。

## 设计决策

- [ADR 0001](docs/adr/0001-single-node-k3s-with-the-nvidia-device-plugin.md)：单节点 k3s 加 NVIDIA device plugin
- [ADR 0002](docs/adr/0002-vllm-as-the-first-engine.md)：第一个推理引擎选 vLLM，镜像版本锁定
- [ADR 0003](docs/adr/0003-own-load-generator-cross-checked-with-guidellm.md)：仓库内自带压测器，用 GuideLLM 交叉校验

每份决策记录都列出了备选方案和取舍。

## 路线图

| 里程碑 | 目标 | 状态 |
| --- | --- | --- |
| M1 | 推理基线：一个模型、一张显卡、可信的曲线 | 进行中 |
| M2 | 网关与多副本智能路由 | 计划中 |
| M4 | 可观测性与故障演练 | 计划中 |
| M3 | 调度与 GPU 共享，含模拟集群 | 计划中 |

细节见 [docs/roadmap.md](docs/roadmap.md)。

## 开发

```bash
make test      # 不需要显卡，不访问网络
make lint
make help      # 列出所有命令
```

## 许可证

MIT
