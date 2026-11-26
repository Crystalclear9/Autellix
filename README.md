# Autellix

对论文 [Autellix: An Efficient Serving Engine for LLM Agents as General Programs（arXiv:2502.13965v1）](https://arxiv.org/abs/2502.13965v1) 的**复现尝试**。

本项目实现程序感知调度、PLAS/ATLAS、KV 交换、多步调度与预备请求，并接入 vLLM 和 SGLang 推理后端。

## 文档

详细使用与开发说明从 [文档导航](docs/README.md) 开始：

| 需要做什么 | 文档 |
| --- | --- |
| 安装、启动、第一次调用 | [安装与首次使用](docs/getting-started.md) |
| 接入程序、HTTP、流式响应和取消 | [接口说明](docs/api.md) |
| 调整策略、窗口、显存和副本 | [配置参考](docs/configuration.md) |
| 理解论文机制和代码职责 | [架构与论文方法](docs/architecture.md) |
| 修改代码、升级后端、验证回归 | [开发与验证](docs/development.md) |
| 构建程序 DAG、回放论文实验 | [工作负载与指标](docs/workloads.md)、[vLLM 回放](integrations/vllm/README.md#论文实验回放) |
| 排查故障、管理状态和输出 | [运行维护与排错](docs/operations.md) |

后端专项说明：[vLLM](integrations/vllm/README.md)、[SGLang](integrations/sglang/README.md)；传输微基准：[KV benchmark](cuda/README.md)。

## 实验入口

- H20 的原生 SGLang、vLLM-opt 和 Autellix 配对，以及共享权重、原始检查点位置。 [H20 复现记录](docs/h20-reproduction.md) · [最近的接纳与尾延迟复测](docs/h20-reproduction.md#sglang-部分解码保留复测)
- 完整 ShareGPT、BFCL、LATS、Mixed 的输入制作、负载曲线和论文机制核对。 [工作负载复现](docs/paper-workload-experiment.md) · [论文核对](docs/autellix-paper-audit.md)
- P40 的功能验证、32B 三卡流水线和三副本路由。 [P40 验收](docs/p40-validation-20261115.md) · [三卡实验](docs/three-gpu-experiment.md) · [服务器目录](docs/server-layout.md)

## 快速开始

真实推理需 Linux/Ubuntu WSL、NVIDIA GPU 和独立 Python 3.10 环境。先安装 `uv`，在仓库根目录执行：

```bash
bash scripts/setup_backend.sh vllm
source ~/autellix-envs/vllm/bin/activate

autellix-serve --backend vllm --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 \
  --engine-args '{"max_model_len":512,"max_num_seqs":4,"gpu_memory_utilization":0.35,"swap_space":0.25}'
```

这是小模型功能验证配置，较大模型需要调整资源。SGLang 使用另一套独立依赖环境。安装前置条件和 SGLang 命令见[安装文档](docs/getting-started.md)。

```python
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    result = client.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        temperature=0, max_tokens=32,
    )
    print(result["choices"][0]["message"]["content"])
```

## 实现与运行

真实推理使用 `autellix.runtime`；`autellix.core`、`autellix.frontend`、`autellix.cli` 和旧的 `AutellixClient` / `AsyncMultiLLMEngine` 用于 CPU 离散时间模拟。

当前默认在线 ATLAS、8 步窗口、1 个预备请求，另有 PLAS、MLFQ、FCFS。`paper` 模式使用多步调度窗口，`compat` 保留旧后端包装路径。SGLang 是项目增加的后端适配。

GPU 验证涵盖真实生成、抢占恢复、预备请求、HTTP、流式输出和程序状态。具体硬件、模型、实测结果与后续工作见[实验报告](docs/autellix-paper-audit.md)。
