# Autellix

对论文 [Autellix: An Efficient Serving Engine for LLM Agents as General Programs（arXiv:2502.13965v1）](https://arxiv.org/abs/2502.13965v1) 的**非官方、独立复现尝试**。

本项目实现程序感知调度，并接入真实 vLLM 和 SGLang 推理后端。可将项目描述为：**独立实现了 Autellix 的核心方法，并完成小模型功能验证。** 这不代表作者原始代码、完全相同的 CUDA 实现或论文性能结果。SGLang 是在论文 vLLM 方案之外增加的适配。

## 文档

详细使用与开发说明从 [文档导航](docs/README.md) 开始：

| 需要做什么 | 文档 |
| --- | --- |
| 安装、启动、第一次调用 | [安装与首次使用](docs/getting-started.md) |
| 接入程序、HTTP、流式响应和取消 | [接口说明](docs/api.md) |
| 调整策略、窗口、显存和副本 | [配置参考](docs/configuration.md) |
| 理解论文机制和代码职责 | [架构与论文方法](docs/architecture.md) |
| 修改代码、升级后端、验证回归 | [开发与验证](docs/development.md) |
| 构建程序 DAG、测量响应和吞吐 | [工作负载与指标](docs/workloads.md) |
| 排查故障、管理状态和输出 | [运行维护与排错](docs/operations.md) |

后端专项说明：[vLLM](integrations/vllm/README.md)、[SGLang](integrations/sglang/README.md)；传输微基准：[KV benchmark](cuda/README.md)。

## 快速开始

真实推理需 Linux/Ubuntu WSL、NVIDIA GPU 和独立 Python 3.10 环境。先安装 `uv`，在仓库根目录执行：

```bash
bash scripts/setup_backend.sh vllm
source ~/autellix-envs/vllm/bin/activate

autellix-serve --backend vllm --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 \
  --engine-args '{"max_model_len":512,"max_num_seqs":4,"gpu_memory_utilization":0.35,"swap_space":0.25}'
```

这是小模型功能验证配置，较大模型需要调整资源。SGLang 使用另一套独立环境，不能直接复用 vLLM 依赖。安装前置条件和 SGLang 命令见[安装文档](docs/getting-started.md)。

```python
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    result = client.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        temperature=0, max_tokens=32,
    )
    print(result["choices"][0]["message"]["content"])
```

## 项目边界

真实推理使用 `autellix.runtime`；`autellix.core`、`autellix.frontend`、`autellix.cli` 和旧的 `AutellixClient` / `AsyncMultiLLMEngine` 属于 CPU 模拟工具，不能替代真实模型执行。

当前默认在线 ATLAS、8 步窗口、1 个预备请求，另有 PLAS、MLFQ、FCFS。默认数值是项目选择；`paper` 模式表示实现目标，不是完全等价认证。SGLang 是论文方案之外的扩展。

已有小模型功能验证，包括真实生成、抢占恢复、HTTP、流式输出、程序状态和同卡双副本；多 GPU TP 未完成硬件验证，未复现论文的大规模性能结果。当前没有完整生产服务所需的鉴权、配额、自动故障恢复等能力。详细支持范围见后端说明和[开发边界](docs/development.md#后续开发边界)。
