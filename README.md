# Autellix

对论文 [Autellix: An Efficient Serving Engine for LLM Agents as General Programs（arXiv:2502.13965v1）](https://arxiv.org/abs/2502.13965v1) 的**复现尝试**。

本项目实现程序感知调度、PLAS/ATLAS、KV 交换、多步调度与预备请求，并接入 vLLM 和 SGLang 推理后端。源码核对、实测结果和改进方向统一记录在实验报告中。

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

2026-11-15 的 [P40 部署与验收记录](docs/p40-validation-20261115.md)记录了
SmolLM2-135M 的真实生成、抢占恢复和 HTTP 验证，以及后续修改清单。
P40 使用 PyTorch 原生算子路径，默认 FP32，也可显式选择 FP16。
论文机制、原模型/数据集差距和直接 SGLang 对照协议见
[论文核对与 SGLang 对照](docs/autellix-paper-audit.md)。对照使用 `sglang-native` 原生调度路径。
单卡的[完整对话与负载实验](docs/experiment-design-review.md)修复了 ShareGPT 分片、
重复 BOS 和合成统计问题，并完成三个到达率、策略及独立参数消融。
Llama-3.1-8B 高负载下，扩大服务历史区间后，程序响应/输出 token 下降 11.21%，
平均程序响应增加 5.37%，吞吐下降 0.52%。核心机制已接入真实 GPU。

最新的[原始工作负载复现](docs/paper-workload-experiment.md)使用论文 8B 模型系列、
完整 ShareGPT、固定版本 BFCL 工具循环和 LATS 搜索，加入 Mixed 与三卡共享模型
对照。ShareGPT 的 16 次运行（14 次单卡、2 次 PP=3）已完成：中/高负载平均程序响应下降 20.32%/9.60%，
吞吐提高 1.05%/1.21%；低负载和短程序消融仍有小幅负收益。
三卡大 KV 配对吞吐提高 8.96%、平均程序响应下降 3.02%；BFCL/LATS/Mixed 实验继续执行。
长上下文 KV 交换和接纳顺序的修复已通过 P40 验证；
论文专用 CUDA 聚合交换内核仍待实现。服务器文件统一在
`autellix/`；新实例入口为 `/root/workspace/autellix/`，连接端口为 47541，
迁移与续跑见[服务器迁移](docs/server-migration-20261117.md)。目录和复跑入口见[服务器目录](docs/server-layout.md)。

[三卡大模型实验](docs/three-gpu-experiment.md)增加了 SGLang PP=3：Qwen3-32B
的性能回放按 20/22/22 层分布在全部三张 P40 上，程序调度、GPU 计时和 KV 交换在各阶段同步。
在 8 个完整 ShareGPT 程序的配对回放中，相对原生 SGLang 吞吐提高 8.84%，
平均程序响应下降 1.54%，程序响应/输出 token 下降 18.76%，显存峰值为 68.94 GiB。
同轮 Llama-3.1-8B 三副本实验中，program 路由的两个长对话集中到同一卡，
平均程序响应增加 23.79%、吞吐下降 25.31%，归一化程序延迟下降 5.84%。
报告保留不同配置下的全部结果和逐卡运行图。常驻服务为长输入使用 20/23/21 层、
KV=8192 和较小的 attention 分块，启动入口为 `bash scripts/serve_p40_three_gpu.sh`。

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
