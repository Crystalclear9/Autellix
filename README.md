# Autellix

对论文 [Autellix: An Efficient Serving Engine for LLM Agents as General Programs（arXiv:2502.13965v1）](https://arxiv.org/abs/2502.13965v1) 的**非官方、独立复现尝试**。

本项目实现程序感知调度，并接入真实 vLLM 和 SGLang 推理后端。可将项目描述为：**独立实现了 Autellix 的核心方法，并完成小模型功能验证。** 这不代表作者原始代码、完全相同的 CUDA 实现或论文性能结果。SGLang 是在论文 vLLM 方案之外增加的适配。

## 实现范围

- **PLAS / ATLAS**：按已完成调用累计服务时间或最长已观察服务路径，为新调用确定初始优先级。在线 ATLAS 对应算法 1，不要求用户提供调用 DAG。
- **程序状态表**：跨副本共享服务时间、等待时间、活动调用、线程信息及到达和完成时间。
- **分级 FIFO 队列**：时间片耗尽后降级，逐调用结合程序历史判断防饥饿；同时移动多个调用时保留当前队列顺序。
- **抢占与恢复**：保存生成进度，通过 GPU/CPU KV 传输恢复执行；批量打包采用 PyTorch CUDA 操作和锁页内存。
- **多步调度与预备请求**：每个窗口冻结调度顺序，运行 N 个解码步；活动请求提前结束时，由已准备的 GPU 请求补位。
- **程序感知路由**：输入不超过 2048 token 时分配给负载最小副本，更长输入保持程序的引擎亲和性。
- **有状态客户端**：自动复用程序会话，标注调用和线程 ID，支持普通响应与流式响应。

已在小模型上验证真实生成、抢占恢复结果一致、预备请求补位、HTTP、路由、取消和会话清理。双副本测试是在同一张 GPU 上运行两个独立进程；多 GPU 张量并行尚未完成硬件验证。未复现论文的大规模实验和性能提升倍数。

## 安装

真实后端需要 Linux 或 Ubuntu WSL、可用的 NVIDIA GPU，以及 Python 3.10。两个后端使用不同依赖环境，不要安装到同一个环境。

| 后端 | 固定版本 | 说明 |
| --- | --- | --- |
| vLLM | 0.6.1 | [支持范围与实现](integrations/vllm/README.md) |
| SGLang | 0.4.9.post6 | [支持范围与实现](integrations/sglang/README.md) |

在仓库根目录执行，需预先安装 `uv`：

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
bash scripts/setup_backend.sh vllm
source ~/autellix-envs/vllm/bin/activate
```

安装 SGLang 时将 `vllm` 换成 `sglang`。脚本使用 `requirements/` 中的完整依赖锁文件，并以可编辑方式安装本项目。`AUTELLIX_ENV_DIR` 可指定环境路径；环境内的安装元数据应保留。

## 启动真实推理服务

以下配置用于小模型功能验证。`--model` 可以是模型名称或本地模型目录，显存和上下文参数需按模型调整。聊天请求要求模型 tokenizer 提供 chat template。

### vLLM

```bash
autellix-serve --backend vllm --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 --state-dir /tmp/autellix-server \
  --engine-args '{"max_model_len":512,"max_num_seqs":4,"gpu_memory_utilization":0.35,"swap_space":0.25}'
```

### SGLang

在 SGLang 环境执行：

```bash
autellix-serve --backend sglang --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 \
  --engine-args '{"context_length":512,"max_running_requests":4,"mem_fraction_static":0.35}'
```

`--devices 0,1` 表示两个独立单 GPU 副本。vLLM 张量并行的配置和限制见其后端说明。共享 SQLite 状态应放在 Linux 本地文件系统，不要放在网络盘或 WSL 的 Windows 挂载路径。

## 客户端

真实推理使用 `autellix.runtime`。同一客户端的自动会话可供多个线程共享；上下文退出时结束会话，已接纳的调用完成或取消后释放程序状态。

```python
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    answer = client.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        temperature=0, max_tokens=32,
    )
    print(answer["choices"][0]["message"]["content"])
```

流式响应返回迭代器。提前停止时应关闭迭代器，让服务端能够取消未完成生成：

```python
from contextlib import closing
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    with closing(client.chat(
        [{"role": "user", "content": "Explain program-aware scheduling."}],
        stream=True, max_tokens=64,
    )) as chunks:
        for chunk in chunks:
            print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

`client.session()` 可显式创建独立会话作用域；自定义程序 ID 支持中文和 `/`。fork 后需创建新客户端，已关闭的客户端不能继续发请求。流式错误或缺少 `[DONE]` 会抛出异常；清理失败不会掩盖已有的应用异常。解释器退出清理是尽力执行，建议使用上下文管理器。

绕过 HTTP 的真实 Python 示例见 [examples/real_inference.py](examples/real_inference.py)。`InferenceEngine.submit()` 返回由工作进程结果完成的 Future；创建引擎的代码需放在 `if __name__ == "__main__":` 中。

| HTTP 接口 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | 普通或 SSE 流式聊天 |
| `GET /v1/models` | 已加载模型 |
| `POST /sessions`、`GET /sessions` | 创建会话、查看程序状态 |
| `DELETE /sessions/{id}` | 结束会话 |
| `DELETE /requests/{id}` | 取消活动请求 |
| `GET /health` | 副本健康状态和负载 |

原始 HTTP 请求若不携带 `session_id`，每个请求会成为独立程序，无法继承此前调用的服务历史。

## 调度参数

| 参数 | 真实运行默认值 |
| --- | --- |
| 策略 | `atlas`；另有 `plas`、`mlfq`、`fcfs` |
| 实现模式 | `paper` |
| 解码窗口 N | 8 |
| 预备请求数 K | 1 |
| 优先级区间边界（秒） | `0,.02,.04,.08,.16,.32,.64,inf` |
| 各队列时间片（秒） | `.01,.02,.04,.08,.16,.32,.64` |
| 防饥饿阈值 beta | 8 |

`--schedule-interval`、`--overprovision`、`--implementation` 可在服务和真实工作负载命令中设置；其余策略参数通过 `PolicyConfig` 配置。这些数值是项目选择，不是论文公布的完整参数表。

`paper` 表示按论文描述实现的调度语义，不是完全等价认证。`compat` 是可选的旧后端调度包装方式。窗口内仍需逐 token 更新执行元数据；预备请求需要真实 prefill（包括首个采样 token）或 KV 换入，并占用显存。

## 验证

不加载模型的测试：

```bash
python -m unittest discover -s tests -v
```

真实 GPU 测试，在对应后端环境执行：

```bash
AUTELLIX_GPU_BACKEND=vllm AUTELLIX_TEST_MODEL=/absolute/path/to/model \
  AUTELLIX_TEST_STEPS=8 AUTELLIX_TEST_RESERVE=1 \
  python -m unittest discover -s tests -p test_gpu_runtime.py -v
```

SGLang 将后端变量改为 `sglang`。可选项：`AUTELLIX_TEST_REPLICAS=1` 验证双进程路由和取消；`AUTELLIX_TEST_CUDA_SWAP=1` 验证实际 KV 传输；`AUTELLIX_TEST_TP=1` 启用需要两张 GPU 的 vLLM 张量并行测试；`AUTELLIX_TEST_IMPLEMENTATION=compat` 验证旧模式。

完整的小模型验证入口会依次运行测试、真实 HTTP 流程和四策略 DAG 工作负载：

```bash
bash scripts/validate_backend.sh vllm /absolute/path/to/model
```

SGLang 同样替换后端名称。可用 `AUTELLIX_PYTHON` 指定 Python 路径，`AUTELLIX_ENGINE_ARGS` 调整 HTTP 和 DAG 验证参数。生成结果写入 `outputs/validation/`，不纳入版本控制。KV 传输单独测量见 [cuda/README.md](cuda/README.md)。

## 真实工作负载

[examples/real_workload.json](examples/real_workload.json) 提供小型顺序和分支依赖示例，不是论文数据集：

```bash
python -m autellix.runtime.benchmark --backend vllm \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --workload examples/real_workload.json \
  --policies fcfs,mlfq,plas,atlas \
  --engine-args '{"max_model_len":512,"max_num_seqs":2,"gpu_memory_utilization":0.35,"swap_space":0.25}' \
  --output outputs/real/results.json
```

工作负载使用含真实文本提示的 JSON/JSONL，不能用只有 token 长度的模拟数据替代。`parents` 描述调用依赖；`arrival_time` 和 `think_time` 单位为秒；已有完整历史的提示可设 `append_parent_outputs=false`。多个 `--workload` 文件可组成混合负载。

`--arrival-rates 1,2,4 --programs 100 --seed 42` 可选地生成泊松到达流，不是使用本项目的必要步骤。结果包含响应时间、输出吞吐、TTFT、程序关键路径 token 延迟、配置、依赖版本和输入哈希。零输出程序的 token 延迟为 null。vLLM 原生对照模式见后端说明。

## 模拟器

`autellix.core`、`autellix.frontend` 和 `autellix.cli` 是独立的 CPU 模拟功能，不能用于真实模型推理。旧的 `AutellixClient`、`AsyncMultiLLMEngine` 及根包导入路径保留兼容用途。

```bash
python -m autellix.cli compare --workload figure2 --policies fcfs,mlfq,plas
python -m autellix.cli paper-preset --preset workload-analysis --dataset tests/fixtures/tiny_workload.jsonl
```

在线 `atlas` 对应算法 1，显式父节点版本 `atlas-dag` 对应公式 2。模拟窗口按模型 tick 而非真实解码步计数；执行、缓存命中率及交换开销都是成本模型。模拟器中的 `vllm` / `vllm-opt` 标签和 paper-style presets 不代表真实后端测量或论文实验复现。

## 目录

| 路径 | 内容 |
| --- | --- |
| `autellix/runtime/` | 真实推理协调、策略、客户端、HTTP 与 KV 传输 |
| `integrations/` | 固定版本的 vLLM / SGLang 后端 |
| `autellix/core/` | 调度模拟器和执行成本模型 |
| `autellix/frontend/`、`autellix/experiments/` | 模拟客户端、数据导入与实验工具 |
| `autellix/*.py` | CLI 和旧导入兼容接口 |
| `scripts/`、`requirements/` | 安装、验证脚本和依赖锁文件 |
| `examples/`、`tests/` | 真实推理示例、回归测试和小型输入 |
| `cuda/` | 真实 KV 传输微基准 |

本地模型、PDF、临时文件、缓存、安装元数据和生成输出不提交。保留仍被使用的模型与环境；测试和验证输出可以重新生成。
