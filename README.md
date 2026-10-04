# Autellix

对论文 [Autellix: An Efficient Serving Engine for LLM Agents as General Programs（arXiv:2502.13965v1）](https://arxiv.org/abs/2502.13965v1) 的**非官方、独立复现尝试**。

本项目实现程序感知调度，并接入真实 vLLM 和 SGLang 推理后端。可将项目描述为：**独立实现了 Autellix 的核心方法，并完成小模型功能验证。** 这不代表作者原始代码、完全相同的 CUDA 实现或论文性能结果。SGLang 是在论文 vLLM 方案之外增加的适配。

## 阅读导航

首次使用：[实现范围](#实现范围) → [安装](#安装) → [启动服务](#启动真实推理服务) → [客户端](#客户端) → [验证](#验证)。

应用接入：[直接 Python 接入](#直接-python-接入)、[HTTP 请求约定](#http-请求约定)、[工作负载与指标](#工作负载数据格式与指标)。

继续开发：[环境检查](#环境检查与开发安装)、[架构与维护入口](#架构与维护入口)、[运维与排错](#运维与排错)、[后续开发边界](#后续开发边界)。

本指南以当前代码为准。后端内部实现和约束分别保留在两个后端 README 中；不要求通过阅读历次对话才能使用或维护项目。

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

## 环境检查与开发安装

下面的命令在 Ubuntu/WSL 中执行。服务端需要 GPU，CPU 模拟器和大部分逻辑测试不需要 GPU。

```bash
nvidia-smi
python --version
command -v uv
```

后端安装完成并激活环境后，确认当前解释器确实来自该环境：

```bash
which python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -c 'from importlib.metadata import version; print(version("vllm"))'
```

SGLang 环境将最后一行的 `vllm` 改为 `sglang`。不要仅根据终端提示符判断环境。锁文件是固定版本依赖清单，不包含模型权重、操作系统、驱动、下载源或包文件哈希，因此不能单独保证所有机器上完全一致的安装结果。

只开发模拟器、策略或 HTTP 协议时，可以创建独立 CPU 环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server]' httpx matplotlib
python -B -m unittest discover -s tests -v
```

Windows PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`。没有 Torch 时，依赖 CPU tensor 的测试会跳过；没有显式 GPU 环境变量时，真实推理测试会跳过。检查输出中的跳过原因，不能把 `OK (skipped=...)` 当成所有路径均已验证。

可编辑安装使源码修改立即生效，但运行中的服务和工作进程仍需重启。修改 `pyproject.toml`、入口点或依赖后需重新安装；升级后端版本不能只修改锁文件中的版本号。

## 直接 Python 接入

下面是一个完整的真实推理脚本骨架。保存为 Python 脚本运行，不要在交互式解释器中直接创建多进程引擎。

```python
from autellix.runtime import InferenceEngine, PolicyConfig, ReplicaConfig


def main():
    replica = ReplicaConfig(
        backend="vllm",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        device="0",
        engine_args={
            "max_model_len": 512,
            "max_num_seqs": 2,
            "gpu_memory_utilization": 0.35,
            "swap_space": 0.25,
        },
    )
    policy = PolicyConfig(policy="atlas", schedule_interval=8, overprovision=1)
    with InferenceEngine([replica], policy=policy) as engine:
        pid = engine.start_session()
        future = None
        try:
            future = engine.submit(
                pid,
                prompt="The capital of France is",
                call_id="first",
                thread_id="main",
                sampling={"temperature": 0, "max_tokens": 16},
            )
            result = future.result(timeout=180)
            print(result["text"])
            print(result["engine_id"], result["metrics"])
        finally:
            if future is not None and not future.done():
                future.cancel()
            engine.end_session(pid)
            engine.wait_idle(timeout=30)


if __name__ == "__main__":
    main()
```

### 对象与生命周期

| 接口 | 行为与约束 |
| --- | --- |
| `ReplicaConfig(backend, model, device, engine_args)` | 描述一个副本；同一协调器内各副本须使用同一模型标识 |
| `InferenceEngine(replicas, policy=..., state_dir=..., locality_threshold=2048)` | 启动独立工作进程、IPC 队列和结果监听线程 |
| `wait_ready(timeout=600)` | 等待所有副本完成启动；引擎上下文进入时自动调用 |
| `start_session(program_id=None)` | 创建程序，未指定 ID 时自动生成；未结束的 ID 不能重复创建 |
| `submit(pid, ...)` | `prompt`、`messages`、`input_ids` 三选一；返回 Future |
| `tokenize(prompt=... / messages=...)` | 使用真实后端 tokenizer；两种输入只能选一种 |
| `end_session(pid)` | 禁止继续向该程序提交；不会主动取消已接纳调用 |
| `wait_idle(timeout=600)` | 等待整个引擎的活动调用及取消确认完成，不只等待某个程序 |
| `shutdown()` | 停止工作进程、结束会话并关闭 IPC 和数据库；上下文退出时自动调用 |

`submit()` 对模型执行是异步的，但字符串/消息输入需要先完成分词，提交本身不保证立即返回。`input_ids` 可省去这次分词，调用者须确保 token 与所加载模型匹配。

`call_id` 在一个程序会话内必须唯一，包括已完成的调用；省略时自动生成。`thread_id` 省略时由当前线程标识生成。`metadata` 用于随请求记录可 JSON 序列化的信息，不参与优先级公式。

一个会话代表一个逻辑程序，不是全体用户的公共桶。不同程序应使用不同会话，同一程序的连续和并行调用应复用会话。服务历史只在调用完成后更新；并行提交的调用可能继承相同历史，这是在线算法的行为。

### 返回值、超时和取消

真实结果主要包括 `text`、`token_ids`、`prompt_tokens`、`finish_reason`、`metrics`、`program_id`、`call_id` 和 `engine_id`。SGLang 还可能提供 `completion_tokens`，其 `token_ids` 可能为空；直接接口中的 `finish_reason` 也可能是后端结构，不能假定所有后端返回同一种字符串。HTTP 层将结束原因统一为 `stop` 或 `length`。

`future.result(timeout=...)` 超时仅停止调用者等待，不会自动取消 GPU 请求。需要主动调用 `future.cancel()`。取消请求发出后，必须等待工作进程确认才能认定负载和程序状态已回收；可用 `wait_idle()` 验证。

进程失败会使对应未完成调用以异常结束。协调器不会自动重启失败副本，也不会把失败的在途调用重新提交到其他引擎。是否重试、怎样处理业务重复执行，需要由上层应用明确决定。

## HTTP 请求约定

这是项目实现的聊天接口子集，并非完整 OpenAI API。未实现的工具调用、结构化输出、多模态及其他扩展字段，不应作为可用功能依赖。

| 请求字段 | 用途 |
| --- | --- |
| `messages` | 必需，非空消息列表；格式还必须适合模型 chat template |
| `model` | 可省略；提供时须与服务配置的模型标识一致 |
| `session_id` | 已创建的程序会话 ID；省略则自动创建一次性会话 |
| `request_id` | 可选的 HTTP 请求标识；活动请求间不能重复，可用于取消 |
| `call_id`、`thread_id` | 程序内调用标识、线程标识，非空字符串 |
| `metadata` | 可选 JSON 对象 |
| `temperature`、`top_p`、`top_k`、`max_tokens`、`stop`、`n` | 当前转发给后端的采样字段；支持范围由后端约束，`n` 只能为 1 |
| `stream` | true 时返回 SSE；Python 客户端返回可迭代的数据块 |

HTTP `request_id` 与协调器内部请求 ID 是两个标识，日志关联时还应保留 `call_id`。HTTP 取消入口使用 HTTP 请求 ID，不使用内部 ID。未列出的采样字段不会自动转发，不能假定请求中的字段都已生效。

完整的最小 HTTP 会话流程：

```bash
curl -sS http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' -d '{"program_id":"demo"}'

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","request_id":"demo-request-1","call_id":"first","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

curl -sS -X DELETE http://127.0.0.1:8000/sessions/demo
```

会话 ID 在 URL 路径中必须正确编码。流式请求可给 curl 加 `-N` 并在 JSON 中设置 `"stream":true`；流以 `data: [DONE]` 结束，中途错误可能作为 `error` 事件返回。

常见状态码：400 表示输入或会话参数问题，404 表示模型不匹配或取消目标不存在，409 表示活动请求 ID 冲突，499 表示请求取消，500 表示执行失败；没有健康副本时健康接口返回 503。部分后端参数错误发生在工作进程内，会表现为执行失败，不能仅凭状态码判断能否重试。

## 工作负载数据格式与指标

真实 benchmark 的 JSON 顶层可以是程序列表，或 `{"programs": [...]}`；JSONL 每行一个程序。程序字段如下：

| 层级 | 字段 | 说明 |
| --- | --- | --- |
| 程序 | `program_id`、`calls` | 必需，程序 ID 唯一，调用列表非空 |
| 程序 | `arrival_time` | 相对测试起点的到达秒数，默认 0 |
| 程序 | `dataset` | 可选数据集标签，用于混合采样 |
| 调用 | `call_id`、`prompt` | 必需，程序内 ID 唯一，prompt 必须是文本 |
| 调用 | `parents` | 同程序内父调用 ID 列表，默认空；不能有环 |
| 调用 | `think_time` | 父调用均完成或根调用到达后的外部等待秒数，默认 0 |
| 调用 | `max_tokens` | 默认 32 |
| 调用 | `sampling` | 覆盖默认采样参数的对象，使用后端支持字段 |
| 调用 | `append_parent_outputs` | 默认 true，将父调用文本附到提示末尾 |

DAG 在 benchmark 驱动侧控制何时提交调用；真实在线 ATLAS 调度器并不读取这些父节点计算精确 DAG 优先级。区分这一点有助于避免把驱动侧的依赖执行与调度器本身混为一谈。

| 指标 | 当前含义 |
| --- | --- |
| `program_latency_seconds` | 从程序计划到达到所有调用完成的响应时间分布 |
| `output_tokens_per_second` | 全部输出 token 数除以工作负载总耗时，包括到达间隔与外部等待 |
| `program_critical_path_seconds` | 沿依赖路径累计实测调用延迟与 `think_time` 的最大值 |
| `program_token_latencies` | 程序关键路径时间除以该程序所有线程总输出 token 数 |
| 调用 `latency_seconds` | 协调器创建 Future 到收到最终结果的耗时，不含此前同步分词 |
| 调用 `ttft_seconds` | 创建 Future 到收到第一段非空文本的耗时；不是逐 token GPU 内核时间 |
| `metrics.executed` | 已记录的模型执行服务时间；批次耗时分别计入参与调用，不按批大小均分 |
| `metrics.inherited` | 该调用接纳时继承的程序服务历史 |

benchmark 另行从 `submit()` 前开始计时，所以其关键路径调用延迟包括提交过程中的分词。后端、模型、提示模板、采样、窗口和缓存状态都会影响结果；比较性能时应同时记录这些配置。

## 架构与维护入口

```text
InferenceClient / HTTP 请求 / 直接 Python 调用
                     |
             InferenceEngine 协调器
        分词、会话、路由、Future、取消与 IPC
              /                    \
      vLLM 工作进程             SGLang 工作进程
      原生执行器 + 调度钩子     原生执行器 + 调度钩子
              \                    /
             SQLite 程序表与各副本 trace
```

| 文件 | 维护职责 |
| --- | --- |
| `autellix/runtime/policy.py` | 配置校验、跨进程程序表、服务历史、时间片、防饥饿和 FIFO |
| `autellix/runtime/window.py` | N 步窗口、活动/预备请求集合、补位和容量截断 |
| `autellix/runtime/engine.py` | 副本进程、路由、Future、取消、错误和会话回收 |
| `autellix/runtime/client.py` | 会话注解、HTTP 请求、SSE 解析和客户端生命周期 |
| `autellix/runtime/server.py` | HTTP 路由、请求 ID 预留、流式输出和断连处理 |
| `autellix/runtime/swap.py` | vLLM KV 打包、锁页内存传输与映射校验 |
| `integrations/vllm/paper.py` | vLLM 默认模式接纳、显存预留及窗口继续执行 |
| `integrations/vllm/backend.py`、`executor.py` | 引擎初始化、计时、请求转换及 TP worker 钩子 |
| `integrations/sglang/scheduler.py` | SGLang 队列接纳、重排、补位和生命周期钩子 |
| `integrations/sglang/swap.py`、`reserve.py`、`pressure.py` | 主机备份、radix 锁、预备 KV 和显存压力处理 |
| `autellix/runtime/benchmark.py` | 真实文本工作负载、依赖驱动与实测指标 |

### 开发时必须保持的约束

1. **服务统计与优先级窗口分开**：防饥饿可以重置局部等待/运行窗口，不能清除调用整个生命周期的实际执行统计。
2. **按当前队列顺序移动请求**：不能用最初到达顺序替代降级、提升后的 FIFO 顺序。
3. **窗口内不随意重新选批次**：新调用等待边界，提前完成由窗口已有预备请求补位；容量不足按有定义的顺序释放资源。
4. **先保全 KV，再释放所有权**：CPU 备份失败不能丢失原有 GPU KV；恢复数据在正式接纳前不能被其他恢复操作逐出。
5. **传输屏障保持顺序**：不要让新换入覆盖尚未完成换出的 GPU block，也不要重复重放原生多步的换入。
6. **取消以确认完成为准**：Future 被取消不代表 GPU 或数据库已回收，错误和取消路径都必须完成负载及会话清理。
7. **路由不能使用未知输出长度**：仅按实际输入 token 数判断 2048 阈值。
8. **保留接口分层**：模拟器数据、估算耗时和 fake worker 不能替代真实后端验证。

### 修改与升级流程

先阅读受影响模块及对应测试，在现有目录内修改；通常不需要增加新的说明文件。修改策略应同时检查真实 runtime 与模拟器，若两者有刻意差异，在现有 README 中解释。

升级 vLLM/SGLang 时，在独立环境验证原生 scheduler、block/token pool、radix lock、输入输出结构、worker 启动方式和 CUDA 计时接口。更新版本检查、锁文件及适配代码后，再执行下列验证，不能仅放宽版本检查。

| 修改范围 | 至少需要的验证 |
| --- | --- |
| 文档、示例命令 | 路径/链接检查、对应 `--help`、Python 示例语法检查 |
| PLAS/ATLAS、窗口、路由 | `test_runtime.py`、`test_runtime_regressions.py`、`test_paper_plan.py`、`test_simulator_paper.py` |
| 客户端/HTTP | `test_client_lifecycle.py`、`test_client_stream.py`、`test_http.py`、`scripts/http_smoke.py` |
| KV 或 SGLang 锁/抢占 | host swap、reserve、decode pressure、SGLang scheduler 测试，加真实 GPU 恢复一致性测试 |
| IPC、取消、副本协调 | `test_transport.py`，加 `AUTELLIX_TEST_REPLICAS=1` 真实双进程测试 |
| TP 适配 | 两张及以上 GPU 的 `AUTELLIX_TEST_TP=1`，没有设备时明确保留未验证状态 |

定向运行示例：

```bash
python -B -m unittest discover -s tests -p test_runtime.py -v
python -B -m unittest discover -s tests -p test_http.py -v
python -B scripts/http_smoke.py --help
```

增加新策略需更新 `PolicyConfig` 的合法值、继承和完成统计、队列行为、服务/benchmark CLI 的策略选项及测试；若需要模拟支持，再修改 `autellix/core/schedulers.py`。新增后端还需实现工作进程使用的初始化、分词、接纳、step、cancel、has_work 和 close 接口，并接入副本配置及 CLI 校验。当前没有自动注册后端的插件接口。

## 运维与排错

服务默认监听 `127.0.0.1:8000`。当前没有应用层鉴权、租户隔离、请求配额或持久化任务恢复；修改为公网监听不等于具备生产部署能力。需要多人或外网使用时，应先完成相应接入控制、容量限制和故障处理设计。

| 现象 | 检查顺序 |
| --- | --- |
| 找不到 `autellix-serve` 或包 | 检查当前 Python 和激活环境，重新执行该环境的可编辑安装 |
| 后端版本拒绝启动 | 确认使用对应锁文件和独立环境，不要删除版本校验规避问题 |
| CUDA 不可用 | 检查 `nvidia-smi`、Torch CUDA 状态及进程设备配置；后端不提供假推理回退 |
| 初始化显存不足 | 检查模型大小、其他 GPU 进程、上下文和内存比例；共享 GPU 的副本预算需要合计考虑 |
| 整段 prefill 超过 token 预算 | vLLM paper 模式增大 `max_num_batched_tokens` 或缩短输入，同时确认显存足够 |
| 窗口 KV 预留不足 | 降低批次、K 或 N，或增加可用 KV 显存；记录调整后的配置 |
| SGLang 主机 KV 预算耗尽 | 调整 `autellix_swap_space`，确认系统 RAM 足够，不要靠丢弃活动 KV 恢复运行 |
| 聊天模板错误 | 使用提供 chat template 的模型；直接 Python 的 `prompt`/`input_ids` 路径可用于原始文本 |
| 后续调用的继承历史为 0 | 检查是否复用了 session，前一调用是否完成，以及是否选择了 `plas`/`atlas` |
| 流式内容不完整 | 检查服务器错误、连接关闭和 `[DONE]`；不要把部分文本当成完整成功结果 |
| 状态表残留或 SQLite 错误 | 确认无活动服务使用该目录，检查进程退出情况及文件系统位置，换用新的状态目录复测 |

### 状态与日志

指定 `state_dir` 后，目录通常含 `programs.sqlite`、SQLite 的 WAL/SHM 文件，以及 `replica-*.jsonl`。未指定时会创建临时状态目录。目录用于进程协调与诊断，不是崩溃后可恢复生成任务的检查点；服务重启建议使用新目录，旧目录在确认无进程使用后再清理。

通过 `GET /sessions` 查看程序服务历史、活动调用、引擎和时间戳。通过 trace 的 `admit`、`demote`、`promote`、`schedule_window`、`execute`、`swap_out`、`resume`/`swap_in`、`reserve`、`refill` 等事件检查实际路径；事件是否出现取决于后端、配置和负载。

记录 bug 时保留：Git 提交、后端及 Torch 版本、模型标识/版本、启动参数、最小输入、相关异常和 trace。日志可能包含业务标识或请求 metadata，分享前应检查内容。不要上传模型权重、整个虚拟环境或无关数据。

## 后续开发边界

现有测试和文档提供继续开发的起点，不能保证未来依赖、任意模型和硬件升级后自动兼容。继续维护时，应优先保留固定环境中的可运行基线，再逐项扩展支持范围。

当前需要额外开发或验证的方向包括：多 GPU TP 的硬件覆盖、更广模型/KV 布局支持、后端版本迁移，以及鉴权、限流、失败副本重启等服务能力。论文大规模性能实验是独立验证目标，不是现有功能测试已经完成的结论。对外说明应持续区分“已经实现”“已经验证”和“尚未支持”。
