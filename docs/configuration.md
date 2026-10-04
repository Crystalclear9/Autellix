# 配置参考

[文档导航](README.md) · [项目首页](../README.md)

本文档中的命令均从仓库根目录执行。

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

## 服务端命令行

入口 `autellix-serve` 对应 `python -m autellix.runtime.server`。`autellix-sglang` 是默认选择 SGLang 的便捷入口。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--backend` | `vllm` | `vllm` 或 `sglang` |
| `--model` | 必需 | 模型名称或本地路径；同一协调器的副本模型须一致 |
| `--devices` | `0` | 逗号分隔，每项启动一个独立副本 |
| `--device-groups` | 无 | 分号分隔 TP 副本设备组，提供时优先于 `--devices` |
| `--policy` | `atlas` | 当前策略之一 |
| `--implementation` | `paper` | `paper` 或 `compat` |
| `--schedule-interval` | `8` | 正整数，解码窗口长度 |
| `--overprovision` | `1` | 非负整数，预备请求数 |
| `--engine-args` | `{}` | 传给当前后端的 JSON 对象，不是通用跨后端配置 |
| `--state-dir` | 自动创建临时目录 | SQLite 和 trace 所在目录；每个服务实例使用独立目录 |
| `--host`、`--port` | `127.0.0.1`、`8000` | HTTP 监听地址和端口 |

配置区间与时间片用 Python 设置，而不是塞进 `--engine-args`：

```python
import math
from autellix.runtime import PolicyConfig

policy = PolicyConfig(
    policy="atlas",
    boundaries=(0, 0.02, 0.08, math.inf),
    quanta=(0.01, 0.04, 0.16),
    beta=8.0,
    schedule_interval=8,
    overprovision=1,
    implementation="paper",
)
# 将 policy 传给完整脚本中的 InferenceEngine(..., policy=policy)。
```

边界必须从 0 严格递增至正无穷，长度比 quanta 多 1；时间片和 beta 必须有限且大于 0。以上区间仅演示接口，不是性能最优参数。路由阈值可通过 `InferenceEngine(locality_threshold=...)` 调整，当前服务 CLI 没有暴露该选项。

## 后端参数与覆盖关系

| 概念 | vLLM | SGLang |
| --- | --- | --- |
| 上下文长度 | `max_model_len` | `context_length` |
| 活动请求容量 B | `max_num_seqs`，适配默认 8 | `max_running_requests`，适配默认 8 |
| GPU 内存比例 | `gpu_memory_utilization` | `mem_fraction_static` |
| 主机 KV 空间 | `swap_space`，适配默认 1 GiB | `autellix_swap_space`，默认 1 GiB |
| 整批 token 预算 | `max_num_batched_tokens` | 沿用固定版本原生接纳参数和检查 |

上下文长度和 token 预算不是同一个概念；GPU 内存比例也不能直接在两个后端间等价换算。B 控制活动请求，K 个预备请求另占显存；适配内部会扩展原生容量到 B+K。

vLLM 默认 paper 模式强制关闭异步输出处理，启用支持混合阶段的执行配置，内部 `num_scheduler_steps` 为 1，由 Autellix 自己维护 N 步调度窗口。因此原生启动日志显示 1 不代表 Autellix N=8 未生效。若显式传入 `num_scheduler_steps`，它必须与 `PolicyConfig.schedule_interval` 一致，随后仍按适配路径设置执行参数。不要把原生参数覆盖当成禁用 Autellix 逻辑的方法。

SGLang 适配强制 `disable_overlap_schedule=True`、`chunked_prefill_size=-1`、`page_size=1`。TP/PP/DP、LoRA、HiCache、推测解码等不受支持，具体校验见后端说明。其他未强制覆盖的参数仍需符合固定版本及当前适配的要求。

## 设备分组

```bash
# 两张 GPU，各运行一个独立副本；将模型及 engine-args 换成实际配置。
autellix-serve --backend vllm --model /absolute/path/to/model --devices 0,1

# 两张 GPU 共同执行一个 TP=2 副本；该路径仍需实际硬件验证。
autellix-serve --backend vllm --model /absolute/path/to/model \
  --device-groups '0,1' --engine-args '{"tensor_parallel_size":2}'
```

前者能把不同请求分配给不同副本，后者是单个模型副本跨 GPU。`--device-groups '0,1;2,3'` 是两个 TP=2 副本，需要四个可用设备。设备组通过工作进程的 `CUDA_VISIBLE_DEVICES` 控制可见性。

## 工具环境变量

| 变量 | 使用位置 | 含义 |
| --- | --- | --- |
| `AUTELLIX_ENV_DIR` | `scripts/setup_backend.sh` | 环境安装位置，默认 `~/autellix-envs/<backend>` |
| `AUTELLIX_PYTHON` | `scripts/validate_backend.sh` | 验证使用的解释器，默认 `python` |
| `AUTELLIX_ENGINE_ARGS` | 同上 | 覆盖 HTTP 和 DAG 验证的 JSON 参数；GPU unittest 使用各自测试配置 |
| `AUTELLIX_GPU_BACKEND`、`AUTELLIX_TEST_MODEL` | GPU unittest | 启用指定后端和模型的真实推理测试 |
| `AUTELLIX_TEST_STEPS`、`AUTELLIX_TEST_RESERVE` | GPU unittest | 特定生成测试的 N、K，测试默认 1、0，与服务默认 8、1 不同 |
| `AUTELLIX_TEST_REPLICAS`、`AUTELLIX_TEST_CUDA_SWAP`、`AUTELLIX_TEST_TP` | GPU unittest | 可选双副本、CUDA KV、TP 测试开关 |
| `AUTELLIX_TEST_IMPLEMENTATION` | GPU unittest | `paper` 或 `compat` |

可选测试开关按环境变量是否为非空判断；禁用时应取消设置，不要设成字符串 `0`。执行 CPU 测试前确认未继承上一轮 GPU 验证变量。
