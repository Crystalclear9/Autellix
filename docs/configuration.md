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

`--schedule-interval`、`--overprovision`、`--implementation` 可在服务和真实工作负载命令中设置。
服务 CLI 另有 `--quantum-scale`、`--boundary-scale` 缩放默认时间片和区间，默认 1；
完整策略参数通过 `PolicyConfig` 配置。这些数值是项目选择，不是论文公布的完整参数表。

`paper` 使用论文描述的调度语义。`compat` 是可选的旧后端调度包装方式。窗口内仍需逐 token 更新执行元数据；预备请求需要真实 prefill（包括首个采样 token）或 KV 换入，并占用显存。

## 服务端命令行

入口 `autellix-serve` 对应 `python -m autellix.runtime.server`。`autellix-sglang` 是默认选择 SGLang 的便捷入口。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--backend` | `vllm` | `vllm` 或 `sglang` |
| `--model` | 必需 | 模型名称或本地路径；同一协调器的副本模型须一致 |
| `--devices` | `0` | 逗号分隔，每项启动一个独立副本 |
| `--device-groups` | 无 | 分号分隔 TP/PP 副本设备组，提供时优先于 `--devices` |
| `--policy` | `atlas` | 当前策略之一 |
| `--routing` | `program` | 程序亲和路由；消融可选 `least-used`、`round-robin` |
| `--implementation` | `paper` | `paper` 或 `compat` |
| `--schedule-interval` | `8` | 正整数，解码窗口长度 |
| `--overprovision` | `1` | 非负整数，预备请求数 |
| `--engine-args` | `{}` | 传给当前后端的 JSON 对象，不是通用跨后端配置 |
| `--state-dir` | 自动创建临时目录 | SQLite 和 trace 所在目录；每个服务实例使用独立目录 |
| `--host`、`--port` | `127.0.0.1`、`8000` | HTTP 监听地址和端口 |

自定义完整区间与时间片用 Python 设置：

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

vLLM 的 paper 模式关闭异步输出处理，启用混合阶段执行，内部 `num_scheduler_steps=1`；Autellix 自己维护 N 步调度窗口。显式传入的 `num_scheduler_steps` 与 `PolicyConfig.schedule_interval` 保持一致。

SGLang 适配设置 `disable_overlap_schedule=True`、`page_size=1`，`chunked_prefill_size` 默认 `-1`。支持单卡与 `pp_size>1` 的流水线并行；TP、DP attention、LoRA、HiCache 和推测解码尚未接入 Autellix 调度。

| 单卡优化 | vLLM | SGLang |
| --- | --- | --- |
| 暂停期间保留 KV | `keep_resident_kv` 在有空余 GPU blocks 时保留暂停请求 | `autellix_cache_paused_kv` 把暂停请求的已计算前缀交给可回收的 GPU radix 缓存 |
| 窗口边界发布活动视图 | `publish_activity_at_window` | `autellix_window_activity` |
| 补齐闲置槽位 | `refill_idle_slots` | `autellix_refill_idle_slots` |
| 候补从驻留 KV 选择 | `resident_reserve_only` | `autellix_resident_reserve_only` |
| 分块预填充 | `paper_chunked_prefill` 复用原生预算及块对齐 | `chunked_prefill_size` 为正时接入原生分块，进行中的块优先续跑 |
| 跨层 KV 打包 | `fused_swap` | `autellix_fused_swap` |

活动视图延后写入 SQLite，优先级和调用完成统计仍使用逐步计量的服务时间。补位先使用已有
候补请求，候补耗尽后才按已知优先级补空位。它保留已接纳次序和原窗口边界，遇到容量限制时
等下一窗口再尝试接纳。H20 回放脚本通过 `--publish-activity-at-window` 与 `--refill-idle-slots`
开启这些选项。SGLang 通过 `--cache-paused-kv` 开启 radix 保留，通过 `--optimized-kernels`
使用原生后端选择的 attention 和 CUDA Graph。
两个后端的回放脚本都使用 `--fused-swap` 选择 Triton 跨层打包。离线图 16 使用同一份
完成统计，关闭用于副本路由的瞬时活动视图发布。
`autellix_mix_prefill_decode=false` 把 prefill 和 decode 分开执行，可用于对照同一接纳策略下的模型执行成本。
单阶段现代 GPU 的 `paper` 模式可设置 `autellix_mix_prefill_max_tokens`。正整数开启分段预填充
上的自适应合批，按实际接纳的未缓存 token 数选择是否混合执行，并检查当前分块和解码 KV 余量。
默认 `0`，使用 `autellix_mix_prefill_decode` 指定的固定方式。
`autellix_fill_chunk_tail=true` 允许当前窗口的普通等待请求使用最后一个 prefill 分块的余量。
默认值为 `false`。开启后，原生接纳继续检查 token 预算和 KV 容量。
`autellix_decode_cached_resumes=true` 启用单卡缓存恢复的 decode 路径，默认值为 `false`。
原生接纳完成后，完全命中已计算 KV、只需执行一个 token 的贪心恢复请求可与同批解码一起执行。
`autellix_min_decode_steps` 设置单阶段 SGLang `paper` 模式中，GPU 时间片耗尽后降级所需的最少解码步数。
默认 `0`。预填充和解码都按完整 GPU 时间结算。混合批次只有解码部分计步。
打开 `scheduler_trace` 和 `autellix_measure_timing` 可另外记录真实预填充与恢复预填充的 token 数。
单阶段 `paper` 模式可开启 `autellix_partial_retention`，默认关闭。整组低优先级解码与
高优先级预填充无法同时驻留时，按窗口顺序保留可安全共存的最长解码前缀。
`autellix_prefill_frontier=true` 在现代单阶段分段预填充下，按当前前向能到达的等待前缀
估算解码保留预算。使用固定分离或自适应合批，并保留 `autellix_retain_running=true`。
遇到已准备的 host KV 或候补请求时，预算仍覆盖原有候选集合。默认关闭。
矩阵计划的单个用例可设 `last_queue_fifo=true`。最低队列的时间片用尽时保留原入队顺序，
调用仍可因达到防饥饿阈值而提升。
两个回放脚本均支持 `--resident-reserve-only`。该模式保持前 B 个活动请求的优先级顺序，
K 个候补从余下请求中按同样的优先级选取，候补需已有驻留 KV。

PP 配置中，B=`max_running_requests` 是整个副本的活动容量，按 PP 阶段数均分到微批次，B 取 PP 的整数倍。K 是每个微批次的预备数，原生槽位总数为 B+PP×K。各阶段使用相同 token 容量，仅保存本阶段层的 KV。`SGLANG_PP_LAYER_PARTITION` 可指定分层，例如 Qwen3-32B 的 `20,22,22`。

SGLang 可显式设置 `autellix_legacy_gpu=true`（默认关闭），用于实验性 P40 兼容。
它默认 `dtype=float32`，保留显式指定的 `dtype=float16`；固定使用
`attention_backend=torch_native`、`sampling_backend=pytorch`、
`disable_cuda_graph=True`、`disable_custom_all_reduce=True`，并在 scheduler 进程禁用
Dynamo、替换标准模型算子。已测模型与配置见[验收记录](p40-validation-20261115.md)。

`autellix_attention_score_elements` 控制 P40 前缀 attention 的单块 score 元素预算，
默认 `67108864`。三卡 32B 服务使用 `16777216`，将每个 FP32 score 块上限从
256 MiB 降到 64 MiB；同时设置 KV=8192、`max_prefill_tokens=4096`，供长输入运行。
原生与 Autellix 两条路径都支持这个参数，性能回放双方使用相同值。

`autellix_retain_running=true` 默认在选中的 prefill 与当前 N 步解码窗口能放入
KV 预算时保留已选中的 decode。显存检查按实际 radix 路径锁定候选前缀，共享部分只扣一次。
空间不足时按优先级释放。
设为 false 可运行旧接纳方式的消融。`compare_sglang.py` 的 `--quantum-scale` 与
`--boundary-scale` 分别缩放时间片和服务历史区间，后者缺省时沿用前者。

## 设备分组

```bash
# 两张 GPU，各运行一个独立副本；将模型及 engine-args 换成实际配置。
autellix-serve --backend vllm --model /absolute/path/to/model --devices 0,1

# 两张 GPU 共同执行一个 TP=2 副本；该路径仍需实际硬件验证。
autellix-serve --backend vllm --model /absolute/path/to/model \
  --device-groups '0,1' --engine-args '{"tensor_parallel_size":2}'

# 三张 P40 共同执行 Qwen3-32B，包含完整的精度、缓存及分层配置。
bash scripts/serve_p40_three_gpu.sh
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
| `AUTELLIX_TEST_PP_MODEL` | GPU unittest | 三卡流水线回归使用的模型目录 |
| `AUTELLIX_TEST_IMPLEMENTATION` | GPU unittest | `paper` 或 `compat` |
| `AUTELLIX_TEST_LEGACY_GPU` | SGLang GPU unittest | 严格等于 `1` 时启用原生算子兼容模式及 2048 KV token 测试预算 |
| `AUTELLIX_MODEL`、`AUTELLIX_DEVICES`、`AUTELLIX_PORT` | `scripts/serve_p40.sh` | 模型目录、设备列表、端口；默认模型为持久盘 SmolLM2-135M，设备 `0`、端口 `8000` |
| `AUTELLIX_QUANTUM_SCALE`、`AUTELLIX_BOUNDARY_SCALE` | `scripts/serve_p40.sh` | 时间片倍数默认 1，服务区间倍数默认跟随时间片；生产服务可直接复用实验参数 |
| `AUTELLIX_PYTHON`、`AUTELLIX_STATE_DIR` | `scripts/serve_p40.sh` | 解释器及状态目录；默认独立持久盘环境，每次新建 `/tmp/autellix-p40-XXXXXX` 状态目录 |

除严格判断 `1` 的 `AUTELLIX_TEST_LEGACY_GPU` 外，可选测试开关按环境变量是否为非空判断；禁用时应取消设置，不要设成字符串 `0`。执行 CPU 测试前确认未继承上一轮 GPU 验证变量。
