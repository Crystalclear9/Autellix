# vLLM 后端

本项目独立实现的 Autellix 适配，固定使用 **vLLM 0.6.1**。安装、服务、客户端和验证入口见 [安装与首次使用](../../docs/getting-started.md)。

## 支持范围

- Decoder-only 文本生成，每次调用仅支持一个输出序列（`n=1`、`best_of=1`）。
- 前缀缓存、GPU/CPU KV 交换、批量 KV 传输和独立引擎副本。
- 张量并行让每个 rank 安装传输钩子，已在 H20 上用 Llama‑3.1‑70B 验证 TP=2 和 TP=4 生成。
- 不支持流水线并行；推测解码等扩展不属于已验证的运行配置。

后端检查版本，并在自己的进程内安装钩子，不修改已安装的 vLLM 文件。

单卡可直接读取本地 Qwen3-32B BF16 检查点。Qwen3 模型沿用 0.6.1 的 Qwen2 权重加载和
执行接口，使用 Qwen3 的独立 head dimension 及 RoPE 前的 Q/K RMSNorm。原生 vLLM
和 Autellix 共用这份适配。回放入口见 [H20 单卡 Qwen3-32B](../../docs/h20-reproduction.md#单卡-qwen3-32b)。
`qwen3_fused_qk_rope=true` 在单个 Triton kernel 内完成 Q/K 的 RMSNorm 与 RoPE，保留
原算子的中间舍入，并原样保留 V。该参数可同时应用于原生调度和 PLAS 对照。

## 调度与显存

默认 `paper` 模式由 `paper.py` 按全局优先级接纳请求，不使用原生 `_schedule()` 决定批次。窗口开始时预留后续 KV 槽位；窗口内更新 token 和 block 元数据，预备请求可在活动请求结束后补位。

`--resident-reserve-only` 将候补限定为已有 GPU KV 的请求，按论文的候补选择方式运行。

混合 prefill/decode 执行按程序优先级选批。`paper_chunked_prefill=true` 复用 vLLM 的 token 预算与前缀缓存块对齐规则，把长输入分到多个迭代。最后一段完成后才预留后续 decode 的 KV 槽位。默认整段 prefill，`max_num_batched_tokens` 需覆盖单条输入。预备请求占用 KV 显存，成为活动请求后参与 decode。

分块 prefill 耗尽当步 token 预算时，待接纳请求和已入选的 decode 留在窗口内。实际 KV 空间不足时，调度窗口按优先级收缩。

`exact_prefix_cache=true` 按实际算完的完整 KV 块发布可复用前缀。启用前缀缓存的全上下文
PLAS 分块 prefill 默认使用这一路径。未完成的 KV 块释放时直接
归还分配器，换入时按已计算长度恢复前缀状态。开启该选项后，PLAS 分块 prefill 用实际需要
执行的 token 计入单步预算，并把逻辑进度连同已缓存的 token 一起交给 vLLM worker。
接纳新请求时，活跃请求已经占用且已算完的共享前缀块可直接复用，按新增物理块核算
KV 容量。未使用的缓存块继续纳入分配器的空闲容量。

分块 prefill 可用 `resident_decode_headroom=true`，给当前窗口中已驻留的活动 decode 各留一个单步 token。这只改变窗口内的混合批次构造，程序优先级和 KV 容量检查保持一致。

抢占沿用原生 block 所有权转换。换出和换入分开处理，各自可与已接纳请求的模型执行放在同一步。等待下一次传输的请求保留在窗口中，传输后继续检查其接纳条件。实际批量传输由 `autellix/runtime/swap.py` 完成。

## 配置

[安装文档](../../docs/getting-started.md)的启动命令使用默认 `paper`、N=8、K=1。`--implementation compat` 选择旧包装实现：单 GPU 且 K=0 时使用原生缓存多步执行，其余情况在筛选后的请求集合中调用原生调度。

`--devices 0,1` 启动两个独立副本。张量并行使用 `--device-groups '0,1;2,3'`，并在 `--engine-args` 中设置 `"tensor_parallel_size":2`；每组设备数必须匹配并行度。

## 原生基准对照

真实工作负载命令可选择以下原生模式。性能回放保留原生调度和 KV 传输，默认关闭观察钩子。`timing` 启用观察钩子和执行计时。

| 模式 | 配置 |
| --- | --- |
| `vllm` | 基础原生配置 |
| `vllm-opt` | 前缀缓存和分块 prefill |
| `vllm-opt-multistep` | 前缀缓存和原生多步执行 |

vLLM 0.6.1 的分块 prefill 与原生多步执行使用两个独立配置，分别测试。用 `--interval` 配置原生多步长度和 Autellix 窗口长度。[vLLM 0.6.1 参数检查](https://github.com/vllm-project/vllm/blob/v0.6.1/vllm/engine/arg_utils.py#L900)

## 论文实验回放

使用 `scripts/run_paper_vllm.py` 和预处理好的 ShareGPT、BFCL、LATS、Mixed 轨迹。
每次测量完成后，运行目录保存计划和原子写入的 JSON 检查点。以同一组模型、参数和轨迹
再次运行即可从中断处继续。不同实验、模型尺寸和离线批量使用独立的 `--run-dir`。
H20 的 vLLM-opt 与 Autellix 回归使用相同模型、到达轨迹、设备、KV 预算及逐调用 token 数。
最近的 8B、14B 配对见 [H20 窗口边界续跑复测](../../docs/h20-reproduction.md#sglang-窗口边界续跑复测)，
其他模型的计划和检查点见 [H20 复现记录](../../docs/h20-reproduction.md)。

| 入口 | 对应的论文实验 |
| --- | --- |
| `online` | 图 12–13，泊松到达、程序级延迟和 P95/P99，对比 vLLM、两个优化配置、MLFQ 和 Autellix |
| `routing` | 图 14，固定 PLAS/ATLAS，对比程序亲和性、最少调用数和轮询 |
| `scaling` | 图 15，按不同副本数运行同一组到达率，以同一延迟目标比较 |
| `offline` | 图 16，所有程序在 t=0 到达，扫 `--programs` 并对比总用时 |
| `timing` | 图 17，额外记录调度、prefill、decode、混合批次和 KV 传输时间 |
| `swap` | 图 19，固定调度策略，对比 vLLM 默认 KV 搬运和当前批量打包实现 |

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_paper_vllm.py online sharegpt \
  --model "$model_8b" --datasets "$datasets" --run-dir "$runs/online-sharegpt" \
  --device "$gpu" --rates .48 .96 1.92
python scripts/run_paper_vllm.py offline sharegpt \
  --model "$model_8b" --datasets "$datasets" --run-dir "$runs/offline-1000" \
  --device "$gpu" --programs 1000
```

`--device '0,1,2,3'` 表示一个 TP=4 引擎，`--device '0;1;2;3'` 表示四个单卡副本。
长队列离线测量可用 `--window-activity` 将等待中请求的状态更新合并到调度窗口。
`--keep-resident-kv` 保留暂停请求的 GPU KV，当前调度窗口需要显存块时再换出。
使用 `scripts/prepare_sharegpt_replay.py --programs 4000` 准备 4000 条不同的完整对话，
再用 `offline sharegpt --trace "$trace" --all-programs` 回放每条对话一次。
基础模型的 tokenizer 可通过 `--chat-template` 读取现有的聊天模板。
`--request-timeout` 按秒设置单次调用允许等待和执行的总时长，默认 14,400 秒。
各组使用相同活动请求上限和 BF16。`--cuda-graphs` 为各组启用原生 decode CUDA Graph，
`--kv-blocks` 固定相同的 GPU KV block 数。`--paper-chunked-prefill` 为 Autellix 与 MLFQ
启用分块，`--batch-tokens` 控制每步 token 预算。`--no-scheduler-trace` 用于性能对照，
`timing` 配合默认开启的调度轨迹记录各阶段开销。
使用 `scripts/summarize_paper_suite.py` 汇总各目录的测量 JSON，计时数据可用
`scripts/analyze_scheduler_trace.py --state-root "$run/scheduler-state"` 读取。图 18 的等步长模拟使用
`python -m autellix.cli paper-preset --preset optimal-scheduling --workload sharegpt --dataset "$trace"`，
同一命令可切换到 LATS。KV 内核微基准使用 `cuda/batched_swap_benchmark.py`。

H20 运行 vLLM 0.6.1 和 PyTorch 2.4.0 时，预载 cuBLAS 12.6.4.1 的两份库，
以避开 cuBLAS 12.1 在该设备上的矩阵乘法崩溃。同一组实验的所有策略使用同一对库。

```bash
export LD_PRELOAD="$cublas_lib/libcublas.so.12:$cublas_lib/libcublasLt.so.12"
```

[H20 上的 PyTorch/cuBLAS 问题记录](https://github.com/pytorch/pytorch/issues/133548) · [Autellix 论文第 6 节](https://arxiv.org/html/2502.13965v1#S6)

`autellix_adapter.py` 保留旧元数据接口，其 `create_backend()` 可构造真实后端。回归测试包括抢占前后贪心结果一致、程序历史继承和无效输入隔离；双副本及张量并行测试需显式启用，见[开发与验证](../../docs/development.md)。
