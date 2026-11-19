# vLLM 后端

本项目独立实现的 Autellix 适配，固定使用 **vLLM 0.6.1**。安装、服务、客户端和验证入口见 [安装与首次使用](../../docs/getting-started.md)。

## 支持范围

- Decoder-only 文本生成，每次调用仅支持一个输出序列（`n=1`、`best_of=1`）。
- 前缀缓存、GPU/CPU KV 交换、批量 KV 传输和独立引擎副本。
- 张量并行让每个 rank 安装传输钩子，已在 H20 上用 Llama‑3.1‑70B 验证 TP=2 和 TP=4 生成。
- 不支持流水线并行；推测解码等扩展不属于已验证的运行配置。

后端检查版本，并在自己的进程内安装钩子，不修改已安装的 vLLM 文件。

## 调度与显存

默认 `paper` 模式由 `paper.py` 按全局优先级接纳请求，不使用原生 `_schedule()` 决定批次。窗口开始时预留后续 KV 槽位；窗口内更新 token 和 block 元数据，预备请求可在活动请求结束后补位。

启用混合 prefill/decode 执行，避免原生阶段偏好覆盖请求优先级。接纳按整段 prefill 计算；输入超过 `max_num_batched_tokens` 会在接纳前报错，需增大该参数。预备请求占用 KV 显存，在补位前不消耗 decode token 配额。

抢占沿用原生 block 所有权转换。换出和换入分开处理，保留生成序列；实际批量传输由 `autellix/runtime/swap.py` 完成。预留窗口所需显存不足时会明确报错，需调整模型、显存预算、批次或窗口长度。

## 配置

[安装文档](../../docs/getting-started.md)的启动命令使用默认 `paper`、N=8、K=1。`--implementation compat` 选择旧包装实现：单 GPU 且 K=0 时使用原生缓存多步执行，其余情况在筛选后的请求集合中调用原生调度。

`--devices 0,1` 启动两个独立副本。张量并行使用 `--device-groups '0,1;2,3'`，并在 `--engine-args` 中设置 `"tensor_parallel_size":2`；每组设备数必须匹配并行度。

## 原生基准对照

真实工作负载命令可选择以下原生模式，保留原生调度和 KV 传输，仅观测执行时间：

| 模式 | 配置 |
| --- | --- |
| `vllm` | 基础原生配置 |
| `vllm-opt` | 前缀缓存和分块 prefill |
| `vllm-opt-multistep` | 前缀缓存和原生多步执行 |

vLLM 0.6.1 的分块 prefill 与原生多步执行使用两个独立配置，分别测试。`--native-opt-steps` 配置原生多步长度。

## 论文实验回放

使用 `scripts/run_paper_vllm.py` 和预处理好的 ShareGPT、BFCL、LATS、Mixed 轨迹。
每次测量完成后，运行目录保存计划和原子写入的 JSON 检查点。以同一组模型、参数和轨迹
再次运行即可从中断处继续。不同实验、模型尺寸和离线批量使用独立的 `--run-dir`。

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
使用 `scripts/prepare_sharegpt_replay.py --programs 4000` 准备 4000 条不同的完整对话，
再用 `offline sharegpt --trace "$trace" --all-programs` 回放每条对话一次。
基础模型的 tokenizer 可通过 `--chat-template` 读取现有的聊天模板。
主对照使用相同的活动请求上限、BF16 和 eager 模式。Autellix 可额外预填充一个预备请求。
`timing` 记录各阶段开销。
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
