# Autellix 论文复现尝试 · H20

`h20-env-zy-fjr` 用于代码和数据同步，`h20-gpu-zy-fjr` 用于计算。代码位于
`/home/nice/qiuzeyu/stu/fjr/code/autellix`，数据、计划、结果保存在
`/home/nice/qiuzeyu/stu/fjr/data/autellix`，环境为 `fjr` 的 `autellix-sglang`。
SQLite 和进程间通信使用本机 `/tmp/fjr-autellix-*`，运行记录和调度轨迹在结束后保存到共享目录。

实验直接读取共享模型。8B 回放使用 `Meta-Llama-3.1-8B`，大模型配对使用
`Meta-Llama-3.1-70B`，接口验证使用 `Llama-3.2-1B-Instruct`。四个工作负载使用从
P40 迁移的完整 ShareGPT、BFCL、LATS 及其混合集合。Mixed 默认抽取 12 个程序，
每个来源 4 个，混排后按泊松过程到达。每个配对共用程序、到达时间、
GPU、模型、KV 容量和批次上限；候选组使用 PLAS 或 ATLAS。正式配对在 BF16 下使用
SGLang 的 Triton attention，保留 radix 缓存。两组关闭 overlap、CUDA Graph 和
chunked prefill。到达率按 H20 的执行速度设置，结果包含实际积压和排空时间。
汇总记录 SGLang 的累计缓存 token 计数。请求换出后再次接纳会重新累计，因而该计数可
超过原始输入 token 数。[SGLang 0.4.9.post6 缓存计数实现](https://github.com/sgl-project/sglang/blob/v0.4.9.post6/python/sglang/srt/managers/schedule_batch.py)
`--native-default` 增加采用 SGLang 上游默认优化的对照，使用相同模型权重、批次和
KV 上限。`--native-default .48` 只在这个到达率补测。`--device '2;3' --routing-ablation` 在两个单卡副本上，固定调度策略对比
程序亲和性、最少调用数和轮询。多卡单副本使用 `--device '2,3'`。上述卡号为命令示例。
`--fcfs-control .48` 使 Autellix 保留相同钩子、窗口和预备槽，以 FCFS 配对测量调度路径开销。
BFCL 压测的恢复参数为 `--host-kv-gib 32 --host-kv-recompute`，只作用于 Autellix，
主机缓存满时复用 radix 前缀并重算剩余 KV。

下面的命令在 H20 上执行，`gpu` 使用本次分配且空闲的物理卡号。

```bash
source /home/nice/qiuzeyu/stu/fjr/tools/activate.sh
conda activate autellix-sglang
cd /home/nice/qiuzeyu/stu/fjr/code/autellix
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$(mktemp -d /tmp/fjr-autellix-XXXXXX)"
data=/home/nice/qiuzeyu/stu/fjr/data/autellix
model=/data-hdd/opt/models/llama3.1/Meta-Llama-3.1-8B
python scripts/run_h20_paper.py sharegpt --model "$model" --datasets "$data/datasets" \
  --run-dir "$data/runs/h20-sharegpt" --device "$gpu"
```

长任务用 `fjr-autellix-*` 命名的 tmux 会话运行。接口回归启动的 HTTP 服务会随测试退出，
每次配对结束后关闭引擎。结果保存后，通过 `nvidia-smi` 核对任务进程是否已释放 GPU。

论文第 6 节按程序级秒/token 对照可承载的程序到达率，另报 P95 和 P99。结果汇总记录
这三项，并记录实际程序完成率、输出 token/s、到达阶段积压和排空时间。8B 使用单张
H20，70B 在本实验使用流水线并行，论文使用 4 张 A100。论文的主要对照包括 vLLM
0.6.1、启用分块 prefill 和前缀缓存的 vLLM-opt，以及 MLFQ。
[Autellix 论文，第 6 节](https://arxiv.org/html/2502.13965v1#S6)

## 回放输入核对

| 工作负载 | 程序池 | 平均调用数 / 论文 | 平均输出 token/调用 / 论文 | 最长完整上下文 |
| --- | ---: | ---: | ---: | ---: |
| ShareGPT | 128 | 6.10 / 6.66 | 283.97 / 277 | 23,688 |
| BFCL v3 | 64 | 10.77 / 10.75 | 46.39 / 34.14 | 12,482 |
| LATS | 6 | 320.00 / 159.70 | 75.79 / 72.60 | 4,417 |

固定提示包含此前轮次和工具上下文。以最长直接父调用的共同前缀作参考，三个程序池的
平均新增输入分别为 313、693、1717 token/调用。实际 GPU prefill 还取决于 radix
保留的前缀。LATS 采用 30 次搜索迭代和并行扩展，现有样本的程序深度和平均输入都更大。
Mixed 按程序等量抽取，LATS 在模型调用和生成 token 中仍占较高比重。
输入来源与采集配置见[原始工作负载](paper-workload-experiment.md)。

## GPU 功能验收

节点 H20-2 装有 8 张 NVIDIA H20，每张可见 97,871 MiB，CUDA 算力 9.0，驱动
570.133.20。推理环境使用 Python 3.10、PyTorch 2.7.1+cu126 和 SGLang 0.4.9.post6。
8B 单卡配对使用当时空闲的 2、3、7 号卡，各配对在同一张卡上顺序执行。

| 检查 | H20 结果 |
| --- | --- |
| 单卡回归 | 152 项，148 通过，4 项需要 vLLM 或额外 GPU 的测试跳过 |
| KV 往返 | 17,408 token、64 个 K/V buffer，共 2.125 GiB，逐项一致；临时显存峰值 16 MiB |
| 接纳、换出和恢复 | K=0 与 K=1 各换出 15 次，恢复生成与连续生成一致 |
| 主机 KV 满后的重算 | 两个 K 配置各触发 15 次重算，输出一致，会话回收 |
| HTTP 与会话 | Llama-3.2-1B-Instruct 返回 `The capital of France is Paris.`，流式与普通响应、服务历史继承、会话清理通过 |

单卡验收使用 GPU 2，退出后该卡显存为 0 MiB。环境包 `decord 0.6.0` 的内部
wheel 标签为 `cp36`，Python 3.10 上实测可以导入；验收日志保留 `pip check`、
导入输出和完整依赖列表。

代码、固定回放与历史 P40 证据包在 `fjr` 的项目目录内。72 个迁移文件的大小和
SHA-256 已核对，记录位于 `data/autellix/import/migration-verification.json`。
模型直接从 `/data-hdd/opt/models` 读取。

## 论文机制与实现

| 机制 | 实现入口与核对 |
| --- | --- |
| PLAS、ATLAS、MLFQ | `autellix/runtime/policy.py` 以实际模型服务时间更新程序历史，按多级 FIFO 队列执行降级和防饥饿；在线 ATLAS 使用论文算法 1 的最长路径标量 |
| N 步与 K 个预备请求 | `autellix/runtime/window.py` 管理窗口和补位；H20 轨迹记录 `schedule_window`、`continue_plan` 和逐步执行 |
| GPU 调度 | `integrations/sglang/scheduler.py` 在 SGLang 的 GPU 调度进程中控制请求接纳、运行批次和 KV 锁；CUDA event 计入模型执行时间 |
| KV 交换 | `integrations/sglang/swap.py` 逐请求执行分块 gather/copy/scatter；论文的跨请求专用 CUDA 聚合交换仍待实现 |
| 多引擎和会话 | `autellix/runtime/engine.py` 管理每副本进程、异步结果和程序亲和性；短输入优先选择请求最少的副本，长输入沿用程序所在副本 |
| 论文实验 | 覆盖完整 ShareGPT、BFCL、LATS、Mixed、同到达率配对和程序级 SLO 曲线；论文的 Falcon-180B 和大规模离线批量曲线仍待补齐 |

## 已测的单卡配对

下表以同资源、同 Triton 配置的原生 SGLang 为基准。响应和秒/token 越低越好，
输出吞吐越高越好。

| 工作负载与到达率（程序/秒） | 策略 | 平均程序响应 | 论文秒/token | 输出吞吐 |
| --- | --- | ---: | ---: | ---: |
| ShareGPT 0.12 | PLAS | +10.31% | +26.21% | −3.60% |
| ShareGPT 0.24 | PLAS | +14.57% | +18.99% | −6.65% |
| ShareGPT 0.48 | PLAS | +15.36% | +19.06% | −7.80% |
| BFCL 0.12 | PLAS | +11.54% | +12.10% | −0.07% |
| BFCL 0.24 | PLAS | +19.32% | +21.35% | −0.08% |
| BFCL 0.48 | PLAS | +46.96% | +52.22% | −1.33% |
| BFCL 0.48 | Autellix FCFS | +46.04% | +49.70% | −1.21% |
| LATS 0.035 | ATLAS | +12.10% | +12.17% | −7.45% |
| LATS 0.07 | ATLAS | +14.16% | +14.22% | −10.22% |

BFCL 在 0.48 程序/秒时，原生、Autellix FCFS 和 PLAS 的平均程序响应依次为
9.62、14.04、14.13 秒。原生在到达窗口内完成 46/48 个程序，PLAS 只换出 3 次，
5890 个 GPU 批次有 739 次窗口决策。FCFS 仍承担接纳钩子、每步计时、状态记录和
预填充与解码混合。该对照定位了共用执行路径的成本，PLAS 的排序在这个到达率下
只使平均响应用时额外增加约 0.09 秒。

LATS 在 0.07 程序/秒时有 58 次预备请求接纳和 44 次窗口内补位，实际执行了 N/K
机制。约 78% 的调用进入最后两级队列，且发生 54 次换出，重排与搬运增加了成本。
当前 LATS 搜索样本比论文更深，Mixed 的固定样本中 LATS 占 4/12 个程序、约 95%
的调用和 93% 的输出 token。

论文用统一的程序级秒/token 比较可承载的到达率，基础 vLLM 对照没有前缀缓存。
H20 配对使用已经有 radix 前缀缓存的 SGLang。把 H20 上的模型执行和调度开销分开后，
优先优化 GPU 调度热路径：把状态写入集中到窗口边界和调用完成，合并 CUDA 计时同步，
并把同批被抢占请求的 KV 聚合传输。测量时同时保留原生、Autellix FCFS 和 PLAS/ATLAS。
[Autellix 论文，第 6 节](https://arxiv.org/html/2502.13965v1#S6)
