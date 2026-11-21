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

长任务在 `tmux -L fjr-autellix` 的独立会话中运行。接口回归启动的 HTTP 服务会随测试退出，
每次配对结束后关闭引擎。结果保存后，通过 `nvidia-smi` 核对任务进程是否已释放 GPU。

论文第 6 节按程序级秒/token 对照可承载的程序到达率，另报 P95 和 P99。结果汇总记录
这三项，并记录实际程序完成率、输出 token/s、到达阶段积压和排空时间。8B 使用单张
H20，70B 在 SGLang 使用流水线并行，在 vLLM 使用四卡张量并行。论文的主要对照包括 vLLM
0.6.1、启用分块 prefill 和前缀缓存的 vLLM-opt，以及 MLFQ。
[Autellix 论文，第 6 节](https://arxiv.org/html/2502.13965v1#S6)

## vLLM 离线回放与续跑

在 `fjr` 的 `autellix-vllm061` 环境中使用同一份 ShareGPT 程序回放。所有程序同时提交，
用整批完成时间衡量吞吐，并记录程序级秒/token。下面的命令沿用上面的 `model`、`data`
和 `gpu`，直接读取共享权重与已准备好的 4,000 程序输入。

```bash
conda activate autellix-vllm061
lib="$HOME/stu/fjr/miniconda3/envs/autellix-sglang/lib/python3.10/site-packages/nvidia/cublas/lib"
export LD_PRELOAD="$lib/libcublas.so.12:$lib/libcublasLt.so.12"
python scripts/run_paper_vllm.py offline sharegpt --model "$model" --device "$gpu" \
  --trace "$data/datasets/sharegpt-whole4000.json" --all-programs --window-activity \
  --label offline-4000 --run-dir "$data/runs/offline-4000" --cases vllm vllm-opt
```

回放在每个对照组结束时保存检查点。续跑时复用原计划和输出文件，完成的组按版本、
配置和输入哈希校验后跳过；中断的组从头回放。Autellix 的主机 KV 容量和时间片通过
`--swap-gib` 和 `--quantum-scale` 设置，`--keep-resident-kv` 使被抢占的 KV 在有空位时
留在 GPU。4,000 程序的 Autellix 补测使用 `--cases autellix --keep-resident-kv
--quantum-scale 8 --swap-gib 384 --request-timeout 79200`，并指定独立的运行目录。
原生 SGLang 对照沿用同一份输入、随机种子和批次上限，汇总时核对输入哈希和输出 token 数。

## 单卡 Qwen3-32B

在 `autellix-sglang` 环境中直接读取 `stu/ljh/models/Qwen3-32B` 的 BF16 权重。数据和
结果沿用 `fjr` 的目录。ShareGPT 用 Qwen 模板重建完整会话，BFCL 保留已采集的工具反馈，
LATS 保留搜索树，三类输入都用 Qwen tokenizer 编码。Mixed 从三类输入各取两个程序。

```bash
model="$HOME/stu/ljh/models/Qwen3-32B"
out="$data/runs/paper-vllm-20261119"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"

python scripts/prepare_h20_qwen32.py --model "$model" --datasets "$data/datasets" \
  --run-dir "$out" --device "$gpu"
for family in sharegpt bfcl sharegpt-offline128 lats mixed; do
  python scripts/run_sglang_matrix.py --plan "$out/qwen32-$family-plan.json" \
    --output "$out/qwen32-$family.json" --resume
done
```

同一张卡依次运行匹配 Triton 配置的 SGLang、Autellix PLAS 或 ATLAS，以及采用上游默认优化的
SGLang。功能验收使用 `check_prefill_priority_gpu.py --modern --mem-fraction-static .88`，
分别验证 KV 换入换出与 `--host-kv-recompute`，检查恢复生成和连续生成的输出是否一致。

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

持久盘的 `data/autellix/runs/` 保存 34 次已完成的性能测量和对应计划。回放命令自动使用
`--resume`，按配置、模型、版本和轨迹哈希匹配已完成的测量。[机器可读汇总](results/h20-20261119.json)

## 论文机制与实现

| 论文条目 | 当前 vLLM 实现与实验核对 |
| --- | --- |
| §4.2.1 程序表、PLAS、ATLAS | [`ProgramTable` 和 `RuntimeScheduler`](../autellix/runtime/policy.py) 保存会话、调用及累计等待时间。PLAS 继承已完成调用的累计模型时间。在线 ATLAS 继承程序当前的最长已观察服务路径标量，完成时以 `max(旧值, 继承值 + 本次模型时间)` 更新。DAG 回放按依赖提交调用，在线调度使用程序表标量。 |
| §4.2.2 队列、抢占、防饥饿 | [`PaperPlan`](../integrations/vllm/paper.py) 按程序优先级选取全局批次，遇到 token 或 KV 容量限制时保留可接纳的队首。时间片用尽后降级，较低队列按程序历史和调用当前窗口的等待/服务比提升。请求压入 vLLM 的实际执行批次，CUDA event 记录模型时间。BFCL 1.92 的轨迹含 3,321 次提升和 1,365 次换出及恢复。 |
| §4.2.2 N 步和 K 个预备请求 | [`SchedulingWindow`](../autellix/runtime/window.py) 冻结 N 步的优先次序，预填充或恢复 K 个候补请求，活动请求完成后窗口内补位。ShareGPT 1.92 记录了 74 次补位。接入层仍使用 vLLM 的逐步执行与 KV 元数据接口，保留每步的元数据构造和 CUDA 计时同步。 |
| §4.3 路由与 §5 会话 | [`InferenceEngine`](../autellix/runtime/engine.py) 按副本的未完成调用数选择负载最低的引擎，超过 2,048 token 的请求复用程序亲和的副本。已测两个及四个 8B 副本，以及两个双卡 TP 的 70B 副本。 |
| §5 聚合 KV 交换 | [`batched_swap`](../autellix/runtime/swap.py) 在 vLLM 的真实 KV 缓存路径上聚合跨层 K/V，以连续 pinned buffer 传输，每段上限 1 GiB。打包、解包用 PyTorch gather/scatter 和同步保护；论文使用定制 CUDA/C++ 内核。H20 的 8–512 块往返验收逐项一致，优化后微基准较原生 vLLM 快 1.14–1.30 倍。 |
| §6 实验覆盖 | 8B 四类在线负载共 60 个 vLLM 实验，70B 四卡张量并行共 20 个；另有多副本、耗时拆分、KV 微基准和 SRPT 模拟。8B 离线 128 完成全组，1,000 与 4,000 有部分组合完成。论文使用 A100，本轮使用 H20，Falcon-180B 尚未运行。 |

这是一次 Autellix 论文复现尝试。核心调度、路由和 KV 交换均接入了真实推理。实验中的
`vllm-opt` 使用分块 prefill 与前缀缓存，`vllm-opt-multistep` 使用前缀缓存与原生多步，
分开覆盖论文 vLLM-opt 的三项优化。回放使用固定参考输出长度；本地 LATS 平均 320 次调用，
论文为 159.7 次。论文在线吞吐比较固定程序级延迟下可承载的程序到达率，下表报告固定到达率的
输出 token/s 与程序级延迟。 [Autellix 论文，§4–§6](https://arxiv.org/html/2502.13965v1#S4)

## vLLM Autellix 与原生 SGLang

以下配对使用单张 H20、Llama-3.1-8B BF16、活动 batch 上限 16。Autellix 接入 vLLM 0.6.1，
对照使用 SGLang 0.4.9.post6 的上游默认执行优化。GPU KV 预算和依赖版本沿用各自的运行配置。
`online-*` 与 `sglang-high-*` 的四个最高负载配对分时使用同型号的
不同实体卡；后两行在各自同一张实体卡上完成。每对的输入轨迹 SHA-256、权重路径和输出 token
数量一致，输出序列哈希随引擎而异。延迟为论文口径的平均毫秒/token，LATS 按关键路径计算，
Mixed 按程序类型选择端到端时间或关键路径时间。

| 负载（程序/秒） | 输出 token/s，Autellix / SGLang | 吞吐变化 | 毫秒/token，Autellix / SGLang | 延迟变化 |
| --- | ---: | ---: | ---: | ---: |
| ShareGPT 1.92 | 306.08 / 444.84 | −31.2% | 31.87 / 12.31 | +159.0% |
| BFCL 1.92 | 30.43 / 352.54 | −91.4% | 1,931.48 / 48.90 | +3,849.9% |
| LATS 0.28 | 549.90 / 689.46 | −20.2% | 9.37 / 7.42 | +26.4% |
| Mixed 0.48 | 429.58 / 547.06 | −21.5% | 15.46 / 11.14 | +38.7% |
| ShareGPT 1.92，时间片倍率 8，H20 0 | 331.29 / 444.84 | −25.5% | 23.89 / 12.31 | +94.1% |
| BFCL 1.92，保留 GPU 上暂停的 KV，H20 5 | 272.24 / 347.65 | −21.7% | 60.74 / 50.23 | +20.9% |

原 BFCL 组换出 1,365 次，启用 `--keep-resident-kv` 后换出降至 0，吞吐由 30.43 升至 272.24
token/s。ShareGPT 时间片倍率 8 的补测记录 71 次换出。要量化换出归零后的差距，可对齐两边的
KV token 预算，在同一张卡上分别测量模型执行、每步批次构建、CUDA 计时同步和 KV 传输。原始计划与结果位于
`data/autellix/runs/paper-vllm-20261119/` 的 `online-*`、`sglang-high-*`、`tune-sharegpt-q8`、
`resident-bfcl` 和 `sglang-bfcl-resident-card`。

离线 1,000 程序调优组和 vLLM-opt 使用同型号的不同 H20。相同 ShareGPT 输入的平均延迟由 vLLM-opt 的
2.316 秒/token 降至 0.690 秒/token（−70.2%），整批时间由 2,453 秒增至 2,604 秒（+6.2%）。
4,000 程序的基础 vLLM 已完成，原生 SGLang 与 Autellix 的整批配对尚未完成。

## SGLang 适配的单卡配对

下表以同资源、同 Triton 配置的原生 SGLang 为基准。响应和秒/token 越低越好，
输出吞吐越高越好。

![H20 8B 四类工作负载程序级延迟](results/h20-workload-curves.png)

| 工作负载与到达率（程序/秒） | 策略 | 平均程序响应 | 论文秒/token | 输出吞吐 |
| --- | --- | ---: | ---: | ---: |
| ShareGPT 0.12 | PLAS | +10.31% | +26.21% | −3.60% |
| ShareGPT 0.24 | PLAS | +14.57% | +18.99% | −6.65% |
| ShareGPT 0.48 | PLAS | +15.36% | +19.06% | −7.80% |
| BFCL 0.12 | PLAS | +11.54% | +12.10% | −0.07% |
| BFCL 0.24 | PLAS | +19.32% | +21.35% | −0.08% |
| BFCL 0.48 | PLAS | +46.96% | +52.22% | −1.33% |
| BFCL 0.48 | Autellix FCFS | +46.04% | +49.70% | −1.21% |
| BFCL 0.96 | PLAS，32 GiB + 重算 | +541.67% | +780.23% | −56.07% |
| BFCL 1.92 | PLAS，32 GiB + 重算 | +339.38% | +429.37% | −55.53% |
| LATS 0.035 | ATLAS | +12.10% | +12.17% | −7.45% |
| LATS 0.07 | ATLAS | +14.16% | +14.22% | −10.22% |
| Mixed 0.06 | ATLAS | +10.76% | +16.15% | −5.68% |
| Mixed 0.12 | ATLAS | +11.73% | +19.61% | −7.89% |
| Mixed 0.30 | ATLAS | +11.59% | +21.65% | −8.43% |

在 ShareGPT 0.48 的补测中，上游默认 SGLang 为 17.65 秒、369.46 输出 token/s，
匹配 Triton 的原生组为 25.00 秒、281.43 token/s，PLAS 为 28.84 秒、259.49 token/s。
PLAS 相对上游默认 SGLang 的平均响应增加 63.37%，吞吐下降 29.76%。

BFCL 在 0.48 程序/秒时，原生、Autellix FCFS 和 PLAS 的平均程序响应依次为
9.62、14.04、14.13 秒。原生在到达窗口内完成 46/48 个程序，PLAS 只换出 3 次，
5890 个 GPU 批次有 739 次窗口决策。FCFS 仍承担接纳钩子、每步计时、状态记录和
预填充与解码混合。该对照定位了共用执行路径的成本，PLAS 的排序在这个到达率下
只使平均响应用时额外增加约 0.09 秒。

BFCL 在 0.96 程序/秒时，原生和 PLAS 在到达窗口分别完成 36 和 2 个程序，排空用时
13.29 和 107.95 秒。在平均程序级延迟目标 0.025 秒/token 下，原生 SGLang 和
PLAS 的最高实测到达率分别为 0.96 和 0.24 程序/秒。
1.92 程序/秒的原生组为 20.71 秒、344.75 token/s，PLAS 为 90.97 秒、153.29
token/s。原 16 GiB 主机 KV 配置触到预算后退出，复跑使用同一 GPU 和输入轨迹，保留
已完成的原生结果。第一次中断日志保存在 `logs/autellix/h20-bfcl-stress-16gib.log`。
0.96 和 1.92 程序/秒分别产生 792 和 1016 次完整 KV 往返，以及 26 和 30 次
主机缓存满后的重算。高负载下的主要增量来自频繁的抢占恢复。

LATS 在 0.07 程序/秒时有 58 次预备请求接纳和 44 次窗口内补位，实际执行了 N/K
机制。约 78% 的调用进入最后两级队列，且发生 54 次换出，重排与搬运增加了成本。
当前 LATS 搜索样本比论文更深，Mixed 的固定样本中 LATS 占 4/12 个程序、约 95%
的调用和 93% 的输出 token。

论文用统一的程序级秒/token 比较可承载的到达率，基础 vLLM 对照没有前缀缓存。
H20 配对使用已经有 radix 前缀缓存的 SGLang。把 H20 上的模型执行和调度开销分开后，
优先优化 GPU 调度热路径：把状态写入集中到窗口边界和调用完成，合并 CUDA 计时同步，
并把同批被抢占请求的 KV 聚合传输。测量时同时保留原生、Autellix FCFS 和 PLAS/ATLAS。
[Autellix 论文，第 6 节](https://arxiv.org/html/2502.13965v1#S6)

## 双卡模型与副本验证

两个独立的 Llama-3.2-1B-Instruct 副本已通过短输入负载分流、长输入程序亲和和取消清理。
双卡流水线的初次验收定位到 SGLang 固定版本的 Llama 共享词嵌入装载：最后一阶段需要从
模型的输入嵌入权重初始化输出头。原 P40 的装载逻辑已移入公共流水线路径，供原生 SGLang
和 Autellix 共用。70B 模型的输入嵌入和输出头分别存储。

2026-11-19 在 H20 0、1 号卡重新运行 Llama-3.2-1B-Instruct 双级流水线验收，耗时
40.73 秒。80 次 KV 换出均完成恢复，恢复后的贪心输出与连续生成一致，服务历史继承、
两级调度轨迹一致性及 2 次取消后的会话清理均通过。两张卡在验收退出后均为 0 MiB。

| 70B，BFCL 12 程序、0.48 程序/秒 | 原生 SGLang | PLAS | 变化 |
| --- | ---: | ---: | ---: |
| 平均程序响应（秒） | 109.20 | 172.98 | +58.42% |
| 平均程序级秒/token | 0.16125 | 0.28158 | +74.63% |
| 输出 token/s | 46.15 | 35.27 | −23.56% |
| GPU 3 / 7 峰值显存（MiB） | 87,599 / 86,247 | 88,835 / 87,857 | |

两张 H20 各承担 40 层，使用 BF16、每个流水线阶段 65,536 个 KV token 槽、总批次
上限 8。PLAS 记录到 173 次换出并全部恢复。该组结束后，两张卡回到 0 MiB。
8B 和 70B 的均值按各自配对解读，
两组使用的程序数、并行方式和 KV 配置不同。

### 双副本路由

两张 H20 各运行一个 8B 副本，对同一批 12 个 ShareGPT 程序回放，提交速率为每秒
0.96 个程序。

| 调度与路由 | 平均程序响应（秒） | 输出 token/s |
| --- | ---: | ---: |
| 原生 SGLang | 28.21 | 148.00 |
| PLAS，程序亲和性 | 29.64 | 141.96 |
| PLAS，最少调用数 | 29.79 | 140.46 |
| PLAS，轮询 | 30.61 | 136.61 |

程序亲和性是三种 PLAS 路由中最快的一组，比轮询多输出 3.91% 的 token/s。
相对原生 SGLang，程序亲和性组的平均响应增加 5.05%，吞吐下降 4.08%。
