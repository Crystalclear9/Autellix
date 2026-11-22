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

同一份 Qwen 轨迹也可以直接生成 vLLM 与原生 SGLang 的对照计划。先在
`autellix-vllm061` 环境测原生 vLLM-opt 和 PLAS，再切到 `autellix-sglang` 环境测原生
SGLang。两个环境都读取共享目录中的相同权重。每组只用一张卡，保留 BF16、
32,768-token 上下文、8 个活动请求、2,048-token prefill 分块和 65,536-token KV 容量。

```bash
comparison="$data/runs/qwen32-comparison"
python scripts/prepare_h20_qwen32.py --model "$model" --comparison-inputs "$out" \
  --run-dir "$comparison" --device "$gpu"
for family in bfcl sharegpt; do
  python scripts/run_sglang_matrix.py --plan "$comparison/qwen32-$family-vllm-plan.json" \
    --output "$comparison/qwen32-$family-vllm.json" --resume
done
# 切换至 autellix-sglang 环境并清除 vLLM 专用的 LD_PRELOAD 后运行。
for family in bfcl sharegpt; do
  python scripts/run_sglang_matrix.py --plan "$comparison/qwen32-$family-sglang-plan.json" \
    --output "$comparison/qwen32-$family-sglang.json" --resume
done
```

BFCL 按泊松到达回放，比较程序级毫秒/token、完成率和排空时间。ShareGPT 所有程序在
t=0 提交，比较整批耗时和尾延迟。对照使用同一个随机种子和请求顺序，按 trace 哈希与
逐调用 token 数核对工作量。关闭调度轨迹的测量会在引擎退出时清理临时 SQLite 状态。
需要测 BFCL 积压时，可在新目录生成 `--bfcl-rates .24 .60 --resident-decode-headroom`
计划。该选项给已选中且驻留的 decode 留出一步 token，长提示仍可分块进入同一次前向。

在新目录生成带 `--exact-prefix-cache --qwen3-fused-qk-rope` 的计划，可以让两个 vLLM
对照共用精确的前缀 KV 状态和 Qwen3 融合算子。全上下文的 PLAS 分块 prefill 默认使用
精确 KV 状态。原生 vLLM-opt 的上游配置用无上述两项参数的计划执行，原生 SGLang
沿用同一程序集合和批次预算。
ShareGPT 的 24 程序复测使用 `--comparison-programs 24 --comparison-seed 2029`；
从已有的 128 程序轨迹取样，与默认种子下的 16 个程序互不重叠。
默认 CUDA Graph 捕获范围为 ShareGPT 8,192 token 和 BFCL 32,768 token，可用
`--graph-context` 按同一轨迹对照。两个工作负载的模型上下文上限均为 32,768 token。

同一张卡依次运行匹配 Triton 配置的 SGLang、Autellix PLAS 或 ATLAS，以及采用上游默认优化的
SGLang。功能验收使用 `check_prefill_priority_gpu.py --modern --mem-fraction-static .88`，
分别验证 KV 换入换出与 `--host-kv-recompute`，检查恢复生成和连续生成的输出是否一致。

优化版补测使用 BFCL 12 程序、到达率 0.48、活动批次上限 8、KV 上限 65,536 token。Autellix
使用倍率 16、可回收的 radix KV、窗口发布和空位补齐。原生 SGLang 与 Autellix 分别达到
73.06 和 70.20 token/s，平均程序级延迟分别为 82.94 和 85.94 毫秒/token。优化版吞吐低
3.9%，平均延迟高 3.6%。P95 分别为 110.43 和 132.09 毫秒/token。

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
| §5 聚合 KV 交换 | [`PackedKVLayout`](../autellix/runtime/packed_kv.py) 接入两个后端的真实 KV 交换。每个载荷用一个 Triton kernel 完成跨层 GPU 打包或解包，再通过锁页缓冲区批量传输。此前的 PyTorch 路径保留作消融，其 H20 8–512 块往返微基准较原生 vLLM 快 1.14–1.30 倍。 |
| §6 实验覆盖 | 8B 四类在线负载共 60 个 vLLM 实验，70B 四卡张量并行共 20 个；另有多副本、耗时拆分、KV 微基准和 SRPT 模拟。8B 离线 128 完成全组，1,000 与 4,000 有部分组合完成。论文使用 A100，本轮使用 H20，Falcon-180B 尚未运行。 |

这是一次 Autellix 论文复现尝试。核心调度、路由和 KV 交换均接入了真实推理。实验中的
`vllm-opt` 使用分块 prefill 与前缀缓存，`vllm-opt-multistep` 使用前缀缓存与原生多步，
分开覆盖论文 vLLM-opt 的三项优化。回放使用固定参考输出长度；本地 LATS 平均 320 次调用，
论文为 159.7 次。论文在线吞吐比较固定程序级延迟下可承载的程序到达率，下表报告固定到达率的
输出 token/s 与程序级延迟。 [Autellix 论文，§4–§6](https://arxiv.org/html/2502.13965v1#S4)

## 按论文图号运行单卡实验

`scripts/run_h20_paper_suite.py` 将实卡测量、模拟和每个配置的检查点组织在同一目录。它只接受
一个物理卡号，逐组启动和关闭引擎。vLLM 和 SGLang 使用各自的 Python 环境，共用模型权重和
准备好的轨迹。`--plan-only` 生成清单和配置，`--figures`、`--cases` 与 `--offline-sizes`
选择本次执行的图和对照组。

| 原论文 | 单卡工作流 |
| --- | --- |
| 图 4–6，负载动机 | 一小时 ShareGPT 到达记录、活动调用数、不同负载的排队占比，并按调用输出 token 数和每程序调用数统计阻塞 |
| 图 7，前缀局部性 | 从 ShareGPT 和 LATS 的实际输入 token 逐块计算同程序及跨程序前缀复用机会 |
| 图 11，工作负载分布 | 从 ShareGPT、BFCL、LATS 与 Mixed 回放提取程序长度与调用统计 |
| 图 12、13，在线均值与尾延迟 | 四类负载的泊松到达率扫描，包含原生 vLLM、两种 vLLM-opt、同路径 FCFS、MLFQ、Autellix，以及原生 SGLang 和它的接入版。汇总同时记录到达窗口完成率和固定延迟目标内完成率 |
| 图 14、15，多副本路由与扩展 | 原数据的多副本模拟。论文的多物理 GPU 性能依赖各副本拥有独立计算资源 |
| 图 16，离线 makespan | 从同一 4,000 程序池抽取 1,000、2,000、3,000 和 4,000 个不同程序，每组同时提交，记录整批完成时间。对照包含三项组合优化的 FCFS、原生 vLLM、原生 SGLang、MLFQ 和两种 Autellix 接入。关闭 SQLite 的瞬时活动发布，调度和完成统计继续更新 |
| 图 17，开销拆分 | vLLM-opt、MLFQ 和 Autellix 分别测量 ShareGPT、LATS 的模型、调度、KV 搬运和请求累计排队时间，按实际处理窗口排除预热 |
| 图 18，最优调度对照 | ShareGPT 与 LATS 的等步长模拟，含可提前获知剩余服务量的 SRPT |
| 图 19，KV 交换 | 跨层 fused 打包与解包，测原生 vLLM 搬运、PyTorch 聚合和 fused 路径。推理消融同时记录逻辑换出、主机与设备之间的 KV 载荷传输次数及耗时，并比较 N=1 窗口 |

SGLang 单步预填充预算耗尽后保留已选请求与 decode，GPU KV 或请求槽位耗尽时按优先级收缩窗口。
PLAS 与 ATLAS 在窗口边界只读取已累计服务或等待时间的程序，其余程序使用零历史。

图 12、13 保留 H20 的低负载测点，并扫描论文 8B 图中的高到达率区间。ShareGPT、LATS 和
Mixed 的默认上限分别为 4、0.30 和 0.80 程序/秒。可用 `--rates` 指定测点，用
`--online-programs` 增加每条到达轨迹的程序数。配对各组使用同一轨迹和随机种子。

图 12 的 Llama-3.1-70B 和 Falcon-180B 原配置分别使用 4 张和 8 张 A100。图 14 的副本组数
分别为 4 和 2，图 15 比较独立引擎扩容。上述 GPU 结果与单卡 8B、CPU 模拟分别汇总。
[Autellix 论文，第 6 节](https://arxiv.org/html/2502.13965v1#S6)

每组完成后直接保存原始结果和续跑检查点。跨目录核对时，`summarize_paper_suite.py --matrix-only`
以轨迹哈希、逐调用 token 数及共同的 GPU、上下文、批次和 KV 容量核对配对，再计算
makespan、程序完成率和归一化程序延迟。

单卡矩阵的失败尝试保存在 `*-failures.json`。显卡空闲时继续执行其余组，
下次续跑从尚未产生成功记录的组恢复。计时汇总的退出码单独写入 `exit-codes.json`。

图 16 的 vLLM Autellix 可用 `--recompute-on-swap-full` 补跑主机 KV 空间耗尽的配置。
主机块充足时沿用换出，空间耗尽时释放被抢占请求的 GPU KV，并在再次调度时按已有 token 重建。
补跑沿用原轨迹、GPU KV 容量和批次上限，使用独立输出目录保留原始失败记录。

ShareGPT 的 4,000 条回放含 23,575 次调用，平均输出 279 token、带完整历史的平均输入
2,962 token。BFCL 轨迹保留每次输入的工具定义和上下文。现有 LATS 轨迹来自六个
HotpotQA 程序，平均 320 次调用，论文报告 159.7 次。输入长度与程序调用数采用图 11
的工作负载清单核对，结果保留对应轨迹版本。

```bash
python scripts/run_h20_paper_suite.py --model "$model" --datasets "$data/datasets" --gpu "$gpu" \
  --run-dir "$data/runs/paper-replay" --vllm-python "$vp" --sglang-python "$sp" \
  --vllm-preload "$lib/libcublas.so.12:$lib/libcublasLt.so.12"
```

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

### BFCL 同卡消融

在 H20 3 号卡上顺序运行 BFCL 1.92 程序/秒的同一条轨迹。每组处理 48 个程序，生成
27,118 个 token，活动批次上限 16。SGLang 的 `max_total_tokens=262144`，vLLM 使用
16,384 个 16-token GPU 块。Autellix 的两种保留 KV 配置都使用 N=8、K=1 和相同的时间片。

| 配置 | 输出 token/s | 平均毫秒/token | P95 毫秒/token | KV 换出 |
| --- | ---: | ---: | ---: | ---: |
| 原生 SGLang 默认 | 346.03 | 50.53 | 74.91 | — |
| 原生 SGLang，匹配 Triton 配置 | 347.87 | 43.56 | 59.64 | — |
| 原生 vLLM-opt，eager | 323.73 | 61.56 | 96.49 | — |
| 原生 vLLM-opt，eager 加观测钩子 | 313.03 | 63.56 | 101.48 | — |
| 原生 vLLM-opt，启用 CUDA Graph | 319.67 | 63.33 | 98.56 | — |
| Autellix 执行路径，FCFS，保留暂停 KV | 310.80 | 64.24 | 99.04 | 0 |
| Autellix PLAS，保留暂停 KV | 292.92 | 61.04 | 114.28 | 0 |
| Autellix PLAS，换出窗口外 KV | 42.45 | 1,333.13 | 1,889.48 | 1,337 |

原生 vLLM-opt eager 比原生默认 SGLang 少输出 6.4% 的 token/s。保持 vLLM 的原生调度，增加
包括 CUDA event 同步、状态更新和轨迹记录在内的观测钩子后，再少 3.3%。单独启用 CUDA Graph
时吞吐变化为 −1.3%。

在同一 Autellix 执行路径上，PLAS 与 FCFS 均未换出 KV。PLAS 的平均活动批次从 6.87 降至 5.84，
执行批次从 3,947 增至 4,642，吞吐低 5.8%。平均程序级延迟降约 5.0%，P95 升约 15.4%。这个配置
下 PLAS 的平均程序级延迟接近原生 vLLM-opt eager，整批仍更慢。与原生默认 SGLang 相比，
PLAS 保留 KV 组的吞吐低 15.3%，平均程序级延迟高 20.8%。

只切换窗口外 KV 的处理方式，整批时间从 92.58 秒增至 638.82 秒，记录到 1,337 次换出和恢复、
3,301 次优先级提升。CUDA event 累计模型执行时间为 76.34 和 77.55 秒。新增时间主要落在模型执行
之外，涉及换入换出、调度与排队。八组的输入哈希和输出 token 数已逐组核对。计划、结果和运行日志
保存在 `data/autellix/runs/paper-vllm-20261119/bfcl-aligned-20261121.tar.gz`。

### 改进后的单卡回放

Llama-3.1-8B BF16、H20 3 号卡、活动批次上限 16。SGLang 的 KV 容量为 262,144 token。
优化组将暂停请求的 KV 交给可回收的 GPU radix 缓存，在窗口边界发布 SQLite 活动视图，
启用原生 attention 与 CUDA Graph，并将时间片倍率从 2 提至 8。原生组启用上游默认优化。

| 负载（程序/秒） | 原生 SGLang（token/s） | 优化后 Autellix（token/s） | 吞吐变化 | 平均毫秒/token，原生 / Autellix |
| --- | ---: | ---: | ---: | ---: |
| BFCL 1.92 | 349.13 | 301.90 | −13.5% | 50.04 / 53.05 |
| ShareGPT 1.92 | 448.28 | 408.53 | −8.9% | 12.29 / 14.50 |
| LATS 0.28 | 682.57 | 621.30 | −9.0% | 7.50 / 8.30 |
| Mixed 0.48 | 545.64 | 499.15 | −8.5% | 11.16 / 12.47 |
| ShareGPT 离线 128 程序 | 692.50 | 614.50 | −11.3% | 65.75 / 72.26 |

同轮 BFCL 的旧 Autellix 配置为 159.55 token/s，发生 1,011 次主机 KV 换出。启用 radix
保留和窗口写入后主机换出归零，吞吐升至 215.59 token/s。时间片倍率和执行配置的联合调整
把吞吐进一步推至 301.90 token/s。即使主机换出归零，本轮 BFCL 的计划中仍有 529 次撤回
发生在窗口内已有的请求上，其中包含候补转换。这是继续检查 prefill 容量估计和批次形状的依据。
vLLM 的配对保留相同的 KV 与策略配置，只调整活动视图写入频率，吞吐由 293.35 升至 302.15 token/s。

本地 Figure 2 模拟用相同的四个程序和双槽批次，FCFS、MLFQ、PLAS 的累计等待分别是 18、19、14，
论文示意为 18、18、12。两组均观察到 PLAS 降低累计等待。[Autellix 论文，图 2](https://arxiv.org/html/2502.13965v1#S1)

在 `autellix-sglang` 环境中运行下面的命令，可在 BFCL 上回放启用补齐的版本，并保留原生对照。
`model`、`data`、`gpu` 沿用前面的单卡设置。

```bash
python scripts/run_h20_paper.py bfcl --model "$model" --datasets "$data/datasets" \
  --run-dir "$data/runs/h20-bfcl-refill" --device "$gpu" --programs 48 --rates 1.92 \
  --native-default --host-kv-gib 32 --host-kv-recompute --quantum-scale 8 \
  --cache-paused-kv --publish-activity-at-window --refill-idle-slots --optimized-kernels
```

在第二轮，prefill 按未缓存 token 核算预算，GPU KV 容量仍按完整前缀核算。窗口内的空位在
候补耗尽后才接纳新调用。BFCL 1.92 用相同的时间片倍率 8，`paper-fast` 的 617 次撤回降为
191 次；时间片倍率 32 时为 175 次。倍率 32 的 BFCL 对照如下。

| 到达率与重复 | token/s，原生 / Autellix | 平均毫秒/token，原生 / Autellix | 平均延迟变化 |
| --- | ---: | ---: | ---: |
| 1.92，首次 | 349.13 / 312.49 | 50.04 / 46.16 | −7.7% |
| 1.92，反向执行顺序 | 351.41 / 311.06 | 48.91 / 47.57 | −2.7% |
| 3.84，首次 | 346.17 / 310.49 | 74.93 / 69.67 | −7.0% |
| 3.84，反向执行顺序 | 345.60 / 309.15 | 75.32 / 68.80 | −8.6% |
| 1.92，种子 2027 | 350.53 / 318.31 | 75.79 / 87.38 | +15.3% |

默认种子 2026 的高负载两次回放中，Autellix 的平均程序级延迟都低于 70 毫秒/token，原生
SGLang 的均值均在 74 毫秒/token 以上。种子 2027 改变了所选程序和泊松到达序列。在 1.92
程序/秒下，换种子后 Autellix 的撤回从 166 次增至 253 次，按总输出长度排在最短四分之一的
程序由平均 96.84 升至 128.02 毫秒/token。同一类短程序在默认种子的反向复测中由 59.38
降至 43.82 毫秒/token。调度收益对输入程序和到达序列敏感。

同一 BFCL 轨迹、时间片倍率 32，分开执行 prefill 和 decode 得到 299.54 token/s、56.29
毫秒/token，保留混合执行时分别为 312.49 和 46.16。vLLM 在窗口发布活动视图的基础上加入
空位补齐，平均批次由 5.99 增至 6.41，吞吐由 302.15 增至 306.65 token/s。

同样启用空位补齐后，ShareGPT、LATS 与 Mixed 的在线吞吐分别为 403.35、629.67 和 501.26
token/s，较各自原生对照低 10.0%、7.8% 和 8.1%，平均程序级延迟分别增加 14.3%、9.0%
和 8.9%。

离线 128 程序中，在同一输入上，倍率 8 的补齐组为 613.45 token/s、平均 70.03 毫秒/token、
P95 98.40 毫秒/token；倍率 32 分别为 619.31、68.10 和 103.46。原生 SGLang 是
692.50 token/s、平均 65.75、P95 168.97 毫秒/token。两个补齐组的 P95 下降 38.8%–41.8%，
整批吞吐仍低 10.6%–11.4%。

单卡复测包含 33 个性能配置，GPU 验收涵盖优先级抢占、KV 换回、缓存溢出重算、生成结果一致性及
HTTP 调用。每组对照核对了输入哈希、程序数和输出 token 数。实验计划、日志和调度轨迹保存在
`data/autellix/runs/paper-vllm-20261119/single-card-recheck-20261121.tar.gz`。

### 单卡论文基线对照

固定同一张 H20、Llama-3.1-8B BF16、活动批次上限 16 与 262,144 token 的 GPU KV 容量。
BFCL 与 ShareGPT 使用 PLAS，LATS 与 Mixed 使用 ATLAS。候选的调度窗口为 N=8，预备槽 K=1，
vLLM 同引擎对照共用 CUDA Graph 和同步输出，关闭逐步 JSONL 轨迹。SGLang 原生组沿用上游默认优化。

vLLM 0.6.1 将前缀缓存加分块 prefill，以及前缀缓存加原生多步，作为两个原生配置测量。
Autellix 在自己的 N 步窗口内复用 vLLM 的分块预算，长输入可在多个迭代中完成 prefill。
解码窗口已准备就绪时，仅物化当前批次。显存不足时，把为同一调用腾出的多个暂停请求合为一次换出。
[vLLM 0.6.1 参数检查](https://github.com/vllm-project/vllm/blob/v0.6.1/vllm/engine/arg_utils.py#L900)

在 `autellix-vllm061` 环境中按上文预载 cuBLAS，以下命令回放 BFCL 的四个对照配置和 Autellix。

```bash
python scripts/run_paper_vllm.py online bfcl --model "$model" --datasets "$data/datasets" \
  --run-dir "$data/runs/bfcl-single-card-baseline" --device "$gpu" --programs 48 --rates 1.92 3.84 \
  --batch-size 16 --kv-blocks 16384 --interval 8 --quantum-scale 8 --boundary-scale 16 \
  --keep-resident-kv --publish-activity-at-window --refill-idle-slots --paper-chunked-prefill \
  --cuda-graphs --no-scheduler-trace
```

汇总分别记录到达率、到达窗口的完成率、整批完成率与排空时间。同一档位的平均程序级延迟需在每次
已执行重复中都满足目标，才计入该目标的最高已测到达率。延迟统计保留均值、P95 和 P99。
靠近单卡完成率的 BFCL 对照使用 `--rates .48 .72 --batch-tokens 2048 --cases vllm-opt autellix`，
单独指定运行目录，两组均按 2,048 token 的单步预算分块。

#### BFCL 在同一 vLLM 引擎上的对照

| 随机种子与到达率（程序/秒） | vLLM-opt / Autellix 吞吐（token/s） | vLLM-opt / Autellix 平均延迟（毫秒/token） | vLLM-opt / Autellix P95（毫秒/token） |
| --- | ---: | ---: | ---: |
| 2026，1.92 | 319.66 / 302.21 | 62.83 / 56.29 | 97.24 / 92.26 |
| 2026，3.84 | 320.40 / 305.37 | 87.96 / 80.03 | 134.30 / 149.88 |
| 2027，1.92 | 304.62 / 295.98 | 93.73 / 97.27 | 135.64 / 168.03 |
| 2027，3.84 | 303.96 / 293.30 | 111.52 / 115.20 | 175.27 / 176.18 |

默认原生 vLLM 在种子 2026 的两档 BFCL 压测分别为 46.76、46.74 token/s，平均 843.52、
864.11 毫秒/token。启用前缀缓存和分块后，平均延迟降至 62.83、87.96 毫秒/token。
同执行路径的 MLFQ 在这两档为 55.93、80.62 毫秒/token，与 PLAS 的均值接近。
种子 2026 的 Autellix 整批完成率是 0.535、0.541 程序/秒，压测时仍有明显积压。

原生多步配置在 1.92 的完整轨迹上复跑两次。两次均生成 27,118 token，与分块版 vLLM-opt
逐调用文本完全一致的分别只有 66/533 和 65/533。3.84 的原生多步组则为 458/533。
六条长输入逐条及成批生成的复核中，多步加 CUDA Graph 有 9/12 份输出与分块版逐 token
一致。改用 eager 或关闭前缀缓存后分别为 11/12。主表使用开启分块和前缀缓存的 vLLM-opt。

#### BFCL 与原生 SGLang 的直接对照

SGLang 接入在每次预算检查中按 radix 的实际锁定路径计算空间，多个候选共享的前缀只计一次。
以下使用时间片倍率 32、同一 BFCL 轨迹和 CUDA Graph。

| 随机种子与到达率（程序/秒） | 原生 SGLang / Autellix 吞吐（token/s） | 原生 SGLang / Autellix 平均延迟（毫秒/token） |
| --- | ---: | ---: |
| 2026，1.92 | 348.85 / 319.27 | 49.87 / 47.49 |
| 2026，3.84 | 345.66 / 316.64 | 75.26 / 71.21 |
| 2027，1.92 | 350.63 / 318.11 | 75.83 / 88.45 |
| 2027，3.84 | 348.32 / 318.11 | 94.75 / 107.55 |

在 1.92 程序/秒下，按输出 token 数排在最短四分之一的程序，默认种子的平均延迟从 59.67
降到 45.04 毫秒/token，换样后从 96.62 升到 124.91。换样组只发生一次主机 KV 换出，
48 个程序中有 10 个较原生 SGLang 更早完成。减少主机换出以后，程序组合、批次利用率与调度
次序仍决定程序级延迟的变化。

#### BFCL 接近单卡容量的回放

| 后端与到达率（程序/秒） | 原生 / Autellix 实际完成率（程序/秒） | 原生 / Autellix 平均延迟（毫秒/token） | 原生 / Autellix 排空（秒） |
| --- | ---: | ---: | ---: |
| vLLM-opt，0.48 | 0.385 / 0.384 | 17.86 / 19.82 | 2.95 / 3.23 |
| vLLM-opt，0.72 | 0.521 / 0.462 | 27.30 / 28.09 | 11.03 / 22.65 |
| SGLang，0.48 | 0.389 / 0.388 | 13.73 / 15.33 | 1.66 / 2.03 |
| SGLang，0.72 | 0.557 / 0.543 | 20.30 / 22.78 | 5.03 / 7.14 |

#### 其他在线工作负载

表中每个单元按原生 / Autellix 列出。同一工作负载的两种后端使用相同回放轨迹。

| 工作负载（程序/秒） | vLLM-opt 吞吐（token/s） | vLLM-opt 平均延迟（毫秒/token） | SGLang 吞吐（token/s） | SGLang 平均延迟（毫秒/token） |
| --- | ---: | ---: | ---: | ---: |
| ShareGPT，1.92 | 360.18 / 356.80 | 18.51 / 16.55 | 448.02 / 402.56 | 12.23 / 13.52 |
| LATS，0.28 | 581.18 / 574.53 | 8.90 / 9.00 | 685.18 / 633.16 | 7.46 / 8.12 |
| Mixed，0.48 | 451.41 / 447.10 | 14.19 / 14.44 | 544.55 / 498.90 | 11.18 / 12.02 |

#### ShareGPT 离线 128 程序

128 条对话各回放一次，共生成 221,779 token。同引擎对照沿用上述 vLLM 配置。

| 配置 | 整批耗时（秒） | 吞吐（token/s） | 平均延迟（毫秒/token） | P95（毫秒/token） | P99（毫秒/token） |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM-opt | 364.82 | 607.91 | 89.02 | 225.19 | 398.63 |
| Autellix PLAS，vLLM | 377.00 | 588.27 | 73.41 | 98.97 | 207.79 |
| 原生 SGLang，同卡上轮 | 320.26 | 692.50 | 65.75 | 168.97 | 296.73 |

PLAS 相对 vLLM-opt 的均值、P95、P99 分别下降 17.5%、56.0%、47.9%，整批耗时增加 3.3%。
上轮原生 SGLang 对照与本轮输入哈希均为 `e82e74c90b…`。本轮 vLLM 接入的 P95 较该组低
41.4%，整批用时多 56.74 秒。

#### Qwen3-32B BFCL 复测

复用 `stu/ljh/models/Qwen3-32B` 和同一组 12 程序。原生 SGLang 与接入版分别达到 73.02 和
69.72 token/s。平均程序级延迟为 83.15 和 86.76 毫秒/token，P95 为 110.88 和 135.46。

#### 论文的 GPU 驻留候补

`--resident-reserve-only` 从已在 GPU 上持有 KV 的请求里挑选候补。相同的活动批次上限、优先级、
窗口长度和空槽补齐设置用于两种候补策略，分别在 vLLM 和 SGLang 上用两个 BFCL 种子复测。
[Autellix 第 4.2.2 节](https://arxiv.org/html/2502.13965v1#S4)

八组配对中，vLLM 吞吐变化为 −0.6% 至 +1.0%，平均延迟变化为 −0.5% 至 +1.3%。SGLang
吞吐变化为 −0.9% 至 +1.2%，平均延迟变化为 −2.7% 至 +2.9%。在 3.84 程序/秒下，驻留
候补版 SGLang 相对原生的平均延迟为 69.28 / 75.26 毫秒/token（种子 2026）和
106.70 / 94.75 毫秒/token（种子 2027）。

复现对照使用 vLLM 的 Autellix、MLFQ 和分块版 vLLM-opt。需要按论文限定候补范围时，在上述
命令加入 `--resident-reserve-only` 并使用独立运行目录。跨引擎对照使用原生 SGLang 的默认
优化。SGLang 的 Autellix 路径关闭了 overlap 调度和分块 prefill，保留 CUDA Graph。

单卡性能测试共 55 组，按轨迹核对了输入哈希及逐调用输入和输出长度。真实 GPU 验收通过了
vLLM 分块与整段生成一致性、抢占恢复与服务历史继承、SGLang 暂停 KV 和溢出重算，以及批量 CUDA
KV 往返。计划、日志与原始数据保存在 `data/autellix/runs/paper-vllm-20261119/single-card-paper-baseline-20261121.tar.gz`。

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
