# Autellix 论文复现尝试：源码核对与 GPU 实验

后续的完整对话恢复、BOS 修正、调度时间尺度消融和负载曲线见
[完整对话与负载实验](experiment-design-review.md)。本页保留上一轮协议与结果。
后续工作负载、修复和目录见[原始工作负载复现](paper-workload-experiment.md)与
[服务器目录](server-layout.md)。最近的 SGLang 和 vLLM 配对见 [H20 解码元数据融合复测](h20-reproduction.md#sglang-解码元数据融合复测)。
下文的服务进程、实现清单和差距对应 P40 验收时的版本。

复现目标：[Autellix arXiv:2502.13965v1](https://arxiv.org/html/2502.13965v1)。
本项目实现论文的程序调度方法，推理后端为 vLLM 0.6.1 和 SGLang 0.4.9.post6。
本轮在 3×Tesla P40 服务器检查源码、测量开销、修正实现并重新运行对照。

## 源码与论文对应

检查覆盖 runtime、两种后端、模拟器、客户端、HTTP、数据导入、实验工具、
启动脚本和回归测试。84 个 Python/启动脚本的列表及 SHA256 见[源码清单](results/p40-source-inventory.json)。
[机器可读实验汇总](results/p40-20261115.json)记录每组参数、指标、输出一致性、模型哈希和 trace 计数。

| 论文机制 | 源码入口 | 实现情况 |
| --- | --- | --- |
| PLAS 程序累计服务时间 | `runtime/policy.py` | 新调用继承已完成调用的服务时间；完成时累加真实模型执行秒数 |
| 在线 ATLAS，算法 1 | `runtime/policy.py` | 接纳时继承程序标量 h；完成时更新 max(program.service, h + executed) |
| 多级 FIFO、时间片降级 | `runtime/policy.py` | 按服务时间分箱，时间片耗尽后降级，并保留队列内 FIFO 顺序 |
| 程序级防饥饿 | `runtime/policy.py` | 根据历史等待/服务与当前局部窗口的比值提升；完整服务统计独立保存 |
| N 步调度 | `runtime/window.py`、`sglang/scheduler.py`、`vllm/paper.py` | 窗口边界选定请求，窗口内继续执行，提前结束由预备请求补位 |
| GPU 接纳与抢占 | `sglang/scheduler.py`、`pressure.py` | 钩子安装在实际 GPU scheduler 子进程，直接控制 running batch、waiting queue 和 KV 所有权 |
| KV 换出/恢复 | `sglang/swap.py`、`runtime/swap.py` | CUDA gather/stack 打包、锁页主机内存、整块传输与 scatter；目前使用 PyTorch kernels |
| 程序路由，算法 2 | `runtime/engine.py` | ≤2048 输入 token 选择未完成调用最少的副本；长输入保持程序亲和性 |
| 状态客户端、IPC | `runtime/client.py`、`engine.py`、`server.py` | 程序会话、调用/线程标记、独立副本进程、Future、流式响应和取消 |
| 模拟实验 | `core/`、`frontend/`、`experiments/` | 离散时间调度、显式 DAG、SRPT 与成本模型，和 GPU 执行路径分开 |

GPU 调度确实执行上述策略：请求选择在 CPU scheduler 上完成，选出的 batch 交给
真实 GPU 模型执行；KV 在 GPU 与主机间搬运。在线 ATLAS 使用论文算法 1 的程序
标量；公式 2 的显式父节点版本位于模拟器 `atlas-dag`。

实现上的主要差距：论文专用 CUDA 交换内核尚未实现；当前打包以请求为单位，
多个被抢占请求分别传输。多步窗口减少了优先级重算，但每 token 仍有元数据更新、
执行计时和 SQLite 写入。BFCL ReAct、LATS MCTS 和等比例 Mixed 的真实轨迹仍待接入。

## 本轮修正

1. **缓存前缀 attention**：固定版 SGLang 的 torch_native extend 为完整历史分配
   Query，然后丢弃前缀输出。恢复请求即使只续写 1 个 token，也会计算完整历史的
   attention。新增 `integrations/sglang/attention.py`，仅计算新增 Query，按前缀偏移
   构造因果遮罩。原生 SGLang 与 Autellix 共用这个修正。
2. **FP16 配置**：P40 默认仍用 FP32，显式 `dtype=float16` 现在会保留，用于装载
   Llama-3.1-8B；原代码会把用户设置覆盖为 FP32。
3. **数据导入**：显式 `parents=[]` 保留为并行根节点；原模拟数据导入器会把空列表
   当成缺省值，错误地串接到前一个调用。
4. **原长度回放**：`prepare_sharegpt_replay.py --output-cap 0 --min-calls 1 --max-calls 0`
   保留参考回答长度和选中记录的全部调用，取消此前 32 token 和 2–8 调用筛选。
5. **真实数据模拟实验**：修复 `make_paper_workload(dataset_path=...)` 忽略到达率和
   程序数的问题，按随机种子抽取完整程序、生成泊松到达，并为重复样本分配独立 ID。
6. **诊断入口**：新增 scheduler CPU profile、attention GPU 微基准，以及时间片比例
   `--quantum-scale`。这些入口用于定位计算、交换和调度各项成本。

## Figure 2 模拟核对

输入采用论文表中的 A=[4,3,1,1]、B=[3,3,4]、C=[1,2]、D=[4]，batch=2。
当前默认离散队列配置下，FCFS/MLFQ/PLAS 的总等待时间为 **18/19/14**；
论文图注为 **18/18/12**。FCFS 一致，另两项需要继续对齐队列边界、时间片及
同一时刻的到达/降级顺序。源码保留实际模拟结果，没有为匹配图注硬编码调度表。

## 旧实现负收益的原因

同一组 12 程序、48 调用、1532 输出 token、高到达率 1 程序/秒的 CPU profile：

| 项目 | 原生 SGLang | Autellix |
| --- | ---: | ---: |
| 调度进程总计 | 68.63 秒 | 93.70 秒 |
| 模型 decode 路径 | 61.37 秒 | 65.10 秒 |
| 模型 prefill/恢复路径 | 6.01 秒，35 批 | 19.68 秒，94 批 |
| KV save + restore | — | 4.35 秒，135 次换出/换入 |
| SQLite execute | — | 1.80 秒 |
| trace 写入 | — | 0.46 秒 |

上表为累计调用时间，父函数与子函数有包含关系。profile 用于开销分析，普通
性能对照关闭 profile。原始文件位于服务器 `audit-rerun/profiles-full/`。

主要原因依次是：

- **恢复计算放大**：频繁恢复把请求重新带入 extend，旧 attention 重新处理整个
  前缀。prefill/恢复路径多出约 13.67 秒，是这组总耗时差距的最大部分。
- **时间片与 P40 步长不匹配**：默认时间片 0.01–0.64 秒，N=8；本机一个窗口
  约数百毫秒，多个低级队列的时间片实际都在一个窗口内耗尽，退化为频繁轮转。
- **短输出削弱调度收益**：此前大部分请求只有 32 token，长短请求差异被压缩；
  同时保留了抢占、KV 交换和接纳成本。
- **额外控制开销**：135 次 KV 往返约 4.35 秒，SQLite 与 trace 合计约 2.26 秒。
  数据库有优化空间，但本次最大的计算问题在 attention 恢复路径。

## 已完成验证

- 本地回归：115 项，110 通过；5 项 GPU 用例在本地跳过。
- P40/SGLang：真实抢占与服务历史继承、CUDA KV 往返、两张物理 GPU 的路由和取消
  共 3 项通过；另 2 项为 vLLM 专用用例。
- HTTP：普通/流式输出一致，自定义会话关闭、自动会话历史继承、共享时间戳和状态清理通过。
- K=1 单独验证：预备请求驻留、窗口内补位、恢复生成及会话回收通过。
- attention：CPU 覆盖 GQA、因果/非因果、不同前缀长度及异长 batch 的非连续 KV 映射；
  GPU 数值比较最大绝对误差 1.35e-7。

P40 上 attention 局部微基准（FP32，4 个 Query head、1 个 KV head、head_dim=64）：

| 历史+新增长度 / 新增 Query | 旧实现 | 修正后 | 加速 |
| --- | ---: | ---: | ---: |
| 512 / 1 | 1.00 ms | 0.59 ms | 1.69× |
| 2048 / 1 | 4.70 ms | 0.59 ms | 7.90× |
| 2048 / 64 | 4.69 ms | 0.89 ms | 5.25× |
| 2048 / 2048 | 4.65 ms | 4.35 ms | 1.07× |

退出时仍有 SGLang 子进程 semaphore 清理警告；请求/会话回收与 GPU 释放测试通过。

## 服务器与论文实验条件

| 项目 | 当前实验 | 论文实验 |
| --- | --- | --- |
| GPU | Tesla P40 ×3，CC 6.1；单引擎比较独占一张卡 | A100 80GB，按模型使用 1/4/8 张卡 |
| 模型 | SmolLM2-135M FP32；Llama-3.1-8B-Instruct FP16 | Llama-3.1-8B/70B、Falcon-180B |
| 后端 | SGLang 0.4.9.post6 + 原生 PyTorch 算子 | 基于 vLLM 的 Autellix |
| CPU / RAM | Xeon E5-2680 v4，56 逻辑核、容器 18 核配额；72 GiB | 按论文服务器配置 |
| 环境 | Ubuntu 22.04.5，Python 3.10.12，Torch 2.7.1+cu126，驱动 550.120 | 论文固定实验环境 |

当前 P40 第 0 张卡可见 23040 MiB，第 1、2 张各 24576 MiB。
代码、环境、模型与日志保存在 `/mnt/masfiles/`；SQLite 运行状态放在 `/tmp/`。
8B 权重来自 [NousResearch 的 Llama-3.1-8B-Instruct 仓库](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct)，
revision 为 `d10aef7999a2b5ba950ab3974312feeedbfe0b77`；四个 safetensors 分片
已完整装载索引，共 291 个张量；四个文件的 SHA256 均与仓库 LFS 元数据一致。
使用未量化 FP16 权重。

## 数据与实验设置

数据来源：[ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)、
[BFCLv3](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/blob/main/BFCL_v3_multi_turn_base.json)、
[LATS](https://github.com/andyz245/LanguageAgentTreeSearch)。
ShareGPT 源文件有 94145 条记录，SHA256 为
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

| 工作负载 | 设置 |
| --- | --- |
| 原短回放 | seed=42，12 程序、48 调用、1532 输出 token，context=2048、输出最多32 token |
| 原长度回放 | SmolLM2-135M，seed=42，12 程序、29 调用、10916 输出 token，context=8192、无输出长度截断 |
| 论文模型回放 | Llama-3.1-8B-Instruct，seed=42，4 程序、9 调用、3423 输出 token，context=4096、无输出长度截断 |

回放按数据集参考历史构造下一轮提示；到达时间采用泊松过程。每个配对使用同一
GPU、模型、输入与输出长度，依次运行两种策略，重复时轮换顺序。新建引擎后做
相同预热。比较完整程序响应时间、程序级秒/token、P95 和输出吞吐。
源文件本身已经拆分对话；本次以其中一条完整记录作为程序。

`sglang-native` 路径保留上游原生 scheduler 和 KV 管理；`atlas` 安装程序调度钩子。
双方采用相同 P40 原生算子、PyTorch sampling、radix cache、FP32/FP16 配置，
overlap、CUDA Graph 与 chunked prefill 关闭。外层客户端、IPC 与会话封装相同。

## 历史结果

此前 6 组 K=0 配对的平均变化：

| 到达率 | 平均程序响应 | 程序级秒/token | 输出吞吐 |
| --- | ---: | ---: | ---: |
| 0.1 程序/秒 | +25.76% | +25.02% | -3.71% |
| 1.0 程序/秒 | +42.41% | +53.45% | -20.20% |

K=1 的单组补测：程序响应 +48.19%，吞吐 -16.24%。上述结果使用旧 attention。
原始结果保存在服务器 `/mnt/masfiles/autellix/runs/paper-20261115/`，
本地副本为 `outputs/archive/p40/paper-sglang-20261115/`。

## 本轮结果与后续修改

本轮结果保存在服务器 `/mnt/masfiles/autellix/runs/audit-rerun/`，本地归档为
`outputs/archive/p40/audit-results.tar.gz`，独有的补充结果位于同级的 `audit-rerun/`。
短回放结果如下，延迟变化为正表示更慢，吞吐变化为正表示更快。

| 实验 | 配对数 | 平均程序响应变化 | 程序级秒/token 变化 | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| 修正 attention，N=8、B=2、K=0 | 3 | +25.91% | +37.95% | −8.54% |
| 时间片与分箱边界 ×64，其他同上 | 1 | +28.08% | +47.23% | −2.85% |
| 原长度回放，N=8、B=4、K=1 | 1 | +26.62% | +36.31% | −5.53% |
| Llama-3.1-8B 原长度回放，N=8、B=2、K=0 | 1 | −4.30% | −16.97% | −2.70% |

三组普通时间片实验的吞吐变化范围为 −12.42% 至 −5.39%，响应变化范围为
+10.56% 至 +42.36%。每个配对的工作负载、输出 token 数、文本哈希、模型和
共享后端参数一致，修正前后的逐调用文本哈希也一致。

原长度回放的 SGLang/Autellix 吞吐为 **32.57/30.77 token/s**，平均程序响应为
**134.25/169.99 秒**；全部 29 次调用的输出一致。

8B 模型的 SGLang/Autellix 吞吐为 **9.65/9.39 token/s**，平均程序响应为
**227.83/218.04 秒**；P95 增加 2.90%。9 次调用的输入和输出长度一致，8 次
文本完全一致，`RwNX5pB_0/4` 从字符索引 1081 处开始不同。
对该 1791 输入 / 365 输出 token 请求的单独检查结果：

- 原生 SGLang 单独生成 = Autellix 单独生成。
- Autellix 强制抢占、KV 恢复后的完整输出 = 同引擎不中断生成。
- 原生 SGLang 单独生成 = 原 Autellix 回放，但与原生 SGLang 自己的并发回放不同。

因此这条输出差异指向 FP16 批次/前缀缓存计算路径的数值变化；直接抢占恢复检查通过。
8B 回放共执行 168 次 KV 换出/换入，平均 batch 为 1.37。收益集中在程序响应：
小程序更早结束，而额外交换使总吞吐略降。

普通时间片每次回放发生 116–130 次 KV 换出/换入；×64 后降为 33 次。
吞吐差距缩小但响应未改善，说明减少交换解决了部分开销，优先级分箱和完成顺序
仍需针对模型步长调整。下一轮应分别扫描时间片和分箱边界，而不是始终一起放大。

原长度回放共发生 **602 次 KV 往返、149 次预备请求驻留、13 次窗口内补位**，
平均执行 batch 为 3.24。12 个程序中只有 2 个响应变快。单调用程序
`aSGXnFA_0`（785 输出 token）从 89.71 秒变为 185.68 秒，
`5yOfC65_0`（692 token）从 78.18 秒变为 168.47 秒。
默认最大服务分箱边界只有 0.64 秒，P40 上一次完整调用往往运行数秒到数十秒；
长短调用很快落到同一低优先级队列，反复轮转推迟了原本较早完成的请求。
这解释了吞吐只差 5.53%，平均响应却差 26.62% 的现象。

后续修改顺序：

1. 按实际模型 batch 步长选择时间片，扫描 N、B、K，记录请求完成顺序和 P95；
   将时间片与服务历史分箱的尺度独立配置。
2. 在 KV 足够时保留已选中的低优先级运行请求，仅在接纳高优先级请求需要空间时
   释放；当前的运行后缀重接纳会引入额外 extend。
3. 将同一批被抢占请求的 KV gather 合并、复用锁页缓冲区，用一批传输替代逐请求
   save/restore；随后实现论文的专用 CUDA gather/scatter 内核。
4. 将共享状态发布集中到窗口边界和调用完成处，保留本地逐步计时，减少 SQLite 写入。
5. 接入 BFCL 工具执行与 LATS 搜索轨迹，加入等比例 Mixed；继续核对 Figure 2 的
   离散调度设置，并补齐多引擎负载曲线。

## 复跑

激活项目 SGLang 环境，设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`TORCHDYNAMO_DISABLE=1`、`OMP_NUM_THREADS=4`。

```bash
python scripts/prepare_sharegpt_replay.py --source /path/to/sharegpt.json \
  --model /path/to/model --output /path/to/replay.json \
  --context 8192 --output-cap 0 --min-calls 1 --max-calls 0
python scripts/compare_sglang.py --model /path/to/model \
  --workload /path/to/replay.json --output /path/to/results.json
python scripts/benchmark_prefix_attention.py --output /path/to/attention.json
```

设置 `AUTELLIX_PROFILE_DIR=/path/to/profiles` 启用调度进程 cProfile；删除该环境变量
恢复普通测量。`autellix_prefix_attention=false` 仅用于复测旧 attention。


启动论文 8B 模型的服务：

```bash
cd /mnt/masfiles/autellix/code
AUTELLIX_DEVICES=2 \
AUTELLIX_MODEL=/mnt/masfiles/autellix/models/Meta-Llama-3.1-8B-Instruct \
AUTELLIX_ENGINE_ARGS='{"autellix_legacy_gpu":true,"dtype":"float16","context_length":4096,"max_running_requests":2,"max_total_tokens":8192,"mem_fraction_static":0.9,"random_seed":0}' \
bash scripts/serve_p40.sh
```

该轮验收时的常驻服务为 **GPU 2 / Llama-3.1-8B-Instruct / FP16 / ATLAS / N=8 / K=1**，
PID 为 `22496`，状态目录 `/tmp/autellix-p40-Sk7dM2`，日志为
`/mnt/masfiles/autellix/runs/audit-rerun/server.log`。
三路并发 HTTP 请求完成并回收会话，法国首都与加法问题分别返回 `Paris` 和 `5`。

服务监听 `127.0.0.1:8000`。本机使用 `ssh -p 46644 -L 18000:127.0.0.1:8000 root@cn-north-b.ssh.damodel.com`
建立隧道后，客户端地址为 `http://127.0.0.1:18000`。
