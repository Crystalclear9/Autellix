# 真实工作负载与指标

[文档导航](README.md) · [项目首页](../README.md)

本文档中的命令均从仓库根目录执行。

## 真实工作负载

[examples/real_workload.json](../examples/real_workload.json) 提供小型顺序和分支依赖示例，不是论文数据集：

```bash
python -m autellix.runtime.benchmark --backend vllm \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --workload examples/real_workload.json \
  --policies fcfs,mlfq,plas,atlas \
  --engine-args '{"max_model_len":512,"max_num_seqs":2,"gpu_memory_utilization":0.35,"swap_space":0.25}' \
  --output outputs/real/results.json
```

工作负载使用含真实文本提示的 JSON/JSONL，不能用只有 token 长度的模拟数据替代。`parents` 描述调用依赖；`arrival_time` 和 `think_time` 单位为秒；已有完整历史的提示可设 `append_parent_outputs=false`。多个 `--workload` 文件可组成混合负载。

`--arrival-rates 1,2,4 --programs 100 --seed 42` 可选地生成泊松到达流，不是使用本项目的必要步骤。结果包含响应时间、输出吞吐、TTFT、程序关键路径 token 延迟、配置、依赖版本和输入哈希。零输出程序的 token 延迟为 null。vLLM 原生对照模式见后端说明。

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
| 调用 | `input_ids` | 可选的固定输入 token；不追加父输出时直接提交，避免重复编码模板 |

DAG 在 benchmark 驱动侧控制何时提交调用；真实在线 ATLAS 调度器并不读取这些父节点计算精确 DAG 优先级。区分这一点有助于避免把驱动侧的依赖执行与调度器本身混为一谈。

| 指标 | 当前含义 |
| --- | --- |
| `program_latency_seconds` | 从程序计划到达到所有调用完成的响应时间分布 |
| `output_tokens_per_second` | 全部输出 token 数除以工作负载总耗时，包括到达间隔与外部等待 |
| `program_critical_path_seconds` | 沿依赖路径累计实测调用延迟与 `think_time` 的最大值 |
| `program_token_latencies` | 程序关键路径时间除以该程序所有线程总输出 token 数 |
| `arrival_horizon_seconds`、`drain_seconds` | 最后一次到达的时刻，以及此后排空队列的耗时 |
| `completions_during_arrivals` | 最后一次到达前已完成的程序数 |
| `first_submission_lag_seconds` | 第一次提交相对程序计划到达的延迟 |
| 调用 `latency_seconds` | 协调器创建 Future 到收到最终结果的耗时，不含此前同步分词 |
| 调用 `ttft_seconds` | 创建 Future 到收到第一段非空文本的耗时；不是逐 token GPU 内核时间 |
| `metrics.executed` | 已记录的模型执行服务时间；批次耗时分别计入参与调用，不按批大小均分 |
| `metrics.inherited` | 该调用接纳时继承的程序服务历史 |

benchmark 另行从 `submit()` 前开始计时，所以其关键路径调用延迟包括提交过程中的分词。后端、模型、提示模板、采样、窗口和缓存状态都会影响结果；比较性能时应同时记录这些配置。

## 最小输入示例

将下面内容保存到自己的 JSON 输入文件，或直接使用仓库的 [real_workload.json](../examples/real_workload.json)：

```json
[
  {
    "program_id": "demo",
    "arrival_time": 0,
    "calls": [
      {"call_id": "root", "prompt": "Name a fruit.", "max_tokens": 8},
      {"call_id": "describe", "parents": ["root"], "prompt": "Describe this fruit.", "think_time": 0.1, "max_tokens": 16}
    ]
  }
]
```

运行时创建的 session ID 与输入的 `program_id` 不必相同；结果的 `workload_program` 和 `workload_call` 用来关联原始数据。benchmark 对每种策略创建新的引擎并做短预热，然后执行同一输入或同一随机种子的采样 trace。比较结果时检查 `workload_sha256`、`engine_args`、版本、N/K 和模式，而不是只比较输出文件名。

结果文件顶层为记录列表，每完成一个到达率/策略组合就保存一次。中断后保留已完成记录，重跑剩余组合并合并结果。

## 原生 SGLang 配对对照

`--backend sglang --policies sglang-native,atlas --overprovision 0` 可在通用 benchmark
中选择无 Autellix 调度钩子的 SGLang 基线。`fcfs` 是保留 Autellix 统计开销的内部
策略，用于策略消融。P40 上双方使用相同的 `autellix_legacy_gpu=true` 配置。

`scripts/prepare_sharegpt_replay.py` 与 `scripts/compare_sglang.py` 提供可复核的
ShareGPT 子集回放和配对重复实验，保存原始结果及输出哈希；
`scripts/summarize_sglang_comparison.py` 拒绝缺失配对/重复，保留负收益和输出不一致标记。
正的 latency reduction / throughput gain 表示改善，负值表示退化；汇总保留全部配对。
源码与论文对应、数据设置和实测结果见 [论文核对与实验](autellix-paper-audit.md)。

## 完整对话与负载曲线

ShareGPT 原始文件的 `<对话ID>_<数字>` 后缀是消息偏移。准备脚本默认先合并同一
对话的片段，再按整段对话过滤上下文；重叠消息逐项核对。默认保留 1–80 次调用，
上下文 32768，输出长度不截短。`--source-records` 可复测旧的分片协议。

```bash
python scripts/prepare_sharegpt_replay.py \
  --source /mnt/masfiles/autellix/datasets/paper/sharegpt.json \
  --model /mnt/masfiles/autellix/models/Meta-Llama-3.1-8B-Instruct \
  --programs 128 --seed 42 --output outputs/sharegpt-whole128.json
python scripts/run_sglang_matrix.py --plan examples/p40-sharegpt-plan.json \
  --output outputs/load-curve.json
python scripts/summarize_sglang_matrix.py outputs/load-curve.json \
  --output outputs/load-curve-summary.json
```

计划中的模型、数据和设备路径可按机器修改。所有策略在同一到达率、重复编号下使用
相同 trace；每次重启引擎、预热后计时。该顺序回放保留参考历史、固定参考输出长度，
生成内容不反馈到下轮输入。用 `paper_sequential_token_latency` 比较程序响应秒数/输出
token 随程序到达率的变化；`output_tokens_per_second` 包含到达间隔和尾部排空时间。
这是两个不同指标。汇总分别报告输入/长度匹配和生成文本一致数。

CPU 合成负载按论文表校准平均调用数和 token 数，用于检查策略与参数。
对数正态分布的均值使用 `mu=-sigma²/2` 校正；sigma、裁剪阈值和模拟执行成本
由项目设定。其输出标记为 `measurement=simulation`，真实 GPU 输出为 `real_backend`。

## BFCL、LATS 与混合工作负载

`fetch_paper_sources.py` 固定 BFCL v3、LATS 源码及 HotpotQA 数据版本。
`collect_bfcl.py` 使用官方 Llama 工具提示和状态式模拟器，运行本地模型的多轮
工具循环；`collect_lats.py` 执行官方 MCTS，将每组并行采样和后续评价的依赖保存为 DAG。
两者记录实际提示、生成长度、工具/控制延迟和程序结束原因。

回放使用这些固定输入与长度，输出文本不改变后续工作量。混合采样先等概率选择
数据集，再在该数据集中选取完整程序。`summarize_paper_suite.py` 汇总配对结果、
到达窗口内完成量、排空时间和固定延迟目标对应的实测到达率。
参数与运行结果见[原始工作负载实验](paper-workload-experiment.md)，准备及重跑命令见
[服务器目录](server-layout.md)。
