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

结果文件顶层为记录列表，每个记录对应一个到达率/策略组合。中途失败时可能只保留此前完成的组合；记录数少于计划组合数不表示其余组合已经通过。现有 benchmark 不自动恢复中断实验。
