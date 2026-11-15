# Autellix v1 方法核对与 SGLang 对照实验

审计对象：[Autellix arXiv:2502.13965v1](https://arxiv.org/html/2502.13965v1)，
2025-02-19 版本。后续以 Agentix 名称发表的版本不能自动视为本仓库的复现目标。
本文区分方法实现、功能测试、性能验证和完整论文复现；四者不是同一结论。

## 对照论文的结论

**尚未完整复现。** 本项目是核心方法的独立实现，SGLang/P40 是额外扩展。
截至本次实验前，只建立了小模型功能基线，没有建立相对于原生 SGLang 的收益结论。
以前的 `fcfs` 是经过 Autellix 钩子的 FCFS，不能作为原生 SGLang 基线。

| 论文部分 | 本仓库现状 | 证据与边界 |
| --- | --- | --- |
| §4.2 PLAS | 按程序累计完成调用服务时间 | 已实现并有单元测试；不是作者代码 |
| §4.2 在线 ATLAS / 算法 1 | 继承程序标量，完成时取 max(inherited + executed) | 符合在线标量方案；不是精确父节点 DAG 调度，不能混称公式 2 的逐父节点实现 |
| 多级 FIFO、降级、防饥饿 | 已实现，局部计时重置与生命周期统计分开 | 有回归测试；beta、队列阈值、时间片为本项目选择，未核实为作者实验参数 |
| 多步调度和预备请求 | 有窗口冻结、准备和补位 | N=8/K=1 小模型验证通过；逐 token 元数据、CUDA 同步和 SQLite 更新仍存在 |
| KV 批量交换 | PyTorch gather/stack、锁页 CPU 内存和传输 | 验证了恢复一致性；未复现作者自定义 CUDA 内核及其带宽/消融结果 |
| §4.3 路由 | 2048 token 阈值、短请求最少未完成调用数、长请求亲和 | 有同卡双副本测试；跨 GPU 吞吐、负载偏斜和数据局部性收益未验证 |
| §5 系统实现 | IPC 副本协调、有状态客户端、SQLite 进程表 | 机制相近但实现不同；SQLite 为同机协调，不是分布式集群服务 |
| vLLM 0.6.1 实现 | 有独立适配及历史小模型测试 | 本轮 P40 测试的是 SGLang，不是原论文后端实验 |
| §6 模型/硬件 | 论文：LLaMA-3.1-8B/70B、Falcon-180B；A100 80GB/NVLink | 本机 3×P40；当前 FP32 单卡路径无法容纳 8B（仅权重约 32GB），70B/180B 更不可行 |
| ShareGPT | 获取了公开原始数据，回放选中记录的全部轮次 | 不是作者原始 trace；源数据已 split，未重建切分前的整段对话 |
| BFCLv3 | 已下载官方组织的多轮 base 题目文件 | 尚无论文等价的 ReAct/工具调用执行轨迹；题目回放不算 BFCL 工具任务复现 |
| LATS/HotpotQA | 核对了 LATS 官方代码仓库 | 问答数据不等于 MCTS 调用 DAG；本次未取得作者实验轨迹，也未运行真实 LATS |
| Mixed | 未完成 | 缺 BFCL/LATS 轨迹时，不能用任意拼接的提示声称复现等比例混合 |
| §6 性能实验 | 新增受限 SGLang 对照，见实验结果 | 尚未复现论文单/多引擎、尾延迟、扩展性、交换内核及 SRPT 消融 |

原模型访问检查：尝试 Llama-3.1-8B-Instruct 候选配置时，服务器访问 Hugging Face
官方地址超时，镜像返回 403。论文的模型名称不等于已确定的 checkpoint/revision；
本次也未确认作者使用的确切权重版本。
未获取/绕过受限权重。本次继续使用已验证的 SmolLM2-135M-Instruct，明确作为替代模型。
使用半精度、量化、CPU offload 或 TP 都会改变现有兼容路径，不能未经验证直接采用。

## 新增的 SGLang 对照方式

`autellix_native_baseline=true` 保留固定版本 SGLang 原生 scheduler、请求接纳和
KV 管理，不安装 `integrations.sglang.scheduler.install`。P40 必需的模型算子替换
双方相同。客户端、IPC、分词和外层会话记账保持一致，比较的是这个共同封装内
的推理调度路径，而非原生 SGLang HTTP 产品端到端性能。

这仍不是原封不动的上游优化性能：双方都使用 FP32、torch_native attention、
PyTorch sampling，关闭 CUDA Graph、overlap、chunked prefill，并开启 radix cache。
原生基线使用 SGLang FCFS。本轮隔离调度策略，双方活动容量 B=2、K=0、KV 预算
4096 token；Autellix N=8。K=0 与当前常驻服务 K=1 有区别，不能把结果直接推广至
所有配置。论文的主要基线是 vLLM/vLLM-opt/MLFQ；用户要求的直接 SGLang 对照
属于扩展实验。

另在主实验完成后串行补测一组 `--rates 1.0 --repeats 1 --reserve 1`，检验
服务默认 N=8/K=1 结构。该组仍使用 2048 上下文/4096 KV token 的回放预算，
不是原常驻服务所有参数的逐项复制。双方解码活动容量 B=2、GPU KV token 预算
一致；Autellix 额外有一个预备请求槽位（其底层 SGLang 容量 B+K=3，基线为 2），因此仅作为预备请求
机制的敏感性检查，不把它混入 K=0 的等槽位主实验或当作多次重复证据。

公平性要求：同一 GPU 串行执行、同样模型/权重/输入、固定输出 token 数和随机种子、
同一到达 trace、每次新建引擎与空缓存并做相同预热、轮换执行顺序、不删除负收益。
输出文本哈希和输出 token 数另行比较；未满足可比条件时不作收益结论。

## 数据和指标

来源：[ShareGPT 原始数据仓库](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)、
[BFCLv3 官方组织数据](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/blob/main/BFCL_v3_multi_turn_base.json)、
[LATS 作者仓库](https://github.com/andyz245/LanguageAgentTreeSearch)。

准备脚本 `scripts/prepare_sharegpt_replay.py` 从 94145 条记录中按 seed=42 打乱，
保留 2–8 调用且完整历史能放进 2048 上下文的记录，选择前 12 条，共 48 次调用。
不截断记录轮次，但将输出长度设为原回答 token 数与 32 的较小值，ignore_eos=true，
总输出 1532 token。后续调用使用数据集参考历史，避免不同生成内容改变对照的输入。
该回放不是答案质量评测，不能代表原始长尾长度分布或实时工具反馈。

`scripts/compare_sglang.py` 使用每秒 0.1 和 1.0 程序的泊松到达，各做 3 次配对重复。
每个重复使用相同的 12 个程序，但改变顺序和到达 trace；因此它衡量的是重复性，
不是大样本泛化。报告程序响应时间、P95/P99、输出吞吐和程序级秒/token。
单线程程序的论文指标应使用“完整程序响应时间 / 总输出 token”，包括初始接纳及
驱动延迟；本脚本明确输出 `paper_sequential_token_latency`。既有通用 benchmark 的
`program_token_latency_seconds` 采用调用关键路径累计时间，不应在单线程实验中
无说明地当作同一指标。TTFT、平均请求延迟和 token/s 都不能替代程序级指标。

原始数据、生成提示、权重、日志均不纳入 Git；保留源文件 SHA256、筛选规则、
选中 ID、trace 哈希、版本和完整结果，保证可审查。只有 12 程序时 P95/P99
接近最大值，不能据此声明长尾 SLO 达标。

## 主实验结果（2026-11-15）

12 次运行全部完成。6 个配对的输入 trace、模型、依赖版本、共同资源配置和输出
token 数一致，每个配对的 48 次调用输出文本哈希全部一致。每次运行输出 1532 token。
没有丢弃负收益或失败样本；本轮没有失败运行。

下表的百分比为各配对相对变化的算术平均，不是两个总体均值相除；括号为三次
观察到的范围，不是置信区间。延迟增加、吞吐下降均表示更差。

| 程序到达率 | 平均程序响应增加 | 程序级秒/token 增加 | P95 程序响应增加 | 输出吞吐下降 |
| --- | ---: | ---: | ---: | ---: |
| 0.1 /秒 | 25.76%（6.88–35.40%） | 25.02%（3.42–36.48%） | 32.42%（10.99–54.42%） | 3.71%（0.31–5.97%） |
| 1.0 /秒 | 42.41%（28.47–49.93%） | 53.45%（28.33–66.71%） | 30.74%（24.54–37.65%） | 20.20%（18.04–21.98%） |

| 到达率 | 重复 | 原生 SGLang 平均程序秒 | ATLAS 平均程序秒 |
| --- | --- | ---: | ---: |
| 0.1 | 0 | 11.043 | 14.952 |
| 0.1 | 1 | 10.626 | 11.358 |
| 0.1 | 2 | 12.521 | 16.905 |
| 1.0 | 0 | 32.082 | 48.100 |
| 1.0 | 1 | 37.647 | 48.363 |
| 1.0 | 2 | 33.293 | 49.555 |

**当前主实验没有测到正向收益。** 它否定了以此前小型内部 FCFS 表格推断
“优于原生 SGLang”的说法，但不能否定论文在 A100、大模型和真实长尾负载上的结果。
此次输出截短、程序筛选和原生算子实现改变了 workload 与成本结构，必须保留限制。

合理的下一步是先定位开销，再扩大实验：当前代码存在逐步 CUDA 同步、SQLite
状态更新、trace 写入和同步 KV 打包/传输；这些是值得测量的候选开销，尚未通过
消融证实各自贡献。不能直接宣称某一项已经解释全部退化，也不应先调整数据筛选
或参数来追求正数。优化后应重跑固定输入和输出一致性检查，再独立验证新负载。

服务器证据：`/mnt/masfiles/runs/autellix/paper-20261115/comparison.json`、
`summary.json`、`comparison.log`；本地副本：`outputs/paper-sglang-20261115/`。
数据与筛选证据：服务器 `/mnt/masfiles/datasets/autellix-paper-20261115/` 下的
`sharegpt-replay.manifest.json`。源 ShareGPT SHA256：
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`；回放文件 SHA256：
`23829e8bd82646c2395f4ae8b618841b895418821fd043fdfcb1ad1e7a3c9fdd`。

### K=1 补测与完成状态

高到达率 1.0/秒下另完成 1 个配对（2 次运行）。原生 SGLang 平均程序响应
32.945 秒，ATLAS 48.821 秒：响应增加 48.19%，程序级秒/token 增加 71.09%，
P95 增加 21.60%，输出吞吐下降 16.24%。48 次调用的输出文本哈希、1532 个
输出 token 及共同输入/配置一致。该组有额外的 Autellix 预备槽位且仅一次重复，
不能当作等槽位主实验，也不能由它确定 K=1 相对 K=0 的因果收益。
证据为同目录 `default-reserve-comparison.json`、`default-reserve-summary.json`
与 `default-reserve.log`。本次共完成 14 次真实 GPU 运行，没有测到正向收益。
`provenance-sha256.txt` 记录模型、分词器、BFCL 文件和执行脚本哈希；
`compare-main-executed.py`、`compare-reserve.py` 保存实际运行入口，
`executed-source.tar.gz` 保存最终同步前的服务器代码快照。后续版本的参数检查和
状态目录记录有所补充，因此不能把最终 Git 文件逐字等同于已运行入口。

已完成修正：增加真正绕过 Autellix 调度钩子的 SGLang 基线、固定输入回放、
配对顺序轮换、输出一致性校验、论文单线程程序指标，以及保留负收益的汇总。
以前内部 FCFS 与原生 SGLang 的混淆已在历史验证文档中标明。
最终本地验证 `python -m unittest discover -s tests -q`：110 项，105 通过、
5 项 GPU 专用检查跳过；真实 GPU 证据来自上述服务器实验，不能用本地跳过替代。
直接运行 pytest 时本机全局第三方 Hydra/ANTLR 插件加载冲突，因此采用仓库
使用的 unittest 测试入口。实验退出时仍有 multiprocessing 信号量清理警告；
不能把本轮成功退出等同于长期运行无资源泄漏。

尚未完成：消除性能退化、精确定位各项开销、原论文模型与完整 agent 轨迹、
跨 GPU 性能/扩展性、完整消融及长期稳定性。当前适合称为可运行的方法原型，
不能称为完整论文复现或已验证的 SGLang 性能优化。

## 复跑入口

在项目独立 SGLang 环境中设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`TORCHDYNAMO_DISABLE=1`、`OMP_NUM_THREADS=4`，然后运行：

```bash
python scripts/prepare_sharegpt_replay.py --source /path/to/sharegpt.json \
  --model /path/to/model --output /path/to/replay.json
python scripts/compare_sglang.py --model /path/to/model \
  --workload /path/to/replay.json --output /path/to/comparison.json
```

通用 benchmark 也支持 `--policies sglang-native,atlas`；要作等容量对照应显式
设置 `--overprovision 0`。该工具不自动替代完整的多轮配对实验。
