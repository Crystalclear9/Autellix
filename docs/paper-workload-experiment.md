# Autellix 论文复现尝试：原始工作负载与 SGLang 对照

本轮围绕 [Autellix v1 §6](https://arxiv.org/html/2502.13965v1) 的实验方法，
使用完整程序、Poisson 到达和程序级 token 延迟，比较现有 PLAS/ATLAS 与原生
SGLang 调度。推理在三张 P40 上执行；每个配对点的两种策略使用同一张物理卡。

## 数据与工作流

| 工作负载 | 来源与处理 |
| --- | --- |
| ShareGPT | 原始 94,145 个片段合并为 50,142 个完整对话；按整段上下文筛选，不裁剪回答长度；从 128 个程序的固定池中按程序采样 |
| BFCL v3 | 论文发表前的官方提交 `8bf675cc5f92e99971ba1f6bbb15bcf1d1d2eb1c`，2025-02-12；采集 64 个 `multi_turn_base` 任务，使用函数定义和官方状态式工具模拟器 |
| LATS | 官方提交 `853d81614607dd27433faf17c7b0a7d660f95d22`，2024-07-30；运行官方树搜索和提示模板，使用 HotpotQA 训练集问题及配套文档 |
| Mixed | 在三个工作负载类别中等概率选择，再在对应类别中抽取完整程序 |

BFCL 采集直接调用固定版本 `LlamaFCHandler._format_prompt` 和函数定义预处理，
保留 JSON 工具调用及工具返回后的继续生成。温度为 0.001，每次最多生成 4096
token，并受 32768 上下文约束。官方循环在工具步数超过 20 时停止，采集器相应使用
每轮最多 21 次调用。另有 `prepare_bfcl_replay.py` 用官方参考调用执行工具，生成
64 个任务、365 次调用的参考轨迹，供数据准备检查。

采集完成后，用同一固定版本的 `multi_turn_checker` 检查各轮工具状态与返回结果，
并执行 `multi_turn_irrelevance_checker`。官方参考动作的 64 个任务自检全部通过；
实际模型采集的正确率单独汇总，性能回放保留采集到的全部程序。

LATS 采集使用 30 次迭代、5 个扩展样本、1 个评价样本、温度 1.0、每次最多
100 个输出 token，保留官方深度限制、UCT、rollout、评价和反思。每个五样本组
并发提交，组间保留执行依赖。Search/Lookup 在 HotpotQA 随题提供的文档上运行，
Finish 按原始答案判断；采集使用本地 Llama，提示长短版本按服务模型的 tokenizer
选择。搜索树耗尽时结束，单程序调用预算为 400。采集清单记录每个程序的结束原因。

ShareGPT 固定参考对话历史；BFCL/LATS 固定实际采集的输入、输出长度和依赖关系，
然后分别回放两种调度。后续提示使用固定历史，实际生成文本差异不会改变下一次调用
的工作量。调度器在程序到达、调用依赖满足时才接收请求。

数据与代码来源：[ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)、
[BFCL 固定版本](https://github.com/ShishirPatil/gorilla/tree/8bf675cc5f92e99971ba1f6bbb15bcf1d1d2eb1c/berkeley-function-call-leaderboard)、
[LATS 固定版本](https://github.com/andyz245/LanguageAgentTreeSearch/tree/853d81614607dd27433faf17c7b0a7d660f95d22)、
[HotpotQA 固定版本](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)。

## 配对设置

- 模型：Llama-3.1-8B-Instruct，FP16；每个引擎使用一张 P40。
- 主实验：B=4、context=32768、KV 容量 32768 token、静态显存比例 0.9。
- 两组均使用 radix 缓存、P40 PyTorch 算子，关闭 overlap 和 chunked prefill。
- 原生组运行 SGLang 自身调度；PLAS/ATLAS 使用 N=8、K=1、β=8。
- 时间片为 0.64、1.28、2.56、5.12、10.24、20.48、40.96 秒；
  历史分箱边界为 0、20.48、40.96、81.92、163.84、327.68、655.36、∞ 秒。
- 每组重启引擎并预热；比较输入 trace、逐调用长度、模型、软件版本和资源参数。
- 三张卡同时承担独立实验。GPU 0 的 ECC 开启，GPU 1/2 关闭，配对时不换卡。

论文要求各策略使用相同最大 batch size；P40 上主实验固定 B=4，另外进行 B=8
对照。论文的硬件为 A100，模型还包括 70B 和 180B；本轮复现其 8B 模型系列、
工作负载与程序级测量方法。P40 的时间尺度、KV 预算和后端算子按上述配置固定。

## 实验矩阵

| 实验 | 程序数 | 到达率（程序/秒） | GPU | 策略 |
| --- | ---: | --- | --- | --- |
| ShareGPT 主曲线，seed=2026 | 每点 48 | 0.004 / 0.008 / 0.016 | 分别 0 / 1 / 2 | native、PLAS |
| ShareGPT 第二种子，seed=2027 | 24 | 0.008 | 0 | native、PLAS |
| ShareGPT 策略消融，seed=2027 | 8 | 0.016 | 0 | native、PLAS、内部 FCFS、MLFQ |
| ShareGPT B=8，seed=2027 | 8 | 0.016 | 0 | native、PLAS |
| BFCL，seed=2026 | 每点 48 | 0.02 / 0.05 / 0.1 | 2 | native、PLAS |
| LATS，seed=2026 | 6 | 0.004 | 1 | native、ATLAS |
| Mixed，seed=2026 | 12 | 0.008 | 1 | native、ATLAS |
| ShareGPT 三卡共享模型，seed=2026 | 48 | 0.048 | 0、1、2，PP=3 | native、PLAS |

ShareGPT 主曲线每组均为 297 次调用、83,455 个输出 token；三个到达窗口分别
约为 14,614、7,307、3,654 秒。第二种子为 95 次调用、28,324 个输出 token。
这轮将到达窗口延长，并通过单引擎配对消除上一轮两个长程序被分到同一卡的路由变量。

三卡共享模型组使用同一个 Llama-3.1-8B，按 10/11/11 层切分，B=12（每个
pipeline microbatch 为 4）、KV 预算 327680 token、静态显存比例 0.9，attention
分块预算为 16777216 个元素，两组使用相同设置。
该组把三张卡的显存用于同一个模型及其 KV，检验更接近论文大显存条件时的调度效果。
全部单卡工作结束后再运行此组。

主曲线中的两个最长程序分别有 34/48 次调用、15,959/12,171 个输出 token，
合计占全部输出的 33.71%。两者最大提示分别为 20,010/23,466 token，合计
43,476，超过单卡的 32,768 KV 槽位。即使尾部只剩这两个程序，单卡也无法同时
保留它们的完整长上下文；三卡共享模型组用于检验扩大 KV 空间后的表现。

## 指标

同时报告平均/P95/P99 程序响应、论文程序级 token 延迟、输出吞吐、到达期间的
完成程序数、尾部排空时间、按到达顺序分四组的延迟，以及十个时间点的在途程序数。
顺序程序使用程序响应/总输出 token；LATS/Mixed 的论文指标使用关键路径响应/
所有线程的总输出 token。

SLO 表取实测负载点中满足给定平均程序级 token 延迟的最大到达率。到达率、
整段完成速率和包含排空时间的 token/s 分别记录。每个主曲线点对应的 GPU 编号
保留在结果中，第二种子和批大小消融单独汇总。

## 运行记录

长对话首次触发 KV 换出时，旧实现的整层 `stack` 额外申请 2.13 GiB，超过剩余
显存。现已改为按层、按 token 的 16 MiB 分块传输，换入使用相同方式。
P40 上 17,408 token、64 个 K/V buffer、共 2.125 GiB 的往返逐块相等，
临时 GPU 内存峰值为 16 MiB。失败日志保存在 `paper-suite-swap-oom/`，
中负载 PLAS 配对从头重跑。原生组不经过此换出路径。

重跑至第 258 次调用时，又发现低优先级恢复项的缓存锁占用了高优先级新调用
所需空间。修复后，新调用已有的缓存前缀先被保护，接纳失败时按优先级释放后缀
的临时锁并重新接纳；被延后的调用保留主机 KV。已晋升为活动请求的低优先级
预备项也会让出空间。GPU 小模型测试覆盖 K=0/1、512 个 KV 槽位容纳两条各
300 token 的输入，实际触发换出、恢复和接纳重试，恢复后的生成与连续生成一致。
失败与中断日志保存在 `paper-suite-prefill-capacity/`。

`run_sglang_matrix.py --resume` 保留同一计划、后端版本和输入 trace 的已完成项。
高负载原生 SGLang 的完整结果保留；受影响的 PLAS 项从头重跑。

正式实验正在执行，结果完成后补入此节。原始输出写入
`/mnt/masfiles/autellix/runs/paper-suite/`。首次启动发现 SQLite 状态位于网络持久盘，
在正式配对完成前停止并重启；启动日志保存在 `paper-suite-networkfs-start/`，
正式运行使用指向本机磁盘的 `autellix/tmp/`。

目录组织和重跑入口见[服务器目录](server-layout.md)。
