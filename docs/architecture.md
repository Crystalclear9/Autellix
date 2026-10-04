# 架构与论文方法

[文档导航](README.md) · [项目首页](../README.md)

本文档中的命令均从仓库根目录执行。

## 实现范围

- **PLAS / ATLAS**：按已完成调用累计服务时间或最长已观察服务路径，为新调用确定初始优先级。在线 ATLAS 对应算法 1，不要求用户提供调用 DAG。
- **程序状态表**：跨副本共享服务时间、等待时间、活动调用、线程信息及到达和完成时间。
- **分级 FIFO 队列**：时间片耗尽后降级，逐调用结合程序历史判断防饥饿；同时移动多个调用时保留当前队列顺序。
- **抢占与恢复**：保存生成进度，通过 GPU/CPU KV 传输恢复执行；批量打包采用 PyTorch CUDA 操作和锁页内存。
- **多步调度与预备请求**：每个窗口冻结调度顺序，运行 N 个解码步；活动请求提前结束时，由已准备的 GPU 请求补位。
- **程序感知路由**：输入不超过 2048 token 时分配给负载最小副本，更长输入保持程序的引擎亲和性。
- **有状态客户端**：自动复用程序会话，标注调用和线程 ID，支持普通响应与流式响应。

已在小模型上验证真实生成、抢占恢复结果一致、预备请求补位、HTTP、路由、取消和会话清理。双副本测试是在同一张 GPU 上运行两个独立进程；多 GPU 张量并行尚未完成硬件验证。未复现论文的大规模实验和性能提升倍数。

## 架构与维护入口

```text
InferenceClient / HTTP 请求 / 直接 Python 调用
                     |
             InferenceEngine 协调器
        分词、会话、路由、Future、取消与 IPC
              /                    \
      vLLM 工作进程             SGLang 工作进程
      原生执行器 + 调度钩子     原生执行器 + 调度钩子
              \                    /
             SQLite 程序表与各副本 trace
```

| 文件 | 维护职责 |
| --- | --- |
| `autellix/runtime/policy.py` | 配置校验、跨进程程序表、服务历史、时间片、防饥饿和 FIFO |
| `autellix/runtime/window.py` | N 步窗口、活动/预备请求集合、补位和容量截断 |
| `autellix/runtime/engine.py` | 副本进程、路由、Future、取消、错误和会话回收 |
| `autellix/runtime/client.py` | 会话注解、HTTP 请求、SSE 解析和客户端生命周期 |
| `autellix/runtime/server.py` | HTTP 路由、请求 ID 预留、流式输出和断连处理 |
| `autellix/runtime/swap.py` | vLLM KV 打包、锁页内存传输与映射校验 |
| `integrations/vllm/paper.py` | vLLM 默认模式接纳、显存预留及窗口继续执行 |
| `integrations/vllm/backend.py`、`executor.py` | 引擎初始化、计时、请求转换及 TP worker 钩子 |
| `integrations/sglang/scheduler.py` | SGLang 队列接纳、重排、补位和生命周期钩子 |
| `integrations/sglang/swap.py`、`reserve.py`、`pressure.py` | 主机备份、radix 锁、预备 KV 和显存压力处理 |
| `autellix/runtime/benchmark.py` | 真实文本工作负载、依赖驱动与实测指标 |

## 模拟器

`autellix.core`、`autellix.frontend` 和 `autellix.cli` 是独立的 CPU 模拟功能，不能用于真实模型推理。旧的 `AutellixClient`、`AsyncMultiLLMEngine` 及根包导入路径保留兼容用途。

```bash
python -m autellix.cli compare --workload figure2 --policies fcfs,mlfq,plas
python -m autellix.cli paper-preset --preset workload-analysis --dataset tests/fixtures/tiny_workload.jsonl
```

在线 `atlas` 对应算法 1，显式父节点版本 `atlas-dag` 对应公式 2。模拟窗口按模型 tick 而非真实解码步计数；执行、缓存命中率及交换开销都是成本模型。模拟器中的 `vllm` / `vllm-opt` 标签和 paper-style presets 不代表真实后端测量或论文实验复现。

## 目录

| 路径 | 内容 |
| --- | --- |
| `autellix/runtime/` | 真实推理协调、策略、客户端、HTTP 与 KV 传输 |
| `integrations/` | 固定版本的 vLLM / SGLang 后端 |
| `autellix/core/` | 调度模拟器和执行成本模型 |
| `autellix/frontend/`、`autellix/experiments/` | 模拟客户端、数据导入与实验工具 |
| `autellix/*.py` | CLI 和旧导入兼容接口 |
| `scripts/`、`requirements/` | 安装、验证脚本和依赖锁文件 |
| `examples/`、`tests/` | 真实推理示例、回归测试和小型输入 |
| `cuda/` | 真实 KV 传输微基准 |

本地模型、PDF、临时文件、缓存、安装元数据和生成输出不提交。保留仍被使用的模型与环境；测试和验证输出可以重新生成。

## 程序级服务历史

以下时间均为真实 runtime 记录的模型执行秒数，不是输出 token 数，也不包括程序在外部工具中的耗时。

- **PLAS**：新调用继承程序已完成调用的累计服务时间；调用完成时，将本次完整执行时间加到程序历史中。
- **在线 ATLAS**：接纳时记录继承值 `h`；完成时执行 `program.service = max(program.service, h + executed)`。
- **MLFQ**：新调用从最高队列开始，防饥饿不使用程序历史。
- **FCFS**：按入队顺序调度，不做时间片降级或防饥饿提升。

例如，历史服务时间为 2 秒，两个并行调用均继承 2 秒，分别执行 3 秒和 1 秒。两者完成后的 ATLAS 历史为 `max(2, 2+3, 2+1)=5` 秒，PLAS 为 `2+3+1=6` 秒。之后到来的调用继承新的历史；尚未完成的其他调用不会追溯修改其接纳时的继承值。

这是一种在线历史估计。即使输入程序是 DAG，它也不等于在服务端维护精确的父子图；精确父节点路径只存在于模拟器 `atlas-dag` 或工作负载驱动的指标计算中。

## 调度窗口与防饥饿

优先级值按半开区间 `[low, high)` 分箱，队列编号越小优先级越高。每个调用维护队列位置、FIFO 顺序号、剩余时间片、累计执行/等待时间，以及用于防饥饿的局部时间窗口。

在窗口边界刷新状态：时间片耗尽则降至下一队列，最低队列中的调用移到该队列尾部；当前不在最高队列的调用若满足 `(program_wait + call_wait_window) / (program_service + call_run_window) >= beta`，则提升到最高队列并重置局部窗口。实现使用很小的正数保护零分母。完整执行和等待统计不因提升而清零。

窗口从当前可用请求按队列/FIFO 顺序取最多 B+K 个；前 B 个活动，其余为预备请求。预备请求准备 KV 后保持驻留，活动调用提前结束时补位。整个集合耗尽可以提前开启新窗口，不必让 GPU 空等到原定边界。内存或 token 预算限制可以截断接纳，不能跳过放不下的高优先级请求去接纳其后的低优先级请求。

多步冻结的是请求选择，不是全部 CPU 工作：IPC、输出、token/block 元数据和部分内存检查仍会执行。vLLM 预留未来 KV 槽位；SGLang 依赖 radix 锁和按优先级释放内存，二者不是逐行相同的实现。

## 请求从提交到完成

1. 客户端选择自动或显式会话，附加调用与线程 ID；直接 Python 接口由调用者管理程序会话。
2. 协调器通过真实 tokenizer 获取输入 token；使用已有 `input_ids` 则跳过分词。
3. 短输入按当前未完成调用数选择最少负载副本；长输入复用该程序绑定的健康副本，必要时选择新的健康副本。
4. 协调器记录活动信息、增加负载、创建 Future，经队列发送接纳命令。
5. 工作进程将请求接入调度器，读取程序历史；选中后由真实模型执行器运行。
6. 增量输出由 IPC 返回，Future 保存一个有界、合并的最新输出队列；HTTP 层据此形成流式文本。
7. 完成时更新程序历史，返回最终结果；协调器减少负载、删除活动信息，若会话已关闭且无剩余调用，则回收程序。

负载是未完成调用数量，而不是 GPU 利用率、剩余 token 预测或显存估计。流式队列会合并进度，因此一个数据块不一定对应一个 token。

## 状态表及所有权

| 表 | 作用 | 主要写入方 |
| --- | --- | --- |
| `programs` | 程序服务、等待、活动数、关闭标志、最近到达/完成时间 | 协调器和后端调度器 |
| `requests` | 已被调度器接纳的内部请求到程序的对应关系 | 后端调度器，错误路径由协调器幂等清理 |
| `contexts` | 跨进程转交调用、线程、引擎和用户 metadata | 协调器写入，后端取走 |
| `activity` | 从提交起可观测的活动请求、等待/运行状态 | 协调器记录与清理，调度器更新时间 |
| `results` | SGLang 完成统计的跨 IPC 通道交接 | 调度器写入，后端读取回收 |

SQLite 使用 WAL 和事务，各连接内还使用线程锁。`activity` 中的已提交调用可能尚未进入调度器，不能要求它与 `requests` 或程序 active 数在每个瞬间一一相同。

会话 ID、HTTP 请求 ID、内部请求 ID、调用 ID 和线程 ID 各有用途，不应混用。SGLang 内部请求 ID 编码程序与外部请求信息；trace 关联应保留 metadata 中的原始标识。状态数据库只用于当前运行协调，不保存完整模型 KV 或待执行消息队列。

## 论文、实现选择与扩展

| 类别 | 当前定位 |
| --- | --- |
| 核心方法 | PLAS/在线 ATLAS、分级 FIFO、时间片、防饥饿、程序路由、KV 交换、多步与预备请求 |
| 实现选择 | SQLite 协调、PyTorch CUDA 打包、具体默认阈值、固定版本钩子、HTTP 子集 |
| 额外扩展 | SGLang 适配、CPU 模拟器、真实小模型验证工具 |
| 未建立的结论 | 作者 CUDA kernel 等价、全部硬件/模型支持、论文规模性能复现、生产可靠性 |

论文方法与代码的对应关系用于理解设计，不是新的审计结论。改变实现选择时，应验证是否保留方法所需的不变量，并同步更新[开发文档](development.md)。
