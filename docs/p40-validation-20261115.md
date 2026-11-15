# P40 服务器部署与验证（2026-11-15）

已在 3 张 Tesla P40 的服务器上完成真实 SGLang 推理验证。当前常驻服务使用
GPU 2、ATLAS、SmolLM2-135M-Instruct，监听 `127.0.0.1:8000`，显存约 845 MiB。
服务通过 nohup 保持运行；未配置机器重启后的自动启动。

最后在线复核：2026-11-15 16:19（Asia/Shanghai），`/health` 返回 `ok`，
`/v1/models` 返回持久盘中的 SmolLM2-135M-Instruct。这里的模型指服务器推理模型，
与执行部署工作的 Codex 对话模型无关。结论是**小模型功能基线已完成**，
不是大模型、论文性能或生产环境的完整验收。

## 服务器资源快照

| 项目 | 当前实测 |
| --- | --- |
| 系统 | Ubuntu 22.04.5 LTS，Linux 5.15，容器环境 |
| 宿主 CPU | 2 × Xeon E5-2680 v4，28 物理核 / 56 逻辑线程 |
| 容器 CPU 配额 | `cpu.max=1800000 100000`，相当于 18 核 CPU 时间，并非独享 56 核 |
| 容器内存限制 | 72 GiB（77309411328 字节），检查时可用约 68 GiB，无 swap |
| GPU 0 | Tesla P40，报告总显存 23040 MiB，当前空闲 |
| GPU 1 | Tesla P40，报告总显存 24576 MiB，当前空闲 |
| GPU 2 | Tesla P40，报告总显存 24576 MiB，服务占用约 845 MiB |
| 持久盘 | `/mnt/masfiles` 网络文件系统，500G，总可用约 102G |
| 系统盘 | overlay 100G，可用约 86G，不视作持久存储 |
| 服务配置 | GPU 2 单副本，FP32，ATLAS/paper，N=8，K=1，活动容量 B=2 |
| 长度/预算 | context=512，max_total_tokens=2048，mem_fraction_static=0.35 |

显存、内存和磁盘可用量会随任务变化。3 张卡并未合并成一个模型副本；
当前服务只有一个模型进程使用 GPU 2，历史“双副本”测试发生在同一张 GPU。

## 路径与环境

- 源码：`/mnt/masfiles/code/autellix-20261115`
- 独立环境：`/mnt/masfiles/env/autellix-sglang-py310`
- 模型：`/mnt/masfiles/models/SmolLM2-135M-Instruct`
- 原始日志/JSON：`/mnt/masfiles/runs/autellix/20261115`
- 服务日志/PID：上述目录中的 `server.log` / `server.pid`
- 当前运行实例 SQLite：`/tmp/autellix-p40`，避开网络持久盘；更新后的启动脚本
  在下次启动默认新建 `/tmp/autellix-p40-XXXXXX`，不影响当前进程
- 本地结果副本：`outputs/server-p40-20261115/`

Python 3.10.12，SGLang 0.4.9.post6，Torch 2.7.1+cu126，Transformers 4.54.0，
驱动 550.120。依赖使用项目锁文件、清华 PyPI 镜像安装，额外安装 matplotlib
用于测试，以及系统 python3.10-dev 编译头文件。原服务器的其他实验环境未覆盖。

P40 算力为 6.1。原有 Torch 2.8/cu128 环境的基础 GPU 运算失败；新环境的基础
GPU 运算及真实生成均成功。SGLang 默认拒绝 sm75 以下硬件，新增的显式
`autellix_legacy_gpu` 模式采用 FP32/native PyTorch，详情见后端 README。
该模式属于本项目实验性适配，不代表上游正式支持 P40。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 常规回归 | 101 项：96 通过、5 跳过 |
| GPU 专项 | 5 项：2 通过、3 个非本次场景跳过 |
| 强制抢占恢复 | 7 次 swap_out / swap_in；恢复输出与不中断贪心输出一致 |
| 双独立副本 | 同卡双进程路由、取消、清理通过；不是 TP 测试 |
| N=8、K=1 专项 | 1 通过、4 跳过；reserve/refill/resume_cached 均实际发生 |
| HTTP | 普通生成、SSE、客户端流式一致性、程序继承、中文会话 ID、清理均通过 |
| 常驻服务 | 健康检查 ok，GPU 0/1 在实验结束后均无显存占用 |

常规回归中跳过的 5 项是 GPU 专项；后续分别启用其中适用于 SGLang 的场景。
这些运行复用同一个测试文件，不能把次数相加当作新增的独立测试。
GPU 专项没有执行 vLLM 超长 prefill、vLLM TP 或独立 CUDA KV 微基准；
SGLang 的真实换入换出由抢占恢复测试验证，不能替代所有 KV 布局的覆盖。

证据对应：`cpu-tests-final.log`、`gpu-tests.log`、`gpu-window8-reserve1.log`、
`http.json`、`live-demo.json`、`benchmark.json` 位于上述服务器结果目录。
本地常规回归副本名为 `cpu-tests.log`；生成日志按仓库规则不进入 Git。

## 已知不足与后续验收清单

记录不足不等于修复不足，下表未完成项不得标记为通过。

| 状态 | 不足 / 待检验项 | 后续完成条件 |
| --- | --- | --- |
| 已修复并验证 | 取消测试直接轮询 pending，可能早于 SQLite 清理完成 | 已改用 wait_idle；本地 3 项 IPC 测试及服务器回归通过 |
| 已修复启动脚本 | 固定状态路径可能复用旧会话 | 每次默认创建新本地目录；脚本语法和两次启动参数检查通过，当前服务未重启 |
| 已知未修复 | 部分退出路径出现 semaphore 资源清理警告 | 定位资源所有者；正常退出、取消、启动失败等重复运行不再出现警告，并检查残留资源 |
| 待验证 | P40 原生算子与独立实现的数值等价 | 固定模型权重/输入，比较 Transformers FP32 的 logits 与贪心输出；现有中断一致性不能代替该验证 |
| 待验证 | 其他模型、中文任务、质量和长输出 | 为目标模型建立任务集、预期答案与指标；本次只测 135M 小模型，不支持宣称业务质量达标 |
| 待验证 | 长上下文、高并发、显存压力和长期稳定性 | 扩展长度/并发扫描，覆盖 OOM、CPU KV 预算耗尽、断连/取消及长时间负载，核对错误和资源回收 |
| 待验证 | 调度策略性能差异 | 独占资源、重复多种到达率/负载，报告延迟分布、方差及对照；当前单次两程序样本只能作冒烟测试 |
| 待验证 / 未支持 | 跨 GPU 独立副本；SGLang TP/PP/DP | 独立副本需多卡实测；TP/PP/DP 当前明确禁止，需先实现再验证，不能仅修改参数 |
| 待开发 | 自动重启、故障恢复、鉴权、限流、监控 | 完成对应实现和故障注入/恢复验收；当前仅回环监听、SSH 隧道访问及 nohup 常驻 |
| 维护限制 | AST 运行时补丁依赖内部源码，非上游 P40 支持 | 保持版本锁定；升级前检查补丁匹配范围并重跑数值/生成/抢占测试；未知模型不视作兼容 |
| 待完善 | 完整环境可复现与模型来源固定 | 记录模型 revision/文件哈希、安装源和系统依赖，干净环境重建；仅依赖包版本锁不足以证明可复现 |

优先顺序：资源清理警告与数值对照 → 目标模型/压力稳定性 → 可重复性能实验 →
生产运维能力。无需等待更换 GPU 才能开展前几项，但不得把 P40 原生算子性能当作
常规 SGLang 优化内核性能。

实际请求：`What is the capital of France?` 返回 `The capital of France is Paris.`，
8 个输出 token，端到端 0.815 秒。天空颜色问题生成 48 token 用时 3.714 秒，
因设置的 token 上限而截断。小模型只用于功能验证，长回答可能包含事实错误。

## 小型 DAG 工作负载

使用仓库 `examples/real_workload.json`，2 个程序、6 次调用，每个策略输出
81 token。GPU 1，FP32，context=512，max_running_requests=2，max_total_tokens=2048，
N=8、K=1，random_seed=0；每种策略单次运行。

| 策略 | 输出 token/s | 平均程序响应秒 |
| --- | ---: | ---: |
| FCFS | 14.41 | 4.005 |
| MLFQ | 14.85 | 4.444 |
| PLAS | 17.43 | 3.312 |
| ATLAS | 18.18 | 3.128 |

样本很小，实验期间其他 GPU 有验证/服务启动活动，共享主机 CPU；没有重复试验、
置信区间或上游原生 SGLang 对照，不能由此证明性能提升或复现论文结果。
上述 FCFS 与 ATLAS 都安装了本项目调度钩子，故不能由这张表声称优于原生 SGLang。
后续新增的直接对照协议与结果见 [论文对照审计](autellix-paper-audit.md)。
这也是原生 PyTorch 兼容路径，不能与优化内核性能混为一谈。
部分测试/benchmark 在退出时报告 multiprocessing semaphore 清理警告，原始日志已保留；
请求、会话和 GPU 显存的清理检查均通过，此警告尚未单独修复。

## 调用与重启

在本机建立 SSH 隧道，按提示输入密码：

```bash
ssh -p 46644 -L 18000:127.0.0.1:8000 root@cn-north-b.ssh.damodel.com
```

然后本机客户端使用 `http://127.0.0.1:18000`。

```python
from autellix.runtime import InferenceClient
with InferenceClient("http://127.0.0.1:18000") as client:
    result = client.chat([{"role": "user", "content": "What is the capital of France?"}],
                         temperature=0, max_tokens=32)
    print(result["choices"][0]["message"]["content"])
```

服务停止后，在服务器执行：

```bash
cd /mnt/masfiles/code/autellix-20261115
AUTELLIX_DEVICES=2 nohup bash scripts/serve_p40.sh \
  > /mnt/masfiles/runs/autellix/20261115/server.log 2>&1 < /dev/null &
echo $! > /mnt/masfiles/runs/autellix/20261115/server.pid
```

不要在已有服务运行时重复启动。模型/上下文/并发数需要更大规模验证；现有
14B/32B 权重没有在这条 FP32 单卡路径上运行。
