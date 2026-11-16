# P40 初次部署与验证（2026-11-15）

本页保存初次部署记录；后续源码修正、原模型实验与性能分析见[论文复现实验报告](autellix-paper-audit.md)。
三卡运行与当前部署配置见[三卡大模型实验](three-gpu-experiment.md)。

已在 3 张 Tesla P40 的服务器上完成真实 SGLang 推理验证。当时常驻服务使用
GPU 2、ATLAS、SmolLM2-135M-Instruct，监听 `127.0.0.1:8000`，显存约 845 MiB。
服务通过 nohup 保持运行；未配置机器重启后的自动启动。

初次部署在线复核：2026-11-15 16:19（Asia/Shanghai），`/health` 返回 `ok`，
`/v1/models` 返回持久盘中的 SmolLM2-135M-Instruct。这里的模型指服务器推理模型，
与执行部署工作的 Codex 对话模型无关。本次完成了小模型功能基线。

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

- 源码：`/mnt/masfiles/autellix/code`
- 独立环境：`/mnt/masfiles/autellix/env/sglang`
- 模型：`/mnt/masfiles/autellix/models/SmolLM2-135M-Instruct`
- 原始日志/JSON：`/mnt/masfiles/autellix/runs/20261115`
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
该模式由本项目适配。

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
上述专项运行复用同一组测试用例。
GPU 专项没有执行 vLLM 超长 prefill、vLLM TP 或独立 CUDA KV 微基准；
SGLang 的真实换入换出由抢占恢复测试验证。

证据对应：`cpu-tests-final.log`、`gpu-tests.log`、`gpu-window8-reserve1.log`、
`http.json`、`live-demo.json`、`benchmark.json` 位于上述服务器结果目录。
本地常规回归副本名为 `cpu-tests.log`；生成日志按仓库规则不进入 Git。

## 已知不足与后续验收清单

以下为初次部署时的待办快照。

| 状态 | 不足 / 待检验项 | 后续完成条件 |
| --- | --- | --- |
| 已修复并验证 | 取消测试直接轮询 pending，可能早于 SQLite 清理完成 | 已改用 wait_idle；本地 3 项 IPC 测试及服务器回归通过 |
| 已修复启动脚本 | 固定状态路径可能复用旧会话 | 每次默认创建新本地目录；脚本语法和两次启动参数检查通过，本轮服务更新见最新实验报告 |
| 已知未修复 | 部分退出路径出现 semaphore 资源清理警告 | 定位资源所有者；正常退出、取消、启动失败等重复运行不再出现警告，并检查残留资源 |
| 待验证 | P40 原生算子与独立实现的数值等价 | 固定模型权重/输入，比较 Transformers FP32 的 logits 与贪心输出 |
| 待验证 | 其他模型、中文任务、质量和长输出 | 为目标模型建立任务集、预期答案与指标 |
| 待验证 | 长上下文、高并发、显存压力和长期稳定性 | 扩展长度/并发扫描，覆盖 OOM、CPU KV 预算耗尽、断连/取消及长时间负载，核对错误和资源回收 |
| 已补测 | 调度策略性能差异 | 原生 SGLang 配对、三组短回放重复、原长度回放和论文 8B 模型结果见最新实验报告 |
| 已补测 / 待开发 | 跨 GPU 独立副本；SGLang TP/PP/DP | 本轮两张物理 GPU 的独立副本测试已通过；TP/PP/DP 尚未实现 |
| 待开发 | 自动重启、故障恢复、鉴权、限流、监控 | 完成对应实现和故障注入/恢复验收；当前仅回环监听、SSH 隧道访问及 nohup 常驻 |
| 维护限制 | AST 运行时补丁依赖内部源码，非上游 P40 支持 | 保持版本锁定；升级前检查补丁匹配范围并重跑数值/生成/抢占测试 |
| 待完善 | 完整环境可复现与模型来源固定 | 记录模型 revision/文件哈希、安装源和系统依赖，干净环境重建 |

优先顺序：资源清理警告与数值对照 → 目标模型/压力稳定性 → 可重复性能实验 →
服务运维能力。

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

本表是 2 程序单次运行，所有策略均安装 Autellix 钩子，使用 P40 原生 PyTorch 算子。
原生 SGLang 配对结果见[论文复现实验报告](autellix-paper-audit.md)。
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
cd /mnt/masfiles/autellix/code
AUTELLIX_DEVICES=2 nohup bash scripts/serve_p40.sh \
  > /mnt/masfiles/autellix/runs/20261115/server.log 2>&1 < /dev/null &
echo $! > /mnt/masfiles/autellix/runs/20261115/server.pid
```

不要在已有服务运行时重复启动。本轮已增加 Llama-3.1-8B FP16 和长回放；最新运行配置见[实验报告](autellix-paper-audit.md)。
