# P40 服务器迁移（2026-11-17）

新连接为 `ssh -p 47541 root@cn-north-b.ssh.damodel.com`。
旧服务器的 Autellix 目录约 133 GB，整体迁入新实例；已完成的实验结果保留原始硬件
记录，新采集和后续配对在新实例上继续。

## 迁移验收

2026-11-17 15:41:22（Asia/Shanghai）完成逐文件校验：主目录 164949 个文件/链接、
142286847353 字节文件内容，运行状态目录 148 项、379074711 字节，SHA256 比较
均无差异。完整清单保存在服务器 `logs/migration-47541-*-files.json`，本地副本位于
`outputs/archive/p40/migration-47541/`。[精简验收结果](results/server-migration-20261117.json)记录
清单哈希、硬件、检查点及 GPU 验证结果。

新实例补装系统依赖 `libnuma1`，SGLang/Torch 与 vLLM/Torch 两套环境均成功导入。
146 项回归中 139 项通过、7 项跳过。GPU 0 完成 2.125 GiB KV 精确往返，临时 GPU
内存峰值 16 MiB；GPU 1 完成 K=0/1 两种抢占恢复检查，恢复生成与连续生成一致，
程序会话全部回收。

旧实例保存了 25 个 BFCL 程序和 1 个 LATS 程序后停止实验进程。新实例已按下表
续跑：GPU 0 采集 BFCL，GPU 1 采集 LATS，随后执行配对回放及 Mixed。
采集前的完整程序哈希保存在 `logs/migration-47541-capture-prefixes.json`。
最终采集得到 64 个 BFCL 程序和 6 个 LATS 程序，迁移前的 25 个 BFCL 程序与
1 个 LATS 程序的内容哈希全部匹配。复核记录保存在 `runs/paper-suite/capture-resume-verification.json`。

| 项目 | 新实例 |
| --- | --- |
| GPU | 2×Tesla P40，各 23040 MiB，合计 45 GiB 可见显存 |
| CPU 配额 / 内存 | 12 核 / 48 GiB |
| 系统 / 驱动 | Ubuntu 22.04.5 / 550.120 |
| 统一项目入口 | `/root/workspace/autellix/` |
| 代码、数据、历史日志和实验归档 | 20 GB 持久工作区内的对应子目录 |
| Python 环境 | `autellix/env/` → `/root/damodel-tmp/autellix/env/` |
| Llama-3.1-8B 权重 | `autellix/models/Meta-Llama-3.1-8B-Instruct/` → 50 GB 数据盘 |
| Qwen3-14B/32B、SmolLM2 权重 | `autellix/models/` → `/root/autellix-system/models/` |
| SQLite、IPC 与运行 trace | `autellix/tmp/` → `/tmp/autellix-runtime/` |
| 历史绝对路径入口 | `/mnt/masfiles/autellix` → `/root/workspace/autellix` |

目录链接保留统一入口，历史计划中的模型、数据、环境和状态路径继续可用。
完整迁移包含四个模型、两套 Python 环境、数据与上游固定版本、全部结果和归档。
迁移记录、文件内容 SHA256 和校验结果存放在 `autellix/logs/`。

采集器的 `--resume` 按完整程序检查点接续 BFCL/LATS；已保存程序及实际输入、
生成长度、工具结果继续保留。迁移时尚未完成的程序从该程序开头重新采集。
`run_paper_followups.py --gpu` 标识原实验流程，`--device` 指定新实例的物理卡：

```bash
cd /root/workspace/autellix/code
python_bin=/root/workspace/autellix/env/sglang/bin/python
"$python_bin" -u scripts/run_paper_followups.py --gpu 2 --device 0
"$python_bin" -u scripts/run_paper_followups.py --gpu 1 --device 1
```

两条命令分别安排 BFCL 与 LATS/Mixed，可并行执行。两种调度策略在同一物理卡上
使用相同输入 trace 与资源配置进行配对。此前三张卡上的 ShareGPT 和 PP=3
实验已完成，结果见[论文工作负载报告](paper-workload-experiment.md)。
